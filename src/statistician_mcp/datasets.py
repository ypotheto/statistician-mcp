from __future__ import annotations

import io
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from statistician_mcp.errors import DatasetNotFoundError, QuotaExceededError, ValidationError
from statistician_mcp.storage import StorageBackend
from statistician_mcp.workspace import DEFAULT_QUOTAS

# A dataset file can vanish between an existence/enumeration step and the actual
# read, if another concurrent call deletes it in between (see get_dataframe,
# get_info, list below). FileNotFoundError is the obvious case; PermissionError is
# included because Windows enforces mandatory file-sharing locks and can raise it
# when one thread's read races a concurrent unlink of the same file (a threaded
# stress test reproduced this) -- Linux, the actual production target, has no such
# restriction (unlinking an open file is always safe there), but treating both as
# "this file is gone" is correct and harmless on every platform.
_TRANSIENT_READ_ERRORS = (FileNotFoundError, PermissionError)


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # numeric | categorical | datetime | text
    n_missing: int
    example: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetInfo:
    handle: str
    name: str
    created_at: float
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetInfo:
        return cls(
            handle=raw["handle"],
            name=raw["name"],
            created_at=raw["created_at"],
            n_rows=raw["n_rows"],
            n_columns=raw["n_columns"],
            columns=[ColumnProfile(**c) for c in raw["columns"]],
        )


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        non_null = series.dropna()
        example = None if non_null.empty else str(non_null.iloc[0])
        extra: dict[str, Any] = {}

        if pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            dtype = "categorical"
            extra = {"levels": sorted(str(v) for v in non_null.unique())[:20]}
        elif pd.api.types.is_numeric_dtype(series):
            dtype = "numeric"
            if not non_null.empty:
                extra = {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                }
        else:
            n_unique = int(non_null.nunique())
            if 0 < n_unique <= max(20, int(0.2 * len(non_null))):
                dtype = "categorical"
                extra = {
                    "n_levels": n_unique,
                    "levels": sorted(str(v) for v in non_null.unique())[:20],
                }
            else:
                dtype = "text"

        profiles.append(
            ColumnProfile(
                name=str(col), dtype=dtype, n_missing=n_missing, example=example, extra=extra
            )
        )
    return profiles


class DatasetStore:
    def __init__(self, backend: StorageBackend, max_cache_entries: int = 16) -> None:
        self._backend = backend
        self._cache: OrderedDict[tuple[str, str], pd.DataFrame] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_entries = max_cache_entries

    def _dataset_path(self, workspace_id: str, handle: str) -> str:
        return f"workspaces/{workspace_id}/datasets/{handle}.parquet"

    def _meta_path(self, workspace_id: str, handle: str) -> str:
        return f"workspaces/{workspace_id}/datasets/{handle}.meta.json"

    def create(self, workspace_id: str, df: pd.DataFrame, name: str) -> DatasetInfo:
        if len(df) > DEFAULT_QUOTAS.max_rows_per_dataset:
            raise ValidationError(
                f"dataset has {len(df)} rows, exceeding the "
                f"{DEFAULT_QUOTAS.max_rows_per_dataset}-row limit",
                hint="aggregate or sample the data before loading it",
            )
        if len(self.list(workspace_id)) >= DEFAULT_QUOTAS.max_datasets:
            raise QuotaExceededError(
                f"workspace already has {DEFAULT_QUOTAS.max_datasets} datasets",
                hint="delete unused datasets before loading more",
            )

        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
        if len(data) > DEFAULT_QUOTAS.max_bytes_per_dataset:
            raise ValidationError(
                f"dataset is {len(data)} bytes, exceeding the "
                f"{DEFAULT_QUOTAS.max_bytes_per_dataset}-byte limit"
            )

        handle = "ds_" + uuid.uuid4().hex[:8]
        info = DatasetInfo(
            handle=handle,
            name=name,
            created_at=time.time(),
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=profile_columns(df),
        )
        self._backend.write_bytes(self._dataset_path(workspace_id, handle), data)
        self._backend.write_bytes(
            self._meta_path(workspace_id, handle), json.dumps(info.to_dict()).encode("utf-8")
        )
        self._cache_put((workspace_id, handle), df)
        return info

    def get_dataframe(self, workspace_id: str, handle: str) -> pd.DataFrame:
        # A concurrent miss on the same uncached key can redundantly re-read the
        # backend (the cache-hit check and the eventual _cache_put aren't one
        # atomic section) -- accepted as a rare, harmless cost. What the lock
        # guarantees is that self._cache itself is never corrupted by concurrent
        # compound mutations (check + move_to_end + evict).
        key = (workspace_id, handle)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        path = self._dataset_path(workspace_id, handle)
        try:
            raw = self._backend.read_bytes(path)
        except _TRANSIENT_READ_ERRORS as exc:
            raise DatasetNotFoundError(handle) from exc
        df = pd.read_parquet(io.BytesIO(raw))
        self._cache_put(key, df)
        return df

    def get_info(self, workspace_id: str, handle: str) -> DatasetInfo:
        path = self._meta_path(workspace_id, handle)
        try:
            raw = self._backend.read_bytes(path)
        except _TRANSIENT_READ_ERRORS as exc:
            raise DatasetNotFoundError(handle) from exc
        return DatasetInfo.from_dict(json.loads(raw))

    def list(self, workspace_id: str) -> list[DatasetInfo]:
        # Enumerating then reading each file is inherently a two-step, non-atomic
        # sequence -- a concurrent delete() on another thread can remove a file
        # between the two steps. That's a benign race (the dataset genuinely is
        # gone), not an error, so a vanished file is skipped rather than raised.
        prefix = f"workspaces/{workspace_id}/datasets/"
        infos = []
        for p in self._backend.list(prefix):
            if not p.endswith(".meta.json"):
                continue
            try:
                raw = self._backend.read_bytes(p)
            except _TRANSIENT_READ_ERRORS:
                continue
            infos.append(DatasetInfo.from_dict(json.loads(raw)))
        return sorted(infos, key=lambda i: i.created_at)

    def delete(self, workspace_id: str, handle: str) -> None:
        path = self._dataset_path(workspace_id, handle)
        if not self._backend.exists(path):
            raise DatasetNotFoundError(handle)
        self._backend.delete(path)
        self._backend.delete(self._meta_path(workspace_id, handle))
        with self._cache_lock:
            self._cache.pop((workspace_id, handle), None)

    def _cache_put(self, key: tuple[str, str], df: pd.DataFrame) -> None:
        with self._cache_lock:
            self._cache[key] = df
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)


# Storage/inspection tools (load, describe, sample, list) may work with datasets up
# to DEFAULT_QUOTAS.max_rows_per_dataset (500k) since they're cheap, near-O(n) single
# passes. Compute-heavy analysis tools (regression, ANOVA, control charts, Gauge R&R,
# ...) are far more expensive per row -- O(n log n) or worse, or per-row Python loops
# (Nelson rules) -- so they share this tighter cap instead. Every analysis module
# should resolve its dataset through this helper rather than calling
# `store.get_dataframe()` directly, so the cap can't be silently bypassed by a new
# module forgetting to add its own check.
MAX_ANALYSIS_ROWS = 200_000


def get_dataframe_for_analysis(
    store: DatasetStore, workspace_id: str, handle: str, max_rows: int = MAX_ANALYSIS_ROWS
) -> pd.DataFrame:
    df = store.get_dataframe(workspace_id, handle)
    if len(df) > max_rows:
        raise ValidationError(
            f"dataset has {len(df)} rows, exceeding the {max_rows}-row analysis limit",
            hint="aggregate or sample the dataset with transform_dataset first",
        )
    return df
