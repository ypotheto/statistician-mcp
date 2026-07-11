from __future__ import annotations

import io
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from statistician_mcp.errors import DatasetNotFoundError, QuotaExceededError, ValidationError
from statistician_mcp.storage import StorageBackend
from statistician_mcp.workspace import DEFAULT_QUOTAS


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
        key = (workspace_id, handle)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self._dataset_path(workspace_id, handle)
        if not self._backend.exists(path):
            raise DatasetNotFoundError(handle)
        df = pd.read_parquet(io.BytesIO(self._backend.read_bytes(path)))
        self._cache_put(key, df)
        return df

    def get_info(self, workspace_id: str, handle: str) -> DatasetInfo:
        path = self._meta_path(workspace_id, handle)
        if not self._backend.exists(path):
            raise DatasetNotFoundError(handle)
        return DatasetInfo.from_dict(json.loads(self._backend.read_bytes(path)))

    def list(self, workspace_id: str) -> list[DatasetInfo]:
        prefix = f"workspaces/{workspace_id}/datasets/"
        infos = [
            DatasetInfo.from_dict(json.loads(self._backend.read_bytes(p)))
            for p in self._backend.list(prefix)
            if p.endswith(".meta.json")
        ]
        return sorted(infos, key=lambda i: i.created_at)

    def delete(self, workspace_id: str, handle: str) -> None:
        path = self._dataset_path(workspace_id, handle)
        if not self._backend.exists(path):
            raise DatasetNotFoundError(handle)
        self._backend.delete(path)
        self._backend.delete(self._meta_path(workspace_id, handle))
        self._cache.pop((workspace_id, handle), None)

    def _cache_put(self, key: tuple[str, str], df: pd.DataFrame) -> None:
        self._cache[key] = df
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
