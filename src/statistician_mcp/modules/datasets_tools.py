from __future__ import annotations

import io
import json
from typing import Any, Literal

import pandas as pd
from mcp.server.fastmcp import FastMCP

from statistician_mcp import envelope
from statistician_mcp.datasets import DatasetStore
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.url_fetch import fetch_tabular_from_url
from statistician_mcp.utils.formulas import FormulaError, evaluate_expression
from statistician_mcp.workspace import get_current_workspace_id

MAX_INLINE_CSV_BYTES = 2 * 1024 * 1024


def register_dataset_tools(mcp: FastMCP, store: DatasetStore) -> None:
    @mcp.tool()
    @envelope.tool("load_dataset_from_csv")
    def load_dataset_from_csv(csv_text: str, name: str = "dataset") -> dict[str, Any]:
        """Load inline CSV text (up to 2 MB) into a new dataset and return its handle
        with a column profile. Use for small-to-medium data pasted directly by the user."""
        if len(csv_text.encode("utf-8")) > MAX_INLINE_CSV_BYTES:
            raise ValidationError(
                "csv_text exceeds the 2 MB inline limit",
                hint="use load_dataset_from_url for larger files",
            )
        df = _read_csv_text(csv_text)
        info = store.create(get_current_workspace_id(), df, name)
        return envelope.ok_envelope(
            info.to_dict(), meta={"dataset": info.handle, "n_rows_used": info.n_rows}
        )

    @mcp.tool()
    @envelope.tool("load_dataset_from_url")
    async def load_dataset_from_url(url: str, name: str = "dataset") -> dict[str, Any]:
        """Fetch a CSV, Excel, or Parquet file from a public https URL (up to 50 MB) into
        a new dataset. Use when the data lives at a link rather than being pasted inline."""
        df = await fetch_tabular_from_url(url)
        info = store.create(get_current_workspace_id(), df, name)
        return envelope.ok_envelope(
            info.to_dict(), meta={"dataset": info.handle, "n_rows_used": info.n_rows}
        )

    @mcp.tool()
    @envelope.tool("list_datasets")
    def list_datasets() -> dict[str, Any]:
        """List every dataset handle in the current workspace with its name, row count,
        and column count. Use to see what data is already loaded before loading more."""
        infos = store.list(get_current_workspace_id())
        return envelope.ok_envelope([i.to_dict() for i in infos])

    @mcp.tool()
    @envelope.tool("describe_dataset")
    def describe_dataset(handle: str) -> dict[str, Any]:
        """Return the full column profile (dtype, missing count, example value, and
        summary stats) for a dataset. Use before analyzing a dataset you haven't inspected."""
        info = store.get_info(get_current_workspace_id(), handle)
        return envelope.ok_envelope(
            info.to_dict(), meta={"dataset": handle, "n_rows_used": info.n_rows}
        )

    @mcp.tool()
    @envelope.tool("sample_dataset_rows")
    def sample_dataset_rows(
        handle: str, n: int = 10, mode: Literal["head", "random"] = "head", seed: int = 0
    ) -> dict[str, Any]:
        """Return the first N rows (mode='head') or a random sample of N rows
        (mode='random') from a dataset. Use to eyeball actual raw values."""
        df = store.get_dataframe(get_current_workspace_id(), handle)
        n_used = max(1, min(n, len(df))) if len(df) else 0
        sample = df.head(n_used) if mode == "head" else df.sample(n=n_used, random_state=seed)
        rows = json.loads(sample.to_json(orient="records", date_format="iso"))
        return envelope.ok_envelope({"rows": rows}, meta={"dataset": handle, "n_rows_used": n_used})

    @mcp.tool()
    @envelope.tool("transform_dataset")
    def transform_dataset(
        handle: str,
        op: Literal["filter", "select", "rename", "derive", "stack", "unstack"],
        expression: str | None = None,
        columns: list[str] | None = None,
        mapping: dict[str, str] | None = None,
        new_column: str | None = None,
        id_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
        index: list[str] | None = None,
        pivot_column: str | None = None,
        value_column: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Apply one safe, declarative transformation to a dataset and return a NEW
        dataset handle — the original is left untouched. Supported ops: filter (boolean
        expression), select (keep columns), rename (column mapping), derive (new column
        from a restricted expression), stack (wide-to-long via id_vars/columns), unstack
        (long-to-wide via index/pivot_column/value_column). Expressions may reference
        only existing columns, numeric literals, + - * /, comparisons, & | ~, and the
        functions log/log10/sqrt/exp/abs."""
        workspace_id = get_current_workspace_id()
        df = store.get_dataframe(workspace_id, handle)
        result = _apply_transform(
            df,
            op,
            expression=expression,
            columns=columns,
            mapping=mapping,
            new_column=new_column,
            id_vars=id_vars,
            var_name=var_name,
            value_name=value_name,
            index=index,
            pivot_column=pivot_column,
            value_column=value_column,
        )
        info = store.create(workspace_id, result, name or f"{op}({handle})")
        return envelope.ok_envelope(
            info.to_dict(),
            meta={"dataset": info.handle, "n_rows_used": info.n_rows, "source_dataset": handle},
        )

    @mcp.tool()
    @envelope.tool("delete_dataset")
    def delete_dataset(handle: str) -> dict[str, Any]:
        """Permanently delete a dataset handle and free its storage. Use for cleanup once
        a dataset is no longer needed."""
        store.delete(get_current_workspace_id(), handle)
        return envelope.ok_envelope({"deleted": handle})


def _read_csv_text(csv_text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise ValidationError(f"could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValidationError("CSV produced zero rows")
    return df


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _safe_eval(df: pd.DataFrame, expression: str) -> pd.Series:
    try:
        return evaluate_expression(df, expression)
    except FormulaError as exc:
        raise ValidationError(
            str(exc),
            hint="allowed: column names, + - * /, comparisons, & | ~, log/log10/sqrt/exp/abs",
        ) from exc


def _apply_transform(
    df: pd.DataFrame,
    op: str,
    *,
    expression: str | None,
    columns: list[str] | None,
    mapping: dict[str, str] | None,
    new_column: str | None,
    id_vars: list[str] | None,
    var_name: str,
    value_name: str,
    index: list[str] | None,
    pivot_column: str | None,
    value_column: str | None,
) -> pd.DataFrame:
    if op == "filter":
        if not expression:
            raise ValidationError("op='filter' requires 'expression'")
        mask = _safe_eval(df, expression)
        if mask.dtype != bool:
            raise ValidationError("filter expression must evaluate to a boolean series")
        return df[mask].reset_index(drop=True)

    if op == "select":
        if not columns:
            raise ValidationError("op='select' requires 'columns'")
        _require_columns(df, columns)
        return df[columns].copy()

    if op == "rename":
        if not mapping:
            raise ValidationError("op='rename' requires 'mapping'")
        _require_columns(df, list(mapping.keys()))
        return df.rename(columns=mapping)

    if op == "derive":
        if not expression or not new_column:
            raise ValidationError("op='derive' requires 'expression' and 'new_column'")
        result = df.copy()
        result[new_column] = _safe_eval(df, expression)
        return result

    if op == "stack":
        if not id_vars:
            raise ValidationError("op='stack' requires 'id_vars'")
        _require_columns(df, id_vars)
        if columns:
            _require_columns(df, columns)
        return df.melt(
            id_vars=id_vars, value_vars=columns, var_name=var_name, value_name=value_name
        )

    if op == "unstack":
        if not index or not pivot_column or not value_column:
            raise ValidationError(
                "op='unstack' requires 'index', 'pivot_column', and 'value_column'"
            )
        _require_columns(df, [*index, pivot_column, value_column])
        pivoted = df.pivot_table(
            index=index, columns=pivot_column, values=value_column, aggfunc="first"
        )
        pivoted.columns = [str(c) for c in pivoted.columns]
        return pivoted.reset_index()

    raise ValidationError(f"unsupported op '{op}'")
