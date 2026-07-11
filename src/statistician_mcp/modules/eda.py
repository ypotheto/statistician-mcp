from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from scipy import stats as sp_stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore, get_dataframe_for_analysis
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats.assumptions import check_normality
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id


def register_eda_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("summarize_columns")
    def summarize_columns(handle: str) -> dict[str, Any]:
        """Return descriptive statistics for every column of a dataset: n, mean, sd,
        quartiles, skew, and kurtosis for numeric columns; level counts for categorical
        columns. Use this first, before running any specific analysis."""
        df = _get_df(store, handle)
        numeric_summary, categorical_summary = _summarize(df)
        interpretation = _summarize_interpretation(numeric_summary, categorical_summary)
        return envelope.ok_envelope(
            {"numeric": numeric_summary, "categorical": categorical_summary},
            interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("plot_distribution")
    def plot_distribution(handle: str, column: str) -> dict[str, Any]:
        """Render a 3-panel plot (histogram, boxplot, normal Q-Q plot) for one numeric
        column. Use to visually inspect a column's shape before choosing a test."""
        df = _get_df(store, handle)
        data = _numeric_column(df, column)

        fig, axes = new_figure(1, 3, figsize=(12.0, 4.0))
        axes[0].hist(data, bins="auto", color="#4c72b0", edgecolor="white")
        axes[0].set_title("Histogram")
        axes[0].set_xlabel(column)

        axes[1].boxplot(data, tick_labels=[column])
        axes[1].set_title("Boxplot")

        (osm, osr), (slope, intercept, r) = sp_stats.probplot(data, dist="norm")
        axes[2].scatter(osm, osr, s=12, color="#4c72b0")
        axes[2].plot(osm, slope * osm + intercept, color="#c44e52", linewidth=1.5)
        axes[2].set_title(f"Normal Q-Q (r={r:.3f})")

        fig.suptitle(f"Distribution of '{column}' (n={len(data)})")
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename=f"{column}_distribution.png",
            data=render_png(fig),
            media_type="image/png",
        )
        return envelope.ok_envelope(
            {"n": len(data), "qq_correlation": float(r)},
            artifacts=[record],
            meta={"dataset": handle, "n_rows_used": len(data)},
        )

    @mcp.tool()
    @envelope.tool("test_normality")
    def test_normality(handle: str, column: str) -> dict[str, Any]:
        """Test whether a numeric column is normally distributed (Shapiro-Wilk for
        n<=5000, Anderson-Darling above that). Use before choosing a parametric test."""
        df = _get_df(store, handle)
        data = _numeric_column(df, column)
        result = check_normality(data, column)
        return envelope.ok_envelope(
            {"n": len(data)},
            assumptions=[result.to_dict()],
            interpretation=_normality_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(data)},
        )

    @mcp.tool()
    @envelope.tool("detect_outliers")
    def detect_outliers(handle: str, column: str) -> dict[str, Any]:
        """Flag potential outliers in a numeric column via IQR fences (1.5xIQR beyond
        Q1/Q3) and Grubbs' test for a single most-extreme value. Use to sanity-check
        raw data before analysis."""
        df = _get_df(store, handle)
        series = df[column]
        if column not in df.columns:
            raise ColumnNotFoundError(column, list(map(str, df.columns)))
        if not pd.api.types.is_numeric_dtype(series):
            raise ValidationError(f"column '{column}' is not numeric")

        non_null = series.dropna()
        data = non_null.to_numpy(dtype=float)
        q1, q3 = np.percentile(data, [25, 75]) if len(data) else (float("nan"), float("nan"))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (non_null < lower) | (non_null > upper)
        iqr_flagged = non_null.index[iqr_mask].tolist()

        grubbs = _grubbs_test(data)
        return envelope.ok_envelope(
            {
                "iqr_fences": {"lower": float(lower), "upper": float(upper)},
                "iqr_flagged_row_indices": iqr_flagged,
                "grubbs_test": grubbs,
            },
            interpretation=_outlier_interpretation(iqr_flagged, grubbs),
            meta={"dataset": handle, "n_rows_used": len(data)},
        )

    @mcp.tool()
    @envelope.tool("compute_correlations")
    def compute_correlations(
        handle: str,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> dict[str, Any]:
        """Compute a correlation matrix (Pearson or Spearman) across a dataset's numeric
        columns, with a heatmap artifact, flagging pairs with |r| > 0.7. Use to spot
        collinearity or candidate relationships before modeling."""
        df = _get_df(store, handle)
        numeric_df = df.select_dtypes(include="number")
        if columns:
            missing = [c for c in columns if c not in numeric_df.columns]
            if missing:
                raise ColumnNotFoundError(missing[0], list(map(str, numeric_df.columns)))
            numeric_df = numeric_df[columns]
        if numeric_df.shape[1] < 2:
            raise ValidationError(
                "need at least two numeric columns to compute correlations",
                hint=f"numeric columns available: {list(map(str, numeric_df.columns))}",
            )

        corr = numeric_df.corr(method=method)
        cols = list(map(str, corr.columns))
        strong_pairs = [
            {"a": cols[i], "b": cols[j], "r": float(corr.iloc[i, j])}
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
            if abs(corr.iloc[i, j]) > 0.7
        ]

        fig, ax = new_figure(figsize=(max(4.0, 0.6 * len(cols) + 2), max(4.0, 0.6 * len(cols) + 2)))
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{method.title()} correlation")
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="correlation_heatmap.png",
            data=render_png(fig),
            media_type="image/png",
        )

        interpretation = _correlation_interpretation(strong_pairs, method)
        return envelope.ok_envelope(
            {"matrix": corr.round(6).to_dict(), "strong_pairs": strong_pairs},
            artifacts=[record],
            interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": len(numeric_df)},
        )

    @mcp.tool()
    @envelope.tool("plot_scatter")
    def plot_scatter(
        handle: str,
        x: str,
        y: str,
        group_column: str | None = None,
        overlay: Literal["none", "ols", "loess"] = "none",
    ) -> dict[str, Any]:
        """Scatter-plot one numeric column against another, optionally colored by a
        grouping column and/or overlaid with an OLS or LOESS smoother. Use to inspect
        the relationship between two variables before modeling."""
        if x == y:
            raise ValidationError("x and y must be different columns")
        df = _get_df(store, handle)
        _require_columns(df, [x, y, *([group_column] if group_column else [])])

        fig, ax = new_figure(figsize=(6.5, 5.0))
        if group_column:
            for level, subset in df.groupby(group_column, dropna=False):
                ax.scatter(subset[x], subset[y], s=15, label=str(level))
            ax.legend(title=group_column, fontsize=8)
        else:
            ax.scatter(df[x], df[y], s=15, color="#4c72b0")

        pair = df[[x, y]].dropna()
        if overlay == "ols" and len(pair) >= 2:
            slope, intercept = np.polyfit(pair[x], pair[y], 1)
            xs = np.linspace(pair[x].min(), pair[x].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="#c44e52", linewidth=1.5)
        elif overlay == "loess" and len(pair) >= 2:
            sorted_pair = pair.sort_values(x)
            smoothed = lowess(sorted_pair[y], sorted_pair[x], frac=0.5)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#c44e52", linewidth=1.5)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename=f"{x}_vs_{y}_scatter.png",
            data=render_png(fig),
            media_type="image/png",
        )
        return envelope.ok_envelope(
            {"n": len(pair)}, artifacts=[record], meta={"dataset": handle, "n_rows_used": len(pair)}
        )

    @mcp.tool()
    @envelope.tool("plot_time_series")
    def plot_time_series(
        handle: str, column: str, time_column: str | None = None
    ) -> dict[str, Any]:
        """Plot a run chart of one column against row order or a datetime column. Use
        to check for trends, shifts, or cycles before setting up an SPC chart."""
        df = _get_df(store, handle)
        _require_columns(df, [column, *([time_column] if time_column else [])])

        ordered = df.sort_values(time_column) if time_column else df
        x_values = ordered[time_column] if time_column else range(len(ordered))

        fig, ax = new_figure(figsize=(8.0, 4.0))
        ax.plot(x_values, ordered[column], marker="o", markersize=3, linewidth=1, color="#4c72b0")
        ax.set_xlabel(time_column or "row order")
        ax.set_ylabel(column)
        ax.set_title(f"Run chart of '{column}'")
        if time_column:
            fig.autofmt_xdate()
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename=f"{column}_run_chart.png",
            data=render_png(fig),
            media_type="image/png",
        )
        return envelope.ok_envelope(
            {"n": len(ordered)},
            artifacts=[record],
            meta={"dataset": handle, "n_rows_used": len(ordered)},
        )

    @mcp.tool()
    @envelope.tool("crosstab")
    def crosstab(
        handle: str,
        row: str,
        col: str,
        normalize: Literal["none", "row", "col", "all"] = "none",
    ) -> dict[str, Any]:
        """Build a contingency table of two categorical columns, optionally normalized
        to row/column/overall percentages. Use to inspect the relationship between two
        categorical variables before a proportion or chi-square test."""
        df = _get_df(store, handle)
        _require_columns(df, [row, col])

        counts = pd.crosstab(df[row], df[col])
        results: dict[str, Any] = {"counts": _stringify_index(counts).to_dict()}
        if normalize != "none":
            norm_arg = {"row": "index", "col": "columns", "all": "all"}[normalize]
            pct = pd.crosstab(df[row], df[col], normalize=norm_arg) * 100
            results["percent"] = _stringify_index(pct.round(2)).to_dict()

        return envelope.ok_envelope(
            results, meta={"dataset": handle, "n_rows_used": int(counts.to_numpy().sum())}
        )


def _get_df(store: DatasetStore, handle: str) -> pd.DataFrame:
    return get_dataframe_for_analysis(store, get_current_workspace_id(), handle)


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(df, [column])
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(f"column '{column}' is not numeric")
    return series.dropna().to_numpy(dtype=float)


def _stringify_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def _summarize(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    numeric_summary: dict[str, Any] = {}
    categorical_summary: dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        name = str(col)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            non_null = series.dropna()
            n = int(non_null.count())
            numeric_summary[name] = {
                "n": n,
                "n_missing": int(series.isna().sum()),
                "mean": float(non_null.mean()) if n else None,
                "sd": float(non_null.std(ddof=1)) if n > 1 else None,
                "min": float(non_null.min()) if n else None,
                "q1": float(non_null.quantile(0.25)) if n else None,
                "median": float(non_null.median()) if n else None,
                "q3": float(non_null.quantile(0.75)) if n else None,
                "max": float(non_null.max()) if n else None,
                "skew": float(non_null.skew()) if n > 2 else None,
                "kurtosis": float(non_null.kurtosis()) if n > 3 else None,
            }
        else:
            non_null = series.dropna()
            counts = non_null.value_counts().head(20)
            categorical_summary[name] = {
                "n": int(non_null.count()),
                "n_missing": int(series.isna().sum()),
                "n_levels": int(non_null.nunique()),
                "top_levels": {str(k): int(v) for k, v in counts.items()},
            }
    return numeric_summary, categorical_summary


def _summarize_interpretation(numeric: dict[str, Any], categorical: dict[str, Any]) -> str:
    parts = [f"{len(numeric)} numeric and {len(categorical)} categorical column(s) summarized."]
    missing_cols = [
        name
        for name, s in {**numeric, **categorical}.items()
        if s["n_missing"] > 0
    ]
    if missing_cols:
        parts.append(f"Missing values found in: {', '.join(missing_cols)}.")
    skewed = [
        name for name, s in numeric.items() if s.get("skew") is not None and abs(s["skew"]) > 1
    ]
    if skewed:
        parts.append(f"Notably skewed (|skew|>1): {', '.join(skewed)}.")
    return " ".join(parts)


def _normality_interpretation(result: Any) -> str:
    if result.status == "pass":
        return f"{result.detail}. No evidence against normality; parametric tests are appropriate."
    if result.status == "warn":
        return (
            f"{result.detail}. Borderline evidence against normality; "
            "interpret parametric results with caution."
        )
    return f"{result.detail}. Evidence against normality; consider a nonparametric alternative."


def _grubbs_test(x: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    n = len(x)
    if n < 3:
        return {"applicable": False, "reason": "Grubbs' test requires at least 3 values"}
    mean, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return {"applicable": False, "reason": "zero variance"}
    deviations = np.abs(x - mean)
    idx = int(np.argmax(deviations))
    g_stat = float(deviations[idx] / sd)
    t_crit = float(sp_stats.t.ppf(1 - alpha / (2 * n), n - 2))
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return {
        "applicable": True,
        "candidate_index": idx,
        "candidate_value": float(x[idx]),
        "g_statistic": g_stat,
        "g_critical": float(g_crit),
        "is_outlier": bool(g_stat > g_crit),
        "alpha": alpha,
    }


def _outlier_interpretation(iqr_flagged: list[int], grubbs: dict[str, Any]) -> str:
    parts = [f"{len(iqr_flagged)} value(s) flagged by the 1.5xIQR fence."]
    if grubbs.get("applicable"):
        if grubbs["is_outlier"]:
            parts.append(
                f"Grubbs' test flags the most extreme value ({grubbs['candidate_value']:.4g}) "
                f"as a statistically significant outlier (G={grubbs['g_statistic']:.3f} > "
                f"critical {grubbs['g_critical']:.3f})."
            )
        else:
            parts.append("Grubbs' test does not find the most extreme value significant.")
    return " ".join(parts)


def _correlation_interpretation(strong_pairs: list[dict[str, Any]], method: str) -> str:
    if not strong_pairs:
        return f"No {method} correlations exceed |r|=0.7."
    described = ", ".join(f"{p['a']}~{p['b']} (r={p['r']:.2f})" for p in strong_pairs)
    return f"Strong {method} correlations (|r|>0.7): {described}."
