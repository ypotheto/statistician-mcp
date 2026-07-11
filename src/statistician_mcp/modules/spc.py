from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore, get_dataframe_for_analysis
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats import capability as stats_capability
from statistician_mcp.stats import control_charts as cc
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id

ChartType = Literal["xbar_r", "xbar_s", "i_mr", "p", "np", "c", "u", "ewma", "cusum"]

# Charts with a single evolving series and constant-or-varying sigma per point;
# Nelson rules apply here. EWMA/CUSUM have their own built-in violation logic
# (time-varying limits / decision interval) and are excluded.
NELSON_ELIGIBLE = {"xbar_r", "xbar_s", "i_mr", "p", "np", "c", "u"}


def register_spc_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("create_control_chart")
    def create_control_chart(
        handle: str,
        chart_type: ChartType,
        value_column: str | None = None,
        subgroup_column: str | None = None,
        nonconforming_column: str | None = None,
        sample_size_column: str | None = None,
        sample_size: int | None = None,
        count_column: str | None = None,
        unit_column: str | None = None,
        target: float | None = None,
        sigma: float | None = None,
        lam: float = 0.2,
        k: float = 0.5,
        h: float = 5.0,
    ) -> dict[str, Any]:
        """Build a Shewhart/EWMA/CUSUM control chart and flag out-of-control points.
        chart_type + required columns: xbar_r/xbar_s need `value_column` +
        `subgroup_column`; i_mr/ewma/cusum need `value_column` only; p needs
        `nonconforming_column` + `sample_size_column`; np needs
        `nonconforming_column` + `sample_size`; c needs `count_column`; u needs
        `count_column` + `unit_column`. Limits are computed from the data unless
        `target`/`sigma` are given (historical limits). Nelson/Western-Electric
        rules 1-8 are applied except for ewma/cusum (which use their own limits)."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        chart, secondary_chart, row_index = _build_chart(
            df,
            chart_type,
            value_column=value_column,
            subgroup_column=subgroup_column,
            nonconforming_column=nonconforming_column,
            sample_size_column=sample_size_column,
            sample_size=sample_size,
            count_column=count_column,
            unit_column=unit_column,
            target=target,
            sigma=sigma,
            lam=lam,
            k=k,
            h=h,
        )

        violations: dict[str, Any]
        if chart_type in NELSON_ELIGIBLE:
            rule_hits = cc.nelson_rules(chart["points"], chart["cl"], chart["sigma"])
            violations = {
                f"rule_{rule}": [row_index[i] for i in idxs]
                for rule, idxs in rule_hits.items()
                if idxs
            }
        else:
            violations = {"decision_interval": [row_index[i] for i in chart.get("violations", [])]}

        record = _render_chart(artifacts, chart_type, chart, secondary_chart)
        n_violations = sum(len(v) for v in violations.values())
        interpretation = (
            f"{n_violations} rule violation(s) flagged."
            if n_violations
            else "No rule violations; process appears in statistical control."
        )
        results = {"chart_type": chart_type, **chart, "violations": violations}
        if secondary_chart is not None:
            results["secondary"] = secondary_chart
        return envelope.ok_envelope(
            results,
            artifacts=[record],
            interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("assess_capability")
    def assess_capability(
        handle: str,
        column: str,
        lsl: float | None = None,
        usl: float | None = None,
        subgroup_size: int | None = None,
    ) -> dict[str, Any]:
        """Cp/Cpk (within-subgroup sigma) and Pp/Ppk (overall sigma) against spec
        limits, DPMO, and process sigma level. Runs a normality check first (with a
        Box-Cox-transformed alternative if it fails) and renders a capability
        histogram with spec/limit lines."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)
        try:
            result = stats_capability.process_capability(
                df[column].to_numpy(), lsl, usl, subgroup_size
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        record = _render_capability_histogram(artifacts, df[column].to_numpy(), lsl, usl, result)
        return envelope.ok_envelope(
            result,
            assumptions=[result["normality"]],
            artifacts=[record],
            interpretation=_capability_interpretation(result),
            meta={"dataset": handle, "n_rows_used": result["n"]},
        )

    @mcp.tool()
    @envelope.tool("run_stability_check")
    def run_stability_check(
        handle: str, value_column: str, cl: float, ucl: float, lcl: float
    ) -> dict[str, Any]:
        """Apply Nelson/Western-Electric rules to new data against EXISTING
        (historical) control limits, without re-deriving limits from this data.
        Use to check whether a process is still in control relative to a baseline."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, value_column)
        points = df[value_column].to_numpy()
        sigma = (ucl - cl) / 3
        if sigma <= 0:
            raise ValidationError("ucl must be greater than cl to derive a sigma unit")

        rule_hits = cc.nelson_rules(points.tolist(), cl, sigma)
        violations = {f"rule_{rule}": idxs for rule, idxs in rule_hits.items() if idxs}
        n_violations = sum(len(v) for v in violations.values())

        return envelope.ok_envelope(
            {"cl": cl, "ucl": ucl, "lcl": lcl, "violations": violations},
            interpretation=(
                f"{n_violations} rule violation(s) against the baseline limits."
                if n_violations
                else "No rule violations; process remains in control vs. the baseline."
            ),
            meta={"dataset": handle, "n_rows_used": len(points)},
        )


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    _require_columns(df, [column])
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValidationError(f"column '{column}' is not numeric")


def _build_chart(
    df: pd.DataFrame,
    chart_type: ChartType,
    *,
    value_column: str | None,
    subgroup_column: str | None,
    nonconforming_column: str | None,
    sample_size_column: str | None,
    sample_size: int | None,
    count_column: str | None,
    unit_column: str | None,
    target: float | None,
    sigma: float | None,
    lam: float,
    k: float,
    h: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[int]]:
    """Returns (primary_chart, secondary_chart_or_None, row_index). For xbar_r/
    xbar_s the primary is the Xbar chart and the secondary is the companion R/S
    chart -- both are rendered, but Nelson rules are only evaluated against the
    primary (standard practice: the R/S chart's own limits already flag dispersion
    problems, and its points are far fewer / not independent trials in the same
    sense Nelson's rules assume)."""
    if chart_type in ("xbar_r", "xbar_s"):
        if not value_column or not subgroup_column:
            raise ValidationError(
                f"chart_type='{chart_type}' requires value_column and subgroup_column"
            )
        _require_numeric(df, value_column)
        _require_columns(df, [subgroup_column])
        groups = list(df.groupby(subgroup_column, sort=False))
        subgroups = [g[value_column].tolist() for _, g in groups]
        row_index = [int(g.index[0]) for _, g in groups]
        fn = cc.xbar_r_limits if chart_type == "xbar_r" else cc.xbar_s_limits
        result = fn(subgroups)
        secondary_key = "r" if chart_type == "xbar_r" else "s"
        return result["xbar"], result[secondary_key], row_index

    if chart_type == "i_mr":
        if not value_column:
            raise ValidationError("chart_type='i_mr' requires value_column")
        _require_numeric(df, value_column)
        result = cc.i_mr_limits(df[value_column].tolist())
        return result["individuals"], result["moving_range"], list(range(len(df)))

    if chart_type == "p":
        if not nonconforming_column or not sample_size_column:
            raise ValidationError(
                "chart_type='p' requires nonconforming_column and sample_size_column"
            )
        _require_columns(df, [nonconforming_column, sample_size_column])
        result = cc.p_chart_limits(
            df[nonconforming_column].astype(int).tolist(),
            df[sample_size_column].astype(int).tolist(),
        )
        return result, None, list(range(len(df)))

    if chart_type == "np":
        if not nonconforming_column or not sample_size:
            raise ValidationError("chart_type='np' requires nonconforming_column and sample_size")
        _require_columns(df, [nonconforming_column])
        result = cc.np_chart_limits(df[nonconforming_column].astype(int).tolist(), sample_size)
        return result, None, list(range(len(df)))

    if chart_type == "c":
        if not count_column:
            raise ValidationError("chart_type='c' requires count_column")
        _require_columns(df, [count_column])
        result = cc.c_chart_limits(df[count_column].astype(int).tolist())
        return result, None, list(range(len(df)))

    if chart_type == "u":
        if not count_column or not unit_column:
            raise ValidationError("chart_type='u' requires count_column and unit_column")
        _require_columns(df, [count_column, unit_column])
        result = cc.u_chart_limits(df[count_column].astype(int).tolist(), df[unit_column].tolist())
        return result, None, list(range(len(df)))

    if chart_type == "ewma":
        if not value_column:
            raise ValidationError("chart_type='ewma' requires value_column")
        _require_numeric(df, value_column)
        result = cc.ewma_chart(df[value_column].tolist(), target, sigma, lam)
        return result, None, list(range(len(df)))

    if not value_column:
        raise ValidationError("chart_type='cusum' requires value_column")
    _require_numeric(df, value_column)
    result = cc.cusum_chart(df[value_column].tolist(), target, sigma, k, h)
    return result, None, list(range(len(df)))


_SECONDARY_TITLES = {"xbar_r": "R chart", "xbar_s": "S chart", "i_mr": "Moving range chart"}


def _plot_chart_panel(ax: Any, chart_type: str, chart: dict[str, Any], title: str) -> None:
    if chart_type == "cusum":
        idx = range(1, len(chart["c_plus"]) + 1)
        ax.plot(idx, chart["c_plus"], marker="o", markersize=3, color="#4c72b0", label="C+")
        ax.plot(idx, chart["c_minus"], marker="o", markersize=3, color="#c44e52", label="C-")
        ax.axhline(
            chart["decision_interval"], color="#55a868", linestyle="--", label="decision interval"
        )
        ax.axhline(0, color="black", linewidth=0.8)
    else:
        points = chart["points"]
        idx = range(1, len(points) + 1)
        ax.plot(idx, points, marker="o", markersize=4, color="#4c72b0")
        ax.axhline(chart["cl"], color="black", linewidth=1, label="CL")
        ucl, lcl = chart["ucl"], chart["lcl"]
        if isinstance(ucl, list):
            ax.plot(idx, ucl, color="#c44e52", linestyle="--", linewidth=1, label="UCL")
            ax.plot(idx, lcl, color="#c44e52", linestyle="--", linewidth=1, label="LCL")
        else:
            ax.axhline(ucl, color="#c44e52", linestyle="--", linewidth=1, label="UCL")
            ax.axhline(lcl, color="#c44e52", linestyle="--", linewidth=1, label="LCL")
    ax.set_xlabel("subgroup / observation")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _render_chart(
    artifacts: ArtifactStore,
    chart_type: str,
    chart: dict[str, Any],
    secondary_chart: dict[str, Any] | None,
) -> dict[str, Any]:
    if secondary_chart is None:
        fig, ax = new_figure(figsize=(8.0, 4.5))
        _plot_chart_panel(ax, chart_type, chart, f"{chart_type} control chart")
    else:
        fig, axes = new_figure(2, 1, figsize=(8.0, 8.0))
        _plot_chart_panel(axes[0], chart_type, chart, f"{chart_type} control chart (Xbar)")
        secondary_title = _SECONDARY_TITLES.get(chart_type, "secondary chart")
        _plot_chart_panel(axes[1], "secondary", secondary_chart, secondary_title)
        fig.tight_layout()

    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename=f"{chart_type}_chart.png",
        data=render_png(fig),
        media_type="image/png",
    )


def _render_capability_histogram(
    artifacts: ArtifactStore,
    data: np.ndarray,
    lsl: float | None,
    usl: float | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    fig, ax = new_figure(figsize=(7.0, 4.5))
    ax.hist(data, bins="auto", color="#4c72b0", edgecolor="white", density=True)
    if lsl is not None:
        ax.axvline(lsl, color="#c44e52", linestyle="--", label="LSL")
    if usl is not None:
        ax.axvline(usl, color="#c44e52", linestyle="--", label="USL")
    ax.axvline(result["mean"], color="black", linewidth=1, label="mean")
    cpk = result["overall"]["cpk"]
    title = f"Capability histogram (Cpk={cpk:.3f})" if cpk is not None else "Capability histogram"
    ax.set_title(title)
    ax.legend(fontsize=8)
    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename="capability_histogram.png",
        data=render_png(fig),
        media_type="image/png",
    )


def _capability_interpretation(result: dict[str, Any]) -> str:
    cpk = result["overall"]["cpk"]
    parts = []
    if cpk is not None:
        if cpk >= 1.33:
            verdict = "capable"
        elif cpk >= 1.0:
            verdict = "marginally capable"
        else:
            verdict = "not capable"
        parts.append(f"Overall Cpk={cpk:.3f} ({verdict}).")
    parts.append(f"DPMO~={result['dpmo']:.0f}, sigma level~={result['sigma_level']:.2f}.")
    if result["normality"]["status"] == "fail":
        parts.append(
            "Data failed the normality check; consider the Box-Cox-transformed indices instead."
        )
    return " ".join(parts)
