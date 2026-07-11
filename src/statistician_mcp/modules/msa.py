from __future__ import annotations

from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore, get_dataframe_for_analysis
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats import gauge_rr as stats_gauge_rr
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id


def register_msa_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("analyze_gauge_rr")
    def analyze_gauge_rr(
        handle: str,
        part_column: str,
        operator_column: str,
        value_column: str,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Crossed Gauge R&R (ANOVA method) for a fully-crossed parts x operators x
        replicates measurement study: variance components, %Contribution,
        %StudyVar, %Tolerance (if `tolerance` given), and ndc (number of distinct
        categories the gauge can resolve). Verdict thresholds: <10% StudyVar
        acceptable, 10-30% marginal, >30% unacceptable. Requires an equal number of
        replicates in every part x operator cell."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_columns(df, [part_column, operator_column, value_column])
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            raise ValidationError(f"column '{value_column}' is not numeric")

        try:
            result = stats_gauge_rr.crossed_gauge_rr(
                df[part_column].tolist(),
                df[operator_column].tolist(),
                df[value_column].tolist(),
                tolerance,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        record = _render_gauge_rr_plots(artifacts, df, part_column, operator_column, value_column)
        grr = result["variance_components"]["gauge_rr (repeatability+reproducibility)"]
        interpretation = (
            f"Gauge R&R = {grr['pct_study_var']:.1f}% of study variation "
            f"({result['verdict']}); ndc={result['ndc']}."
        )
        return envelope.ok_envelope(
            result, artifacts=[record], interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": result["n"]},
        )

    @mcp.tool()
    @envelope.tool("analyze_attribute_agreement")
    def analyze_attribute_agreement(handle: str, rater_columns: list[str]) -> dict[str, Any]:
        """Fleiss' kappa for agreement among 2+ raters classifying the same subjects
        (e.g. pass/fail inspection). Each row is a subject/part; each named column
        is one rater's judgment."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_columns(df, rater_columns)
        if len(rater_columns) < 2:
            raise ValidationError("analyze_attribute_agreement needs at least 2 rater columns")

        ratings = df[rater_columns].to_numpy().tolist()
        result = stats_gauge_rr.fleiss_kappa(ratings)
        interpretation = (
            f"Fleiss' kappa={result['kappa']:.3f} ({result['interpretation']} agreement)."
        )
        return envelope.ok_envelope(
            result,
            interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": result["n_subjects"]},
        )


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _render_gauge_rr_plots(
    artifacts: ArtifactStore,
    df: pd.DataFrame,
    part_column: str,
    operator_column: str,
    value_column: str,
) -> dict[str, Any]:
    fig, axes = new_figure(1, 2, figsize=(11.0, 4.5))

    by_operator = df.groupby(operator_column)[value_column]
    means, sds, names = [], [], []
    for name, group in by_operator:
        means.append(group.mean())
        sds.append(group.std(ddof=1))
        names.append(str(name))
    axes[0].errorbar(names, means, yerr=sds, fmt="o", capsize=4, color="#4c72b0")
    axes[0].set_title("Variation by operator")
    axes[0].set_ylabel(value_column)

    for operator, group in df.groupby(operator_column):
        cell_means = group.groupby(part_column)[value_column].mean()
        axes[1].plot(
            cell_means.index.astype(str), cell_means.to_numpy(), marker="o", label=str(operator)
        )
    axes[1].set_title("Operator x part interaction")
    axes[1].set_xlabel(part_column)
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename="gauge_rr.png",
        data=render_png(fig),
        media_type="image/png",
    )
