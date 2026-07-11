from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from mcp.server.fastmcp import FastMCP
from scipy import stats as sp_stats

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore, get_dataframe_for_analysis
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats import inference as stats_inference
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id

Alternative = Literal["two-sided", "less", "greater"]


def register_inference_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("compare_means")
    def compare_means(
        handle: str,
        column: str,
        mu: float | None = None,
        group_column: str | None = None,
        group_values: list[str] | None = None,
        paired_with_column: str | None = None,
        alternative: Alternative = "two-sided",
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Compare means: 1-sample (vs `mu`), 2-sample independent (via `group_column`,
        Welch by default), or paired (via `paired_with_column`). Always runs normality
        and (for 2-sample) equal-variance checks, always reports Cohen's d + a CI, and
        computes the matching nonparametric test automatically when normality fails.
        Exactly one of mu/group_column/paired_with_column must be given."""
        modes = [mu is not None, group_column is not None, paired_with_column is not None]
        if sum(modes) != 1:
            raise ValidationError(
                "exactly one of mu, group_column, or paired_with_column must be given"
            )

        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)

        if mu is not None:
            result = stats_inference.one_sample_mean_test(
                df[column].to_numpy(), mu, alternative, confidence
            )
        elif group_column is not None:
            _require_columns(df, [group_column])
            levels = group_values or sorted(df[group_column].dropna().unique().tolist())
            if len(levels) != 2:
                raise ValidationError(
                    f"group_column must have exactly 2 levels to compare, found {len(levels)}",
                    hint=f"levels present: {levels}; pass group_values=[a, b] to pick two",
                )
            a = df.loc[df[group_column] == levels[0], column].to_numpy()
            b = df.loc[df[group_column] == levels[1], column].to_numpy()
            result = stats_inference.two_sample_mean_test(
                a, b, str(levels[0]), str(levels[1]), alternative, confidence
            )
        else:
            _require_numeric(df, paired_with_column)  # type: ignore[arg-type]
            result = stats_inference.paired_mean_test(
                df[column].to_numpy(), df[paired_with_column].to_numpy(), alternative, confidence
            )

        assumptions = result.pop("assumptions")
        return envelope.ok_envelope(
            result,
            assumptions=[a.to_dict() for a in assumptions],
            interpretation=_mean_test_interpretation(result, assumptions),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("compare_multiple_groups")
    def compare_multiple_groups(
        handle: str, column: str, group_column: str, confidence: float = 0.95
    ) -> dict[str, Any]:
        """One-way ANOVA across 3+ groups with automatic post-hoc: Tukey HSD when
        variances are equal, Welch ANOVA + Games-Howell when they aren't, or
        Kruskal-Wallis + Dunn when normality fails. Includes a group-means plot."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)
        _require_columns(df, [group_column])

        groups = {
            str(level): sub[column].to_numpy()
            for level, sub in df.groupby(group_column, dropna=True)
        }
        if len(groups) < 3:
            raise ValidationError(
                f"group_column has {len(groups)} level(s); compare_multiple_groups needs at "
                "least 3 (use compare_means for 2 groups)"
            )

        result = stats_inference.one_way_anova(groups, confidence)
        assumptions = result.pop("assumptions")

        fig, ax = new_figure(figsize=(6.5, 4.5))
        names = list(result["groups"].keys())
        means = [result["groups"][n]["mean"] for n in names]
        sds = [result["groups"][n]["sd"] for n in names]
        ax.errorbar(names, means, yerr=sds, fmt="o", capsize=4, color="#4c72b0")
        ax.set_ylabel(column)
        ax.set_title(f"Group means +/- 1 SD by {group_column}")
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="group_means.png",
            data=render_png(fig),
            media_type="image/png",
        )

        return envelope.ok_envelope(
            result,
            assumptions=[a.to_dict() for a in assumptions],
            artifacts=[record],
            interpretation=_anova_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("compare_proportions")
    def compare_proportions(
        count_a: int | None = None,
        nobs_a: int | None = None,
        count_b: int | None = None,
        nobs_b: int | None = None,
        value: float | None = None,
        contingency_table: list[list[int]] | None = None,
        alternative: Alternative = "two-sided",
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Compare proportions: pass `contingency_table` for a chi-square/Fisher-exact
        test of independence; pass `count_a`/`nobs_a`/`count_b`/`nobs_b` for a 2-sample
        proportion z-test; pass `count_a`/`nobs_a`/`value` for a 1-sample test against a
        hypothesized proportion (exact binomial for nobs<30, else a z-test). Not
        dataset-based — pass raw counts."""
        modes = [contingency_table is not None, count_b is not None, value is not None]
        if sum(modes) != 1:
            raise ValidationError(
                "exactly one of contingency_table, count_b/nobs_b, or value must be given"
            )

        if contingency_table is not None:
            result = stats_inference.contingency_table_test(contingency_table)
        elif count_b is not None:
            if count_a is None or nobs_a is None or nobs_b is None:
                raise ValidationError("count_a, nobs_a, and nobs_b are required alongside count_b")
            result = stats_inference.two_sample_proportion_test(
                count_a, nobs_a, count_b, nobs_b, alternative, confidence
            )
        else:
            if count_a is None or nobs_a is None:
                raise ValidationError("count_a and nobs_a are required alongside value")
            result = stats_inference.one_sample_proportion_test(
                count_a, nobs_a, value, alternative, confidence  # type: ignore[arg-type]
            )

        return envelope.ok_envelope(
            result, interpretation=_proportions_interpretation(result), meta={}
        )

    @mcp.tool()
    @envelope.tool("test_equivalence")
    def test_equivalence(
        handle: str,
        column: str,
        low_bound: float,
        high_bound: float,
        mu: float | None = None,
        group_column: str | None = None,
        group_values: list[str] | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """TOST equivalence test: is the mean (1-sample, vs `mu`) or the difference
        between two groups (2-sample, via `group_column`) within [low_bound,
        high_bound] of the reference? Use for 'are these two lots equivalent'
        questions rather than a difference test."""
        if (mu is None) == (group_column is None):
            raise ValidationError("exactly one of mu or group_column must be given")

        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)

        if mu is not None:
            result = stats_inference.tost_equivalence_one_sample(
                df[column].to_numpy(), mu, low_bound, high_bound, alpha
            )
        else:
            _require_columns(df, [group_column])  # type: ignore[list-item]
            levels = group_values or sorted(df[group_column].dropna().unique().tolist())
            if len(levels) != 2:
                raise ValidationError(
                    f"group_column must have exactly 2 levels, found {len(levels)}",
                    hint=f"levels present: {levels}",
                )
            a = df.loc[df[group_column] == levels[0], column].to_numpy()
            b = df.loc[df[group_column] == levels[1], column].to_numpy()
            result = stats_inference.tost_equivalence_two_sample(a, b, low_bound, high_bound, alpha)

        verdict = "equivalent" if result["equivalent"] else "not equivalent"
        interpretation = (
            f"At the {low_bound} to {high_bound} equivalence margin (alpha={alpha}), "
            f"the two are {verdict} (p={result['p_value']:.4g})."
        )
        return envelope.ok_envelope(
            result, interpretation=interpretation, meta={"dataset": handle, "n_rows_used": len(df)}
        )

    @mcp.tool()
    @envelope.tool("compute_confidence_interval")
    def compute_confidence_interval(
        parameter: Literal["mean", "sd", "proportion", "mean_difference"],
        handle: str | None = None,
        column: str | None = None,
        group_column: str | None = None,
        group_values: list[str] | None = None,
        count: int | None = None,
        nobs: int | None = None,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Confidence interval for a mean, standard deviation, or mean difference
        (dataset-based: pass `handle`/`column`, plus `group_column` for a difference)
        or a proportion (pass raw `count`/`nobs`, no dataset needed)."""
        if parameter == "proportion":
            if count is None or nobs is None:
                raise ValidationError("parameter='proportion' requires 'count' and 'nobs'")
            result = stats_inference.confidence_interval_proportion(count, nobs, confidence)
            return envelope.ok_envelope(result, meta={})

        if handle is None or column is None:
            raise ValidationError(f"parameter='{parameter}' requires 'handle' and 'column'")
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)

        if parameter == "mean":
            result = stats_inference.confidence_interval_mean(df[column].to_numpy(), confidence)
        elif parameter == "sd":
            result = stats_inference.confidence_interval_sd(df[column].to_numpy(), confidence)
        else:
            if not group_column:
                raise ValidationError("parameter='mean_difference' requires 'group_column'")
            _require_columns(df, [group_column])
            levels = group_values or sorted(df[group_column].dropna().unique().tolist())
            if len(levels) != 2:
                raise ValidationError(
                    f"group_column must have exactly 2 levels, found {len(levels)}",
                    hint=f"levels present: {levels}",
                )
            a = df.loc[df[group_column] == levels[0], column].to_numpy()
            b = df.loc[df[group_column] == levels[1], column].to_numpy()
            result = stats_inference.confidence_interval_mean_difference(a, b, confidence)

        return envelope.ok_envelope(result, meta={"dataset": handle, "n_rows_used": len(df)})

    @mcp.tool()
    @envelope.tool("test_variance")
    def test_variance(
        handle: str, column: str, group_column: str, group_values: list[str] | None = None
    ) -> dict[str, Any]:
        """Test whether variance is equal across groups: an F-test for exactly 2
        groups, Levene's test for 3+. Use before choosing pooled vs. Welch methods."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_numeric(df, column)
        _require_columns(df, [group_column])

        levels = group_values or sorted(df[group_column].dropna().unique().tolist())
        arrays = [df.loc[df[group_column] == level, column].to_numpy() for level in levels]
        if any(len(arr) < 2 for arr in arrays):
            raise ValidationError("each group needs at least 2 observations")

        if len(levels) == 2:
            result = stats_inference.f_test_variance_ratio(arrays[0], arrays[1])
            interpretation = (
                f"F={result['f_statistic']:.4f}, p={result['p_value']:.4g}: variances are "
                f"{'not ' if result['p_value'] < 0.05 else ''}equal at alpha=0.05."
            )
        elif len(levels) >= 3:
            stat, p = sp_stats.levene(*arrays, center="median")
            result = {
                "test": "Levene",
                "statistic": float(stat),
                "p_value": float(p),
                "groups": levels,
            }
            unequal = "not " if p < 0.05 else ""
            interpretation = (
                f"Levene: statistic={stat:.4f}, p={p:.4g}: variances are {unequal}equal."
            )
        else:
            raise ValidationError("group_column must have at least 2 levels")

        return envelope.ok_envelope(
            result, interpretation=interpretation, meta={"dataset": handle, "n_rows_used": len(df)}
        )


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    _require_columns(df, [column])
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValidationError(f"column '{column}' is not numeric")


def _mean_test_interpretation(result: dict[str, Any], assumptions: list[Any]) -> str:
    parts = [f"{result['test']}: p={result['p_value']:.4g}, Cohen's d={result['cohens_d']:.3f}."]
    if result.get("nonparametric"):
        np_result = result["nonparametric"]
        parts.append(
            f"Normality check failed, so {np_result['test']} was also computed: "
            f"p={np_result['p_value']:.4g}."
        )
    for a in assumptions:
        if a.status == "fail":
            parts.append(f"Caution: {a.check} failed ({a.detail}).")
    return " ".join(parts)


def _anova_interpretation(result: dict[str, Any]) -> str:
    parts = [f"One-way ANOVA: F={result['statistic']:.4f}, p={result['p_value']:.4g}."]
    if result.get("welch_anova"):
        w = result["welch_anova"]
        parts.append(
            f"Variances were unequal, so Welch ANOVA was used instead: p={w['p_value']:.4g}."
        )
    if result.get("kruskal_wallis"):
        k = result["kruskal_wallis"]
        parts.append(
            f"Normality failed for at least one group, so Kruskal-Wallis was also computed: "
            f"p={k['p_value']:.4g}."
        )
    if result.get("posthoc"):
        parts.append(f"Post-hoc: {result['posthoc']['test']}.")
    return " ".join(parts)


def _proportions_interpretation(result: dict[str, Any]) -> str:
    return f"{result['test']}: p={result['p_value']:.4g}."
