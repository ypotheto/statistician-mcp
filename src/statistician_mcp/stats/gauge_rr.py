from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def crossed_gauge_rr(
    parts: list[Any], operators: list[Any], values: list[float], tolerance: float | None = None
) -> dict[str, Any]:
    """Crossed Gauge R&R via the ANOVA method (AIAG convention): two-way ANOVA with
    interaction over parts x operators x replicates, variance components with
    negative estimates clamped to zero, %Contribution, %StudyVar (6-sigma
    convention), %Tolerance (if `tolerance` given), and ndc = 1.41*(part_sd/GRR_sd).

    Sum-of-squares decomposition uses the standard balanced two-way ANOVA identity
    (SS_total = SS_part + SS_operator + SS_interaction + SS_error), not a library
    ANOVA call, so every term here is directly hand-verifiable.
    """
    df = pd.DataFrame({"part": parts, "operator": operators, "value": values})
    part_levels = df["part"].unique()
    operator_levels = df["operator"].unique()
    p, o = len(part_levels), len(operator_levels)
    counts = df.groupby(["part", "operator"]).size()
    if counts.nunique() != 1:
        raise ValueError("crossed Gauge R&R requires an equal number of replicates in every cell")
    r = int(counts.iloc[0])
    n = p * o * r

    grand_mean = float(df["value"].mean())
    part_means = df.groupby("part")["value"].mean()
    operator_means = df.groupby("operator")["value"].mean()
    cell_means = df.groupby(["part", "operator"])["value"].mean()

    ss_total = float(((df["value"] - grand_mean) ** 2).sum())
    ss_part = float(o * r * ((part_means - grand_mean) ** 2).sum())
    ss_operator = float(p * r * ((operator_means - grand_mean) ** 2).sum())

    ss_interaction = 0.0
    for (part, operator), cell_mean in cell_means.items():
        deviation = cell_mean - part_means[part] - operator_means[operator] + grand_mean
        ss_interaction += deviation**2
    ss_interaction *= r

    ss_repeatability = ss_total - ss_part - ss_operator - ss_interaction

    df_part, df_operator = p - 1, o - 1
    df_interaction = df_part * df_operator
    df_repeatability = p * o * (r - 1)

    ms_part = ss_part / df_part
    ms_operator = ss_operator / df_operator
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0.0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0.0

    var_repeatability = max(ms_repeatability, 0.0)
    if df_interaction > 0:
        var_interaction = max((ms_interaction - ms_repeatability) / r, 0.0)
    else:
        var_interaction = 0.0
    var_operator = max((ms_operator - ms_interaction) / (p * r), 0.0)
    var_part = max((ms_part - ms_interaction) / (o * r), 0.0)

    var_grr = var_repeatability + var_operator + var_interaction
    var_total = var_grr + var_part

    def pct_contribution(v: float) -> float:
        return 100 * v / var_total if var_total > 0 else 0.0

    def pct_study_var(v: float) -> float:
        return 100 * np.sqrt(v) / np.sqrt(var_total) if var_total > 0 else 0.0

    sd_grr = float(np.sqrt(var_grr))
    sd_part = float(np.sqrt(var_part))
    ndc = int(1.41 * (sd_part / sd_grr)) if sd_grr > 0 else None

    components = {
        "repeatability (equipment variation)": var_repeatability,
        "reproducibility (operator)": var_operator,
        "operator*part interaction": var_interaction,
        "gauge_rr (repeatability+reproducibility)": var_grr,
        "part-to-part": var_part,
        "total": var_total,
    }

    result: dict[str, Any] = {
        "n": n,
        "p": p,
        "o": o,
        "r": r,
        "anova_table": {
            "part": {"df": df_part, "ss": ss_part, "ms": ms_part},
            "operator": {"df": df_operator, "ss": ss_operator, "ms": ms_operator},
            "interaction": {"df": df_interaction, "ss": ss_interaction, "ms": ms_interaction},
            "repeatability": {
                "df": df_repeatability, "ss": ss_repeatability, "ms": ms_repeatability
            },
        },
        "variance_components": {
            name: {
                "variance": v,
                "sd": float(np.sqrt(v)),
                "pct_contribution": pct_contribution(v),
                "pct_study_var": pct_study_var(v),
            }
            for name, v in components.items()
        },
        "ndc": ndc,
        "verdict": _grr_verdict(pct_study_var(var_grr)),
    }
    if tolerance is not None:
        for comp in result["variance_components"].values():
            comp["pct_tolerance"] = 100 * 6 * comp["sd"] / tolerance if tolerance > 0 else None
    return result


def _grr_verdict(pct_study_var_grr: float) -> str:
    if pct_study_var_grr < 10:
        return "acceptable"
    if pct_study_var_grr < 30:
        return "marginal"
    return "unacceptable"


def fleiss_kappa(ratings: list[list[Any]]) -> dict[str, Any]:
    """Fleiss' kappa for m raters classifying n subjects into k categories
    (raters need not agree on which rater rated what -- only counts per category
    per subject matter, the standard Fleiss formulation)."""
    df = pd.DataFrame(ratings)
    categories = sorted({v for row in ratings for v in row})
    n, m = df.shape
    counts = np.zeros((n, len(categories)))
    for i, row in enumerate(ratings):
        for value in row:
            counts[i, categories.index(value)] += 1

    p_j = counts.sum(axis=0) / (n * m)
    p_i = (np.sum(counts**2, axis=1) - m) / (m * (m - 1))
    p_bar = float(p_i.mean())
    p_e = float(np.sum(p_j**2))
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")

    return {
        "n_subjects": n,
        "n_raters": m,
        "categories": categories,
        "kappa": float(kappa),
        "observed_agreement": p_bar,
        "expected_agreement": p_e,
        "interpretation": _kappa_interpretation(kappa),
    }


def _kappa_interpretation(kappa: float) -> str:
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
