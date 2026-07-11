from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

from statistician_mcp.utils.formulas import validate_model_formula


def build_full_interaction_formula(response: str, factor_names: list[str], order: int = 2) -> str:
    terms = list(factor_names)
    if order >= 2:
        terms.extend(":".join(combo) for combo in combinations(factor_names, 2))
    return f"{response} ~ " + " + ".join(terms)


def _lenth_pse(effects: np.ndarray) -> float:
    """Lenth's (1989) Pseudo Standard Error: a robust estimate of effect noise for
    unreplicated factorial designs, used to flag 'significant' effects without a
    separate error term."""
    abs_effects = np.abs(effects)
    s0 = 1.5 * float(np.median(abs_effects))
    trimmed = abs_effects[abs_effects < 2.5 * s0]
    if len(trimmed) == 0:
        return s0
    return 1.5 * float(np.median(trimmed))


def _lenth_margin_of_error(pse: float, n_effects: int, confidence: float = 0.95) -> float:
    df = max(1, n_effects // 3)
    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df)
    return float(t_crit * pse)


def _lack_of_fit_test(
    df: pd.DataFrame, response: str, factor_names: list[str], resid_ss: float, resid_df: int
) -> dict[str, Any] | None:
    grouped = df.groupby(factor_names)[response]
    pure_error_ss = 0.0
    pure_error_df = 0
    for _, group in grouped:
        if len(group) > 1:
            pure_error_ss += float(((group - group.mean()) ** 2).sum())
            pure_error_df += len(group) - 1
    if pure_error_df == 0:
        return None

    lof_ss = resid_ss - pure_error_ss
    lof_df = resid_df - pure_error_df
    if lof_df <= 0 or pure_error_ss <= 0:
        return None

    lof_ms = lof_ss / lof_df
    pure_error_ms = pure_error_ss / pure_error_df
    f_stat = lof_ms / pure_error_ms
    p_value = float(1 - sp_stats.f.cdf(f_stat, lof_df, pure_error_df))
    return {
        "f_statistic": float(f_stat),
        "p_value": p_value,
        "df_lack_of_fit": lof_df,
        "df_pure_error": pure_error_df,
    }


def _suggest_reduction(coeffs: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    non_intercept = [c for c in coeffs if c["term"] != "Intercept"]
    insignificant = [c["term"] for c in non_intercept if c["p_value"] > alpha]
    insignificant_sorted = sorted(insignificant, key=lambda t: -(t.count(":") + 1))
    return {
        "candidates_to_drop": insignificant_sorted,
        "alpha": alpha,
        "note": (
            "Terms listed highest-order first; drop iteratively and refit, "
            "preserving hierarchy (don't drop a main effect while its interaction remains)."
        ),
    }


def _coefficient_table(model: Any) -> list[dict[str, Any]]:
    conf_int = model.conf_int()
    return [
        {
            "term": term,
            "estimate": float(model.params[term]),
            "effect": None if term == "Intercept" else float(2 * model.params[term]),
            "se": float(model.bse[term]),
            "t": float(model.tvalues[term]),
            "p_value": float(model.pvalues[term]),
            "ci_lower": float(conf_int.loc[term, 0]),
            "ci_upper": float(conf_int.loc[term, 1]),
        }
        for term in model.params.index
    ]


def fit_factorial_model(
    df: pd.DataFrame, response: str, factor_names: list[str], formula: str | None = None
) -> dict[str, Any]:
    columns = set(df.columns.astype(str))
    if formula is None:
        formula = build_full_interaction_formula(response, factor_names)
    validate_model_formula(formula, columns)

    model = smf.ols(formula, data=df).fit()
    coeffs = _coefficient_table(model)
    non_intercept = [c for c in coeffs if c["term"] != "Intercept"]

    effects = np.array([c["effect"] for c in non_intercept])
    abs_effects = np.abs(effects)
    n = len(effects)
    half_normal_order = np.argsort(abs_effects)
    pareto_order = np.argsort(-abs_effects)
    probs = (np.arange(1, n + 1) - 0.5) / n
    half_normal_scores = sp_stats.norm.ppf(0.5 + probs / 2)

    pse = _lenth_pse(effects)
    margin = _lenth_margin_of_error(pse, n)

    return {
        "formula": formula,
        "coefficients": coeffs,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
        "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "half_normal": {
            "terms": [non_intercept[i]["term"] for i in half_normal_order],
            "abs_effects": abs_effects[half_normal_order].tolist(),
            "theoretical_quantiles": half_normal_scores.tolist(),
        },
        "pareto": {
            "terms": [non_intercept[i]["term"] for i in pareto_order],
            "abs_effects": abs_effects[pareto_order].tolist(),
            "lenth_pse": pse,
            "margin_of_error_95": margin,
        },
        "lack_of_fit": _lack_of_fit_test(
            df, response, factor_names, float(model.ssr), int(model.df_resid)
        ),
        "residuals": model.resid.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "model_reduction_suggestion": _suggest_reduction(coeffs),
    }


def fit_response_surface_model(
    df: pd.DataFrame, response: str, factor_names: list[str]
) -> dict[str, Any]:
    working = df.copy()
    sq_terms = []
    for name in factor_names:
        sq_col = f"{name}_sq"
        working[sq_col] = working[name] ** 2
        sq_terms.append(sq_col)

    interaction_terms = [":".join(c) for c in combinations(factor_names, 2)]
    formula = f"{response} ~ " + " + ".join([*factor_names, *interaction_terms, *sq_terms])
    validate_model_formula(formula, set(working.columns.astype(str)))

    model = smf.ols(formula, data=working).fit()
    coeffs = _coefficient_table(model)

    return {
        "formula": formula,
        "coefficients": coeffs,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "lack_of_fit": _lack_of_fit_test(
            df, response, factor_names, float(model.ssr), int(model.df_resid)
        ),
        "stationary_point": _stationary_point(model, factor_names),
        "residuals": model.resid.tolist(),
        "fitted": model.fittedvalues.tolist(),
    }


def _stationary_point(model: Any, factor_names: list[str]) -> dict[str, Any] | None:
    n = len(factor_names)
    quad = np.zeros((n, n))
    linear = np.zeros(n)
    params = model.params

    for i, name in enumerate(factor_names):
        linear[i] = params.get(name, 0.0)
        quad[i, i] = params.get(f"{name}_sq", 0.0)
    for i, j in combinations(range(n), 2):
        term_ij = f"{factor_names[i]}:{factor_names[j]}"
        term_ji = f"{factor_names[j]}:{factor_names[i]}"
        coeff = params.get(term_ij, params.get(term_ji, 0.0))
        quad[i, j] = quad[j, i] = coeff / 2

    # A factor with no quadratic/interaction curvature (purely linear in the fitted
    # model) leaves its row of `quad` at ~0, making the system near-singular rather
    # than exactly singular once regression noise is added -- np.linalg.solve then
    # returns a wildly unstable "solution" instead of raising. Guard on condition
    # number explicitly rather than trusting LinAlgError to catch this.
    if np.linalg.cond(quad) > 1e10:
        return None
    try:
        x_star = np.linalg.solve(quad, -0.5 * linear)
    except np.linalg.LinAlgError:
        return None

    eigenvalues = np.linalg.eigvalsh(quad)
    if np.all(eigenvalues < 0):
        kind = "maximum"
    elif np.all(eigenvalues > 0):
        kind = "minimum"
    else:
        kind = "saddle point"

    predict_row = {name: [x_star[i]] for i, name in enumerate(factor_names)}
    predict_row.update({f"{name}_sq": [x_star[i] ** 2] for i, name in enumerate(factor_names)})
    predicted = model.predict(pd.DataFrame(predict_row))

    return {
        "coded_location": dict(zip(factor_names, x_star.tolist(), strict=True)),
        "kind": kind,
        "predicted_response": float(predicted.iloc[0]),
    }
