from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statistician_mcp.utils.formulas import validate_model_formula

Family = Literal["linear", "logistic"]

_FIT_DISTRIBUTIONS: dict[str, Any] = {
    "normal": sp_stats.norm,
    "lognormal": sp_stats.lognorm,
    "weibull": sp_stats.weibull_min,
    "exponential": sp_stats.expon,
    "gamma": sp_stats.gamma,
}
_POSITIVE_ONLY_DISTRIBUTIONS = {"lognormal", "weibull", "exponential", "gamma"}


def get_distribution(name: str) -> Any:
    """Public accessor for the scipy.stats distribution object behind a fit_distribution
    name, e.g. for plotting the fitted density."""
    return _FIT_DISTRIBUTIONS[name]


def _coefficient_table(model: Any) -> list[dict[str, Any]]:
    conf_int = model.conf_int()
    return [
        {
            "term": term,
            "estimate": float(model.params[term]),
            "se": float(model.bse[term]),
            "statistic": float(model.tvalues[term]),
            "p_value": float(model.pvalues[term]),
            "ci_lower": float(conf_int.loc[term, 0]),
            "ci_upper": float(conf_int.loc[term, 1]),
        }
        for term in model.params.index
    ]


def fit_linear_model(df: pd.DataFrame, formula: str) -> dict[str, Any]:
    validate_model_formula(formula, set(df.columns.astype(str)))
    model = smf.ols(formula, data=df).fit()
    coefficients = _coefficient_table(model)

    exog = model.model.exog
    exog_names = model.model.exog_names
    vif: dict[str, float | None] = {}
    for i, name in enumerate(exog_names):
        if name == "Intercept":
            continue
        try:
            vif[name] = float(variance_inflation_factor(exog, i))
        except Exception:
            vif[name] = None

    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    n = len(cooks_d)
    threshold = 4 / n
    flagged = [i for i, d in enumerate(cooks_d) if d > threshold]

    return {
        "formula": formula,
        "coefficients": coefficients,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
        "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "vif": vif,
        "cooks_distance_threshold": threshold,
        "cooks_distance_flagged_rows": flagged,
        "residuals": model.resid.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "leverage": influence.hat_matrix_diag.tolist(),
    }


def fit_logistic_model(df: pd.DataFrame, formula: str) -> dict[str, Any]:
    validate_model_formula(formula, set(df.columns.astype(str)))
    model = smf.logit(formula, data=df).fit(disp=0)
    conf_int = model.conf_int()

    coefficients = []
    for term in model.params.index:
        coef = float(model.params[term])
        coefficients.append(
            {
                "term": term,
                "estimate": coef,
                "odds_ratio": float(np.exp(coef)),
                "se": float(model.bse[term]),
                "z": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "odds_ratio_ci_lower": float(np.exp(conf_int.loc[term, 0])),
                "odds_ratio_ci_upper": float(np.exp(conf_int.loc[term, 1])),
            }
        )

    y_true = np.asarray(model.model.endog, dtype=float)
    y_score = np.asarray(model.predict(), dtype=float)
    predicted_class = (y_score >= 0.5).astype(float)
    accuracy = float((predicted_class == y_true).mean())
    tp = float(np.sum((predicted_class == 1) & (y_true == 1)))
    fp = float(np.sum((predicted_class == 1) & (y_true == 0)))
    fn = float(np.sum((predicted_class == 0) & (y_true == 1)))
    tn = float(np.sum((predicted_class == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    fpr, tpr, thresholds = _roc_curve(y_true, y_score)
    auc = float(np.trapezoid(tpr, fpr))

    return {
        "formula": formula,
        "coefficients": coefficients,
        "n": len(y_true),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "pseudo_r_squared": float(model.prsquared),
        "classification": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": auc,
        },
    }


def _roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="stable")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    distinct = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_idxs]]

    total_pos = tps[-1] if tps[-1] > 0 else 1.0
    total_neg = fps[-1] if fps[-1] > 0 else 1.0
    return fps / total_neg, tps / total_pos, thresholds


def compare_models(
    df: pd.DataFrame, formulas: list[str], family: Family = "linear"
) -> dict[str, Any]:
    columns = set(df.columns.astype(str))
    models = []
    for formula in formulas:
        validate_model_formula(formula, columns)
        if family == "linear":
            model = smf.ols(formula, data=df).fit()
        else:
            model = smf.logit(formula, data=df).fit(disp=0)
        models.append(model)

    summaries: list[dict[str, Any]] = [
        {
            "formula": formula,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "log_likelihood": float(model.llf),
            "df_model": int(model.df_model),
            "r_squared": float(model.rsquared) if hasattr(model, "rsquared") else None,
        }
        for formula, model in zip(formulas, models, strict=True)
    ]

    nested_f_test = None
    if family == "linear" and len(models) == 2:
        a, b = models
        small, large = (a, b) if a.df_resid > b.df_resid else (b, a)
        small_terms, large_terms = set(small.model.exog_names), set(large.model.exog_names)
        if small_terms.issubset(large_terms) and small.df_resid > large.df_resid:
            df1 = small.df_resid - large.df_resid
            df2 = large.df_resid
            f_stat = ((small.ssr - large.ssr) / df1) / (large.ssr / df2)
            p_value = float(1 - sp_stats.f.cdf(f_stat, df1, df2))
            nested_f_test = {
                "f_statistic": float(f_stat),
                "p_value": p_value,
                "df1": df1,
                "df2": df2,
            }

    best_by_aic = summaries[int(np.argmin([s["aic"] for s in summaries]))]["formula"]
    best_by_bic = summaries[int(np.argmin([s["bic"] for s in summaries]))]["formula"]
    return {
        "models": summaries,
        "nested_f_test": nested_f_test,
        "best_by_aic": best_by_aic,
        "best_by_bic": best_by_bic,
    }


def predict_from_model(
    df: pd.DataFrame,
    formula: str,
    new_data: pd.DataFrame,
    family: Family = "linear",
    confidence: float = 0.95,
) -> dict[str, Any]:
    validate_model_formula(formula, set(df.columns.astype(str)))
    if family == "linear":
        model = smf.ols(formula, data=df).fit()
        prediction = model.get_prediction(new_data)
        summary = prediction.summary_frame(alpha=1 - confidence)
        return {
            "predicted": summary["mean"].tolist(),
            "confidence_interval": {
                "lower": summary["mean_ci_lower"].tolist(),
                "upper": summary["mean_ci_upper"].tolist(),
            },
            "prediction_interval": {
                "lower": summary["obs_ci_lower"].tolist(),
                "upper": summary["obs_ci_upper"].tolist(),
            },
        }

    model = smf.logit(formula, data=df).fit(disp=0)
    predicted_probability = model.predict(new_data)
    return {"predicted_probability": predicted_probability.tolist()}


def _anderson_darling_statistic(sorted_cdf_values: np.ndarray) -> float:
    """Generic Anderson-Darling statistic (works for any continuous CDF, unlike
    scipy.stats.anderson which only supports a handful of named families) via the
    probability integral transform. Used here only for relative ranking across
    candidate fits (lower = better), not for a family-specific p-value."""
    n = len(sorted_cdf_values)
    i = np.arange(1, n + 1)
    reversed_cdf = sorted_cdf_values[::-1]
    s = np.sum((2 * i - 1) * (np.log(sorted_cdf_values) + np.log(1 - reversed_cdf))) / n
    return float(-n - s)


def fit_distribution(data: np.ndarray, distributions: list[str] | None = None) -> dict[str, Any]:
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    candidates = distributions or list(_FIT_DISTRIBUTIONS.keys())

    fits: list[dict[str, Any]] = []
    for name in candidates:
        if name not in _FIT_DISTRIBUTIONS:
            raise ValueError(f"unknown distribution '{name}'")
        if name in _POSITIVE_ONLY_DISTRIBUTIONS and np.any(x <= 0):
            continue
        dist = _FIT_DISTRIBUTIONS[name]
        try:
            params = dist.fit(x, floc=0) if name in _POSITIVE_ONLY_DISTRIBUTIONS else dist.fit(x)
            cdf_values = np.clip(dist.cdf(np.sort(x), *params), 1e-12, 1 - 1e-12)
            ad_statistic = _anderson_darling_statistic(cdf_values)
            log_likelihood = float(np.sum(dist.logpdf(x, *params)))
        except Exception:
            continue
        fits.append(
            {
                "distribution": name,
                "params": [float(p) for p in params],
                "ad_statistic": ad_statistic,
                "log_likelihood": log_likelihood,
            }
        )

    if not fits:
        raise ValueError("no candidate distribution could be fit to this data")

    best = min(fits, key=lambda f: f["ad_statistic"])
    return {"n": len(x), "fits": fits, "best_fit": best["distribution"]}
