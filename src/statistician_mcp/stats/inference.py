from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scikit_posthocs import posthoc_dunn
from scipy import stats as sp_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.weightstats import ttost_ind

from statistician_mcp.stats.assumptions import (
    AssumptionResult,
    check_equal_variance,
    check_normality,
)

Alternative = Literal["two-sided", "less", "greater"]


def _clean(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return values[~np.isnan(values)]


def welch_satterthwaite_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    return float(num / den)


def _t_ci(mean: float, se: float, df: float, confidence: float) -> tuple[float, float]:
    lo, hi = sp_stats.t.interval(confidence, df, loc=mean, scale=se)
    return float(lo), float(hi)


def one_sample_mean_test(
    data: np.ndarray, mu: float, alternative: Alternative = "two-sided", confidence: float = 0.95
) -> dict[str, Any]:
    values = _clean(data)
    n = len(values)
    if n < 2:
        raise ValueError("at least 2 observations are required")

    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    stat, p = sp_stats.ttest_1samp(values, popmean=mu, alternative=alternative)
    ci = _t_ci(mean, se, n - 1, confidence)
    cohens_d = (mean - mu) / sd if sd > 0 else float("nan")

    return {
        "test": "one-sample t-test",
        "n": n,
        "mean": mean,
        "sd": sd,
        "mu": mu,
        "statistic": float(stat),
        "p_value": float(p),
        "df": n - 1,
        "confidence_interval": {"level": confidence, "lower": ci[0], "upper": ci[1]},
        "cohens_d": cohens_d,
        "alternative": alternative,
        "nonparametric": None,
        "assumptions": [check_normality(values, "data")],
    }


def two_sample_mean_test(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str = "group A",
    label_b: str = "group B",
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> dict[str, Any]:
    va, vb = _clean(a), _clean(b)
    n_a, n_b = len(va), len(vb)
    if n_a < 2 or n_b < 2:
        raise ValueError("each group needs at least 2 observations")

    mean_a, mean_b = float(va.mean()), float(vb.mean())
    var_a, var_b = float(va.var(ddof=1)), float(vb.var(ddof=1))

    stat, p = sp_stats.ttest_ind(va, vb, equal_var=False, alternative=alternative)
    diff = mean_a - mean_b
    se_diff = float(np.sqrt(var_a / n_a + var_b / n_b))
    df = welch_satterthwaite_df(var_a, n_a, var_b, n_b)
    ci = _t_ci(diff, se_diff, df, confidence)
    pooled_sd = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else float("nan")

    normality_a = check_normality(va, label_a)
    normality_b = check_normality(vb, label_b)
    equal_var = check_equal_variance([va, vb], f"{label_a} vs {label_b}")

    nonparametric = None
    if normality_a.status == "fail" or normality_b.status == "fail":
        u_stat, u_p = sp_stats.mannwhitneyu(va, vb, alternative=alternative)
        nonparametric = {
            "test": "Mann-Whitney U",
            "statistic": float(u_stat),
            "p_value": float(u_p),
        }

    return {
        "test": "Welch two-sample t-test",
        "groups": {
            label_a: {"n": n_a, "mean": mean_a, "sd": float(np.sqrt(var_a))},
            label_b: {"n": n_b, "mean": mean_b, "sd": float(np.sqrt(var_b))},
        },
        "mean_difference": diff,
        "statistic": float(stat),
        "p_value": float(p),
        "df": df,
        "confidence_interval": {"level": confidence, "lower": ci[0], "upper": ci[1]},
        "cohens_d": cohens_d,
        "alternative": alternative,
        "nonparametric": nonparametric,
        "assumptions": [normality_a, normality_b, equal_var],
    }


def paired_mean_test(
    a: np.ndarray, b: np.ndarray, alternative: Alternative = "two-sided", confidence: float = 0.95
) -> dict[str, Any]:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~np.isnan(va) & ~np.isnan(vb)
    va, vb = va[mask], vb[mask]
    n = len(va)
    if n < 2:
        raise ValueError("at least 2 paired observations are required")

    diffs = va - vb
    mean_diff, sd_diff = float(diffs.mean()), float(diffs.std(ddof=1))
    se = sd_diff / np.sqrt(n)
    stat, p = sp_stats.ttest_rel(va, vb, alternative=alternative)
    ci = _t_ci(mean_diff, se, n - 1, confidence)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")

    normality = check_normality(diffs, "paired differences")
    nonparametric = None
    if normality.status == "fail":
        w_stat, w_p = sp_stats.wilcoxon(diffs, alternative=alternative)
        nonparametric = {
            "test": "Wilcoxon signed-rank",
            "statistic": float(w_stat),
            "p_value": float(w_p),
        }

    return {
        "test": "paired t-test",
        "n": n,
        "mean_difference": mean_diff,
        "sd_difference": sd_diff,
        "statistic": float(stat),
        "p_value": float(p),
        "df": n - 1,
        "confidence_interval": {"level": confidence, "lower": ci[0], "upper": ci[1]},
        "cohens_d": cohens_d,
        "alternative": alternative,
        "nonparametric": nonparametric,
        "assumptions": [normality],
    }


def games_howell(groups: dict[str, np.ndarray], confidence: float = 0.95) -> list[dict[str, Any]]:
    """Pairwise Games-Howell test (Tukey-Kramer generalized to unequal variances):
    per-pair Welch-Satterthwaite df, studentized-range p-values. Reimplemented
    directly from statsmodels' own (sandbox/internal) `tukeyhsd` unequal-variance
    branch rather than depending on that non-public module."""
    names = list(groups.keys())
    means = {k: float(v.mean()) for k, v in groups.items()}
    vars_ = {k: float(v.var(ddof=1)) for k, v in groups.items()}
    ns = {k: len(v) for k, v in groups.items()}
    k = len(names)
    alpha = 1 - confidence

    results = []
    for i in range(k):
        for j in range(i + 1, k):
            name_a, name_b = names[i], names[j]
            var_joint = vars_[name_a] / ns[name_a] + vars_[name_b] / ns[name_b]
            se = float(np.sqrt(var_joint / 2))
            mean_diff = means[name_a] - means[name_b]
            df = welch_satterthwaite_df(vars_[name_a], ns[name_a], vars_[name_b], ns[name_b])
            q_stat = abs(mean_diff) / se if se > 0 else 0.0
            p = float(sp_stats.studentized_range.sf(q_stat, k, df))
            q_crit = float(sp_stats.studentized_range.ppf(confidence, k, df))
            margin = q_crit * se
            results.append(
                {
                    "group_a": name_a,
                    "group_b": name_b,
                    "mean_diff": mean_diff,
                    "p_adj": p,
                    "ci_lower": mean_diff - margin,
                    "ci_upper": mean_diff + margin,
                    "reject_equal": bool(p < alpha),
                }
            )
    return results


def _tukey_result_to_list(tukey: Any) -> list[dict[str, Any]]:
    names = tukey.groupsunique
    idx1, idx2 = np.triu_indices(len(names), 1)
    return [
        {
            "group_a": str(names[i]),
            "group_b": str(names[j]),
            "mean_diff": float(d),
            "p_adj": float(p),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "reject_equal": bool(r),
        }
        for i, j, d, p, (lo, hi), r in zip(
            idx1, idx2, tukey.meandiffs, tukey.pvalues, tukey.confint, tukey.reject, strict=True
        )
    ]


def one_way_anova(groups: dict[str, np.ndarray], confidence: float = 0.95) -> dict[str, Any]:
    cleaned = {name: _clean(values) for name, values in groups.items()}
    if len(cleaned) < 2:
        raise ValueError("at least 2 groups are required")
    if any(len(v) < 2 for v in cleaned.values()):
        raise ValueError("each group needs at least 2 observations")

    names = list(cleaned.keys())
    arrays = [cleaned[name] for name in names]

    normalities = [check_normality(v, name) for name, v in cleaned.items()]
    equal_var = check_equal_variance(arrays, "groups")
    any_nonnormal = any(r.status == "fail" for r in normalities)

    f_stat, p = sp_stats.f_oneway(*arrays)
    group_summary = {
        name: {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        for name, v in cleaned.items()
    }

    welch: dict[str, Any] | None = None
    kruskal: dict[str, Any] | None = None
    posthoc: dict[str, Any] | None = None

    if any_nonnormal:
        h_stat, kp = sp_stats.kruskal(*arrays)
        dunn = posthoc_dunn(arrays, p_adjust="holm")
        dunn.index = names
        dunn.columns = names
        kruskal = {
            "test": "Kruskal-Wallis",
            "statistic": float(h_stat),
            "p_value": float(kp),
            "posthoc": {"test": "Dunn (Holm-adjusted)", "p_values": dunn.round(6).to_dict()},
        }
    elif equal_var.status == "fail":
        welch_res = anova_oneway(arrays, use_var="unequal")
        welch = {
            "test": "Welch ANOVA",
            "statistic": float(welch_res.statistic),
            "p_value": float(welch_res.pvalue),
            "df_num": float(welch_res.df_num),
            "df_denom": float(welch_res.df_denom),
        }
        posthoc = {"test": "Games-Howell", "pairs": games_howell(cleaned, confidence)}
    else:
        tukey = pairwise_tukeyhsd(
            np.concatenate(arrays),
            np.concatenate([[name] * len(cleaned[name]) for name in names]),
            alpha=1 - confidence,
        )
        posthoc = {"test": "Tukey HSD", "pairs": _tukey_result_to_list(tukey)}

    return {
        "test": "one-way ANOVA",
        "groups": group_summary,
        "statistic": float(f_stat),
        "p_value": float(p),
        "df_between": len(arrays) - 1,
        "df_within": sum(len(v) for v in arrays) - len(arrays),
        "welch_anova": welch,
        "kruskal_wallis": kruskal,
        "posthoc": posthoc,
        "assumptions": [*normalities, equal_var],
    }


def one_sample_proportion_test(
    count: int,
    nobs: int,
    value: float,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> dict[str, Any]:
    if nobs < 30:
        result = sp_stats.binomtest(count, nobs, value, alternative=alternative)
        stat, p, test_name = None, result.pvalue, "exact binomial test"
    else:
        stat, p = proportions_ztest(count, nobs, value=value, alternative=alternative)
        test_name = "one-sample z-test for proportions"

    ci_lo, ci_hi = proportion_confint(count, nobs, alpha=1 - confidence, method="wilson")
    return {
        "test": test_name,
        "count": count,
        "nobs": nobs,
        "proportion": count / nobs,
        "value": value,
        "statistic": float(stat) if stat is not None else None,
        "p_value": float(p),
        "confidence_interval": {"level": confidence, "lower": float(ci_lo), "upper": float(ci_hi)},
        "alternative": alternative,
    }


def two_sample_proportion_test(
    count_a: int,
    nobs_a: int,
    count_b: int,
    nobs_b: int,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> dict[str, Any]:
    stat, p = proportions_ztest([count_a, count_b], [nobs_a, nobs_b], alternative=alternative)
    p_a, p_b = count_a / nobs_a, count_b / nobs_b
    diff = p_a - p_b
    se = np.sqrt(p_a * (1 - p_a) / nobs_a + p_b * (1 - p_b) / nobs_b)
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    return {
        "test": "two-sample z-test for proportions",
        "group_a": {"count": count_a, "nobs": nobs_a, "proportion": p_a},
        "group_b": {"count": count_b, "nobs": nobs_b, "proportion": p_b},
        "difference": diff,
        "statistic": float(stat),
        "p_value": float(p),
        "confidence_interval": {
            "level": confidence,
            "lower": float(diff - z * se),
            "upper": float(diff + z * se),
        },
        "alternative": alternative,
    }


def contingency_table_test(table: list[list[int]]) -> dict[str, Any]:
    table_arr = np.asarray(table, dtype=float)
    chi2, p, dof, expected = sp_stats.chi2_contingency(table_arr)
    if table_arr.shape == (2, 2) and (expected < 5).any():
        odds_ratio, fisher_p = sp_stats.fisher_exact(table_arr)
        return {
            "test": "Fisher's exact test",
            "odds_ratio": float(odds_ratio),
            "p_value": float(fisher_p),
            "chi_square_reference": {"statistic": float(chi2), "p_value": float(p), "df": int(dof)},
        }
    return {
        "test": "chi-square test of independence",
        "statistic": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "expected": expected.tolist(),
    }


def tost_equivalence_two_sample(
    a: np.ndarray, b: np.ndarray, low: float, high: float, alpha: float = 0.05
) -> dict[str, Any]:
    va, vb = _clean(a), _clean(b)
    p_overall, (t1, p1, _df1), (t2, p2, _df2) = ttost_ind(va, vb, low, high, usevar="unequal")
    return {
        "test": "TOST equivalence (two one-sided tests, two-sample)",
        "mean_difference": float(va.mean() - vb.mean()),
        "lower_bound": low,
        "upper_bound": high,
        "lower_test": {"statistic": float(t1), "p_value": float(p1)},
        "upper_test": {"statistic": float(t2), "p_value": float(p2)},
        "p_value": float(p_overall),
        "equivalent": bool(p_overall < alpha),
    }


def tost_equivalence_one_sample(
    data: np.ndarray, target: float, low: float, high: float, alpha: float = 0.05
) -> dict[str, Any]:
    values = _clean(data)
    t1, p1 = sp_stats.ttest_1samp(values, popmean=target + low, alternative="greater")
    t2, p2 = sp_stats.ttest_1samp(values, popmean=target + high, alternative="less")
    p_overall = float(max(p1, p2))
    return {
        "test": "TOST equivalence (two one-sided tests, one-sample)",
        "mean": float(values.mean()),
        "target": target,
        "lower_bound": low,
        "upper_bound": high,
        "lower_test": {"statistic": float(t1), "p_value": float(p1)},
        "upper_test": {"statistic": float(t2), "p_value": float(p2)},
        "p_value": p_overall,
        "equivalent": bool(p_overall < alpha),
    }


def confidence_interval_mean(data: np.ndarray, confidence: float = 0.95) -> dict[str, Any]:
    values = _clean(data)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    lo, hi = _t_ci(mean, sd / np.sqrt(n), n - 1, confidence)
    return {
        "parameter": "mean",
        "estimate": mean,
        "n": n,
        "confidence_interval": {"level": confidence, "lower": lo, "upper": hi},
    }


def confidence_interval_proportion(
    count: int, nobs: int, confidence: float = 0.95
) -> dict[str, Any]:
    lo, hi = proportion_confint(count, nobs, alpha=1 - confidence, method="wilson")
    return {
        "parameter": "proportion",
        "estimate": count / nobs,
        "n": nobs,
        "confidence_interval": {"level": confidence, "lower": float(lo), "upper": float(hi)},
    }


def confidence_interval_sd(data: np.ndarray, confidence: float = 0.95) -> dict[str, Any]:
    values = _clean(data)
    n = len(values)
    sd = float(values.std(ddof=1))
    var = sd**2
    chi2_lo = sp_stats.chi2.ppf((1 - confidence) / 2, n - 1)
    chi2_hi = sp_stats.chi2.ppf(1 - (1 - confidence) / 2, n - 1)
    lo, hi = float(np.sqrt((n - 1) * var / chi2_hi)), float(np.sqrt((n - 1) * var / chi2_lo))
    return {
        "parameter": "standard deviation",
        "estimate": sd,
        "n": n,
        "confidence_interval": {"level": confidence, "lower": lo, "upper": hi},
    }


def confidence_interval_mean_difference(
    a: np.ndarray, b: np.ndarray, confidence: float = 0.95
) -> dict[str, Any]:
    result = two_sample_mean_test(a, b, confidence=confidence)
    return {
        "parameter": "difference in means",
        "estimate": result["mean_difference"],
        "n": {
            "group_a": result["groups"]["group A"]["n"],
            "group_b": result["groups"]["group B"]["n"],
        },
        "confidence_interval": result["confidence_interval"],
    }


def f_test_variance_ratio(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> dict[str, Any]:
    va, vb = _clean(a), _clean(b)
    var_a, var_b = float(va.var(ddof=1)), float(vb.var(ddof=1))
    n_a, n_b = len(va), len(vb)
    df1, df2 = n_a - 1, n_b - 1
    f_stat = var_a / var_b
    p_lower = float(sp_stats.f.cdf(f_stat, df1, df2))
    p = 2 * min(p_lower, 1 - p_lower)
    lo = f_stat / sp_stats.f.ppf(1 - (1 - confidence) / 2, df1, df2)
    hi = f_stat * sp_stats.f.ppf(1 - (1 - confidence) / 2, df2, df1)
    return {
        "test": "F-test for equality of variances",
        "variance_a": var_a,
        "variance_b": var_b,
        "f_statistic": float(f_stat),
        "p_value": float(p),
        "df1": df1,
        "df2": df2,
        "confidence_interval_ratio": {"level": confidence, "lower": float(lo), "upper": float(hi)},
    }


def assumptions_to_dicts(results: list[AssumptionResult]) -> list[dict[str, str]]:
    return [r.to_dict() for r in results]
