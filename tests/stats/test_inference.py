from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from statistician_mcp.stats import inference as inf

# NOTE on golden values: rather than transcribe a published dataset (e.g. NIST/
# SEMATECH e-Handbook) from memory and risk silently-wrong numbers, these tests use
# small hand-verifiable integer fixtures and cross-check every result against an
# independent direct scipy/statsmodels call (bypassing this module's own wrapper),
# the same pattern used for Shapiro-Wilk in tests/stats/test_assumptions.py. This
# verifies both the underlying test AND this module's own CI/Cohen's d/df glue code,
# which scipy does not compute directly.

# a: n=5, mean=3, var=2.5. b: n=10, mean=8.5, var=9.1666...7 (same shape as the
# summarize_columns fixture, shifted +3 -- variance is shift-invariant).
GROUP_A = [1, 2, 3, 4, 5]
GROUP_B = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_two_sample_mean_test_matches_scipy_welch_directly() -> None:
    result = inf.two_sample_mean_test(np.array(GROUP_A), np.array(GROUP_B), "A", "B")

    ref_stat, ref_p = sp_stats.ttest_ind(GROUP_A, GROUP_B, equal_var=False)
    assert result["statistic"] == pytest.approx(ref_stat)
    assert result["p_value"] == pytest.approx(ref_p)

    # Hand-verifiable pieces scipy doesn't compute for us:
    assert result["mean_difference"] == pytest.approx(3 - 8.5)
    ref_df = inf.welch_satterthwaite_df(2.5, 5, 9.0 + 1 / 6, 10)
    assert result["df"] == pytest.approx(ref_df)
    pooled_sd = np.sqrt(((5 - 1) * 2.5 + (10 - 1) * (9 + 1 / 6)) / (5 + 10 - 2))
    assert result["cohens_d"] == pytest.approx((3 - 8.5) / pooled_sd)


def test_two_sample_mean_test_flags_nonnormal_with_mann_whitney() -> None:
    rng = np.random.default_rng(0)
    a = rng.exponential(1, 60)
    b = rng.exponential(3, 60)

    result = inf.two_sample_mean_test(a, b, "A", "B")

    assert result["nonparametric"] is not None
    ref_u, ref_p = sp_stats.mannwhitneyu(a, b)
    assert result["nonparametric"]["statistic"] == pytest.approx(ref_u)
    assert result["nonparametric"]["p_value"] == pytest.approx(ref_p)


def test_one_sample_mean_test_matches_scipy_directly() -> None:
    data = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0])
    result = inf.one_sample_mean_test(data, mu=10.0)

    ref_stat, ref_p = sp_stats.ttest_1samp(data, popmean=10.0)
    assert result["statistic"] == pytest.approx(ref_stat)
    assert result["p_value"] == pytest.approx(ref_p)
    assert result["mean"] == pytest.approx(data.mean())
    assert result["cohens_d"] == pytest.approx((data.mean() - 10.0) / data.std(ddof=1))


def test_paired_mean_test_matches_scipy_directly() -> None:
    a = np.array([10.0, 12.0, 9.0, 15.0, 11.0])
    b = np.array([8.0, 11.0, 10.0, 13.0, 9.0])
    result = inf.paired_mean_test(a, b)

    ref_stat, ref_p = sp_stats.ttest_rel(a, b)
    assert result["statistic"] == pytest.approx(ref_stat)
    assert result["p_value"] == pytest.approx(ref_p)
    assert result["mean_difference"] == pytest.approx((a - b).mean())


def test_one_way_anova_equal_variance_matches_scipy_f_oneway_and_uses_tukey() -> None:
    rng = np.random.default_rng(1)
    groups = {
        "g1": rng.normal(0, 1, 30),
        "g2": rng.normal(1, 1, 30),
        "g3": rng.normal(2, 1, 30),
    }

    result = inf.one_way_anova(groups)

    ref_f, ref_p = sp_stats.f_oneway(*groups.values())
    assert result["statistic"] == pytest.approx(ref_f)
    assert result["p_value"] == pytest.approx(ref_p)
    assert result["posthoc"]["test"] == "Tukey HSD"
    assert result["welch_anova"] is None
    assert result["kruskal_wallis"] is None

    tukey = pairwise_tukeyhsd(
        np.concatenate(list(groups.values())),
        np.concatenate([[name] * len(v) for name, v in groups.items()]),
    )
    idx1, idx2 = np.triu_indices(len(tukey.groupsunique), 1)
    ref_pairs = {
        (str(tukey.groupsunique[i]), str(tukey.groupsunique[j])): p
        for i, j, p in zip(idx1, idx2, tukey.pvalues, strict=True)
    }
    got_pairs = {(p["group_a"], p["group_b"]): p["p_adj"] for p in result["posthoc"]["pairs"]}
    assert got_pairs.keys() == ref_pairs.keys()
    for key, ref_p_adj in ref_pairs.items():
        assert got_pairs[key] == pytest.approx(ref_p_adj)


def test_one_way_anova_unequal_variance_uses_welch_and_games_howell() -> None:
    rng = np.random.default_rng(2)
    groups = {
        "g1": rng.normal(0, 1, 30),
        "g2": rng.normal(1, 6, 30),
        "g3": rng.normal(2, 0.3, 30),
    }

    result = inf.one_way_anova(groups)

    assert result["welch_anova"] is not None
    assert result["posthoc"]["test"] == "Games-Howell"
    ref = anova_oneway(list(groups.values()), use_var="unequal")
    assert result["welch_anova"]["statistic"] == pytest.approx(ref.statistic)
    assert result["welch_anova"]["p_value"] == pytest.approx(ref.pvalue)


def test_one_way_anova_nonnormal_uses_kruskal_wallis() -> None:
    rng = np.random.default_rng(3)
    groups = {
        "g1": rng.exponential(1, 40),
        "g2": rng.exponential(1, 40),
        "g3": rng.exponential(4, 40),
    }

    result = inf.one_way_anova(groups)

    assert result["kruskal_wallis"] is not None
    ref_h, ref_p = sp_stats.kruskal(*groups.values())
    assert result["kruskal_wallis"]["statistic"] == pytest.approx(ref_h)
    assert result["kruskal_wallis"]["p_value"] == pytest.approx(ref_p)


def test_contingency_table_uses_fisher_exact_for_small_expected_counts() -> None:
    table = [[8, 2], [1, 9]]
    result = inf.contingency_table_test(table)

    assert result["test"] == "Fisher's exact test"
    ref_or, ref_p = sp_stats.fisher_exact(table)
    assert result["odds_ratio"] == pytest.approx(ref_or)
    assert result["p_value"] == pytest.approx(ref_p)


def test_contingency_table_uses_chi_square_for_large_counts() -> None:
    table = [[80, 20], [30, 70]]
    result = inf.contingency_table_test(table)

    assert result["test"] == "chi-square test of independence"
    ref_chi2, ref_p, ref_dof, _ = sp_stats.chi2_contingency(table)
    assert result["statistic"] == pytest.approx(ref_chi2)
    assert result["p_value"] == pytest.approx(ref_p)
    assert result["df"] == ref_dof


def test_f_test_variance_ratio_matches_hand_computation() -> None:
    a = np.array(GROUP_A, dtype=float)
    b = np.array(GROUP_B, dtype=float)

    result = inf.f_test_variance_ratio(a, b)

    assert result["variance_a"] == pytest.approx(2.5)
    assert result["variance_b"] == pytest.approx(9 + 1 / 6)
    assert result["f_statistic"] == pytest.approx(2.5 / (9 + 1 / 6))


def test_confidence_interval_mean_matches_hand_computation() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = inf.confidence_interval_mean(data, confidence=0.95)

    assert result["estimate"] == pytest.approx(3.0)
    se = data.std(ddof=1) / np.sqrt(5)
    ref_lo, ref_hi = sp_stats.t.interval(0.95, 4, loc=3.0, scale=se)
    assert result["confidence_interval"]["lower"] == pytest.approx(ref_lo)
    assert result["confidence_interval"]["upper"] == pytest.approx(ref_hi)


def test_tost_equivalence_two_sample_declares_equivalent_for_a_tiny_true_difference() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(10.0, 1.0, 200)
    b = rng.normal(10.1, 1.0, 200)

    result = inf.tost_equivalence_two_sample(a, b, low=-1.0, high=1.0)

    assert result["equivalent"] is True


def test_tost_equivalence_two_sample_rejects_for_a_large_true_difference() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(10.0, 1.0, 200)
    b = rng.normal(15.0, 1.0, 200)

    result = inf.tost_equivalence_two_sample(a, b, low=-1.0, high=1.0)

    assert result["equivalent"] is False
