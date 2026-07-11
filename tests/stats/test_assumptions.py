from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from statistician_mcp.stats.assumptions import (
    check_equal_variance,
    check_normality,
    check_sample_size,
)


def test_check_normality_matches_scipy_shapiro_directly() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(10, 2, 200)

    result = check_normality(data, "x")
    stat, p = sp_stats.shapiro(data)

    assert f"statistic={stat:.4f}" in result.detail
    assert f"p={p:.4f}" in result.detail
    assert result.status == ("pass" if p >= 0.05 else "warn" if p >= 0.01 else "fail")


def test_check_normality_flags_skewed_data() -> None:
    rng = np.random.default_rng(1)
    data = rng.exponential(1, 500)

    assert check_normality(data, "skewed").status == "fail"


def test_check_normality_large_sample_uses_anderson_darling() -> None:
    rng = np.random.default_rng(2)
    normal_result = check_normality(rng.normal(0, 1, 6000), "normal")
    skewed_result = check_normality(rng.exponential(1, 6000), "skewed")

    assert "Anderson-Darling" in normal_result.detail
    assert normal_result.status == "pass"
    assert skewed_result.status == "fail"


def test_check_equal_variance_detects_unequal_variance() -> None:
    rng = np.random.default_rng(3)
    g1 = rng.normal(0, 1, 100)
    g2 = rng.normal(0, 5, 100)

    result = check_equal_variance([g1, g2], "g")

    assert result.status == "fail"
    assert "Levene" in result.detail


def test_check_equal_variance_passes_for_equal_variance_groups() -> None:
    rng = np.random.default_rng(4)
    g1 = rng.normal(0, 1, 200)
    g2 = rng.normal(5, 1, 200)

    assert check_equal_variance([g1, g2], "g").status == "pass"


def test_check_sample_size_thresholds() -> None:
    assert check_sample_size(50, 30, "n").status == "pass"
    assert check_sample_size(20, 30, "n").status == "warn"
    assert check_sample_size(2, 30, "n").status == "fail"
