from __future__ import annotations

import pytest
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize

from statistician_mcp.stats import power as pw

# Canonical statsmodels power-analysis tutorial example: effect_size=0.5, alpha=0.05,
# power=0.8 -> n≈63.77 per group for a two-sample t-test (independently reproduced
# here via a direct TTestIndPower call rather than trusting a remembered figure).


def test_solve_power_two_sample_t_matches_canonical_statsmodels_example() -> None:
    result = pw.solve_power_t_test("two_sample_t", effect_size=0.5, alpha=0.05, power=0.8)

    ref_n = TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)
    assert result["n"] == pytest.approx(ref_n)
    assert result["n"] == pytest.approx(63.765610588911635)
    assert result["solved_for"] == "n"


def test_solve_power_two_sample_t_solving_for_power_matches_statsmodels() -> None:
    result = pw.solve_power_t_test("two_sample_t", effect_size=0.3, n=100, alpha=0.05)

    ref_power = TTestIndPower().solve_power(effect_size=0.3, nobs1=100, alpha=0.05)
    assert result["power"] == pytest.approx(ref_power)
    assert result["solved_for"] == "power"


def test_solve_power_one_sample_t_matches_statsmodels() -> None:
    result = pw.solve_power_t_test("one_sample_t", effect_size=0.5, alpha=0.05, power=0.8)

    ref_n = TTestPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)
    assert result["n"] == pytest.approx(ref_n)


def test_solve_power_proportion_matches_statsmodels() -> None:
    result = pw.solve_power_proportion(prop1=0.5, prop2=0.65, alpha=0.05, power=0.8)

    effect_size = proportion_effectsize(0.5, 0.65)
    ref_n = NormalIndPower().solve_power(effect_size=effect_size, alpha=0.05, power=0.8)
    assert result["n"] == pytest.approx(ref_n)
    assert result["effect_size_h"] == pytest.approx(effect_size)


def test_solve_power_anova_matches_statsmodels() -> None:
    result = pw.solve_power_anova(k_groups=4, effect_size=0.25, alpha=0.05, power=0.8)

    ref_n = FTestAnovaPower().solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=4)
    assert result["n"] == pytest.approx(ref_n)


def test_solve_power_rejects_zero_or_more_than_one_unknown() -> None:
    with pytest.raises(ValueError):
        pw.solve_power_t_test("two_sample_t", effect_size=0.5, n=64, alpha=0.05, power=0.8)
    with pytest.raises(ValueError):
        pw.solve_power_t_test("two_sample_t", effect_size=0.5, alpha=0.05)


def test_power_curve_points_are_monotonically_increasing_with_n() -> None:
    points = pw.power_curve_points(
        "two_sample_t", effect_size=0.5, alpha=0.05, n_values=[10, 30, 60, 100]
    )

    powers = [p["power"] for p in points]
    assert powers == sorted(powers)
    assert all(0.0 <= p <= 1.0 for p in powers)
