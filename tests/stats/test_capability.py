from __future__ import annotations

import numpy as np
import pytest

from statistician_mcp.stats.capability import process_capability


def test_capability_matches_hand_computation_for_a_centered_process() -> None:
    # Constructed so the overall sample sd is exactly 2.0 and the mean is exactly
    # 100 (symmetric deviations around 100 summing to zero): Cp=(USL-LSL)/(6*sigma).
    deviations = np.array([-3, -2, -1, 0, 1, 2, 3, 0])
    x = 100 + deviations * (2.0 / deviations.std(ddof=1))
    assert x.std(ddof=1) == pytest.approx(2.0)
    assert x.mean() == pytest.approx(100.0)

    result = process_capability(x, lsl=94, usl=106)

    assert result["overall"]["cp"] == pytest.approx((106 - 94) / (6 * 2.0), rel=1e-6)
    assert result["overall"]["cpu"] == pytest.approx((106 - 100) / (3 * 2.0), rel=1e-6)
    assert result["overall"]["cpl"] == pytest.approx((100 - 94) / (3 * 2.0), rel=1e-6)
    expected_cpk = min(result["overall"]["cpu"], result["overall"]["cpl"])
    assert result["overall"]["cpk"] == pytest.approx(expected_cpk)


def test_capability_cpk_uses_the_closer_spec_limit() -> None:
    # mean shifted toward USL -> Cpk should be governed by CPU (the tighter side).
    rng = np.random.default_rng(2)
    x = rng.normal(100, 1, 200)
    result = process_capability(x, lsl=80, usl=105)
    assert result["overall"]["cpk"] == pytest.approx(result["overall"]["cpu"], rel=1e-3)
    assert result["overall"]["cpu"] < result["overall"]["cpl"]


def test_capability_returns_none_indices_for_zero_variance_data() -> None:
    # Near-constant data (a real, if unusual, input) must not raise -- capability
    # indices are undefined, not infinite, when there's no measurable variation.
    x = np.full(50, 100.0)
    result = process_capability(x, lsl=94, usl=106, subgroup_size=None)
    assert result["overall"]["cpk"] is None
    assert result["dpmo"] != result["dpmo"]  # nan != nan


def test_capability_requires_at_least_one_spec_limit() -> None:
    with pytest.raises(ValueError):
        process_capability(np.array([1.0, 2.0, 3.0]), lsl=None, usl=None)


def test_capability_flags_non_normal_data_and_offers_box_cox() -> None:
    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, 300) + 1  # positive, strongly skewed
    result = process_capability(x, lsl=0.5, usl=20)

    assert result["normality"]["status"] == "fail"
    assert result["box_cox_alternative"] is not None
    assert result["box_cox_alternative"]["overall"]["cpk"] is not None


def test_dpmo_and_sigma_level_are_consistent_for_a_well_centered_process() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(100, 1, 5000)
    result = process_capability(x, lsl=94, usl=106)  # +/- 6 sigma -> very few defects
    assert result["dpmo"] < 100
    assert result["sigma_level"] > 4
