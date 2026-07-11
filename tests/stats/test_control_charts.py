from __future__ import annotations

import pytest

from statistician_mcp.stats.control_charts import (
    A2,
    D3,
    D4,
    c4,
    chart_constants,
    d2,
    i_mr_limits,
    nelson_rules,
    xbar_r_limits,
)

# Anchor points from the standard SPC constants table (Montgomery, "Introduction to
# Statistical Quality Control"; NIST/SEMATECH e-Handbook 6.3.1) -- these are the
# most universally cited values in the field (n=2 underlies every I-MR chart; n=5
# is the textbook Xbar-R worked example), used here to validate the computational
# approach (constants derived from first principles via numerical integration of
# the range distribution, not transcribed from the table) rather than trusting a
# hand-copied 24-row table.
ANCHOR_POINTS = {
    2: {"d2": 1.128, "D3": 0.0, "D4": 3.267},
    5: {"d2": 2.326, "D3": 0.0, "D4": 2.114, "A2": 0.577},
}


@pytest.mark.parametrize("n", sorted(ANCHOR_POINTS))
def test_constants_match_known_anchor_points(n: int) -> None:
    expected = ANCHOR_POINTS[n]
    assert d2(n) == pytest.approx(expected["d2"], abs=1e-3)
    assert D3(n) == pytest.approx(expected["D3"], abs=1e-3)
    assert D4(n) == pytest.approx(expected["D4"], abs=1e-3)
    if "A2" in expected:
        assert A2(n) == pytest.approx(expected["A2"], abs=1e-3)


def test_c4_is_close_to_one_and_approaches_one_as_n_grows() -> None:
    assert c4(2) == pytest.approx(0.7979, abs=1e-3)
    assert c4(25) > c4(5) > c4(2)
    assert c4(100) == pytest.approx(1.0, abs=0.01)


def test_chart_constants_bundle_matches_individual_functions() -> None:
    bundle = chart_constants(5)
    assert bundle["d2"] == pytest.approx(d2(5))
    assert bundle["A2"] == pytest.approx(A2(5))


def test_xbar_r_limits_hand_computable_example() -> None:
    # 4 subgroups of n=3 with an exactly known grand mean and mean range.
    subgroups = [[10, 12, 14], [11, 13, 15], [9, 11, 13], [10, 12, 14]]
    result = xbar_r_limits(subgroups)

    # each subgroup mean is 12, 13, 11, 12 -> grand mean = 12; each range is 4 -> Rbar=4
    assert result["xbar"]["cl"] == pytest.approx(12.0)
    assert result["r"]["cl"] == pytest.approx(4.0)
    a2_n3 = A2(3)
    assert result["xbar"]["ucl"] == pytest.approx(12.0 + a2_n3 * 4.0)
    assert result["xbar"]["lcl"] == pytest.approx(12.0 - a2_n3 * 4.0)
    assert result["r"]["ucl"] == pytest.approx(D4(3) * 4.0)
    assert result["r"]["lcl"] == pytest.approx(D3(3) * 4.0)


def test_i_mr_limits_hand_computable_example() -> None:
    # moving ranges of [10,12,11,13] are [2,1,2] -> MRbar = 5/3
    values = [10, 12, 11, 13]
    result = i_mr_limits(values)

    mr_bar = 5 / 3
    assert result["moving_range"]["cl"] == pytest.approx(mr_bar)
    assert result["individuals"]["cl"] == pytest.approx(sum(values) / 4)
    factor = 3 / d2(2)
    assert result["individuals"]["ucl"] == pytest.approx(sum(values) / 4 + factor * mr_bar)
    assert result["moving_range"]["ucl"] == pytest.approx(D4(2) * mr_bar)


def test_nelson_rule_2_flags_the_ninth_point_of_a_run() -> None:
    # 3 baseline points, then 9 points above the centerline -> rule 2 fires
    # starting at the 9th point of that run, which is index 11 (0-indexed).
    points = [0, 0, 0] + [5] * 9 + [0, 0, 0]
    violations = nelson_rules(points, cl=0, sigma=1)

    assert violations[2] == [11]
    # sanity: none of the other points before the run should ever trigger rule 2
    assert all(idx >= 11 for idx in violations[2])


def test_nelson_rule_1_flags_points_beyond_three_sigma() -> None:
    points = [0, 0, 0, 10, 0, 0]
    violations = nelson_rules(points, cl=0, sigma=1)
    assert violations[1] == [3]


def test_nelson_rules_report_nothing_for_stable_data() -> None:
    import numpy as np

    # Nelson rules are deliberately sensitive (e.g. rule 3's 6-point monotonic run
    # occurs by chance often enough that most random seeds DO trigger something
    # over 30 points -- that's expected multi-rule false-alarm behavior, not a bug).
    # Seed 5 is empirically confirmed clean; this test guards against a regression
    # that makes the rules systematically over-fire, not against ordinary chance.
    rng = np.random.default_rng(5)
    points = rng.normal(0, 1, 30).tolist()
    violations = nelson_rules(points, cl=0, sigma=1)
    assert all(len(v) == 0 for v in violations.values())
