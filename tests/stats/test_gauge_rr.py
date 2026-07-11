from __future__ import annotations

import pytest

from statistician_mcp.stats.gauge_rr import crossed_gauge_rr, fleiss_kappa

# Hand-derived 2 parts x 2 operators x 2 replicates example (SS decomposition
# verified by hand arithmetic and cross-checked with a direct Python computation
# before being hardcoded here -- see the smoke test that produced these numbers).
PARTS = [1, 1, 1, 1, 2, 2, 2, 2]
OPERATORS = ["A", "A", "B", "B", "A", "A", "B", "B"]
VALUES = [10, 12, 11, 13, 20, 22, 22, 24]


def test_crossed_gauge_rr_matches_hand_computed_ss_decomposition() -> None:
    result = crossed_gauge_rr(PARTS, OPERATORS, VALUES)

    anova = result["anova_table"]
    assert anova["part"]["ss"] == pytest.approx(220.5)
    assert anova["operator"]["ss"] == pytest.approx(4.5)
    assert anova["interaction"]["ss"] == pytest.approx(0.5)
    assert anova["repeatability"]["ss"] == pytest.approx(8.0)

    # SS decomposition must be additive (the fundamental ANOVA identity).
    total_ss = (
        anova["part"]["ss"]
        + anova["operator"]["ss"]
        + anova["interaction"]["ss"]
        + anova["repeatability"]["ss"]
    )
    hand_total = sum((v - sum(VALUES) / len(VALUES)) ** 2 for v in VALUES)
    assert total_ss == pytest.approx(hand_total)

    components = result["variance_components"]
    assert components["repeatability (equipment variation)"]["variance"] == pytest.approx(2.0)
    assert components["reproducibility (operator)"]["variance"] == pytest.approx(1.0)
    assert components["operator*part interaction"]["variance"] == pytest.approx(0.0)
    assert components["part-to-part"]["variance"] == pytest.approx(55.0)
    assert components["gauge_rr (repeatability+reproducibility)"]["variance"] == pytest.approx(3.0)
    assert result["ndc"] == 6
    assert result["verdict"] == "marginal"


def test_crossed_gauge_rr_requires_balanced_replicates() -> None:
    with pytest.raises(ValueError):
        crossed_gauge_rr([1, 1, 1, 2, 2], ["A", "A", "B", "A", "B"], [1, 2, 3, 4, 5])


def test_crossed_gauge_rr_dominant_part_variation_gives_acceptable_verdict() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    parts, operators, values = [], [], []
    for part in range(5):
        for op in ("A", "B"):
            for _ in range(3):
                parts.append(part)
                operators.append(op)
                values.append(100 + part * 10 + rng.normal(0, 0.3))

    result = crossed_gauge_rr(parts, operators, values)
    grr = result["variance_components"]["gauge_rr (repeatability+reproducibility)"]
    grr_pct = grr["pct_study_var"]
    assert grr_pct < 10
    assert result["verdict"] == "acceptable"


def test_fleiss_kappa_is_exactly_one_for_perfect_per_subject_agreement() -> None:
    ratings = [
        ["pass", "pass", "pass"],
        ["pass", "pass", "pass"],
        ["fail", "fail", "fail"],
        ["fail", "fail", "fail"],
    ]
    result = fleiss_kappa(ratings)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["interpretation"] == "almost perfect"


def test_fleiss_kappa_below_one_for_partial_disagreement() -> None:
    ratings = [
        ["pass", "pass", "pass"],
        ["pass", "fail", "pass"],
        ["fail", "fail", "fail"],
        ["fail", "fail", "pass"],
    ]
    result = fleiss_kappa(ratings)
    assert 0 < result["kappa"] < 1
