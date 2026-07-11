from __future__ import annotations

import pytest

from statistician_mcp.stats.desirability import (
    individual_desirability,
    optimize_desirability,
    overall_desirability,
)


def test_individual_desirability_maximize_boundaries_and_midpoint() -> None:
    assert individual_desirability(70, "maximize", low=80, high=100) == 0.0
    assert individual_desirability(100, "maximize", low=80, high=100) == 1.0
    assert individual_desirability(90, "maximize", low=80, high=100) == pytest.approx(0.5)


def test_individual_desirability_minimize_boundaries_and_midpoint() -> None:
    assert individual_desirability(15, "minimize", low=0, high=10) == 0.0
    assert individual_desirability(0, "minimize", low=0, high=10) == 1.0
    assert individual_desirability(5, "minimize", low=0, high=10) == pytest.approx(0.5)


def test_individual_desirability_target_shape() -> None:
    assert individual_desirability(5, "target", low=0, high=10, target=5) == 1.0
    assert individual_desirability(
        0, "target", low=0, high=10, target=5
    ) == pytest.approx(0.0, abs=1e-9)
    assert individual_desirability(-1, "target", low=0, high=10, target=5) == 0.0
    assert individual_desirability(11, "target", low=0, high=10, target=5) == 0.0


def test_overall_desirability_is_geometric_mean_and_zero_if_any_zero() -> None:
    assert overall_desirability([0.5, 0.5]) == pytest.approx(0.5)
    assert overall_desirability([1.0, 0.25]) == pytest.approx(0.5)
    assert overall_desirability([0.0, 0.9]) == 0.0


def test_optimize_desirability_finds_the_known_optimum() -> None:
    # Both pred1 (80+20x, "maximize" over [80,100]) and pred2 (10-x, "minimize" over
    # [0,10]) have individual desirability strictly increasing in x over [-1,1], so
    # the true joint optimum is at the upper bound x=1: d1=1.0 exactly, d2=(10-9)/10
    # =0.1, giving overall desirability sqrt(1.0*0.1) ~= 0.3162 -- hand-computable
    # from the Derringer-Suich formula directly, independent of the optimizer.
    def pred1(x):
        return 80 + 20 * x[0]

    def pred2(x):
        return 10 - 1 * x[0]

    result = optimize_desirability(
        [pred1, pred2],
        [
            {"goal": "maximize", "low": 80, "high": 100, "target": None, "weight": 1.0},
            {"goal": "minimize", "low": 0, "high": 10, "target": None, "weight": 1.0},
        ],
        bounds=[(-1.0, 1.0)],
        n_starts=10,
        seed=0,
    )

    assert result["x"][0] == pytest.approx(1.0, abs=1e-3)
    assert result["desirability"] == pytest.approx((1.0 * 0.1) ** 0.5, abs=1e-3)
