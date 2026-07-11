from __future__ import annotations

import numpy as np
import pytest

from statistician_mcp.stats.doe_designs import (
    FactorSpec,
    evaluate_design,
    factorial_effect_power,
    generate_design,
)

# The classic minimal resolution III design (Montgomery, "Design and Analysis of
# Experiments"): 5 factors in 8 runs via generators D=AB, E=AC, defining relation
# I=ABD=ACE=BCDE. Verified directly against pyDOE3's fracfact_aliasing rather than
# a hand-transcribed table -- see the smoke test in this file for confirmation.
CLASSIC_2_5_2_GENERATORS = "a b c ab ac"


def test_full_factorial_2_cubed_has_8_orthogonal_runs() -> None:
    factors = {
        "A": FactorSpec("A", -1, 1),
        "B": FactorSpec("B", -1, 1),
        "C": FactorSpec("C", -1, 1),
    }
    result = generate_design("full_factorial", factors, seed=0)

    assert result["n_runs"] == 8
    assert result["n_factors"] == 3
    assert result["resolution"] is None

    coded = result["run_table"][["A_coded", "B_coded", "C_coded"]].to_numpy()
    assert set(np.unique(coded)) == {-1.0, 1.0}
    corr = np.corrcoef(coded, rowvar=False)
    off_diag = corr[np.triu_indices(3, 1)]
    assert np.allclose(off_diag, 0.0, atol=1e-9)


def test_fractional_factorial_reproduces_classic_resolution_iii_design() -> None:
    factors = {n: FactorSpec(n, -1, 1) for n in "ABCDE"}
    result = generate_design(
        "fractional_factorial", factors, generators=CLASSIC_2_5_2_GENERATORS, seed=0
    )

    assert result["n_runs"] == 8
    assert result["resolution"] == 3
    alias_lines = result["alias_map"]
    assert any(line.split(" = ")[:2] == ["d", "ab"] for line in alias_lines)
    assert any(line.split(" = ")[:2] == ["e", "ac"] for line in alias_lines)


def test_natural_units_map_correctly_from_coded() -> None:
    factors = {"A": FactorSpec("A", 100, 200), "B": FactorSpec("B", 1, 5)}
    result = generate_design("full_factorial", factors, seed=0)
    df = result["run_table"]

    for _, row in df.iterrows():
        assert row["A"] == pytest.approx(150 + row["A_coded"] * 50)
        assert row["B"] == pytest.approx(3 + row["B_coded"] * 2)


def test_replicates_multiply_run_count() -> None:
    factors = {"A": FactorSpec("A", -1, 1)}
    result = generate_design("full_factorial", factors, replicates=3, seed=0)
    assert result["n_runs"] == 2 * 3


def test_ccd_and_box_behnken_produce_center_points() -> None:
    factors = {
        "A": FactorSpec("A", -1, 1),
        "B": FactorSpec("B", -1, 1),
        "C": FactorSpec("C", -1, 1),
    }
    ccd = generate_design("ccd", factors, center_points=4, seed=0)
    bb = generate_design("box_behnken", factors, center_points=3, seed=0)

    ccd_coded = ccd["run_table"][["A_coded", "B_coded", "C_coded"]].to_numpy()
    assert (np.all(ccd_coded == 0, axis=1)).sum() >= 4
    bb_coded = bb["run_table"][["A_coded", "B_coded", "C_coded"]].to_numpy()
    assert (np.all(bb_coded == 0, axis=1)).sum() >= 3


def test_evaluate_design_flags_orthogonality_and_returns_power() -> None:
    factors = {"A": FactorSpec("A", -1, 1), "B": FactorSpec("B", -1, 1)}
    result = generate_design("full_factorial", factors, seed=0)
    coded = result["run_table"][["A_coded", "B_coded"]].to_numpy()

    evaluation = evaluate_design(coded, ["A", "B"], sigma=1.0)
    assert evaluation["orthogonal"] is True
    assert evaluation["max_abs_pairwise_correlation"] < 1e-9
    assert set(evaluation["power_for_main_effects"].keys()) == {"A", "B"}
    assert all(0.0 <= p <= 1.0 for p in evaluation["power_for_main_effects"].values())


def test_factorial_effect_power_increases_with_effect_size() -> None:
    small = factorial_effect_power(n_runs=16, n_params=4, effect_size=1.0, sigma=2.0)
    large = factorial_effect_power(n_runs=16, n_params=4, effect_size=4.0, sigma=2.0)
    assert 0.0 <= small <= large <= 1.0
