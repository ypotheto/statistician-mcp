from __future__ import annotations

import numpy as np
import pytest

from statistician_mcp.stats.doe_analysis import (
    fit_factorial_model,
    fit_response_surface_model,
)
from statistician_mcp.stats.doe_designs import FactorSpec, generate_design
from statistician_mcp.utils.formulas import FormulaError


def test_fit_factorial_model_exactly_recovers_hand_specified_effects() -> None:
    factors = {
        "A": FactorSpec("A", -1, 1),
        "B": FactorSpec("B", -1, 1),
        "C": FactorSpec("C", -1, 1),
    }
    design = generate_design("full_factorial", factors, seed=0)
    df = design["run_table"]
    # y = 10 + 3A + 2B - 1C + 1.5AB (coded units); A:C and B:C are true zero.
    df["y"] = (
        10
        + 3 * df["A_coded"]
        + 2 * df["B_coded"]
        - 1 * df["C_coded"]
        + 1.5 * df["A_coded"] * df["B_coded"]
    )

    result = fit_factorial_model(df, "y", ["A_coded", "B_coded", "C_coded"])
    by_term = {c["term"]: c for c in result["coefficients"]}

    assert by_term["Intercept"]["estimate"] == pytest.approx(10.0)
    assert by_term["A_coded"]["effect"] == pytest.approx(6.0)
    assert by_term["B_coded"]["effect"] == pytest.approx(4.0)
    assert by_term["C_coded"]["effect"] == pytest.approx(-2.0)
    assert by_term["A_coded:B_coded"]["effect"] == pytest.approx(3.0)
    assert by_term["A_coded:C_coded"]["effect"] == pytest.approx(0.0, abs=1e-9)
    assert result["r_squared"] == pytest.approx(1.0)
    # a noiseless, fully-saturated model has no residual df left for lack-of-fit
    assert result["lack_of_fit"] is None


def test_fit_factorial_model_detects_lack_of_fit() -> None:
    factors = {"A": FactorSpec("A", -1, 1), "B": FactorSpec("B", -1, 1)}
    design = generate_design("full_factorial", factors, replicates=3, seed=0)
    df = design["run_table"]
    rng = np.random.default_rng(0)
    # true model has a genuine A*A curvature term that a main-effects-only fit can't
    # capture, plus small noise -- residual variance should exceed pure-error variance.
    df["y"] = (
        5
        + 2 * df["A_coded"]
        + df["B_coded"]
        + 3 * df["A_coded"] ** 2
        + rng.normal(0, 0.05, len(df))
    )

    result = fit_factorial_model(df, "y", ["A_coded", "B_coded"], formula="y ~ A_coded + B_coded")
    assert result["lack_of_fit"] is not None
    assert result["lack_of_fit"]["p_value"] < 0.05


def test_fit_factorial_model_rejects_unsafe_formula() -> None:
    factors = {"A": FactorSpec("A", -1, 1), "B": FactorSpec("B", -1, 1)}
    design = generate_design("full_factorial", factors, seed=0)
    df = design["run_table"]
    df["y"] = df["A_coded"] + df["B_coded"]

    with pytest.raises(FormulaError):
        fit_factorial_model(df, "y", ["A_coded"], formula="y ~ __import__('os').system('dir')")


def test_response_surface_recovers_known_maximum() -> None:
    factors = {"A": FactorSpec("A", -1, 1), "B": FactorSpec("B", -1, 1)}
    design = generate_design("ccd", factors, seed=0)
    df = design["run_table"]
    # true maximum at A=0.5, B=-1.5 (coded): y = 50 + 2A - 3B - 2A^2 - B^2
    df["y"] = (
        50
        + 2 * df["A_coded"]
        - 3 * df["B_coded"]
        - 2 * df["A_coded"] ** 2
        - df["B_coded"] ** 2
    )

    result = fit_response_surface_model(df, "y", ["A_coded", "B_coded"])

    sp = result["stationary_point"]
    assert sp is not None
    assert sp["kind"] == "maximum"
    assert sp["coded_location"]["A_coded"] == pytest.approx(0.5, abs=1e-6)
    assert sp["coded_location"]["B_coded"] == pytest.approx(-1.5, abs=1e-6)
    assert sp["predicted_response"] == pytest.approx(52.75, abs=1e-6)


def test_response_surface_returns_none_for_purely_linear_factor() -> None:
    factors = {
        "A": FactorSpec("A", -1, 1),
        "B": FactorSpec("B", -1, 1),
        "C": FactorSpec("C", -1, 1),
    }
    design = generate_design("ccd", factors, seed=0)
    df = design["run_table"]
    # C only ever appears linearly -- no unique stationary point exists along C.
    df["y"] = (
        50
        + 2 * df["A_coded"]
        - 3 * df["B_coded"]
        - df["C_coded"]
        - 2 * df["A_coded"] ** 2
        - df["B_coded"] ** 2
    )

    result = fit_response_surface_model(df, "y", ["A_coded", "B_coded", "C_coded"])
    assert result["stationary_point"] is None
