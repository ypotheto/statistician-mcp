from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload

FACTORS_3 = {
    "A": {"low": 100, "high": 200},
    "B": {"low": 1, "high": 5},
    "C": {"low": -1, "high": 1},
}


@pytest.mark.asyncio
async def test_design_experiment_analyze_factorial_optimize_response_round_trip(
    settings: Settings,
) -> None:
    """The Phase 4 acceptance round-trip: design -> fill in a synthetic response ->
    analyze_factorial -> optimize_response, all through real tool calls."""
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        designed = payload(
            await session.call_tool(
                "design_experiment",
                {"design_type": "full_factorial", "factors": FACTORS_3, "seed": 1},
            )
        )
        assert designed["ok"] is True
        handle = designed["results"]["handle"]

        transformed = payload(
            await session.call_tool(
                "transform_dataset",
                {
                    "handle": handle,
                    "op": "derive",
                    "new_column": "y",
                    "expression": (
                        "10 + 3*A_coded + 2*B_coded - 1*C_coded + 1.5*A_coded*B_coded"
                    ),
                },
            )
        )
        assert transformed["ok"] is True
        response_handle = transformed["results"]["handle"]

        analyzed = payload(
            await session.call_tool(
                "analyze_factorial",
                {"handle": response_handle, "response": "y", "factor_names": ["A", "B", "C"]},
            )
        )
        assert analyzed["ok"] is True
        by_term = {c["term"]: c for c in analyzed["results"]["coefficients"]}
        assert by_term["A"]["effect"] == pytest.approx(6.0)
        assert by_term["B"]["effect"] == pytest.approx(4.0)
        assert by_term["A:B"]["effect"] == pytest.approx(3.0)
        assert len(analyzed["artifacts"]) == 3

        optimized = payload(
            await session.call_tool(
                "optimize_response",
                {
                    "handle": response_handle,
                    "factor_names": ["A", "B", "C"],
                    "responses": [
                        {
                            "column": "y",
                            "model_type": "linear",
                            "goal": "maximize",
                            "low": 0,
                            "high": 30,
                        }
                    ],
                },
            )
        )
        assert optimized["ok"] is True
        # true model increases in A and B, decreases in C -> optimum at high/high/low
        assert optimized["results"]["coded_settings"]["A"] == pytest.approx(1.0, abs=1e-2)
        assert optimized["results"]["coded_settings"]["B"] == pytest.approx(1.0, abs=1e-2)
        assert optimized["results"]["coded_settings"]["C"] == pytest.approx(-1.0, abs=1e-2)


@pytest.mark.asyncio
async def test_evaluate_design_reports_orthogonality(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        designed = payload(
            await session.call_tool(
                "design_experiment",
                {"design_type": "full_factorial", "factors": FACTORS_3, "seed": 1},
            )
        )
        handle = designed["results"]["handle"]

        evaluated = payload(
            await session.call_tool("evaluate_design", {"handle": handle, "factors": FACTORS_3})
        )

    assert evaluated["ok"] is True
    assert evaluated["results"]["orthogonal"] is True


@pytest.mark.asyncio
async def test_analyze_response_surface_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    factors_2 = {"A": {"low": -1, "high": 1}, "B": {"low": -1, "high": 1}}
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        designed = payload(
            await session.call_tool(
                "design_experiment", {"design_type": "ccd", "factors": factors_2, "seed": 0}
            )
        )
        handle = designed["results"]["handle"]

        transformed = payload(
            await session.call_tool(
                "transform_dataset",
                {
                    "handle": handle,
                    "op": "derive",
                    "new_column": "y",
                    "expression": (
                        "50 + 2*A_coded - 3*B_coded - 2*A_coded*A_coded - 1*B_coded*B_coded"
                    ),
                },
            )
        )
        response_handle = transformed["results"]["handle"]

        analyzed = payload(
            await session.call_tool(
                "analyze_response_surface",
                {"handle": response_handle, "response": "y", "factor_names": ["A", "B"]},
            )
        )

    assert analyzed["ok"] is True
    sp = analyzed["results"]["stationary_point"]
    assert sp["kind"] == "maximum"
    assert sp["predicted_response"] == pytest.approx(52.75, abs=1e-6)
    assert len(analyzed["artifacts"]) == 2


@pytest.mark.asyncio
async def test_analyze_factorial_rejects_malicious_formula(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        designed = payload(
            await session.call_tool(
                "design_experiment",
                {
                    "design_type": "full_factorial",
                    "factors": {"A": {"low": -1, "high": 1}, "B": {"low": -1, "high": 1}},
                    "seed": 0,
                },
            )
        )
        handle = designed["results"]["handle"]
        transformed = payload(
            await session.call_tool(
                "transform_dataset",
                {"handle": handle, "op": "derive", "new_column": "y", "expression": "A_coded"},
            )
        )
        response_handle = transformed["results"]["handle"]

        result = payload(
            await session.call_tool(
                "analyze_factorial",
                {
                    "handle": response_handle,
                    "response": "y",
                    "factor_names": ["A"],
                    "formula": "y ~ __import__('os').system('dir')",
                },
            )
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"
