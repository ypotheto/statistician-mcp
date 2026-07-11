from __future__ import annotations

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload


def _linear_csv() -> str:
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 5 + 2 * x1 - 1 * x2 + rng.normal(0, 0.3, n)
    rows = ["x1,x2,y"] + [f"{a},{b},{c}" for a, b, c in zip(x1, x2, y, strict=True)]
    return "\n".join(rows)


def _logistic_csv() -> str:
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.8 * x1)))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    rows = ["x1,y"] + [f"{a},{b}" for a, b in zip(x1, y, strict=True)]
    return "\n".join(rows)


@pytest.mark.asyncio
async def test_fit_linear_model_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _linear_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "fit_linear_model", {"handle": handle, "formula": "y ~ x1 + x2"}
            )
        )

    assert result["ok"] is True
    assert result["results"]["r_squared"] > 0.9
    assert len(result["artifacts"]) == 1


@pytest.mark.asyncio
async def test_fit_linear_model_rejects_malicious_formula(settings: Settings) -> None:
    """Phase 6 acceptance criterion: formula validation rejects malicious input."""
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _linear_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "fit_linear_model",
                {"handle": handle, "formula": "y ~ __import__('os').system('dir')"},
            )
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_fit_logistic_model_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _logistic_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool("fit_logistic_model", {"handle": handle, "formula": "y ~ x1"})
        )

    assert result["ok"] is True
    assert "roc" not in result["results"]  # rendered as an artifact, not inline
    assert 0 <= result["results"]["classification"]["accuracy"] <= 1
    assert len(result["artifacts"]) == 1


@pytest.mark.asyncio
async def test_compare_models_and_predict_from_model(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _linear_csv()})
        )
        handle = loaded["results"]["handle"]

        cmp = payload(
            await session.call_tool(
                "compare_models", {"handle": handle, "formulas": ["y ~ x1", "y ~ x1 + x2"]}
            )
        )
        one_formula = payload(
            await session.call_tool("compare_models", {"handle": handle, "formulas": ["y ~ x1"]})
        )
        pred = payload(
            await session.call_tool(
                "predict_from_model",
                {"handle": handle, "formula": "y ~ x1 + x2", "new_data": [{"x1": 0.0, "x2": 0.0}]},
            )
        )

    assert cmp["ok"] is True
    assert cmp["results"]["best_by_aic"] == "y ~ x1 + x2"
    assert one_formula["ok"] is False and one_formula["error"]["code"] == "validation_error"
    assert pred["ok"] is True
    assert len(pred["results"]["predicted"]) == 1


@pytest.mark.asyncio
async def test_fit_distribution_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    rng = np.random.default_rng(2)
    csv_text = "v\n" + "\n".join(str(v) for v in rng.weibull(2.0, 300) * 5 + 0.01)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "fit_distribution",
                {"handle": handle, "column": "v", "distributions": ["normal", "weibull", "gamma"]},
            )
        )

    assert result["ok"] is True
    assert result["results"]["best_fit"] == "weibull"
    assert len(result["artifacts"]) == 1
