from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload


@pytest.mark.asyncio
async def test_analyze_gauge_rr_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    rows = ["part,operator,value"]
    parts = [1, 1, 1, 1, 2, 2, 2, 2]
    operators = ["A", "A", "B", "B", "A", "A", "B", "B"]
    values = [10, 12, 11, 13, 20, 22, 22, 24]
    for p, o, v in zip(parts, operators, values, strict=True):
        rows.append(f"{p},{o},{v}")
    csv_text = "\n".join(rows)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "analyze_gauge_rr",
                {
                    "handle": handle,
                    "part_column": "part",
                    "operator_column": "operator",
                    "value_column": "value",
                },
            )
        )

    assert result["ok"] is True
    assert result["results"]["ndc"] == 6
    assert result["results"]["verdict"] == "marginal"
    assert len(result["artifacts"]) == 1


@pytest.mark.asyncio
async def test_analyze_attribute_agreement_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = (
        "rater1,rater2,rater3\n"
        "pass,pass,pass\n"
        "pass,pass,pass\n"
        "fail,fail,fail\n"
        "fail,fail,fail\n"
    )

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "analyze_attribute_agreement",
                {"handle": handle, "rater_columns": ["rater1", "rater2", "rater3"]},
            )
        )

    assert result["ok"] is True
    assert result["results"]["kappa"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_analyze_gauge_rr_rejects_unbalanced_design(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = "part,operator,value\n1,A,10\n1,A,11\n1,B,12\n2,A,20\n2,B,21\n"

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "analyze_gauge_rr",
                {
                    "handle": handle,
                    "part_column": "part",
                    "operator_column": "operator",
                    "value_column": "value",
                },
            )
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"
