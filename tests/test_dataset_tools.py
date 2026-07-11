from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload as _payload

CSV_TEXT = "a,b\n1,4\n2,5\n3,6\n"


@pytest.mark.asyncio
async def test_dataset_lifecycle(settings: Settings) -> None:
    bundle = create_server(settings)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = _payload(
            await session.call_tool(
                "load_dataset_from_csv", {"csv_text": CSV_TEXT, "name": "demo"}
            )
        )
        assert loaded["ok"] is True
        handle = loaded["results"]["handle"]
        assert loaded["results"]["n_rows"] == 3

        described = _payload(await session.call_tool("describe_dataset", {"handle": handle}))
        assert described["ok"] is True
        assert {c["name"] for c in described["results"]["columns"]} == {"a", "b"}

        transformed = _payload(
            await session.call_tool(
                "transform_dataset",
                {"handle": handle, "op": "filter", "expression": "a > 1"},
            )
        )
        assert transformed["ok"] is True
        new_handle = transformed["results"]["handle"]
        assert transformed["results"]["n_rows"] == 2

        sampled = _payload(
            await session.call_tool("sample_dataset_rows", {"handle": new_handle, "n": 10})
        )
        assert sampled["ok"] is True
        assert len(sampled["results"]["rows"]) == 2

        listed = _payload(await session.call_tool("list_datasets", {}))
        assert {d["handle"] for d in listed["results"]} == {handle, new_handle}

        deleted = _payload(await session.call_tool("delete_dataset", {"handle": new_handle}))
        assert deleted["ok"] is True

        missing = _payload(await session.call_tool("describe_dataset", {"handle": new_handle}))
        assert missing["ok"] is False
        assert missing["error"]["code"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_transform_rejects_malicious_expression(settings: Settings) -> None:
    bundle = create_server(settings)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = _payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": CSV_TEXT})
        )
        handle = loaded["results"]["handle"]

        result = _payload(
            await session.call_tool(
                "transform_dataset",
                {
                    "handle": handle,
                    "op": "filter",
                    "expression": "__import__('os').system('dir')",
                },
            )
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_usage_log_records_calls(settings: Settings) -> None:
    bundle = create_server(settings)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        await session.call_tool("load_dataset_from_csv", {"csv_text": CSV_TEXT})

    usage_path = settings.data_dir / "usage" / "usage.jsonl"
    assert usage_path.exists()
    lines = usage_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    event = json.loads(lines[-1])
    assert event["tool"] == "load_dataset_from_csv"
    assert event["ok"] is True
