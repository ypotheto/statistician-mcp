from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server


@pytest.mark.asyncio
async def test_ping_tool_reports_version(settings: Settings) -> None:
    bundle = create_server(settings)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert "ping" in tool_names
        assert "load_dataset_from_csv" in tool_names

        result = await session.call_tool("ping", {})
        assert result.isError is not True
