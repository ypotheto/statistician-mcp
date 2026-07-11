from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import get_settings
from statistician_mcp.server import create_server


@pytest.mark.asyncio
async def test_ping_tool_reports_version() -> None:
    mcp = create_server(get_settings())

    async with create_connected_server_and_client_session(mcp) as session:
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["ping"]

        result = await session.call_tool("ping", {})
        assert result.isError is not True
