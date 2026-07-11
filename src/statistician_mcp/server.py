from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from statistician_mcp import __version__
from statistician_mcp.config import Settings


def create_server(settings: Settings) -> FastMCP:
    mcp = FastMCP(
        "statistician",
        port=settings.port,
        json_response=True,
        stateless_http=True,
    )

    @mcp.tool()
    def ping() -> dict[str, str]:
        """Health check tool: returns server name and version. Use to verify connectivity."""
        return {"server": "statistician", "version": __version__}

    return mcp
