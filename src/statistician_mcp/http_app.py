from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from statistician_mcp import __version__


def create_app(mcp: FastMCP) -> Starlette:
    """Build the Starlette app for HTTP transport: FastMCP's streamable-HTTP
    endpoint at /mcp plus a public /healthz route.

    Auth middleware and the /artifacts route are added in a later phase.
    """

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    return mcp.streamable_http_app()
