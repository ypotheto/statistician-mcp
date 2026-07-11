from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from statistician_mcp import __version__
from statistician_mcp.config import get_settings
from statistician_mcp.http_app import create_app
from statistician_mcp.server import create_server


@pytest.mark.asyncio
async def test_healthz_returns_ok() -> None:
    settings = get_settings()
    app = create_app(create_server(settings))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
