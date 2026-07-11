from __future__ import annotations

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from statistician_mcp.http_app import TimeoutMiddleware


async def _slow_app(scope: Scope, receive: Receive, send: Send) -> None:
    await anyio.sleep(0.3)
    await JSONResponse({"ok": True})(scope, receive, send)


async def _fast_app(scope: Scope, receive: Receive, send: Send) -> None:
    await JSONResponse({"ok": True})(scope, receive, send)


@pytest.mark.asyncio
async def test_timeout_middleware_returns_504_when_the_app_is_too_slow() -> None:
    app = TimeoutMiddleware(_slow_app, timeout_seconds=0.05)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/mcp")
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"


@pytest.mark.asyncio
async def test_timeout_middleware_passes_through_fast_requests() -> None:
    app = TimeoutMiddleware(_fast_app, timeout_seconds=5.0)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/mcp")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_timeout_middleware_only_applies_to_post() -> None:
    # a GET (e.g. /healthz or an artifact download) must never be cut off by the
    # tool-call timeout, even if it happens to be slow.
    app = TimeoutMiddleware(_slow_app, timeout_seconds=0.05)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_timeout_middleware_disabled_when_timeout_is_non_positive() -> None:
    app = TimeoutMiddleware(_slow_app, timeout_seconds=0)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/mcp")
    assert response.status_code == 200
