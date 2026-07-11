from __future__ import annotations

from urllib.parse import urlparse

import pytest
from httpx import ASGITransport, AsyncClient

from statistician_mcp import __version__
from statistician_mcp.config import Settings
from statistician_mcp.http_app import create_app
from statistician_mcp.server import create_server
from statistician_mcp.workspace import resolve_workspace_id


@pytest.mark.asyncio
async def test_healthz_returns_ok(settings: Settings) -> None:
    app = create_app(create_server(settings))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


@pytest.mark.asyncio
async def test_artifact_round_trip_without_auth(settings: Settings) -> None:
    bundle = create_server(settings)
    record = bundle.artifact_store.register(
        "local", kind="plot", filename="chart.png", data=b"fake-png-bytes", media_type="image/png"
    )
    app = create_app(bundle)
    parsed = urlparse(record["url"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(parsed.path)

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"


@pytest.mark.asyncio
async def test_mcp_endpoint_rejects_missing_token(settings_with_token: Settings) -> None:
    app = create_app(create_server(settings_with_token))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/mcp", json={})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_healthz_is_public_even_when_token_configured(settings_with_token: Settings) -> None:
    app = create_app(create_server(settings_with_token))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_artifact_accepts_query_param_token(settings_with_token: Settings) -> None:
    bundle = create_server(settings_with_token)
    workspace_id = resolve_workspace_id(settings_with_token.api_token)
    record = bundle.artifact_store.register(
        workspace_id,
        kind="plot",
        filename="chart.png",
        data=b"fake-png-bytes",
        media_type="image/png",
    )
    app = create_app(bundle)
    parsed = urlparse(record["url"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        no_token = await client.get(parsed.path)
        with_token = await client.get(f"{parsed.path}?t=secret-token")

    assert no_token.status_code == 401
    assert with_token.status_code == 200
    assert with_token.content == b"fake-png-bytes"


@pytest.mark.asyncio
async def test_artifact_not_served_to_a_different_tenant(settings_with_token: Settings) -> None:
    bundle = create_server(settings_with_token)
    record = bundle.artifact_store.register(
        "some-other-workspace",
        kind="plot",
        filename="chart.png",
        data=b"fake-png-bytes",
        media_type="image/png",
    )
    app = create_app(bundle)
    parsed = urlparse(record["url"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"{parsed.path}?t=secret-token")

    assert response.status_code == 404
