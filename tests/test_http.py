from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from statistician_mcp import __version__, apikeys
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


@pytest.mark.asyncio
async def test_keys_mode_rejects_missing_and_unknown_keys(settings_with_keys: Settings) -> None:
    app = create_app(create_server(settings_with_keys))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/mcp", json={})
        unknown = await client.post(
            "/mcp", json={}, headers={"Authorization": "Bearer sk_not_a_real_key"}
        )

    assert missing.status_code == 401
    assert unknown.status_code == 401


@pytest.mark.asyncio
async def test_keys_mode_accepts_an_issued_key(settings_with_keys: Settings) -> None:
    raw_key = apikeys.build_key_store(settings_with_keys).issue_key(
        workspace_id="ws_acme", plan="pro"
    )
    bundle = create_server(settings_with_keys)
    app = create_app(bundle)

    # Unlike the other tests in this file, this request actually reaches FastMCP's
    # real dispatch logic (the others are all rejected by AuthMiddleware first),
    # which needs its session manager's task group running -- something a real
    # server gets for free from uvicorn driving the ASGI lifespan protocol, but
    # which httpx's ASGITransport does not do automatically.
    async with bundle.mcp.session_manager.run():
        transport = ASGITransport(app=app)
        # FastMCP's own DNS-rebinding protection validates the Host header, unlike
        # AuthMiddleware -- "http://test" (fine for the other tests here, which
        # never reach that check) gets rejected with 421, so use a real localhost
        # with the server's configured port (matching what a real uvicorn request
        # always carries -- FastMCP allowlists "localhost:*"/"127.0.0.1:*").
        base_url = f"http://localhost:{settings_with_keys.port}"
        async with AsyncClient(transport=transport, base_url=base_url) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.0"},
                    },
                },
                headers={
                    "Authorization": f"Bearer {raw_key}",
                    "Accept": "application/json, text/event-stream",
                },
            )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_keys_mode_isolates_artifacts_between_two_tenants(
    settings_with_keys: Settings,
) -> None:
    key_store = apikeys.build_key_store(settings_with_keys)
    key_a = key_store.issue_key(workspace_id="ws_a")
    key_b = key_store.issue_key(workspace_id="ws_b")

    bundle = create_server(settings_with_keys)
    record = bundle.artifact_store.register(
        "ws_a", kind="plot", filename="chart.png", data=b"tenant-a-data", media_type="image/png"
    )
    app = create_app(bundle)
    parsed = urlparse(record["url"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        own_tenant = await client.get(parsed.path, headers={"Authorization": f"Bearer {key_a}"})
        other_tenant = await client.get(parsed.path, headers={"Authorization": f"Bearer {key_b}"})

    assert own_tenant.status_code == 200
    assert own_tenant.content == b"tenant-a-data"
    assert other_tenant.status_code == 404


@pytest.mark.asyncio
async def test_keys_mode_rejects_a_disabled_key(settings_with_keys: Settings) -> None:
    key_store = apikeys.build_key_store(settings_with_keys)
    raw_key = key_store.issue_key(workspace_id="ws_acme")
    key_store.disable_key(raw_key)
    app = create_app(create_server(settings_with_keys))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp", json={}, headers={"Authorization": f"Bearer {raw_key}"}
        )

    assert response.status_code == 401


class _StaticSigningKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class _StaticJWKClient:
    """Stands in for `jwt.PyJWKClient` -- see the same pattern (and rationale)
    in test_oauth.py: PyJWKClient's JWKS fetch/cache is PyJWT's own well-tested
    concern, not ours, so tests substitute a real key pair directly rather than
    standing up a fake JWKS-serving endpoint."""

    def __init__(self, key: Any) -> None:
        self._signing_key = _StaticSigningKey(key)

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return self._signing_key


def _make_oauth_token(
    private_key: Any,
    *,
    sub: str = "kp_test_user",
    permissions: list[str] | None = None,
) -> str:
    now = time.time()
    return jwt.encode(
        {
            "iss": "https://test-tenant.kinde.com",
            "aud": "https://statistician-mcp.example/mcp",
            "sub": sub,
            "exp": now + 300,
            "permissions": ["access:statistician-mcp"] if permissions is None else permissions,
        },
        private_key,
        algorithm="RS256",
    )


@pytest.mark.asyncio
async def test_oauth_protected_resource_metadata(settings_with_oauth: Settings) -> None:
    app = create_app(create_server(settings_with_oauth))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 200
    assert response.json() == {
        "resource": "https://statistician-mcp.example/mcp",
        "authorization_servers": ["https://test-tenant.kinde.com"],
    }


@pytest.mark.asyncio
async def test_oauth_mode_401_includes_www_authenticate(settings_with_oauth: Settings) -> None:
    app = create_app(create_server(settings_with_oauth))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/mcp", json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == (
        'Bearer resource_metadata="http://localhost:8347/.well-known/oauth-protected-resource"'
    )


@pytest.mark.asyncio
async def test_oauth_mode_accepts_a_valid_permissioned_token(
    settings_with_oauth: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(jwt, "PyJWKClient", lambda uri: _StaticJWKClient(private_key.public_key()))
    token = _make_oauth_token(private_key)

    bundle = create_server(settings_with_oauth)
    app = create_app(bundle)

    async with bundle.mcp.session_manager.run():
        transport = ASGITransport(app=app)
        base_url = f"http://localhost:{settings_with_oauth.port}"
        async with AsyncClient(transport=transport, base_url=base_url) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.0"},
                    },
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json, text/event-stream",
                },
            )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_oauth_mode_rejects_token_missing_permission(
    settings_with_oauth: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(jwt, "PyJWKClient", lambda uri: _StaticJWKClient(private_key.public_key()))
    token = _make_oauth_token(private_key, permissions=["some:other-permission"])

    app = create_app(create_server(settings_with_oauth))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp", json={}, headers={"Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 401
