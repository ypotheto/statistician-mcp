from __future__ import annotations

import hmac
import mimetypes
from collections.abc import Awaitable, Callable

import anyio
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from statistician_mcp import __version__, apikeys
from statistician_mcp.config import Settings
from statistician_mcp.oauth import OAuthVerifier
from statistician_mcp.server import ServerBundle
from statistician_mcp.workspace import (
    get_current_workspace_id,
    reset_current_workspace_id,
    resolve_workspace_id,
    set_current_workspace_id,
)

TokenVerifier = Callable[[str], Awaitable["str | None"]]


class AuthMiddleware:
    """Plain ASGI middleware (not `BaseHTTPMiddleware`, which breaks streaming
    responses) resolving a bearer token to a workspace id via a pluggable, async
    `verify_token` callable — this is what lets STATMCP_AUTH_MODE=token (a single
    static shared token, hashed into one workspace) and STATMCP_AUTH_MODE=keys (a
    real per-tenant key table, SQLite or Postgres) share one code path. The
    verifier is async so a Postgres-backed lookup's network round trip doesn't
    block the event loop; it runs off-thread via `anyio.to_thread.run_sync` even
    for the SQLite store, which doesn't strictly need it but isn't hurt by it.

    `/healthz` and `/.well-known/oauth-protected-resource` are always public --
    the latter has to be, since a client fetches it *before* it has any token,
    to discover where to authenticate in the first place. `/artifacts/*` also
    accepts the token as a `?t=` query parameter since browsers can't set an
    Authorization header on a plain link.
    """

    _PUBLIC_PATHS = frozenset({"/healthz", "/.well-known/oauth-protected-resource"})

    def __init__(
        self,
        app: ASGIApp,
        verify_token: TokenVerifier | None,
        unauthorized_headers: dict[str, str] | None = None,
    ) -> None:
        self._app = app
        self._verify_token = verify_token
        # Set only in oauth mode: points a client at the Protected Resource
        # Metadata endpoint per the MCP authorization spec, so it can discover
        # where to authenticate rather than just seeing an opaque 401.
        self._unauthorized_headers = unauthorized_headers or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or self._verify_token is None
            or scope["path"] in self._PUBLIC_PATHS
        ):
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        supplied = _extract_bearer_token(request.headers.get("authorization"))
        if supplied is None and scope["path"].startswith("/artifacts/"):
            supplied = request.query_params.get("t")

        workspace_id = await self._verify_token(supplied) if supplied is not None else None
        if workspace_id is None:
            response = JSONResponse(
                {"error": {"code": "unauthorized", "message": "missing or invalid bearer token"}},
                status_code=401,
                headers=self._unauthorized_headers,
            )
            await response(scope, receive, send)
            return

        token = set_current_workspace_id(workspace_id)
        try:
            await self._app(scope, receive, send)
        finally:
            reset_current_workspace_id(token)


class TimeoutMiddleware:
    """Bounds the worst-case duration of a single tool-call request (POST /mcp).
    Scoped to POST only so it never interrupts a legitimate GET (health check,
    artifact download, or a long-lived streamable-HTTP server-push stream)."""

    def __init__(self, app: ASGIApp, timeout_seconds: float) -> None:
        self._app = app
        self._timeout_seconds = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or self._timeout_seconds <= 0:
            await self._app(scope, receive, send)
            return

        try:
            with anyio.fail_after(self._timeout_seconds):
                await self._app(scope, receive, send)
        except TimeoutError:
            response = JSONResponse(
                {"error": {"code": "timeout", "message": "request exceeded the server timeout"}},
                status_code=504,
            )
            await response(scope, receive, send)


def _extract_bearer_token(header_value: str | None) -> str | None:
    if not header_value or not header_value.lower().startswith("bearer "):
        return None
    return header_value[len("bearer ") :].strip()


def _build_token_verifier(settings: Settings) -> TokenVerifier | None:
    if settings.auth_mode == "oauth":
        if not (settings.oauth_issuer and settings.oauth_audience):
            raise ValueError(
                "STATMCP_AUTH_MODE=oauth requires STATMCP_OAUTH_ISSUER and "
                "STATMCP_OAUTH_AUDIENCE to both be set"
            )
        verifier = OAuthVerifier(
            issuer=settings.oauth_issuer,
            audience=settings.oauth_audience,
            required_permission=settings.oauth_required_permission,
        )
        return lambda token: anyio.to_thread.run_sync(verifier.verify, token)

    if settings.auth_mode == "keys":
        key_store = apikeys.build_key_store(settings)
        return lambda token: anyio.to_thread.run_sync(key_store.verify_key, token)

    if settings.api_token:
        static_token = settings.api_token

        async def verify_static_token(token: str) -> str | None:
            if hmac.compare_digest(token, static_token):
                return resolve_workspace_id(token)
            return None

        return verify_static_token

    return None


def _protected_resource_metadata_url(settings: Settings) -> str:
    base_url = settings.public_base_url or f"http://localhost:{settings.port}"
    return f"{base_url.rstrip('/')}/.well-known/oauth-protected-resource"


def create_app(bundle: ServerBundle) -> ASGIApp:
    mcp = bundle.mcp
    settings = bundle.settings

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    if settings.auth_mode == "oauth":

        @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
        async def protected_resource_metadata(_request: Request) -> JSONResponse:
            # RFC 9728 Protected Resource Metadata: tells an MCP client where the
            # authorization server (Kinde) is, so it knows where to log in.
            return JSONResponse(
                {
                    "resource": settings.oauth_audience,
                    "authorization_servers": [settings.oauth_issuer],
                }
            )

    @mcp.custom_route("/artifacts/{workspace_id}/{artifact_id}/{filename}", methods=["GET"])
    async def get_artifact(request: Request) -> Response:
        workspace_id = request.path_params["workspace_id"]
        artifact_id = request.path_params["artifact_id"]
        filename = request.path_params["filename"]
        # The bearer/`?t=` token only proves *a* valid credential; it does not by
        # itself scope access to one tenant while every token maps to one shared
        # workspace (token auth mode). Per-tenant keys resolve to distinct
        # workspace ids, and this check is what stops one tenant's key from
        # reading another's artifacts.
        if workspace_id != get_current_workspace_id():
            return JSONResponse(
                {"error": {"code": "not_found", "message": "artifact not found"}}, status_code=404
            )
        try:
            data = bundle.artifact_store.read(workspace_id, artifact_id, filename)
        except FileNotFoundError:
            return JSONResponse(
                {"error": {"code": "not_found", "message": "artifact not found"}}, status_code=404
            )
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return Response(content=data, media_type=media_type)

    unauthorized_headers = {}
    if settings.auth_mode == "oauth":
        resource_metadata_url = _protected_resource_metadata_url(settings)
        unauthorized_headers["WWW-Authenticate"] = (
            f'Bearer resource_metadata="{resource_metadata_url}"'
        )

    app: ASGIApp = mcp.streamable_http_app()
    app = AuthMiddleware(
        app,
        verify_token=_build_token_verifier(settings),
        unauthorized_headers=unauthorized_headers,
    )
    app = TimeoutMiddleware(app, timeout_seconds=settings.request_timeout_seconds)
    return app
