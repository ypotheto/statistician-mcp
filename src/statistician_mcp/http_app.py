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

    `/healthz` is always public. `/artifacts/*` also accepts the token as a `?t=`
    query parameter since browsers can't set an Authorization header on a plain link.
    """

    def __init__(self, app: ASGIApp, verify_token: TokenVerifier | None) -> None:
        self._app = app
        self._verify_token = verify_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._verify_token is None or scope["path"] == "/healthz":
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


def create_app(bundle: ServerBundle) -> ASGIApp:
    mcp = bundle.mcp

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

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

    app: ASGIApp = mcp.streamable_http_app()
    app = AuthMiddleware(app, verify_token=_build_token_verifier(bundle.settings))
    app = TimeoutMiddleware(app, timeout_seconds=bundle.settings.request_timeout_seconds)
    return app
