from __future__ import annotations

import hmac
import mimetypes

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from statistician_mcp import __version__
from statistician_mcp.server import ServerBundle
from statistician_mcp.workspace import (
    get_current_workspace_id,
    reset_current_workspace_id,
    resolve_workspace_id,
    set_current_workspace_id,
)


class AuthMiddleware:
    """Plain ASGI middleware (not `BaseHTTPMiddleware`, which breaks streaming
    responses) checking a static bearer token and resolving the calling workspace.

    `/healthz` is always public. `/artifacts/*` also accepts the token as a `?t=`
    query parameter since browsers can't set an Authorization header on a plain link.
    """

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._token or scope["path"] == "/healthz":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        supplied = _extract_bearer_token(request.headers.get("authorization"))
        if supplied is None and scope["path"].startswith("/artifacts/"):
            supplied = request.query_params.get("t")

        if supplied is None or not hmac.compare_digest(supplied, self._token):
            response = JSONResponse(
                {"error": {"code": "unauthorized", "message": "missing or invalid bearer token"}},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        token = set_current_workspace_id(resolve_workspace_id(supplied))
        try:
            await self._app(scope, receive, send)
        finally:
            reset_current_workspace_id(token)


def _extract_bearer_token(header_value: str | None) -> str | None:
    if not header_value or not header_value.lower().startswith("bearer "):
        return None
    return header_value[len("bearer ") :].strip()


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
        # workspace (Phase 1-6). Once Phase 7 gives each tenant a distinct key, this
        # check is what stops one tenant's key from reading another's artifacts.
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

    app = mcp.streamable_http_app()
    return AuthMiddleware(app, token=bundle.settings.api_token)
