from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from statistician_mcp import __version__
from statistician_mcp.errors import StatMcpError
from statistician_mcp.usage import log_usage
from statistician_mcp.utils.plotting import close_all_open_figures
from statistician_mcp.workspace import get_current_workspace_id

LOGGER = logging.getLogger(__name__)


def ok_envelope(
    results: Any,
    *,
    assumptions: list[dict[str, Any]] | None = None,
    interpretation: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "results": results,
        "assumptions": assumptions or [],
        "interpretation": interpretation,
        "artifacts": artifacts or [],
        "meta": {"server_version": __version__, **(meta or {})},
    }


def error_envelope(code: str, message: str, hint: str | None = None) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "hint": hint}}


F = TypeVar("F", bound=Callable[..., Any])


def tool(name: str) -> Callable[[F], F]:
    """Wrap a tool implementation with the standard error-handling + usage-logging
    behavior: `StatMcpError` subclasses become structured `error_envelope`s, anything
    else is logged with a correlation id and returned as a generic (but honest) error,
    and every call (success or failure) appends one usage-log line."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            workspace_id = get_current_workspace_id()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except StatMcpError as exc:
                close_all_open_figures()
                log_usage(workspace_id, name, _elapsed_ms(start), ok=False)
                return error_envelope(exc.code, exc.message, exc.hint)
            except Exception:
                close_all_open_figures()
                correlation_id = uuid.uuid4().hex[:12]
                LOGGER.exception(
                    "unhandled error in tool '%s' (correlation_id=%s)", name, correlation_id
                )
                log_usage(workspace_id, name, _elapsed_ms(start), ok=False)
                return error_envelope(
                    "internal_error",
                    f"an unexpected error occurred (correlation_id={correlation_id})",
                    hint="check server logs for this correlation_id",
                )
            else:
                log_usage(workspace_id, name, _elapsed_ms(start), ok=True)
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
