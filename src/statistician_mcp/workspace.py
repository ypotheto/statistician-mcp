from __future__ import annotations

import contextvars
import hashlib
from dataclasses import dataclass

DEFAULT_WORKSPACE_ID = "local"

_current_workspace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "statmcp_workspace_id", default=DEFAULT_WORKSPACE_ID
)


def resolve_workspace_id(token: str | None) -> str:
    """Map a bearer token to a workspace id. Every request in Phase 1-6 shares the one
    configured `STATMCP_API_TOKEN`, so this resolves to a single workspace; Phase 7's
    per-tenant API-key table calls this same function per issued key, at which point
    distinct tokens naturally land in distinct workspaces."""
    if not token:
        return DEFAULT_WORKSPACE_ID
    return "ws_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def get_current_workspace_id() -> str:
    return _current_workspace_id.get()


def set_current_workspace_id(workspace_id: str) -> contextvars.Token[str]:
    return _current_workspace_id.set(workspace_id)


def reset_current_workspace_id(token: contextvars.Token[str]) -> None:
    _current_workspace_id.reset(token)


@dataclass(frozen=True)
class Quotas:
    max_datasets: int = 50
    max_rows_per_dataset: int = 500_000
    max_bytes_per_dataset: int = 50 * 1024 * 1024
    artifact_ttl_days: int = 7


DEFAULT_QUOTAS = Quotas()
