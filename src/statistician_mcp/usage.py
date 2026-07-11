from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_usage_path: Path | None = None


def configure(data_dir: Path) -> None:
    """Point the usage logger at `{data_dir}/usage/usage.jsonl`. Call once at startup."""
    global _usage_path
    usage_dir = data_dir / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    _usage_path = usage_dir / "usage.jsonl"


def log_usage(
    workspace_id: str,
    tool: str,
    duration_ms: float,
    *,
    ok: bool,
    n_rows: int | None = None,
    bytes_in: int | None = None,
) -> None:
    if _usage_path is None:
        return
    event = {
        "ts": time.time(),
        "workspace_id": workspace_id,
        "tool": tool,
        "duration_ms": round(duration_ms, 3),
        "ok": ok,
        "n_rows": n_rows,
        "bytes_in": bytes_in,
    }
    line = json.dumps(event) + "\n"
    with _LOCK, _usage_path.open("a", encoding="utf-8") as f:
        f.write(line)
