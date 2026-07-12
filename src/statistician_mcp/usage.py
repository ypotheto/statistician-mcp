from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from statistician_mcp.config import Settings

logger = logging.getLogger(__name__)

_writer: _UsageWriter | None = None


class _UsageWriter(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...

    def flush(self, timeout: float = 5.0) -> None: ...


class _FileUsageWriter:
    """Append-only JSONL on local disk -- the local/dev default. On ephemeral-
    disk hosting this resets every deploy; hosted deployments should set
    STATMCP_DATABASE_URL so usage lands in Postgres instead."""

    def __init__(self, data_dir: Path) -> None:
        usage_dir = data_dir / "usage"
        usage_dir.mkdir(parents=True, exist_ok=True)
        self._path = usage_dir / "usage.jsonl"
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        line = json.dumps(event) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def flush(self, timeout: float = 5.0) -> None:
        pass  # writes are synchronous; nothing buffered


class _PostgresUsageWriter:
    """INSERTs usage events into `usage_events` in the DSN's default schema.

    Writes go through a bounded queue drained by a daemon thread: `log_usage`
    is called from inside envelope.tool's async wrapper (i.e. on the event
    loop), so the network round trip must not happen inline -- and a usage
    write failing (or the queue overflowing) must only ever cost us the event,
    never fail or slow the tool call it was recording.
    """

    def __init__(self, database_url: str) -> None:
        self._pool = ConnectionPool(database_url, min_size=1, max_size=2, open=True)
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    workspace_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    duration_ms DOUBLE PRECISION NOT NULL,
                    ok BOOLEAN NOT NULL,
                    n_rows INTEGER,
                    bytes_in BIGINT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS usage_events_ts_idx ON usage_events (ts)")
            conn.commit()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        self._thread = threading.Thread(target=self._drain, daemon=True, name="usage-writer")
        self._thread.start()

    def write(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("usage event queue full; dropping event for tool %r", event.get("tool"))

    def _drain(self) -> None:
        while True:
            event = self._queue.get()
            try:
                with self._pool.connection() as conn:
                    conn.execute(
                        "INSERT INTO usage_events "
                        "(ts, workspace_id, tool, duration_ms, ok, n_rows, bytes_in) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            event["ts"],
                            event["workspace_id"],
                            event["tool"],
                            event["duration_ms"],
                            event["ok"],
                            event["n_rows"],
                            event["bytes_in"],
                        ),
                    )
                    conn.commit()
            except Exception:
                logger.exception("failed to write usage event")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Wait (bounded) for queued events to land -- for tests and shutdown."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)

    def close(self) -> None:
        self.flush()
        self._pool.close()


def configure(settings: Settings) -> None:
    """Route usage events to Postgres when STATMCP_DATABASE_URL is set (the
    hosted case -- survives redeploys, queryable with SQL), else to
    `{data_dir}/usage/usage.jsonl`. Call once at startup."""
    global _writer
    if settings.database_url:
        _writer = _PostgresUsageWriter(settings.database_url)
    else:
        _writer = _FileUsageWriter(settings.data_dir)


def log_usage(
    workspace_id: str,
    tool: str,
    duration_ms: float,
    *,
    ok: bool,
    n_rows: int | None = None,
    bytes_in: int | None = None,
) -> None:
    if _writer is None:
        return
    _writer.write(
        {
            "ts": time.time(),
            "workspace_id": workspace_id,
            "tool": tool,
            "duration_ms": round(duration_ms, 3),
            "ok": ok,
            "n_rows": n_rows,
            "bytes_in": bytes_in,
        }
    )


def flush(timeout: float = 5.0) -> None:
    if _writer is not None:
        _writer.flush(timeout)
