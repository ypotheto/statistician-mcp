from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from statistician_mcp import usage
from statistician_mcp.config import Settings


@pytest.fixture(autouse=True)
def _reset_usage_writer() -> None:
    """usage holds module-level writer state; reset it around every test so one
    test's configure() can't leak into another (or into the wider suite)."""
    yield
    usage._writer = None


def _settings(**overrides: object) -> Settings:
    defaults = {"database_url": None, "spaces_bucket": None, "oauth_issuer": None}
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_file_writer_appends_jsonl(tmp_path: Path) -> None:
    usage.configure(_settings(data_dir=tmp_path))
    usage.log_usage("ws_a", "compare_means", 12.345, ok=True, n_rows=100)
    usage.log_usage("ws_b", "design_experiment", 5.0, ok=False)

    lines = (tmp_path / "usage" / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]

    assert len(events) == 2
    assert events[0]["workspace_id"] == "ws_a"
    assert events[0]["tool"] == "compare_means"
    assert events[0]["ok"] is True
    assert events[0]["n_rows"] == 100
    assert events[1]["tool"] == "design_experiment"
    assert events[1]["ok"] is False


def test_log_usage_is_a_noop_before_configure() -> None:
    usage._writer = None
    usage.log_usage("ws_a", "compare_means", 1.0, ok=True)  # must not raise


def test_postgres_writer_inserts_events(tmp_path: Path, postgres_url: str) -> None:
    with psycopg.connect(postgres_url) as conn:
        conn.execute("DROP TABLE IF EXISTS usage_events")
        conn.commit()

    usage.configure(_settings(data_dir=tmp_path, database_url=postgres_url))
    usage.log_usage("ws_a", "compare_means", 12.345, ok=True, n_rows=100, bytes_in=2048)
    usage.log_usage("ws_b", "design_experiment", 5.0, ok=False)
    usage.flush()

    with psycopg.connect(postgres_url) as conn:
        rows = conn.execute(
            "SELECT workspace_id, tool, ok, n_rows, bytes_in FROM usage_events ORDER BY id"
        ).fetchall()

    assert rows == [
        ("ws_a", "compare_means", True, 100, 2048),
        ("ws_b", "design_experiment", False, None, None),
    ]


def test_postgres_writer_failure_never_raises_into_caller(
    tmp_path: Path, postgres_url: str
) -> None:
    """A usage write failing must only cost the event, never the tool call.
    Simulated by dropping the table after the writer starts -- the INSERT in
    the drain thread fails, log_usage itself must stay silent."""
    usage.configure(_settings(data_dir=tmp_path, database_url=postgres_url))
    with psycopg.connect(postgres_url) as conn:
        conn.execute("DROP TABLE usage_events")
        conn.commit()

    usage.log_usage("ws_a", "compare_means", 1.0, ok=True)  # must not raise
    usage.flush()
