from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from statistician_mcp.apikeys import KeyStore, PostgresKeyStore, SqliteKeyStore

_DOCKER_AVAILABLE = shutil.which("docker") is not None
_CONTAINER_NAME = "statmcp-test-postgres"


def _wait_for_postgres(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(url, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001 -- retry on any connect failure
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Postgres did not become ready in time: {last_error}")


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Spins up a throwaway `postgres:16-alpine` container for the test session --
    real Postgres, not a mock, since SQLite/Postgres SQL-dialect differences
    (placeholder syntax, boolean handling) are exactly the kind of thing a mock
    would paper over."""
    if not _DOCKER_AVAILABLE:
        pytest.skip("docker is not available")

    subprocess.run(["docker", "rm", "-f", _CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            _CONTAINER_NAME,
            "-e",
            "POSTGRES_PASSWORD=test",
            "-p",
            "127.0.0.1::5432",
            "postgres:16-alpine",
        ],
        check=True,
        capture_output=True,
    )
    try:
        port_output = subprocess.run(
            ["docker", "port", _CONTAINER_NAME, "5432"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        port = port_output.rsplit(":", 1)[-1]
        url = f"postgresql://postgres:test@127.0.0.1:{port}/postgres"
        _wait_for_postgres(url)
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", _CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture(params=["sqlite", "postgres"])
def key_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[KeyStore]:
    """Runs every test in this file against both backends, so a divergence
    between SqliteKeyStore and PostgresKeyStore shows up as a normal test
    failure. The postgres param only pulls in the (session-scoped, Docker-backed)
    postgres_url fixture on demand, so sqlite tests still run without Docker."""
    if request.param == "sqlite":
        yield SqliteKeyStore(tmp_path / "keys.db")
        return

    url = request.getfixturevalue("postgres_url")
    with psycopg.connect(url) as conn:
        conn.execute("DROP TABLE IF EXISTS api_keys")
        conn.commit()
    store = PostgresKeyStore(url)
    try:
        yield store
    finally:
        store.close()


def test_issue_then_verify_roundtrips(key_store: KeyStore) -> None:
    raw_key = key_store.issue_key("ws_acme", plan="pro")
    assert raw_key.startswith("sk_")
    assert key_store.verify_key(raw_key) == "ws_acme"


def test_verify_unknown_key_returns_none(key_store: KeyStore) -> None:
    assert key_store.verify_key("sk_not_a_real_key") is None


def test_disable_key_then_verify_returns_none(key_store: KeyStore) -> None:
    raw_key = key_store.issue_key("ws_acme")
    assert key_store.disable_key(raw_key) is True
    assert key_store.verify_key(raw_key) is None


def test_disable_unknown_key_returns_false(key_store: KeyStore) -> None:
    assert key_store.disable_key("sk_not_a_real_key") is False


def test_list_keys_reports_all_issued_keys(key_store: KeyStore) -> None:
    key_store.issue_key("ws_a", plan="free")
    key_store.issue_key("ws_b", plan="pro")

    entries = key_store.list_keys()

    assert {(e["workspace_id"], e["plan"], e["disabled"]) for e in entries} == {
        ("ws_a", "free", False),
        ("ws_b", "pro", False),
    }


def test_list_keys_reflects_disabled_status(key_store: KeyStore) -> None:
    raw_key = key_store.issue_key("ws_acme")
    key_store.disable_key(raw_key)

    entries = key_store.list_keys()

    assert entries[0]["disabled"] is True
