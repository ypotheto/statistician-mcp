from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from statistician_mcp.apikeys import KeyStore, PostgresKeyStore, SqliteKeyStore

# The session-scoped, Docker-backed `postgres_url` fixture lives in conftest.py
# (shared with test_usage.py).


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
