from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from statistician_mcp.config import Settings


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class KeyStore(ABC):
    """Per-tenant API-key table. `SqliteKeyStore` is a single local file (dev/
    single-Droplet default); `PostgresKeyStore` is for a hosted deployment backed
    by a shared Postgres cluster (STATMCP_DATABASE_URL)."""

    @abstractmethod
    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        """Generate a new API key, store only its hash, and return the raw key —
        the raw value is shown once, at issuance, and is not recoverable afterward."""

    @abstractmethod
    def verify_key(self, raw_key: str) -> str | None:
        """Return the key's workspace_id if it exists and is not disabled, else None."""

    @abstractmethod
    def disable_key(self, raw_key: str) -> bool: ...

    @abstractmethod
    def list_keys(self) -> list[dict[str, Any]]: ...


class SqliteKeyStore(KeyStore):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'default',
                    created_at REAL NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
            yield conn
        finally:
            conn.close()

    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        raw_key = "sk_" + secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_hash, workspace_id, plan, created_at, disabled) "
                "VALUES (?, ?, ?, ?, 0)",
                (hash_key(raw_key), workspace_id, plan, time.time()),
            )
            conn.commit()
        return raw_key

    def verify_key(self, raw_key: str) -> str | None:
        if not self._db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM api_keys WHERE key_hash = ? AND disabled = 0",
                (hash_key(raw_key),),
            ).fetchone()
        return row[0] if row else None

    def disable_key(self, raw_key: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?", (hash_key(raw_key),)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> list[dict[str, Any]]:
        if not self._db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key_hash, workspace_id, plan, created_at, disabled FROM api_keys "
                "ORDER BY created_at"
            ).fetchall()
        return [
            {
                "key_hash_prefix": key_hash[:12],
                "workspace_id": workspace_id,
                "plan": plan,
                "created_at": created_at,
                "disabled": bool(disabled),
            }
            for key_hash, workspace_id, plan, created_at, disabled in rows
        ]


class PostgresKeyStore(KeyStore):
    """Assumes `database_url` already points at a schema this role owns (or has
    full rights on) via its `search_path` -- table names here are unqualified."""

    def __init__(self, database_url: str) -> None:
        self._pool = ConnectionPool(database_url, min_size=1, max_size=5, open=True)
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'default',
                    created_at DOUBLE PRECISION NOT NULL,
                    disabled BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            conn.commit()

    def issue_key(self, workspace_id: str, plan: str = "default") -> str:
        raw_key = "sk_" + secrets.token_urlsafe(32)
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_hash, workspace_id, plan, created_at, disabled) "
                "VALUES (%s, %s, %s, %s, FALSE)",
                (hash_key(raw_key), workspace_id, plan, time.time()),
            )
            conn.commit()
        return raw_key

    def verify_key(self, raw_key: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM api_keys WHERE key_hash = %s AND disabled = FALSE",
                (hash_key(raw_key),),
            ).fetchone()
        return row[0] if row else None

    def disable_key(self, raw_key: str) -> bool:
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET disabled = TRUE WHERE key_hash = %s", (hash_key(raw_key),)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_keys(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key_hash, workspace_id, plan, created_at, disabled FROM api_keys "
                "ORDER BY created_at"
            ).fetchall()
        return [
            {
                "key_hash_prefix": key_hash[:12],
                "workspace_id": workspace_id,
                "plan": plan,
                "created_at": created_at,
                "disabled": bool(disabled),
            }
            for key_hash, workspace_id, plan, created_at, disabled in rows
        ]

    def close(self) -> None:
        self._pool.close()


def build_key_store(settings: Settings) -> KeyStore:
    if settings.database_url:
        return PostgresKeyStore(settings.database_url)
    return SqliteKeyStore(settings.data_dir / "keys.db")
