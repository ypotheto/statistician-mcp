from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def issue_key(db_path: Path, workspace_id: str, plan: str = "default") -> str:
    """Generate a new API key, store only its hash, and return the raw key —
    the raw value is shown once, at issuance, and is not recoverable afterward."""
    raw_key = "sk_" + secrets.token_urlsafe(32)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, workspace_id, plan, created_at, disabled) "
            "VALUES (?, ?, ?, ?, 0)",
            (hash_key(raw_key), workspace_id, plan, time.time()),
        )
        conn.commit()
    return raw_key


def verify_key(db_path: Path, raw_key: str) -> str | None:
    """Return the key's workspace_id if it exists and is not disabled, else None."""
    if not db_path.exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT workspace_id FROM api_keys WHERE key_hash = ? AND disabled = 0",
            (hash_key(raw_key),),
        ).fetchone()
    return row[0] if row else None


def disable_key(db_path: Path, raw_key: str) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?", (hash_key(raw_key),)
        )
        conn.commit()
        return cursor.rowcount > 0


def list_keys(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
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
