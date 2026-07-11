#!/usr/bin/env python
"""Admin CLI for managing API keys (STATMCP_AUTH_MODE=keys).

Usage:
    python scripts/issue_key.py issue <workspace_id> [--plan PLAN] [--db PATH]
    python scripts/issue_key.py disable <raw_key> [--db PATH]
    python scripts/issue_key.py list [--db PATH]

Targets the same key store the server would use: Postgres if STATMCP_DATABASE_URL
is set, else the SQLite file at {STATMCP_DATA_DIR}/keys.db. `--db PATH` overrides
that and always forces SQLite at the given path, ignoring STATMCP_DATABASE_URL.

The raw key is only ever printed at issuance time — only its hash is stored, so
there is no way to recover a lost key; issue a new one and disable the old one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from statistician_mcp.apikeys import KeyStore, SqliteKeyStore, build_key_store
from statistician_mcp.config import get_settings


def _resolve_key_store(db_override: Path | None) -> KeyStore:
    if db_override is not None:
        return SqliteKeyStore(db_override)
    return build_key_store(get_settings())


def main() -> None:
    parser = argparse.ArgumentParser(prog="issue_key")
    subparsers = parser.add_subparsers(dest="command", required=True)

    issue = subparsers.add_parser("issue", help="Issue a new API key for a workspace")
    issue.add_argument("workspace_id")
    issue.add_argument("--plan", default="default")
    issue.add_argument("--db", type=Path, default=None)

    disable = subparsers.add_parser("disable", help="Disable an existing API key")
    disable.add_argument("raw_key")
    disable.add_argument("--db", type=Path, default=None)

    listing = subparsers.add_parser("list", help="List issued keys (hash prefixes only)")
    listing.add_argument("--db", type=Path, default=None)

    args = parser.parse_args()
    key_store = _resolve_key_store(args.db)

    if args.command == "issue":
        raw_key = key_store.issue_key(args.workspace_id, args.plan)
        print(f"Issued key for workspace '{args.workspace_id}' (plan={args.plan}):")
        print(raw_key)
        print("\nThis is the only time the raw key is shown — store it securely now.")
    elif args.command == "disable":
        if key_store.disable_key(args.raw_key):
            print("Key disabled.")
        else:
            print("No matching (enabled) key found.")
    elif args.command == "list":
        for entry in key_store.list_keys():
            status = "disabled" if entry["disabled"] else "active"
            print(
                f"{entry['key_hash_prefix']}...  workspace={entry['workspace_id']}  "
                f"plan={entry['plan']}  {status}"
            )


if __name__ == "__main__":
    main()
