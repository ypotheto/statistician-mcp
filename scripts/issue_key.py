#!/usr/bin/env python
"""Admin CLI for managing API keys (STATMCP_AUTH_MODE=keys).

Usage:
    python scripts/issue_key.py issue <workspace_id> [--plan PLAN] [--db PATH]
    python scripts/issue_key.py disable <raw_key> [--db PATH]
    python scripts/issue_key.py list [--db PATH]

The raw key is only ever printed at issuance time — only its hash is stored, so
there is no way to recover a lost key; issue a new one and disable the old one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from statistician_mcp import apikeys
from statistician_mcp.config import get_settings


def _default_db_path() -> Path:
    return get_settings().data_dir / "keys.db"


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
    db_path = args.db or _default_db_path()

    if args.command == "issue":
        raw_key = apikeys.issue_key(db_path, args.workspace_id, args.plan)
        print(f"Issued key for workspace '{args.workspace_id}' (plan={args.plan}):")
        print(raw_key)
        print("\nThis is the only time the raw key is shown — store it securely now.")
    elif args.command == "disable":
        if apikeys.disable_key(db_path, args.raw_key):
            print("Key disabled.")
        else:
            print("No matching (enabled) key found.")
    elif args.command == "list":
        for entry in apikeys.list_keys(db_path):
            status = "disabled" if entry["disabled"] else "active"
            print(
                f"{entry['key_hash_prefix']}...  workspace={entry['workspace_id']}  "
                f"plan={entry['plan']}  {status}"
            )


if __name__ == "__main__":
    main()
