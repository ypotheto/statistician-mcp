from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult

from statistician_mcp.config import Settings


def _test_settings(**overrides: Any) -> Settings:
    """`Settings` reads `STATMCP_DATABASE_URL`/`STATMCP_SPACES_BUCKET` from a local
    `.env` (or the real environment) unless explicitly overridden -- pinning both
    to None here keeps every test hermetic (local sqlite / local disk) regardless
    of what a developer's `.env` happens to contain, so tests never silently hit
    a real hosted Postgres cluster or Spaces bucket."""
    return Settings(database_url=None, spaces_bucket=None, **overrides)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", api_token=None)


@pytest.fixture
def settings_with_token(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", api_token="secret-token")


@pytest.fixture
def settings_with_keys(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", auth_mode="keys")


def payload(result: CallToolResult) -> dict[str, Any]:
    return json.loads(result.content[0].text)  # type: ignore[union-attr]
