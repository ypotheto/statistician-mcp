from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult

from statistician_mcp.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_token=None)


@pytest.fixture
def settings_with_token(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_token="secret-token")


@pytest.fixture
def settings_with_keys(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", auth_mode="keys")


def payload(result: CallToolResult) -> dict[str, Any]:
    return json.loads(result.content[0].text)  # type: ignore[union-attr]
