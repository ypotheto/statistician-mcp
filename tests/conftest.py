from __future__ import annotations

from pathlib import Path

import pytest

from statistician_mcp.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_token=None)


@pytest.fixture
def settings_with_token(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_token="secret-token")
