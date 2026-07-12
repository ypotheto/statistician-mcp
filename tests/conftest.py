from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from mcp.types import CallToolResult

from statistician_mcp.config import Settings

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
    would paper over. Shared by test_apikeys.py and test_usage.py."""
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


def _test_settings(**overrides: Any) -> Settings:
    """`Settings` reads `STATMCP_DATABASE_URL`/`STATMCP_SPACES_BUCKET`/
    `STATMCP_OAUTH_ISSUER` from a local `.env` (or the real environment) unless
    explicitly overridden -- defaulting all three to None here keeps every test
    hermetic (local sqlite / local disk / no real Kinde tenant) regardless of
    what a developer's `.env` happens to contain, so tests never silently hit
    real hosted infrastructure."""
    defaults = {"database_url": None, "spaces_bucket": None, "oauth_issuer": None}
    return Settings(**{**defaults, **overrides})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", api_token=None)


@pytest.fixture
def settings_with_token(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", api_token="secret-token")


@pytest.fixture
def settings_with_keys(tmp_path: Path) -> Settings:
    return _test_settings(data_dir=tmp_path / "data", auth_mode="keys")


@pytest.fixture
def settings_with_oauth(tmp_path: Path) -> Settings:
    return _test_settings(
        data_dir=tmp_path / "data",
        auth_mode="oauth",
        oauth_issuer="https://test-tenant.kinde.com",
        oauth_audience="https://statistician-mcp.example/mcp",
    )


def payload(result: CallToolResult) -> dict[str, Any]:
    return json.loads(result.content[0].text)  # type: ignore[union-attr]
