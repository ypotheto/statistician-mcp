from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STATMCP_", extra="ignore")

    port: int = 8347
    data_dir: Path = Path.home() / ".statistician-mcp"
    api_token: str | None = None
    auth_mode: str = "token"
    public_base_url: str | None = None


def get_settings() -> Settings:
    return Settings()
