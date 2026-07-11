from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STATMCP_", extra="ignore", env_file=".env")

    port: int = 8347
    data_dir: Path = Path.home() / ".statistician-mcp"
    api_token: str | None = None
    auth_mode: str = "token"
    public_base_url: str | None = None
    request_timeout_seconds: float = 120.0

    # Dataset/artifact storage backend. Unset spaces_bucket (the default) keeps
    # storage on local disk via LocalDirBackend; set all four spaces_* fields to
    # switch to a DigitalOcean Spaces bucket via SpacesBackend instead.
    spaces_bucket: str | None = None
    spaces_endpoint: str | None = None
    spaces_key: str | None = None
    spaces_secret: str | None = None
    spaces_region: str = "nyc3"
    # Namespaces this app's objects within a bucket shared with other services
    # (each given its own prefix) so their keys can never collide.
    spaces_prefix: str = "statistician-mcp"

    # API-key table (STATMCP_AUTH_MODE=keys). Unset (the default) keeps it on
    # local disk via SqliteKeyStore, at {data_dir}/keys.db; set to a Postgres DSN
    # to switch to PostgresKeyStore instead.
    database_url: str | None = None

    # OAuth (STATMCP_AUTH_MODE=oauth) -- Kinde (or any OIDC provider) as the
    # authorization server; this app only ever plays the resource-server role.
    # oauth_issuer is the provider's base URL (e.g. https://<subdomain>.kinde.com);
    # oauth_audience must match the API's registered Audience in Kinde exactly.
    oauth_issuer: str | None = None
    oauth_audience: str | None = None
    oauth_required_permission: str = "access:statistician-mcp"


def get_settings() -> Settings:
    return Settings()
