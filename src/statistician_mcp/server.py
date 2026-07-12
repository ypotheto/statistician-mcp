from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from statistician_mcp import __version__, usage
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.config import Settings
from statistician_mcp.datasets import DatasetStore
from statistician_mcp.modules.advisor import register_advisor_tools
from statistician_mcp.modules.datasets_tools import register_dataset_tools
from statistician_mcp.modules.doe import register_doe_tools
from statistician_mcp.modules.eda import register_eda_tools
from statistician_mcp.modules.inference import register_inference_tools
from statistician_mcp.modules.msa import register_msa_tools
from statistician_mcp.modules.power import register_power_tools
from statistician_mcp.modules.regression import register_regression_tools
from statistician_mcp.modules.spc import register_spc_tools
from statistician_mcp.storage import LocalDirBackend, SpacesBackend, StorageBackend


@dataclass
class ServerBundle:
    mcp: FastMCP
    settings: Settings
    dataset_store: DatasetStore
    artifact_store: ArtifactStore


def _build_storage_backend(settings: Settings) -> StorageBackend:
    if settings.spaces_bucket is None:
        return LocalDirBackend(settings.data_dir / "storage")
    if not (settings.spaces_endpoint and settings.spaces_key and settings.spaces_secret):
        raise ValueError(
            "STATMCP_SPACES_BUCKET is set, so STATMCP_SPACES_ENDPOINT, "
            "STATMCP_SPACES_KEY, and STATMCP_SPACES_SECRET must be set too"
        )
    return SpacesBackend(
        bucket=settings.spaces_bucket,
        endpoint_url=settings.spaces_endpoint,
        access_key=settings.spaces_key,
        secret_key=settings.spaces_secret,
        region=settings.spaces_region,
        prefix=settings.spaces_prefix,
    )


def _build_transport_security(settings: Settings) -> TransportSecuritySettings:
    """FastMCP constructed without explicit transport security auto-enables
    DNS-rebinding protection with a localhost-only Host allowlist -- correct
    for the local-dev threat model it targets, but it 421s every request that
    carries a real public hostname, so a hosted deployment must allowlist its
    own public host explicitly. Derived from STATMCP_PUBLIC_BASE_URL (which a
    hosted deployment must set anyway, for artifact URLs); localhost stays
    allowed so local runs and tests behave as before."""
    allowed_hosts = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"]
    allowed_origins = ["http://localhost:*", "http://127.0.0.1:*"]
    if settings.public_base_url:
        parsed = urlparse(settings.public_base_url)
        if parsed.netloc:
            allowed_hosts.append(parsed.netloc)
            allowed_origins.append(f"{parsed.scheme}://{parsed.netloc}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def create_server(settings: Settings) -> ServerBundle:
    usage.configure(settings)
    backend = _build_storage_backend(settings)
    dataset_store = DatasetStore(backend)
    base_url = settings.public_base_url or f"http://localhost:{settings.port}"
    artifact_store = ArtifactStore(backend, base_url)

    mcp = FastMCP(
        "statistician",
        port=settings.port,
        json_response=True,
        stateless_http=True,
        transport_security=_build_transport_security(settings),
    )

    @mcp.tool()
    def ping() -> dict[str, str]:
        """Health check tool: returns server name and version. Use to verify connectivity."""
        return {"server": "statistician", "version": __version__}

    register_dataset_tools(mcp, dataset_store)
    register_eda_tools(mcp, dataset_store, artifact_store)
    register_inference_tools(mcp, dataset_store, artifact_store)
    register_power_tools(mcp, artifact_store)
    register_doe_tools(mcp, dataset_store, artifact_store)
    register_spc_tools(mcp, dataset_store, artifact_store)
    register_msa_tools(mcp, dataset_store, artifact_store)
    register_regression_tools(mcp, dataset_store, artifact_store)
    register_advisor_tools(mcp, dataset_store)

    return ServerBundle(
        mcp=mcp, settings=settings, dataset_store=dataset_store, artifact_store=artifact_store
    )
