from __future__ import annotations

from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from statistician_mcp import __version__, usage
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.config import Settings
from statistician_mcp.datasets import DatasetStore
from statistician_mcp.modules.datasets_tools import register_dataset_tools
from statistician_mcp.modules.eda import register_eda_tools
from statistician_mcp.modules.inference import register_inference_tools
from statistician_mcp.modules.power import register_power_tools
from statistician_mcp.storage import LocalDirBackend


@dataclass
class ServerBundle:
    mcp: FastMCP
    settings: Settings
    dataset_store: DatasetStore
    artifact_store: ArtifactStore


def create_server(settings: Settings) -> ServerBundle:
    usage.configure(settings.data_dir)
    backend = LocalDirBackend(settings.data_dir / "storage")
    dataset_store = DatasetStore(backend)
    base_url = settings.public_base_url or f"http://localhost:{settings.port}"
    artifact_store = ArtifactStore(backend, base_url)

    mcp = FastMCP(
        "statistician",
        port=settings.port,
        json_response=True,
        stateless_http=True,
    )

    @mcp.tool()
    def ping() -> dict[str, str]:
        """Health check tool: returns server name and version. Use to verify connectivity."""
        return {"server": "statistician", "version": __version__}

    register_dataset_tools(mcp, dataset_store)
    register_eda_tools(mcp, dataset_store, artifact_store)
    register_inference_tools(mcp, dataset_store, artifact_store)
    register_power_tools(mcp, artifact_store)

    return ServerBundle(
        mcp=mcp, settings=settings, dataset_store=dataset_store, artifact_store=artifact_store
    )
