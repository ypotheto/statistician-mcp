from __future__ import annotations

import uuid
from typing import Any

from statistician_mcp.storage import StorageBackend


class ArtifactStore:
    def __init__(self, backend: StorageBackend, base_url: str) -> None:
        self._backend = backend
        self._base_url = base_url.rstrip("/")

    def _dir(self, workspace_id: str, artifact_id: str) -> str:
        return f"workspaces/{workspace_id}/artifacts/{artifact_id}"

    def register(
        self, workspace_id: str, *, kind: str, filename: str, data: bytes, media_type: str
    ) -> dict[str, Any]:
        artifact_id = "art_" + uuid.uuid4().hex[:10]
        self._backend.write_bytes(f"{self._dir(workspace_id, artifact_id)}/{filename}", data)
        url = f"{self._base_url}/artifacts/{workspace_id}/{artifact_id}/{filename}"
        return {"kind": kind, "description": filename, "media_type": media_type, "url": url}

    def read(self, workspace_id: str, artifact_id: str, filename: str) -> bytes:
        path = f"{self._dir(workspace_id, artifact_id)}/{filename}"
        if not self._backend.exists(path):
            raise FileNotFoundError(path)
        return self._backend.read_bytes(path)
