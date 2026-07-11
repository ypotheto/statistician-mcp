from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Byte-oriented key-value storage. `LocalDirBackend` is the only implementation
    today; a DigitalOcean Spaces (S3-compatible) backend is added in Phase 7 so the
    hosted product can run on ephemeral-disk compute."""

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def delete(self, path: str) -> None: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return the storage-relative paths of every file under `prefix`."""


class LocalDirBackend(StorageBackend):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        full = (self._root / path).resolve()
        root = self._root.resolve()
        if root not in full.parents and full != root:
            raise ValueError(f"invalid storage path: {path!r}")
        return full

    def write_bytes(self, path: str, data: bytes) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def delete(self, path: str) -> None:
        full = self._resolve(path)
        if full.exists():
            full.unlink()

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        root = self._root.resolve()
        return [str(p.relative_to(root)).replace("\\", "/") for p in base.rglob("*") if p.is_file()]
