from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TypeVar

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

T = TypeVar("T")

_RETRY_ATTEMPTS = 8
_RETRY_BASE_DELAY_SECONDS = 0.01


def _retry_on_permission_error(fn: Callable[[], T]) -> T:
    """Windows enforces mandatory file-sharing locks: an operation on a path can
    transiently raise PermissionError if a different thread has that same file
    open at that exact instant (a threaded stress test reproduced this for both
    reads racing a delete and a delete racing a read). Every such open in this
    module is a single open+read/write+close, essentially instantaneous, so a
    short retry reliably clears the contention rather than needing the caller to
    fail outright. Not an issue on the actual Linux production target, which has
    no such restriction."""
    last_error: PermissionError | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except PermissionError as exc:
            last_error = exc
            time.sleep(_RETRY_BASE_DELAY_SECONDS * (attempt + 1))
    assert last_error is not None
    raise last_error


class StorageBackend(ABC):
    """Byte-oriented key-value storage. `LocalDirBackend` writes to a local
    directory (Droplet + volume); `SpacesBackend` writes to a DigitalOcean Spaces
    (S3-compatible) bucket so the same app can run on ephemeral-disk compute."""

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
        """Validate `path` lexically (no `..` traversal, not absolute, no
        backslashes) and join it onto the storage root.

        Deliberately does NOT use `Path.resolve()`: under concurrent directory
        creation, two near-simultaneous `.resolve()` calls for paths in the same
        not-yet-existing directory tree were found (via a threaded stress test)
        to occasionally disagree on the directory's canonical form on Windows
        (e.g. extended-length `\\\\?\\` prefixing kicking in for one call but not
        the other), making a resolve-and-compare containment check spuriously
        reject a perfectly valid path. A pure lexical check has no such race —
        and, as a side benefit, is immune to the TOCTOU/symlink tricks that
        resolve-and-compare traversal checks are notoriously vulnerable to.
        """
        if "\\" in path:
            raise ValueError(f"invalid storage path: {path!r}")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"invalid storage path: {path!r}")
        return self._root / pure

    def write_bytes(self, path: str, data: bytes) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        _retry_on_permission_error(lambda: full.write_bytes(data))

    def read_bytes(self, path: str) -> bytes:
        full = self._resolve(path)
        return _retry_on_permission_error(full.read_bytes)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def delete(self, path: str) -> None:
        # unlink(missing_ok=True) rather than exists()-then-unlink(): the latter is
        # a check-then-act race under concurrent deletes of the same path (two
        # callers can both see exists()==True, then the second unlink() raises).
        # A delete of an already-deleted file is a no-op either way; missing_ok
        # only suppresses FileNotFoundError, so PermissionError still goes through
        # the shared retry helper above.
        full = self._resolve(path)
        _retry_on_permission_error(lambda: full.unlink(missing_ok=True))

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        results = []
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    results.append(str(p.relative_to(self._root)).replace("\\", "/"))
            except OSError:
                # Entry vanished (or, on Windows, was transiently lock-contended)
                # between being yielded by rglob's directory walk and the is_file()
                # stat call, because something else deleted it concurrently. Same
                # benign race as in DatasetStore.list() -- skip it.
                continue
        return results


class SpacesBackend(StorageBackend):
    """DigitalOcean Spaces (S3-compatible). Keys map straight to `path` — no local
    filesystem involved, so the lexical traversal guard in `LocalDirBackend` doesn't
    apply here; a `path` containing `..` is just an ordinary (if odd) object key.

    `prefix` namespaces every key under e.g. `statistician-mcp/` so this backend
    can safely share one bucket with other services (each given its own prefix)
    without their keys ever colliding. The prefix is added/stripped transparently
    -- callers (DatasetStore, ArtifactStore) only ever see `path` relative to this
    backend's own root, exactly as with LocalDirBackend."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
        prefix: str = "",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(s3={"addressing_style": "virtual"}),
        )

    def _key(self, path: str) -> str:
        return f"{self._prefix}/{path}" if self._prefix else path

    @staticmethod
    def _is_not_found(exc: ClientError) -> bool:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in ("404", "NoSuchKey", "NotFound") or status == 404

    def write_bytes(self, path: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)

    def read_bytes(self, path: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._key(path))
        except ClientError as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(path) from exc
            raise
        return response["Body"].read()

    def exists(self, path: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(path))
        except ClientError as exc:
            if self._is_not_found(exc):
                return False
            raise
        return True

    def delete(self, path: str) -> None:
        # S3-style delete is already idempotent -- no error on a missing key.
        self._client.delete_object(Bucket=self._bucket, Key=self._key(path))

    def list(self, prefix: str) -> list[str]:
        results: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        strip_len = len(self._prefix) + 1 if self._prefix else 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                results.append(obj["Key"][strip_len:])
        return results
