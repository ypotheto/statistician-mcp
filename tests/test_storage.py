from __future__ import annotations

from collections.abc import Iterator

import boto3
import pytest
from moto import mock_aws

from statistician_mcp.storage import LocalDirBackend, SpacesBackend, StorageBackend

# moto only intercepts AWS-shaped hostnames (it patches botocore's transport layer
# based on recognized endpoint patterns) -- a fully custom domain like a real DO
# Spaces endpoint would fall through to a real network call, so the fixture points
# at a plain S3 host instead. This still exercises the exact same SpacesBackend
# code path (a boto3 S3 client pointed at an explicit `endpoint_url`); only the
# hostname differs from production.
_FAKE_ENDPOINT = "https://s3.amazonaws.com"
_FAKE_BUCKET = "test-bucket"
_FAKE_REGION = "us-east-1"


@pytest.fixture(params=["local", "spaces"])
def backend(request: pytest.FixtureRequest, tmp_path) -> Iterator[StorageBackend]:
    """Runs every test in this file against both backends, so a divergence between
    `LocalDirBackend` and `SpacesBackend` shows up as a normal test failure."""
    if request.param == "local":
        yield LocalDirBackend(tmp_path)
        return

    with mock_aws():
        boto3.client("s3", region_name=_FAKE_REGION, endpoint_url=_FAKE_ENDPOINT).create_bucket(
            Bucket=_FAKE_BUCKET
        )
        # A non-empty prefix by default (matching the real STATMCP_SPACES_PREFIX
        # default) so every contract test below also exercises the prefix-strip
        # path in list(), not just the empty-prefix case.
        yield SpacesBackend(
            bucket=_FAKE_BUCKET,
            endpoint_url=_FAKE_ENDPOINT,
            access_key="test",
            secret_key="test",
            region=_FAKE_REGION,
            prefix="statistician-mcp",
        )


def test_write_then_read_roundtrips(backend: StorageBackend) -> None:
    backend.write_bytes("workspaces/ws1/datasets/ds_1.parquet", b"hello world")
    assert backend.read_bytes("workspaces/ws1/datasets/ds_1.parquet") == b"hello world"


def test_write_overwrites_existing_key(backend: StorageBackend) -> None:
    backend.write_bytes("a/b.txt", b"first")
    backend.write_bytes("a/b.txt", b"second")
    assert backend.read_bytes("a/b.txt") == b"second"


def test_read_missing_key_raises_file_not_found(backend: StorageBackend) -> None:
    with pytest.raises(FileNotFoundError):
        backend.read_bytes("does/not/exist.txt")


def test_exists_true_after_write_false_before(backend: StorageBackend) -> None:
    assert backend.exists("a/b.txt") is False
    backend.write_bytes("a/b.txt", b"data")
    assert backend.exists("a/b.txt") is True


def test_delete_removes_key(backend: StorageBackend) -> None:
    backend.write_bytes("a/b.txt", b"data")
    backend.delete("a/b.txt")
    assert backend.exists("a/b.txt") is False


def test_delete_missing_key_is_idempotent(backend: StorageBackend) -> None:
    backend.delete("never/written.txt")  # must not raise


def test_list_returns_only_keys_under_prefix(backend: StorageBackend) -> None:
    backend.write_bytes("workspaces/ws1/datasets/ds_1.parquet", b"1")
    backend.write_bytes("workspaces/ws1/datasets/ds_2.parquet", b"2")
    backend.write_bytes("workspaces/ws2/datasets/ds_3.parquet", b"3")

    results = backend.list("workspaces/ws1")

    assert sorted(results) == [
        "workspaces/ws1/datasets/ds_1.parquet",
        "workspaces/ws1/datasets/ds_2.parquet",
    ]


def test_list_empty_prefix_returns_empty(backend: StorageBackend) -> None:
    assert backend.list("workspaces/nothing-here") == []


def test_spaces_prefix_isolates_services_sharing_one_bucket() -> None:
    """Two services (each their own SpacesBackend, own prefix) can share one
    bucket without ever seeing -- or colliding with -- each other's keys."""
    with mock_aws():
        boto3.client("s3", region_name=_FAKE_REGION, endpoint_url=_FAKE_ENDPOINT).create_bucket(
            Bucket=_FAKE_BUCKET
        )
        statistician = SpacesBackend(
            bucket=_FAKE_BUCKET,
            endpoint_url=_FAKE_ENDPOINT,
            access_key="test",
            secret_key="test",
            region=_FAKE_REGION,
            prefix="statistician-mcp",
        )
        chemistry = SpacesBackend(
            bucket=_FAKE_BUCKET,
            endpoint_url=_FAKE_ENDPOINT,
            access_key="test",
            secret_key="test",
            region=_FAKE_REGION,
            prefix="chem-mcp",
        )

        statistician.write_bytes("workspaces/ws1/datasets/ds_1.parquet", b"stats-data")
        chemistry.write_bytes("workspaces/ws1/datasets/ds_1.parquet", b"chem-data")

        # Same root-relative path in both services, no collision: each backend
        # only ever sees its own prefix's copy.
        assert statistician.read_bytes("workspaces/ws1/datasets/ds_1.parquet") == b"stats-data"
        assert chemistry.read_bytes("workspaces/ws1/datasets/ds_1.parquet") == b"chem-data"

        # list() never leaks the other service's keys, or its own prefix.
        assert statistician.list("workspaces/ws1") == ["workspaces/ws1/datasets/ds_1.parquet"]
        assert chemistry.list("workspaces/ws1") == ["workspaces/ws1/datasets/ds_1.parquet"]

        statistician.delete("workspaces/ws1/datasets/ds_1.parquet")
        assert statistician.exists("workspaces/ws1/datasets/ds_1.parquet") is False
        assert chemistry.exists("workspaces/ws1/datasets/ds_1.parquet") is True
