from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from statistician_mcp.datasets import DatasetStore, profile_columns
from statistician_mcp.errors import DatasetNotFoundError
from statistician_mcp.storage import LocalDirBackend

DF = pd.DataFrame(
    {"n": [1, 2, 3, None], "cat": ["x", "y", "x", "x"], "t": ["hi", "there", "world", "!"]}
)


def test_local_dir_backend_blocks_path_traversal(tmp_path: Path) -> None:
    backend = LocalDirBackend(tmp_path)
    with pytest.raises(ValueError):
        backend.write_bytes("../escape.txt", b"data")


def test_profile_columns_classifies_dtypes() -> None:
    profiles = {p.name: p for p in profile_columns(DF)}
    assert profiles["n"].dtype == "numeric"
    assert profiles["n"].n_missing == 1
    assert profiles["cat"].dtype == "categorical"


def test_dataset_store_round_trip(tmp_path: Path) -> None:
    store = DatasetStore(LocalDirBackend(tmp_path))
    info = store.create("local", DF, "demo")

    assert info.n_rows == 4
    fetched = store.get_dataframe("local", info.handle)
    assert fetched.shape == DF.shape

    listed = store.list("local")
    assert [d.handle for d in listed] == [info.handle]

    store.delete("local", info.handle)
    assert store.list("local") == []
    with pytest.raises(DatasetNotFoundError):
        store.get_dataframe("local", info.handle)


def test_workspaces_are_isolated(tmp_path: Path) -> None:
    store = DatasetStore(LocalDirBackend(tmp_path))
    info = store.create("workspace-a", DF, "demo")

    assert store.list("workspace-b") == []
    with pytest.raises(DatasetNotFoundError):
        store.get_dataframe("workspace-b", info.handle)
