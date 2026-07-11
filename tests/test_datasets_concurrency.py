from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from statistician_mcp.datasets import DatasetStore
from statistician_mcp.storage import LocalDirBackend


def test_dataset_store_survives_concurrent_create_read_delete(tmp_path: Path) -> None:
    store = DatasetStore(LocalDirBackend(tmp_path), max_cache_entries=4)
    df = pd.DataFrame({"x": range(20)})
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(25):
                info = store.create("local", df, f"ds-{i}")
                fetched = store.get_dataframe("local", info.handle)
                assert len(fetched) == len(df)
                store.get_info("local", info.handle)
                store.delete("local", info.handle)
        except BaseException as exc:  # noqa: BLE001 - want to see genuinely anything
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(worker, range(16)))

    assert errors == []
    # the cache must never grow past its configured bound, regardless of how many
    # threads raced to populate it concurrently.
    assert len(store._cache) <= store._max_cache_entries
