from __future__ import annotations

import pandas as pd
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.datasets import get_dataframe_for_analysis
from statistician_mcp.errors import ValidationError
from statistician_mcp.server import create_server
from tests.conftest import payload


def test_get_dataframe_for_analysis_enforces_the_row_cap(settings: Settings) -> None:
    bundle = create_server(settings)
    store = bundle.dataset_store
    df = pd.DataFrame({"x": range(10)})
    info = store.create("local", df, "small")

    # a small max_rows override proves the cap is actually enforced without
    # needing to construct/store a literal 200k-row dataset in a fast unit test.
    assert len(get_dataframe_for_analysis(store, "local", info.handle, max_rows=20)) == 10
    with pytest.raises(ValidationError):
        get_dataframe_for_analysis(store, "local", info.handle, max_rows=5)


@pytest.mark.asyncio
async def test_compare_means_rejects_an_oversized_dataset(settings: Settings) -> None:
    """compare_means (modules/inference.py) previously called store.get_dataframe()
    directly, bypassing the 200k-row analysis cap that only eda.py enforced --
    this is a regression test for that fix, using a tiny dataset injected directly
    via DatasetStore.create() (bypassing the inline-CSV tool's own 2MB size cap,
    which a literal 200k-row CSV would otherwise collide with) plus a monkeypatched
    threshold so the test itself stays fast."""
    import statistician_mcp.modules.inference as inference_module

    bundle = create_server(settings)
    df = pd.DataFrame({"value": range(10), "grp": ["a", "b"] * 5})
    info = bundle.dataset_store.create("local", df, "tiny")

    original = inference_module.get_dataframe_for_analysis
    inference_module.get_dataframe_for_analysis = (
        lambda store, workspace_id, handle: original(store, workspace_id, handle, max_rows=5)
    )
    try:
        async with create_connected_server_and_client_session(bundle.mcp) as session:
            result = payload(
                await session.call_tool(
                    "compare_means",
                    {"handle": info.handle, "column": "value", "group_column": "grp"},
                )
            )
    finally:
        inference_module.get_dataframe_for_analysis = original

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"
    assert "row" in result["error"]["message"]
