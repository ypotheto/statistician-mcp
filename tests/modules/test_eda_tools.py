from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload

# Hand-computable fixture: mean=5.5, sample sd=3.0276503540..., quartiles via
# pandas' default linear-interpolation method: Q1=3.25, median=5.5, Q3=7.75.
# Perfectly symmetric -> skew is exactly 0.
SUMMARY_CSV = "n\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"

# b = 2*a (r=+1 exactly), c = -2*a + 12 (r=-1 exactly against both a and b).
CORR_CSV = "a,b,c\n1,2,10\n2,4,8\n3,6,6\n4,8,4\n"


@pytest.mark.asyncio
async def test_summarize_columns_matches_hand_computed_values(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": SUMMARY_CSV})
        )
        handle = loaded["results"]["handle"]

        summarized = payload(await session.call_tool("summarize_columns", {"handle": handle}))

    assert summarized["ok"] is True
    stats = summarized["results"]["numeric"]["n"]
    assert stats["n"] == 10
    assert stats["n_missing"] == 0
    assert stats["mean"] == pytest.approx(5.5)
    assert stats["sd"] == pytest.approx(3.0276503540974917)
    assert stats["q1"] == pytest.approx(3.25)
    assert stats["median"] == pytest.approx(5.5)
    assert stats["q3"] == pytest.approx(7.75)
    assert stats["skew"] == pytest.approx(0.0, abs=1e-9)
    assert isinstance(stats["kurtosis"], float)


@pytest.mark.asyncio
async def test_compute_correlations_matches_hand_computed_values(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": CORR_CSV}))
        handle = loaded["results"]["handle"]

        result = payload(await session.call_tool("compute_correlations", {"handle": handle}))

    assert result["ok"] is True
    matrix = result["results"]["matrix"]
    assert matrix["a"]["b"] == pytest.approx(1.0)
    assert matrix["a"]["c"] == pytest.approx(-1.0)
    assert matrix["b"]["c"] == pytest.approx(-1.0)
    strong_pairs = {(p["a"], p["b"]) for p in result["results"]["strong_pairs"]}
    assert strong_pairs == {("a", "b"), ("a", "c"), ("b", "c")}
    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_eda_tools_return_valid_envelopes_and_render_artifacts(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = (
        "value,other,group,cat,t\n"
        "1.0,5,a,x,2024-01-01\n"
        "2.0,3,a,y,2024-01-02\n"
        "3.5,8,b,x,2024-01-03\n"
        "40.0,1,b,y,2024-01-04\n"
        "2.5,6,a,x,2024-01-05\n"
        "3.0,4,b,y,2024-01-06\n"
    )
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        dist = payload(
            await session.call_tool("plot_distribution", {"handle": handle, "column": "value"})
        )
        normality = payload(
            await session.call_tool("test_normality", {"handle": handle, "column": "value"})
        )
        outliers = payload(
            await session.call_tool("detect_outliers", {"handle": handle, "column": "value"})
        )
        scatter = payload(
            await session.call_tool(
                "plot_scatter", {"handle": handle, "x": "value", "y": "other", "overlay": "ols"}
            )
        )
        time_series = payload(
            await session.call_tool(
                "plot_time_series", {"handle": handle, "column": "value", "time_column": "t"}
            )
        )
        cross = payload(
            await session.call_tool(
                "crosstab", {"handle": handle, "row": "group", "col": "cat", "normalize": "row"}
            )
        )

    for result in (dist, normality, outliers, scatter, time_series, cross):
        assert result["ok"] is True
        assert "meta" in result and result["meta"]["dataset"] == handle

    for result_with_artifact in (dist, scatter, time_series):
        assert len(result_with_artifact["artifacts"]) == 1
        assert result_with_artifact["artifacts"][0]["url"].startswith("http")

    assert normality["assumptions"][0]["check"].startswith("normality")
    assert outliers["results"]["grubbs_test"]["applicable"] is True
    assert set(cross["results"]["percent"]["x"].keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_plot_scatter_rejects_identical_columns(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": "a\n1\n2\n3\n"})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool("plot_scatter", {"handle": handle, "x": "a", "y": "a"})
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"
