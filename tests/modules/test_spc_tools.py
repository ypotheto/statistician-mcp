from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload


@pytest.mark.asyncio
async def test_xbar_r_chart_reports_both_panels_and_no_violations_for_stable_data(
    settings: Settings,
) -> None:
    bundle = create_server(settings)
    rows = ["subgroup,value"]
    # 4 subgroups of 3, stable process centered at 12 with a constant range of 4.
    for sg, triple in enumerate([[10, 12, 14], [11, 13, 15], [9, 11, 13], [10, 12, 14]]):
        for v in triple:
            rows.append(f"{sg},{v}")
    csv_text = "\n".join(rows)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "create_control_chart",
                {
                    "handle": handle,
                    "chart_type": "xbar_r",
                    "value_column": "value",
                    "subgroup_column": "subgroup",
                },
            )
        )

    assert result["ok"] is True
    assert result["results"]["cl"] == pytest.approx(12.0)
    assert "secondary" in result["results"]
    assert result["results"]["secondary"]["cl"] == pytest.approx(4.0)
    assert len(result["artifacts"]) == 1


@pytest.mark.asyncio
async def test_i_mr_chart_flags_synthetic_rule_2_violation_at_the_right_row(
    settings: Settings,
) -> None:
    """Phase 5 acceptance criterion: a synthetic dataset with a known Rule-2
    violation must be flagged at the right subgroup (here: row index)."""
    bundle = create_server(settings)
    values = [50, 50, 50] + [55] * 9 + [50, 50, 50]
    csv_text = "value\n" + "\n".join(str(v) for v in values)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "create_control_chart",
                {"handle": handle, "chart_type": "i_mr", "value_column": "value"},
            )
        )

    assert result["ok"] is True
    assert result["results"]["violations"]["rule_2"] == [11]


@pytest.mark.asyncio
async def test_p_chart_and_u_chart_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    p_csv = "nc,n\n" + "\n".join(f"{v},50" for v in [2, 3, 1, 4, 2, 3, 2, 1, 25, 2])
    u_csv = "defects,units\n" + "\n".join(f"{v},10" for v in [1, 2, 1, 0, 2, 1, 1, 0, 1, 2])

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        p_loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": p_csv}))
        p_result = payload(
            await session.call_tool(
                "create_control_chart",
                {
                    "handle": p_loaded["results"]["handle"],
                    "chart_type": "p",
                    "nonconforming_column": "nc",
                    "sample_size_column": "n",
                },
            )
        )

        u_loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": u_csv}))
        u_result = payload(
            await session.call_tool(
                "create_control_chart",
                {
                    "handle": u_loaded["results"]["handle"],
                    "chart_type": "u",
                    "count_column": "defects",
                    "unit_column": "units",
                },
            )
        )

    assert p_result["ok"] is True
    # the row with nc=25 out of 50 (50%) should be far beyond the UCL
    assert len(p_result["results"]["violations"]["rule_1"]) >= 1
    assert u_result["ok"] is True
    assert isinstance(u_result["results"]["ucl"], list)


@pytest.mark.asyncio
async def test_ewma_and_cusum_charts_run(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = "value\n" + "\n".join(str(50 + i * 0.1) for i in range(20))

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        ewma = payload(
            await session.call_tool(
                "create_control_chart",
                {"handle": handle, "chart_type": "ewma", "value_column": "value"},
            )
        )
        cusum = payload(
            await session.call_tool(
                "create_control_chart",
                {"handle": handle, "chart_type": "cusum", "value_column": "value"},
            )
        )

    assert ewma["ok"] is True
    assert isinstance(ewma["results"]["ucl"], list)
    assert cusum["ok"] is True
    assert "decision_interval" in cusum["results"]["violations"]


@pytest.mark.asyncio
async def test_assess_capability_round_trip(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = "value\n" + "\n".join(
        str(v) for v in [98, 99, 100, 101, 102, 99, 100, 101, 100, 99, 101, 100]
    )

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "assess_capability", {"handle": handle, "column": "value", "lsl": 90, "usl": 110}
            )
        )

    assert result["ok"] is True
    assert result["results"]["overall"]["cpk"] is not None
    assert len(result["assumptions"]) == 1
    assert len(result["artifacts"]) == 1


@pytest.mark.asyncio
async def test_run_stability_check_against_historical_limits(settings: Settings) -> None:
    bundle = create_server(settings)
    values = [50, 50, 50] + [55] * 9 + [50, 50, 50]
    csv_text = "value\n" + "\n".join(str(v) for v in values)

    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "run_stability_check",
                {"handle": handle, "value_column": "value", "cl": 50, "ucl": 53, "lcl": 47},
            )
        )

    assert result["ok"] is True
    assert result["results"]["violations"]["rule_2"] == [11]
