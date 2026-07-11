from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload


def _two_group_csv() -> str:
    rows = ["value,grp"]
    for i in range(30):
        grp = "a" if i % 2 == 0 else "b"
        value = 10 + (i % 5) + (4 if grp == "b" else 0)
        rows.append(f"{value},{grp}")
    return "\n".join(rows)


def _three_group_csv() -> str:
    rows = ["value,grp"]
    for i in range(45):
        grp = ["a", "b", "c"][i % 3]
        value = 10 + (i % 4) + (6 if grp == "c" else 0)
        rows.append(f"{value},{grp}")
    return "\n".join(rows)


@pytest.mark.asyncio
async def test_compare_means_two_sample_mode(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _two_group_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "compare_means", {"handle": handle, "column": "value", "group_column": "grp"}
            )
        )

    assert result["ok"] is True
    assert result["results"]["test"] == "Welch two-sample t-test"
    assert len(result["assumptions"]) == 3
    assert "Welch" in result["interpretation"] or "p=" in result["interpretation"]
    assert result["meta"]["dataset"] == handle


@pytest.mark.asyncio
async def test_compare_means_one_sample_and_paired_modes(settings: Settings) -> None:
    bundle = create_server(settings)
    csv_text = "a,b\n10,9\n12,11\n9,8\n15,14\n11,10\n13,12\n"
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        one_sample = payload(
            await session.call_tool(
                "compare_means", {"handle": handle, "column": "a", "mu": 10.0}
            )
        )
        paired = payload(
            await session.call_tool(
                "compare_means",
                {"handle": handle, "column": "a", "paired_with_column": "b"},
            )
        )
        no_mode = payload(
            await session.call_tool("compare_means", {"handle": handle, "column": "a"})
        )
        two_modes = payload(
            await session.call_tool(
                "compare_means",
                {"handle": handle, "column": "a", "mu": 10.0, "paired_with_column": "b"},
            )
        )

    assert one_sample["ok"] is True and one_sample["results"]["test"] == "one-sample t-test"
    assert paired["ok"] is True and paired["results"]["test"] == "paired t-test"
    assert no_mode["ok"] is False and no_mode["error"]["code"] == "validation_error"
    assert two_modes["ok"] is False and two_modes["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_compare_means_rejects_group_column_with_wrong_level_count(
    settings: Settings,
) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _three_group_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "compare_means", {"handle": handle, "column": "value", "group_column": "grp"}
            )
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"
    assert "3" in result["error"]["message"]


@pytest.mark.asyncio
async def test_compare_multiple_groups_renders_artifact_and_posthoc(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _three_group_csv()})
        )
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "compare_multiple_groups",
                {"handle": handle, "column": "value", "group_column": "grp"},
            )
        )

    assert result["ok"] is True
    assert result["results"]["posthoc"] is not None
    assert len(result["artifacts"]) == 1
    assert len(result["assumptions"]) == 4  # 3 normality checks + 1 equal-variance check


@pytest.mark.asyncio
async def test_compare_proportions_all_three_modes(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        one_sample = payload(
            await session.call_tool(
                "compare_proportions", {"count_a": 8, "nobs_a": 20, "value": 0.5}
            )
        )
        two_sample = payload(
            await session.call_tool(
                "compare_proportions",
                {"count_a": 45, "nobs_a": 100, "count_b": 60, "nobs_b": 100},
            )
        )
        contingency = payload(
            await session.call_tool(
                "compare_proportions", {"contingency_table": [[10, 2], [3, 15]]}
            )
        )
        no_mode = payload(
            await session.call_tool("compare_proportions", {"count_a": 8, "nobs_a": 20})
        )
        missing_count_a = payload(
            await session.call_tool("compare_proportions", {"value": 0.5})
        )

    assert one_sample["ok"] is True and "binomial" in one_sample["results"]["test"]
    assert two_sample["ok"] is True and "two-sample" in two_sample["results"]["test"]
    assert contingency["ok"] is True and "chi-square" in contingency["results"]["test"]
    assert no_mode["ok"] is False and no_mode["error"]["code"] == "validation_error"
    assert missing_count_a["ok"] is False and missing_count_a["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_test_equivalence_two_sample_and_one_sample(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _two_group_csv()})
        )
        handle = loaded["results"]["handle"]

        two_sample = payload(
            await session.call_tool(
                "test_equivalence",
                {
                    "handle": handle,
                    "column": "value",
                    "group_column": "grp",
                    "low_bound": -10,
                    "high_bound": 10,
                },
            )
        )
        one_sample = payload(
            await session.call_tool(
                "test_equivalence",
                {"handle": handle, "column": "value", "mu": 12.0, "low_bound": -5, "high_bound": 5},
            )
        )

    assert two_sample["ok"] is True and "equivalent" in two_sample["interpretation"]
    assert one_sample["ok"] is True and "equivalent" in one_sample["interpretation"]


@pytest.mark.asyncio
async def test_compute_confidence_interval_all_parameters(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _two_group_csv()})
        )
        handle = loaded["results"]["handle"]

        mean_ci = payload(
            await session.call_tool(
                "compute_confidence_interval",
                {"parameter": "mean", "handle": handle, "column": "value"},
            )
        )
        sd_ci = payload(
            await session.call_tool(
                "compute_confidence_interval",
                {"parameter": "sd", "handle": handle, "column": "value"},
            )
        )
        diff_ci = payload(
            await session.call_tool(
                "compute_confidence_interval",
                {
                    "parameter": "mean_difference",
                    "handle": handle,
                    "column": "value",
                    "group_column": "grp",
                },
            )
        )
        prop_ci = payload(
            await session.call_tool(
                "compute_confidence_interval", {"parameter": "proportion", "count": 45, "nobs": 100}
            )
        )
        missing_args = payload(
            await session.call_tool("compute_confidence_interval", {"parameter": "mean"})
        )

    assert mean_ci["ok"] is True and mean_ci["results"]["parameter"] == "mean"
    assert sd_ci["ok"] is True and sd_ci["results"]["parameter"] == "standard deviation"
    assert diff_ci["ok"] is True and diff_ci["results"]["parameter"] == "difference in means"
    assert prop_ci["ok"] is True and prop_ci["results"]["parameter"] == "proportion"
    assert missing_args["ok"] is False


@pytest.mark.asyncio
async def test_test_variance_two_and_three_groups(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        two_loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _two_group_csv()})
        )
        two_handle = two_loaded["results"]["handle"]
        three_loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _three_group_csv()})
        )
        three_handle = three_loaded["results"]["handle"]

        two_group = payload(
            await session.call_tool(
                "test_variance", {"handle": two_handle, "column": "value", "group_column": "grp"}
            )
        )
        three_group = payload(
            await session.call_tool(
                "test_variance", {"handle": three_handle, "column": "value", "group_column": "grp"}
            )
        )

    assert two_group["ok"] is True
    assert two_group["results"]["test"] == "F-test for equality of variances"
    assert three_group["ok"] is True and three_group["results"]["test"] == "Levene"


@pytest.mark.asyncio
async def test_power_tools(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        solved = payload(
            await session.call_tool(
                "compute_power_or_sample_size",
                {"test_family": "two_sample_t", "effect_size": 0.5, "alpha": 0.05, "power": 0.8},
            )
        )
        curve = payload(
            await session.call_tool(
                "plot_power_curve",
                {"test_family": "two_sample_t", "effect_size": 0.5, "alpha": 0.05, "n_max": 150},
            )
        )
        missing_effect_size = payload(
            await session.call_tool(
                "compute_power_or_sample_size",
                {"test_family": "two_sample_t", "alpha": 0.05, "power": 0.8},
            )
        )

    assert solved["ok"] is True
    assert solved["results"]["n"] == pytest.approx(63.765610588911635)
    assert curve["ok"] is True and len(curve["artifacts"]) == 1
    assert missing_effect_size["ok"] is False


@pytest.mark.asyncio
async def test_scripted_conversation_load_describe_compare_means(settings: Settings) -> None:
    """Adapted acceptance test for the Phase 3 'scripted conversation' criterion.

    The plan's Phase 3 acceptance text names a `recommend_analysis` step, but that
    tool is Module I and doesn't exist until Phase 6 -- this exercises the same
    load -> inspect -> analyze flow and checks the interpretation mentions the
    assumption status, without calling a tool that isn't built yet.
    """
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(
            await session.call_tool("load_dataset_from_csv", {"csv_text": _two_group_csv()})
        )
        handle = loaded["results"]["handle"]

        described = payload(await session.call_tool("describe_dataset", {"handle": handle}))
        assert described["ok"] is True

        compared = payload(
            await session.call_tool(
                "compare_means", {"handle": handle, "column": "value", "group_column": "grp"}
            )
        )

    assert compared["ok"] is True
    assert compared["assumptions"], "compare_means must report assumption checks"
    assert any(a["status"] in ("pass", "warn", "fail") for a in compared["assumptions"])
    assert "p=" in compared["interpretation"]
