from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from statistician_mcp.config import Settings
from statistician_mcp.server import create_server
from tests.conftest import payload


@pytest.mark.asyncio
async def test_recommend_analysis_matches_keyword_rules(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        capability = payload(
            await session.call_tool(
                "recommend_analysis",
                {"question": "Is my process capable given these spec limits?"},
            )
        )
        equivalence = payload(
            await session.call_tool(
                "recommend_analysis", {"question": "I want to know if two lots are equivalent"}
            )
        )
        power = payload(
            await session.call_tool(
                "recommend_analysis", {"question": "How many samples do I need for adequate power?"}
            )
        )
        fallback = payload(
            await session.call_tool("recommend_analysis", {"question": "xyzzy plugh"})
        )

    assert capability["ok"] is True
    assert capability["results"]["recommendations"][0]["tool"] == "assess_capability"
    assert equivalence["results"]["recommendations"][0]["tool"] == "test_equivalence"
    assert power["results"]["recommendations"][0]["tool"] == "compute_power_or_sample_size"
    assert fallback["results"]["recommendations"][0]["tool"] == "summarize_columns"


@pytest.mark.asyncio
async def test_recommend_analysis_uses_dataset_profile_when_no_keyword_matches(
    settings: Settings,
) -> None:
    bundle = create_server(settings)
    csv_text = "value,grp\n1,a\n2,b\n3,a\n4,b\n"
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        loaded = payload(await session.call_tool("load_dataset_from_csv", {"csv_text": csv_text}))
        handle = loaded["results"]["handle"]

        result = payload(
            await session.call_tool(
                "recommend_analysis", {"question": "what should I look at", "handle": handle}
            )
        )

    assert result["ok"] is True
    assert result["results"]["recommendations"][0]["tool"] == "compare_multiple_groups"


@pytest.mark.asyncio
async def test_explain_concept_lookup_and_listing(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        listing = payload(await session.call_tool("explain_concept", {"concept": ""}))
        cpk = payload(await session.call_tool("explain_concept", {"concept": "cp_vs_cpk"}))
        alias_form = payload(await session.call_tool("explain_concept", {"concept": "Cp Vs Cpk"}))
        unknown = payload(
            await session.call_tool("explain_concept", {"concept": "not_a_real_concept"})
        )

    assert listing["ok"] is True
    assert len(listing["results"]["available_concepts"]) >= 30
    assert cpk["ok"] is True
    assert cpk["results"]["title"] == "Cp vs. Cpk"
    assert alias_form["ok"] is True and alias_form["results"]["concept"] == "cp_vs_cpk"
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_all_registered_prompts_are_present(settings: Settings) -> None:
    bundle = create_server(settings)
    async with create_connected_server_and_client_session(bundle.mcp) as session:
        prompts = await session.list_prompts()
        names = {p.name for p in prompts.prompts}
        assert names == {"plan_an_experiment", "analyze_my_experiment", "set_up_spc"}

        result = await session.get_prompt(
            "set_up_spc", {"handle": "ds_test", "value_column": "diameter"}
        )
        text = result.messages[0].content.text
        assert "diameter" in text
        assert "i_mr" in text
