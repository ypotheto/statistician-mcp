from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from statistician_mcp import envelope
from statistician_mcp.errors import ValidationError
from statistician_mcp.utils.plotting import new_figure


@pytest.mark.asyncio
async def test_tool_decorator_closes_leaked_figures_on_stat_mcp_error() -> None:
    @envelope.tool("fake_tool_validation_error")
    def fake_tool() -> dict[str, object]:
        new_figure()  # create a figure, then fail before it's ever closed
        raise ValidationError("deliberate failure for this test")

    assert plt.get_fignums() == []
    result = await fake_tool()
    assert result["ok"] is False
    assert plt.get_fignums() == []


@pytest.mark.asyncio
async def test_tool_decorator_closes_leaked_figures_on_unexpected_exception() -> None:
    @envelope.tool("fake_tool_unexpected_error")
    def fake_tool() -> dict[str, object]:
        new_figure()
        raise RuntimeError("deliberate unexpected failure for this test")

    assert plt.get_fignums() == []
    result = await fake_tool()
    assert result["ok"] is False
    assert result["error"]["code"] == "internal_error"
    assert plt.get_fignums() == []
