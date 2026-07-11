from __future__ import annotations

from typing import Any, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.errors import ValidationError
from statistician_mcp.stats import power as stats_power
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id

Alternative = Literal["two-sided", "less", "greater"]
TestFamily = Literal["one_sample_t", "two_sample_t", "paired_t", "proportion", "anova"]


def register_power_tools(mcp: FastMCP, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("compute_power_or_sample_size")
    def compute_power_or_sample_size(
        test_family: TestFamily,
        effect_size: float | None = None,
        n: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
        alternative: Alternative = "two-sided",
        ratio: float = 1.0,
        k_groups: int = 2,
        prop1: float | None = None,
        prop2: float | None = None,
    ) -> dict[str, Any]:
        """Solve for whichever of {effect_size, n, alpha, power} is omitted, for a
        t-test (one/two-sample/paired), a proportion test (pass prop1/prop2 instead
        of effect_size), or a one-way ANOVA (pass k_groups). Exactly one of
        effect_size/n/alpha/power must be omitted."""
        if test_family == "proportion":
            if prop1 is None or prop2 is None:
                raise ValidationError("test_family='proportion' requires 'prop1' and 'prop2'")
            result = stats_power.solve_power_proportion(
                prop1, prop2, n, alpha, power, alternative, ratio
            )
        elif test_family == "anova":
            result = stats_power.solve_power_anova(k_groups, effect_size, n, alpha, power)
        else:
            if effect_size is None:
                raise ValidationError(f"test_family='{test_family}' requires 'effect_size'")
            result = stats_power.solve_power_t_test(
                test_family, effect_size, n, alpha, power, alternative, ratio
            )

        return envelope.ok_envelope(result, interpretation=_power_interpretation(result))

    @mcp.tool()
    @envelope.tool("plot_power_curve")
    def plot_power_curve(
        test_family: TestFamily,
        effect_size: float,
        alpha: float = 0.05,
        n_max: float = 200,
        alternative: Alternative = "two-sided",
        ratio: float = 1.0,
        k_groups: int = 2,
    ) -> dict[str, Any]:
        """Plot statistical power vs. sample size for a given effect size and alpha.
        Use to show a stakeholder how power grows with sample size for a planned study."""
        if n_max <= 4:
            raise ValidationError("n_max must be greater than 4")
        n_values = np.linspace(4, n_max, 60).tolist()
        points = stats_power.power_curve_points(
            test_family,
            effect_size,
            alpha,
            n_values,
            alternative=alternative,
            ratio=ratio,
            k_groups=k_groups,
        )

        fig, ax = new_figure(figsize=(7.0, 4.5))
        ax.plot([p["n"] for p in points], [p["power"] for p in points], color="#4c72b0")
        ax.axhline(0.8, color="#c44e52", linestyle="--", linewidth=1, label="power=0.80")
        ax.set_xlabel("n")
        ax.set_ylabel("power")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Power curve ({test_family}, effect size={effect_size}, alpha={alpha})")
        ax.legend(fontsize=8)
        record = artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="power_curve.png",
            data=render_png(fig),
            media_type="image/png",
        )
        return envelope.ok_envelope({"points": points}, artifacts=[record])


def _power_interpretation(result: dict[str, Any]) -> str:
    solved_for = result["solved_for"]
    value = result[solved_for]
    if solved_for == "n":
        return f"Requires n={value:.1f} per group to detect this effect at the given alpha/power."
    if solved_for == "power":
        return f"This design has power={value:.3f} to detect the given effect size."
    if solved_for == "effect_size":
        return f"The smallest detectable effect size under this design is {value:.3f}."
    return f"Solved alpha={value:.4g} for the given n/power/effect size."
