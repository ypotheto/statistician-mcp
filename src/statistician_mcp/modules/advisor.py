from __future__ import annotations

import importlib.resources
from functools import lru_cache
from typing import Any

import pandas as pd
import yaml
from mcp.server.fastmcp import FastMCP

from statistician_mcp import envelope
from statistician_mcp.datasets import DatasetStore
from statistician_mcp.errors import ValidationError
from statistician_mcp.workspace import get_current_workspace_id


@lru_cache(maxsize=1)
def _load_concepts() -> dict[str, dict[str, str]]:
    resource = importlib.resources.files("statistician_mcp.content").joinpath("concepts.yaml")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


def register_advisor_tools(mcp: FastMCP, store: DatasetStore) -> None:
    @mcp.tool()
    @envelope.tool("recommend_analysis")
    def recommend_analysis(question: str, handle: str | None = None) -> dict[str, Any]:
        """Recommend which tool(s) to use for a described analytical question, with
        rationale. Deterministic keyword/data-shape rules, not an LLM call — pass
        `handle` so the rules can also consider the dataset's column types."""
        profile = None
        if handle is not None:
            df = store.get_dataframe(get_current_workspace_id(), handle)
            profile = _profile_for_recommendation(df)

        recommendations = _recommend(question.lower(), profile)
        return envelope.ok_envelope(
            {"recommendations": recommendations},
            interpretation=(
                f"Top recommendation: {recommendations[0]['tool']}."
                if recommendations
                else "No specific rule matched; start with summarize_columns and describe_dataset."
            ),
        )

    @mcp.tool()
    @envelope.tool("explain_concept")
    def explain_concept(concept: str) -> dict[str, Any]:
        """Look up a short, plain-language explanation of a core statistical/DOE/
        SPC/MSA concept (e.g. 'cp_vs_cpk', 'nelson_rules', 'tost_equivalence').
        Call with an empty string to list all available concepts."""
        concepts = _load_concepts()
        if not concept:
            return envelope.ok_envelope({"available_concepts": sorted(concepts.keys())})
        key = concept.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in concepts:
            raise ValidationError(
                f"unknown concept '{concept}'",
                hint=f"available concepts: {', '.join(sorted(concepts.keys()))}",
            )
        entry = concepts[key]
        return envelope.ok_envelope(
            {"concept": key, "title": entry["title"], "explanation": entry["explanation"].strip()}
        )

    @mcp.prompt()
    def plan_an_experiment(goal: str, factor_names: list[str]) -> str:
        """Walk through planning a designed experiment for a stated goal."""
        factors = ", ".join(factor_names)
        return (
            f"I want to plan a designed experiment to: {goal}\n"
            f"The factors I'm considering are: {factors}\n\n"
            "Please help me by: (1) calling design_experiment to generate a run table "
            "(ask me for each factor's low/high range, and recommend a design_type — "
            "full_factorial for <=4 factors, fractional_factorial with resolution=4 or "
            "5 for more, unless I want to screen many factors cheaply at resolution 3); "
            "(2) calling evaluate_design on the result to confirm it's orthogonal and "
            "has adequate power; (3) showing me the run table so I can go execute the "
            "physical experiment and record the response for each run."
        )

    @mcp.prompt()
    def analyze_my_experiment(handle: str, response: str, factor_names: list[str]) -> str:
        """Walk through analyzing a completed factorial or response-surface experiment."""
        factors = ", ".join(factor_names)
        return (
            f"I've run an experiment (dataset {handle}) with response '{response}' and "
            f"factors: {factors}. Please: (1) call analyze_factorial (or "
            "analyze_response_surface if this was a CCD/Box-Behnken design) to fit the "
            "effects model; (2) tell me which effects are significant using the "
            "half-normal/Pareto-of-effects output and Lenth's margin of error; (3) if "
            "I have a goal for the response, call optimize_response to recommend "
            "factor settings; (4) flag anything the lack-of-fit test or residual "
            "diagnostics suggest I should be cautious about."
        )

    @mcp.prompt()
    def set_up_spc(handle: str, value_column: str, subgroup_column: str | None = None) -> str:
        """Walk through setting up statistical process control for a measured characteristic."""
        subgroup_note = (
            f"grouped into subgroups by '{subgroup_column}' (use chart_type='xbar_r' or 'xbar_s')"
            if subgroup_column
            else "as individual sequential readings (use chart_type='i_mr')"
        )
        return (
            f"I want to set up SPC for '{value_column}' in dataset {handle}, {subgroup_note}. "
            "Please: (1) call summarize_columns and test_normality on this column first; "
            "(2) call create_control_chart with the recommended chart_type and tell me "
            "about any Nelson-rule violations found; (3) if I give you spec limits, also "
            "call assess_capability and tell me the Cpk/Ppk verdict and sigma level."
        )


def _profile_for_recommendation(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    binary_numeric_cols = [c for c in numeric_cols if set(df[c].dropna().unique()) <= {0, 1}]
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "binary_numeric_columns": binary_numeric_cols,
        "n_rows": len(df),
    }


_KEYWORD_RULES: list[tuple[list[str], str, str]] = [
    (["capab", "cpk", "ppk"], "assess_capability", "Question mentions process capability."),
    (
        ["control chart", "in control", "out of control", "spc"],
        "create_control_chart",
        "Question mentions control charts / process stability.",
    ),
    (
        ["gauge", "repeatability", "reproducibility", "measurement system"],
        "analyze_gauge_rr",
        "Question mentions a measurement system study.",
    ),
    (
        ["design an experiment", "doe", "factorial", "factors and levels"],
        "design_experiment",
        "Question mentions designing an experiment.",
    ),
    (
        ["equivalent", "equivalence", "same as"],
        "test_equivalence",
        "Question asks whether two things are equivalent, not just different.",
    ),
    (
        ["sample size", "how many samples", "how much power"],
        "compute_power_or_sample_size",
        "Question is about sample size or power planning.",
    ),
    (
        ["correlat", "relationship between"],
        "compute_correlations",
        "Question asks about relationships between numeric variables.",
    ),
    (
        ["outlier", "unusual value"],
        "detect_outliers",
        "Question asks about unusual/extreme values.",
    ),
    (
        ["distribution", "weibull", "reliability", "time to failure"],
        "fit_distribution",
        "Question is about the shape of a distribution or reliability data.",
    ),
    (
        ["predict", "regression", "model the relationship"],
        "fit_linear_model",
        "Question is about modeling/predicting a numeric response.",
    ),
    (
        ["classify", "logistic", "pass or fail", "probability of"],
        "fit_logistic_model",
        "Question is about modeling a binary outcome.",
    ),
    (
        ["different", "significant difference", "compare"],
        "compare_means",
        "Question asks whether groups differ.",
    ),
]


def _recommend(question: str, profile: dict[str, Any] | None) -> list[dict[str, str]]:
    recommendations = []
    for keywords, tool, rationale in _KEYWORD_RULES:
        if any(kw in question for kw in keywords):
            recommendations.append({"tool": tool, "rationale": rationale})

    if recommendations:
        return recommendations

    has_grouping_columns = (
        profile is not None and profile["categorical_columns"] and profile["numeric_columns"]
    )
    if has_grouping_columns:
        return [
            {
                "tool": "compare_multiple_groups",
                "rationale": "Dataset has both categorical and numeric columns; "
                "consider comparing the numeric column across group levels.",
            }
        ]
    return [
        {
            "tool": "summarize_columns",
            "rationale": "No specific question pattern matched; start by profiling the data.",
        }
    ]
