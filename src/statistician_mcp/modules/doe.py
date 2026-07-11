from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from scipy import stats as sp_stats

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore, get_dataframe_for_analysis
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats import desirability as stats_desirability
from statistician_mcp.stats import doe_analysis, doe_designs
from statistician_mcp.stats.doe_designs import FactorSpec
from statistician_mcp.utils.formulas import FormulaError
from statistician_mcp.utils.plotting import new_3d_figure, new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id

DesignType = Literal[
    "full_factorial", "fractional_factorial", "plackett_burman", "ccd", "box_behnken", "lhs"
]
Goal = Literal["maximize", "minimize", "target"]

MAX_FACTORS = 15
MAX_RUNS = 10_000


def register_doe_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("design_experiment")
    def design_experiment(
        design_type: DesignType,
        factors: dict[str, dict[str, Any]],
        center_points: int = 0,
        replicates: int = 1,
        resolution: int | None = None,
        generators: str | None = None,
        ccd_alpha: Literal["orthogonal", "rotatable"] = "orthogonal",
        ccd_face: Literal["circumscribed", "inscribed", "faced"] = "circumscribed",
        lhs_samples: int = 10,
        seed: int = 0,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Generate a designed experiment and save its run table as a new dataset.
        Each factor is `{"low": .., "high": ..}` (continuous) or `{"levels": [...]}`
        (discrete, full_factorial only). design_type: full_factorial,
        fractional_factorial (needs `resolution` or `generators`),
        plackett_burman, ccd, box_behnken, or lhs. Returns the run table inline, a
        CSV artifact, and — for fractional factorials — the alias structure."""
        if len(factors) > MAX_FACTORS:
            raise ValidationError(f"design_experiment supports at most {MAX_FACTORS} factors")

        parsed_factors = _parse_factors(factors)
        result = doe_designs.generate_design(
            design_type,
            parsed_factors,
            center_points=center_points,
            replicates=replicates,
            resolution=resolution,
            generators=generators,
            ccd_alpha=ccd_alpha,
            ccd_face=ccd_face,
            lhs_samples=lhs_samples,
            seed=seed,
        )
        if result["n_runs"] > MAX_RUNS:
            raise ValidationError(
                f"design has {result['n_runs']} runs, exceeding the {MAX_RUNS}-run limit"
            )

        run_table: pd.DataFrame = result["run_table"]
        info = store.create(get_current_workspace_id(), run_table, name or f"{design_type}_design")

        csv_record = artifacts.register(
            get_current_workspace_id(),
            kind="design",
            filename="design.csv",
            data=run_table.to_csv(index=False).encode("utf-8"),
            media_type="text/csv",
        )

        return envelope.ok_envelope(
            {
                "handle": info.handle,
                "design_type": design_type,
                "n_runs": result["n_runs"],
                "n_factors": result["n_factors"],
                "resolution": result["resolution"],
                "alias_map": result["alias_map"],
                "run_table": json.loads(run_table.to_json(orient="records")),
            },
            artifacts=[csv_record],
            interpretation=_design_interpretation(result),
            meta={"dataset": info.handle, "n_rows_used": info.n_rows},
        )

    @mcp.tool()
    @envelope.tool("evaluate_design")
    def evaluate_design(
        handle: str, factors: dict[str, dict[str, Any]], sigma: float = 1.0
    ) -> dict[str, Any]:
        """Diagnostics for a design already loaded as a dataset: orthogonality (max
        pairwise column correlation), alias structure, and power to detect a main
        effect of the given size at the given noise sigma. Use to sanity-check a
        design before running the experiment."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        parsed_factors = _parse_factors(factors)
        names = list(parsed_factors.keys())
        _require_columns(df, names)

        coded_matrix = _coded_matrix(df, parsed_factors)
        result = doe_designs.evaluate_design(coded_matrix, names, sigma=sigma)
        return envelope.ok_envelope(
            result,
            interpretation=_evaluate_design_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("analyze_factorial")
    def analyze_factorial(
        handle: str, response: str, factor_names: list[str], formula: str | None = None
    ) -> dict[str, Any]:
        """Fit an effects model to factorial response data: coefficient/effect
        table, half-normal and Pareto-of-effects plots, R²/adj-R², a lack-of-fit
        test (if replicates/center points exist), residual diagnostics, and a
        hierarchical model-reduction suggestion. Automatically analyzes in coded
        (-1/+1) units using each factor's `{name}_coded` column if the dataset has
        one (e.g. from design_experiment), else standardizes the raw column."""
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_columns(df, [response, *factor_names])
        working = _coded_working_frame(df, factor_names)

        try:
            result = doe_analysis.fit_factorial_model(working, response, factor_names, formula)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc
        plot_records = _render_factorial_plots(artifacts, result)

        return envelope.ok_envelope(
            {k: v for k, v in result.items() if k not in ("residuals", "fitted")},
            artifacts=plot_records,
            interpretation=_factorial_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("analyze_response_surface")
    def analyze_response_surface(
        handle: str, response: str, factor_names: list[str]
    ) -> dict[str, Any]:
        """Fit a second-order (quadratic) model to response-surface data (CCD/BBD):
        coefficient table, lack-of-fit test, stationary-point analysis (min/max/
        saddle), and contour + 3D surface plots over the first two factors."""
        if len(factor_names) < 2:
            raise ValidationError("analyze_response_surface needs at least 2 factors")
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_columns(df, [response, *factor_names])
        working = _coded_working_frame(df, factor_names)

        try:
            result = doe_analysis.fit_response_surface_model(working, response, factor_names)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc
        plot_records = _render_response_surface_plots(artifacts, result, factor_names)

        return envelope.ok_envelope(
            {k: v for k, v in result.items() if k not in ("residuals", "fitted")},
            artifacts=plot_records,
            interpretation=_response_surface_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("optimize_response")
    def optimize_response(
        handle: str,
        factor_names: list[str],
        responses: list[dict[str, Any]],
        n_starts: int = 20,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Find factor settings optimizing one or more responses via Derringer-Suich
        desirability. Each entry in `responses` is `{"column": str, "model_type":
        "linear"|"quadratic", "goal": "maximize"|"minimize"|"target", "low": float,
        "high": float, "target": float (if goal="target"), "weight": 1.0}`. Models
        are refit from the dataset each call (not persisted). Returns optimal
        factor settings in both coded and natural units, plus predicted responses."""
        if not responses:
            raise ValidationError("at least one response is required")
        df = get_dataframe_for_analysis(store, get_current_workspace_id(), handle)
        _require_columns(df, [r["column"] for r in responses] + factor_names)
        working = _coded_working_frame(df, factor_names)

        predict_fns = []
        goals = []
        response_summaries = []
        for spec in responses:
            column = spec["column"]
            model_type = spec.get("model_type", "linear")
            if model_type == "quadratic":
                fitted = doe_analysis.fit_response_surface_model(working, column, factor_names)
            else:
                fitted = doe_analysis.fit_factorial_model(working, column, factor_names)
            predict_fns.append(_build_predictor(fitted, factor_names, model_type))
            goals.append(
                {
                    "goal": spec["goal"],
                    "low": spec["low"],
                    "high": spec["high"],
                    "target": spec.get("target"),
                    "weight": spec.get("weight", 1.0),
                }
            )
            response_summaries.append({"column": column, "model_type": model_type})

        bounds = [(-1.0, 1.0)] * len(factor_names)
        result = stats_desirability.optimize_desirability(
            predict_fns, goals, bounds, n_starts, seed
        )

        natural_settings = _coded_to_natural_settings(df, factor_names, result["x"])
        return envelope.ok_envelope(
            {
                "coded_settings": dict(zip(factor_names, result["x"], strict=True)),
                "natural_settings": natural_settings,
                "desirability": result["desirability"],
                "predicted_responses": dict(
                    zip(
                        (r["column"] for r in response_summaries),
                        result["predicted_responses"],
                        strict=True,
                    )
                ),
                "responses": response_summaries,
            },
            interpretation=(
                f"Overall desirability {result['desirability']:.3f} achieved at the "
                "reported settings."
                if result["desirability"] > 0
                else "No factor setting satisfies all response goals simultaneously "
                "(desirability 0)."
            ),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )


def _parse_factors(factors: dict[str, dict[str, Any]]) -> dict[str, FactorSpec]:
    parsed = {}
    for factor_name, spec in factors.items():
        try:
            parsed[factor_name] = FactorSpec(
                name=factor_name,
                low=spec.get("low"),
                high=spec.get("high"),
                levels=spec.get("levels"),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    return parsed


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _coded_matrix(df: pd.DataFrame, factors: dict[str, FactorSpec]) -> np.ndarray:
    columns = []
    for factor_name, factor in factors.items():
        coded_col = f"{factor_name}_coded"
        if coded_col in df.columns:
            columns.append(df[coded_col].to_numpy(dtype=float))
            continue
        if factor.low is None or factor.high is None:
            raise ValidationError(
                f"factor '{factor_name}' needs low/high to standardize to coded units"
            )
        center, half = (factor.low + factor.high) / 2, (factor.high - factor.low) / 2
        columns.append(((df[factor_name] - center) / half).to_numpy(dtype=float))
    return np.column_stack(columns)


def _coded_working_frame(df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """A copy of `df` where each factor column has been overwritten with its coded
    (-1/+1-scaled) values, using the `{name}_coded` column if present (e.g. from
    design_experiment) or else standardizing the raw column by its own min/max."""
    working = df.copy()
    for factor_name in factor_names:
        coded_col = f"{factor_name}_coded"
        if coded_col in df.columns:
            working[factor_name] = df[coded_col]
            continue
        col = df[factor_name]
        lo, hi = float(col.min()), float(col.max())
        if hi == lo:
            raise ValidationError(f"factor '{factor_name}' has no variation (constant column)")
        working[factor_name] = (col - (lo + hi) / 2) / ((hi - lo) / 2)
    return working


def _coded_to_natural_settings(
    df: pd.DataFrame, factor_names: list[str], coded: list[float]
) -> dict[str, float]:
    natural = {}
    for factor_name, coded_value in zip(factor_names, coded, strict=True):
        col = df[factor_name]
        lo, hi = float(col.min()), float(col.max())
        center, half = (lo + hi) / 2, (hi - lo) / 2
        natural[factor_name] = center + coded_value * half
    return natural


def _build_predictor(fitted: dict[str, Any], factor_names: list[str], model_type: str) -> Any:
    coeffs = {c["term"]: c["estimate"] for c in fitted["coefficients"]}

    def predict(x: np.ndarray) -> float:
        values = dict(zip(factor_names, x, strict=True))
        total = coeffs.get("Intercept", 0.0)
        for name in factor_names:
            total += coeffs.get(name, 0.0) * values[name]
            if model_type == "quadratic":
                total += coeffs.get(f"{name}_sq", 0.0) * values[name] ** 2
        for i, name_i in enumerate(factor_names):
            for name_j in factor_names[i + 1 :]:
                coeff = coeffs.get(f"{name_i}:{name_j}", coeffs.get(f"{name_j}:{name_i}", 0.0))
                total += coeff * values[name_i] * values[name_j]
        return float(total)

    return predict


def _render_factorial_plots(
    artifacts: ArtifactStore, result: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []

    fig, ax = new_figure(figsize=(6.0, 4.5))
    half_normal = result["half_normal"]
    ax.scatter(half_normal["theoretical_quantiles"], half_normal["abs_effects"], color="#4c72b0")
    hn_terms = half_normal["terms"]
    for x, y, term in zip(
        half_normal["theoretical_quantiles"], half_normal["abs_effects"], hn_terms, strict=True
    ):
        ax.annotate(term, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("half-normal quantile")
    ax.set_ylabel("|effect|")
    ax.set_title("Half-normal plot of effects")
    records.append(
        artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="half_normal.png",
            data=render_png(fig),
            media_type="image/png",
        )
    )

    fig, ax = new_figure(figsize=(6.5, 4.5))
    pareto = result["pareto"]
    y_pos = np.arange(len(pareto["terms"]))
    ax.barh(y_pos, pareto["abs_effects"], color="#4c72b0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pareto["terms"])
    ax.invert_yaxis()
    ax.axvline(
        pareto["margin_of_error_95"], color="#c44e52", linestyle="--", label="Lenth ME (95%)"
    )
    ax.set_xlabel("|effect|")
    ax.set_title("Pareto of effects")
    ax.legend(fontsize=8)
    records.append(
        artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="pareto.png",
            data=render_png(fig),
            media_type="image/png",
        )
    )

    fig, axes = new_figure(2, 2, figsize=(9.0, 7.0))
    residuals = np.array(result["residuals"])
    fitted = np.array(result["fitted"])
    axes[0, 0].scatter(fitted, residuals, color="#4c72b0", s=15)
    axes[0, 0].axhline(0, color="#c44e52", linewidth=1)
    axes[0, 0].set_xlabel("fitted")
    axes[0, 0].set_ylabel("residual")
    axes[0, 0].set_title("Residuals vs. fitted")

    (osm, osr), (slope, intercept, _r) = sp_stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(osm, osr, color="#4c72b0", s=15)
    axes[0, 1].plot(osm, slope * osm + intercept, color="#c44e52", linewidth=1.5)
    axes[0, 1].set_title("Normal Q-Q of residuals")

    axes[1, 0].hist(residuals, bins="auto", color="#4c72b0", edgecolor="white")
    axes[1, 0].set_title("Histogram of residuals")

    axes[1, 1].plot(
        range(1, len(residuals) + 1), residuals, marker="o", markersize=3, color="#4c72b0"
    )
    axes[1, 1].axhline(0, color="#c44e52", linewidth=1)
    axes[1, 1].set_xlabel("run order")
    axes[1, 1].set_title("Residuals vs. run order")

    fig.tight_layout()
    records.append(
        artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="residual_diagnostics.png",
            data=render_png(fig),
            media_type="image/png",
        )
    )
    return records


def _render_response_surface_plots(
    artifacts: ArtifactStore, result: dict[str, Any], factor_names: list[str]
) -> list[dict[str, Any]]:
    coeffs = {c["term"]: c["estimate"] for c in result["coefficients"]}
    x_name, y_name = factor_names[0], factor_names[1]
    grid = np.linspace(-1, 1, 40)
    xx, yy = np.meshgrid(grid, grid)

    def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        total = coeffs.get("Intercept", 0.0)
        total = total + coeffs.get(x_name, 0.0) * x + coeffs.get(y_name, 0.0) * y
        x_sq_coeff = coeffs.get(f"{x_name}_sq", 0.0)
        y_sq_coeff = coeffs.get(f"{y_name}_sq", 0.0)
        total = total + x_sq_coeff * x**2 + y_sq_coeff * y**2
        inter = coeffs.get(f"{x_name}:{y_name}", coeffs.get(f"{y_name}:{x_name}", 0.0))
        return total + inter * x * y

    zz = surface(xx, yy)

    records = []
    fig, ax = new_figure(figsize=(6.0, 5.0))
    contour = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel(f"{x_name} (coded)")
    ax.set_ylabel(f"{y_name} (coded)")
    ax.set_title("Response surface contour")
    records.append(
        artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="contour.png",
            data=render_png(fig),
            media_type="image/png",
        )
    )

    fig3d, ax3d = new_3d_figure()
    ax3d.plot_surface(xx, yy, zz, cmap="viridis")
    ax3d.set_xlabel(f"{x_name} (coded)")
    ax3d.set_ylabel(f"{y_name} (coded)")
    ax3d.set_zlabel("predicted response")
    ax3d.set_title("Response surface (3D)")
    records.append(
        artifacts.register(
            get_current_workspace_id(),
            kind="plot",
            filename="surface_3d.png",
            data=render_png(fig3d),
            media_type="image/png",
        )
    )
    return records


def _design_interpretation(result: dict[str, Any]) -> str:
    parts = [
        f"{result['design_type']} design with {result['n_runs']} runs, "
        f"{result['n_factors']} factors."
    ]
    if result.get("resolution") is not None:
        parts.append(f"Resolution {_roman(result['resolution'])}.")
    return " ".join(parts)


def _roman(n: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(n, str(n))


def _evaluate_design_interpretation(result: dict[str, Any]) -> str:
    orthogonal = "orthogonal" if result["orthogonal"] else "not perfectly orthogonal"
    max_corr = result["max_abs_pairwise_correlation"]
    return f"Design is {orthogonal} (max pairwise |r|={max_corr:.4f})."


def _factorial_interpretation(result: dict[str, Any]) -> str:
    parts = [f"R^2={result['r_squared']:.4f}, adj-R^2={result['adj_r_squared']:.4f}."]
    if result.get("lack_of_fit"):
        lof = result["lack_of_fit"]
        parts.append(f"Lack-of-fit test: p={lof['p_value']:.4g}.")
    top = result["pareto"]["terms"][: min(3, len(result["pareto"]["terms"]))]
    if top:
        parts.append(f"Largest effects: {', '.join(top)}.")
    return " ".join(parts)


def _response_surface_interpretation(result: dict[str, Any]) -> str:
    parts = [f"R^2={result['r_squared']:.4f}."]
    sp = result.get("stationary_point")
    if sp:
        parts.append(
            f"Stationary point is a {sp['kind']} with predicted response "
            f"{sp['predicted_response']:.4g}."
        )
    else:
        parts.append(
            "No unique stationary point (surface is flat or ill-conditioned in at "
            "least one factor)."
        )
    return " ".join(parts)
