from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from scipy import stats as sp_stats

from statistician_mcp import envelope
from statistician_mcp.artifacts import ArtifactStore
from statistician_mcp.datasets import DatasetStore
from statistician_mcp.errors import ColumnNotFoundError, ValidationError
from statistician_mcp.stats import regression as stats_regression
from statistician_mcp.utils.formulas import FormulaError
from statistician_mcp.utils.plotting import new_figure, render_png
from statistician_mcp.workspace import get_current_workspace_id

Family = Literal["linear", "logistic"]


def register_regression_tools(mcp: FastMCP, store: DatasetStore, artifacts: ArtifactStore) -> None:
    @mcp.tool()
    @envelope.tool("fit_linear_model")
    def fit_linear_model(handle: str, formula: str) -> dict[str, Any]:
        """Fit an OLS linear model from a formula (`y ~ A + B + A:B`, restricted to
        column names / ~ + - : * — no function calls). Returns coefficients with
        CIs, R²/adj-R², VIF (multicollinearity), Cook's-distance-flagged rows, and
        4-panel residual diagnostics."""
        df = store.get_dataframe(get_current_workspace_id(), handle)
        try:
            result = stats_regression.fit_linear_model(df, formula)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc

        record = _render_residual_diagnostics(artifacts, result)
        return envelope.ok_envelope(
            {k: v for k, v in result.items() if k not in ("residuals", "fitted", "leverage")},
            artifacts=[record],
            interpretation=_linear_interpretation(result),
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("fit_logistic_model")
    def fit_logistic_model(handle: str, formula: str) -> dict[str, Any]:
        """Fit a logistic regression from a formula (response must be 0/1). Returns
        odds ratios with CIs, a classification summary (accuracy/precision/recall),
        and an ROC curve artifact with AUC."""
        df = store.get_dataframe(get_current_workspace_id(), handle)
        try:
            result = stats_regression.fit_logistic_model(df, formula)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc

        record = _render_roc_curve(artifacts, result["roc"])
        return envelope.ok_envelope(
            {k: v for k, v in result.items() if k != "roc"},
            artifacts=[record],
            interpretation=(
                f"AUC={result['roc']['auc']:.3f}, "
                f"accuracy={result['classification']['accuracy']:.3f}."
            ),
            meta={"dataset": handle, "n_rows_used": result["n"]},
        )

    @mcp.tool()
    @envelope.tool("compare_models")
    def compare_models(
        handle: str, formulas: list[str], family: Family = "linear"
    ) -> dict[str, Any]:
        """Compare 2+ fitted models by AIC/BIC, plus a nested F-test if exactly two
        linear models are given and one's terms are a subset of the other's."""
        if len(formulas) < 2:
            raise ValidationError("compare_models needs at least 2 formulas")
        df = store.get_dataframe(get_current_workspace_id(), handle)
        try:
            result = stats_regression.compare_models(df, formulas, family)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc

        interpretation = (
            f"Best by AIC: {result['best_by_aic']}. Best by BIC: {result['best_by_bic']}."
        )
        return envelope.ok_envelope(
            result,
            interpretation=interpretation,
            meta={"dataset": handle, "n_rows_used": len(df)},
        )

    @mcp.tool()
    @envelope.tool("predict_from_model")
    def predict_from_model(
        handle: str,
        formula: str,
        new_data: list[dict[str, Any]],
        family: Family = "linear",
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Predict at new X values from a model re-fit from `formula` + the
        dataset (models are not persisted between calls). For family='linear',
        also returns confidence and prediction intervals."""
        if not new_data:
            raise ValidationError("new_data must have at least one row")
        df = store.get_dataframe(get_current_workspace_id(), handle)
        new_df = pd.DataFrame(new_data)
        try:
            result = stats_regression.predict_from_model(df, formula, new_df, family, confidence)
        except FormulaError as exc:
            raise ValidationError(str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise ValidationError(f"could not predict at new_data: {exc}") from exc

        return envelope.ok_envelope(result, meta={"dataset": handle, "n_rows_used": len(df)})

    @mcp.tool()
    @envelope.tool("fit_distribution")
    def fit_distribution(
        handle: str, column: str, distributions: list[str] | None = None
    ) -> dict[str, Any]:
        """Fit normal/lognormal/weibull/exponential/gamma distributions to a column
        (positive-only distributions are skipped for non-positive data), rank by
        an Anderson-Darling-style statistic, and report the best fit. Renders a
        histogram with the best-fit density overlay."""
        df = store.get_dataframe(get_current_workspace_id(), handle)
        _require_numeric(df, column)
        try:
            result = stats_regression.fit_distribution(df[column].to_numpy(), distributions)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        record = _render_distribution_fit(artifacts, df[column].to_numpy(), result)
        return envelope.ok_envelope(
            result,
            artifacts=[record],
            interpretation=f"Best fit: {result['best_fit']} (lowest Anderson-Darling statistic).",
            meta={"dataset": handle, "n_rows_used": result["n"]},
        )


def _require_columns(df: pd.DataFrame, names: list[str]) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ColumnNotFoundError(missing[0], list(map(str, df.columns)))


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    _require_columns(df, [column])
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValidationError(f"column '{column}' is not numeric")


def _render_residual_diagnostics(
    artifacts: ArtifactStore, result: dict[str, Any]
) -> dict[str, Any]:
    residuals = np.array(result["residuals"])
    fitted = np.array(result["fitted"])
    leverage = np.array(result["leverage"])
    flagged = set(result["cooks_distance_flagged_rows"])

    fig, axes = new_figure(2, 2, figsize=(9.0, 7.0))
    colors = ["#c44e52" if i in flagged else "#4c72b0" for i in range(len(residuals))]
    axes[0, 0].scatter(fitted, residuals, c=colors, s=15)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set_title("Residuals vs. fitted")

    (osm, osr), (slope, intercept, _r) = sp_stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(osm, osr, color="#4c72b0", s=15)
    axes[0, 1].plot(osm, slope * osm + intercept, color="#c44e52", linewidth=1.5)
    axes[0, 1].set_title("Normal Q-Q of residuals")

    axes[1, 0].scatter(leverage, residuals, c=colors, s=15)
    axes[1, 0].set_xlabel("leverage")
    axes[1, 0].set_ylabel("residual")
    axes[1, 0].set_title("Residuals vs. leverage (flagged Cook's D in red)")

    axes[1, 1].hist(residuals, bins="auto", color="#4c72b0", edgecolor="white")
    axes[1, 1].set_title("Histogram of residuals")
    fig.tight_layout()

    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename="residual_diagnostics.png",
        data=render_png(fig),
        media_type="image/png",
    )


def _render_roc_curve(artifacts: ArtifactStore, roc: dict[str, Any]) -> dict[str, Any]:
    fig, ax = new_figure(figsize=(5.5, 5.0))
    ax.plot(roc["fpr"], roc["tpr"], color="#4c72b0", label=f"AUC={roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8)
    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename="roc_curve.png",
        data=render_png(fig),
        media_type="image/png",
    )


def _render_distribution_fit(
    artifacts: ArtifactStore, data: np.ndarray, result: dict[str, Any]
) -> dict[str, Any]:
    fig, ax = new_figure(figsize=(7.0, 4.5))
    ax.hist(data, bins="auto", density=True, color="#4c72b0", edgecolor="white", alpha=0.7)

    best = next(f for f in result["fits"] if f["distribution"] == result["best_fit"])
    dist = stats_regression.get_distribution(best["distribution"])
    xs = np.linspace(data.min(), data.max(), 200)
    ax.plot(
        xs, dist.pdf(xs, *best["params"]), color="#c44e52", linewidth=2, label=best["distribution"]
    )
    ax.set_title(f"Best fit: {best['distribution']}")
    ax.legend(fontsize=8)
    return artifacts.register(
        get_current_workspace_id(),
        kind="plot",
        filename="distribution_fit.png",
        data=render_png(fig),
        media_type="image/png",
    )


def _linear_interpretation(result: dict[str, Any]) -> str:
    parts = [f"R^2={result['r_squared']:.4f}, adj-R^2={result['adj_r_squared']:.4f}."]
    high_vif = [name for name, v in result["vif"].items() if v is not None and v > 5]
    if high_vif:
        parts.append(f"High multicollinearity (VIF>5): {', '.join(high_vif)}.")
    flagged_rows = result["cooks_distance_flagged_rows"]
    if flagged_rows:
        parts.append(f"{len(flagged_rows)} row(s) flagged by Cook's distance.")
    return " ".join(parts)
