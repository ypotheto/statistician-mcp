from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

Goal = Literal["maximize", "minimize", "target"]


def individual_desirability(
    y: float, goal: Goal, low: float, high: float, target: float | None = None, weight: float = 1.0
) -> float:
    """Derringer-Suich (1980) desirability function: maps a predicted response to
    [0, 1], 1 being fully desirable."""
    if goal == "maximize":
        if y <= low:
            return 0.0
        if y >= high:
            return 1.0
        return ((y - low) / (high - low)) ** weight
    if goal == "minimize":
        if y >= high:
            return 0.0
        if y <= low:
            return 1.0
        return ((high - y) / (high - low)) ** weight

    if target is None:
        raise ValueError("goal='target' requires 'target'")
    if y < low or y > high:
        return 0.0
    if y <= target:
        return ((y - low) / (target - low)) ** weight if target > low else 1.0
    return ((high - y) / (high - target)) ** weight if high > target else 1.0


def overall_desirability(individual: list[float]) -> float:
    """Geometric mean of individual desirabilities; zero if any response is at its
    worst (a single unacceptable response makes the whole combination unacceptable)."""
    arr = np.asarray(individual, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def optimize_desirability(
    predict_fns: list[Any],
    goals: list[dict[str, Any]],
    bounds: list[tuple[float, float]],
    n_starts: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Multi-start L-BFGS-B search (to avoid local optima on a possibly non-convex
    desirability surface) for the factor settings maximizing overall desirability.

    `predict_fns[i](x)` must return response i's prediction at factor vector `x`;
    `goals[i]` is a dict of `individual_desirability` kwargs (goal/low/high/target/weight).
    """
    rng = np.random.default_rng(seed)

    def neg_desirability(x: np.ndarray) -> float:
        preds = [fn(x) for fn in predict_fns]
        ds = [
            individual_desirability(pred, **goal) for pred, goal in zip(preds, goals, strict=True)
        ]
        return -overall_desirability(ds)

    best = None
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(neg_desirability, x0, bounds=bounds, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None  # n_starts >= 1 guarantees at least one result
    predicted = [float(fn(best.x)) for fn in predict_fns]
    return {
        "x": best.x.tolist(),
        "desirability": float(-best.fun),
        "predicted_responses": predicted,
    }
