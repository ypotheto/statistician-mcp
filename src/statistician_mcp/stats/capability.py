from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

from statistician_mcp.stats.assumptions import check_normality
from statistician_mcp.stats.control_charts import d2


def _sigma_within(x: np.ndarray, subgroup_size: int | None) -> float:
    if subgroup_size and subgroup_size > 1:
        usable = len(x) - len(x) % subgroup_size
        groups = x[:usable].reshape(-1, subgroup_size)
        ranges = groups.max(axis=1) - groups.min(axis=1)
        return float(ranges.mean() / d2(subgroup_size))
    moving_ranges = np.abs(np.diff(x))
    return float(moving_ranges.mean() / d2(2))


def _indices(
    mean: float, sigma: float, lsl: float | None, usl: float | None
) -> dict[str, float | None]:
    if sigma <= 0:
        # No (or immeasurably little) variation -- capability indices are undefined
        # rather than "infinite"; a caller-facing tool should surface this plainly.
        return {"cp": None, "cpu": None, "cpl": None, "cpk": None}
    cpu = (usl - mean) / (3 * sigma) if usl is not None else None
    cpl = (mean - lsl) / (3 * sigma) if lsl is not None else None
    cp = (usl - lsl) / (6 * sigma) if (usl is not None and lsl is not None) else None
    candidates = [v for v in (cpu, cpl) if v is not None]
    cpk = min(candidates) if candidates else None
    return {"cp": cp, "cpu": cpu, "cpl": cpl, "cpk": cpk}


def process_capability(
    data: np.ndarray,
    lsl: float | None,
    usl: float | None,
    subgroup_size: int | None = None,
) -> dict[str, Any]:
    """Cp/Cpk (within-subgroup sigma, via R-bar/d2) and Pp/Ppk (overall sample sigma)
    against spec limits, plus DPMO and the corresponding process sigma level. Runs a
    normality check first (capability indices assume normality) and, if that fails,
    also reports Box-Cox-transformed indices as an alternative."""
    if lsl is None and usl is None:
        raise ValueError("at least one of lsl/usl is required")

    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    mean, overall_sigma = float(x.mean()), float(x.std(ddof=1))
    sigma_within = _sigma_within(x, subgroup_size)

    normality = check_normality(x, "data")
    within = _indices(mean, sigma_within, lsl, usl)
    overall = _indices(mean, overall_sigma, lsl, usl)
    dpmo, sigma_level = _dpmo_and_sigma_level(mean, overall_sigma, lsl, usl)

    box_cox = None
    if normality.status == "fail" and np.all(x > 0):
        box_cox = _box_cox_capability(x, lsl, usl, subgroup_size)

    return {
        "n": len(x),
        "mean": mean,
        "sigma_within": sigma_within,
        "sigma_overall": overall_sigma,
        "within": within,
        "overall": overall,
        "dpmo": dpmo,
        "sigma_level": sigma_level,
        "normality": normality.to_dict(),
        "box_cox_alternative": box_cox,
    }


def _dpmo_and_sigma_level(
    mean: float, sigma: float, lsl: float | None, usl: float | None
) -> tuple[float, float]:
    if sigma <= 0:
        return float("nan"), float("nan")
    p_below = float(sp_stats.norm.cdf(lsl, mean, sigma)) if lsl is not None else 0.0
    p_above = float(1 - sp_stats.norm.cdf(usl, mean, sigma)) if usl is not None else 0.0
    p_defect = min(max(p_below + p_above, 0.0), 1.0)
    dpmo = p_defect * 1_000_000
    sigma_level = float(sp_stats.norm.ppf(1 - p_defect)) if 0 < p_defect < 1 else float("inf")
    return dpmo, sigma_level


def _box_cox_capability(
    x: np.ndarray, lsl: float | None, usl: float | None, subgroup_size: int | None
) -> dict[str, Any] | None:
    try:
        transformed, lam = sp_stats.boxcox(x)
    except Exception:
        return None

    def transform(v: float) -> float:
        return float((v**lam - 1) / lam) if lam != 0 else float(np.log(v))

    t_lsl = transform(lsl) if lsl is not None else None
    t_usl = transform(usl) if usl is not None else None
    mean_t, sigma_t = float(transformed.mean()), float(transformed.std(ddof=1))
    within_t = _sigma_within(transformed, subgroup_size)
    return {
        "lambda": float(lam),
        "within": _indices(mean_t, within_t, t_lsl, t_usl),
        "overall": _indices(mean_t, sigma_t, t_lsl, t_usl),
    }
