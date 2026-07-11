from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyDOE3
from scipy import stats as sp_stats

DesignType = Literal[
    "full_factorial", "fractional_factorial", "plackett_burman", "ccd", "box_behnken", "lhs"
]


@dataclass
class FactorSpec:
    name: str
    low: float | None = None
    high: float | None = None
    levels: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.levels is None and (self.low is None or self.high is None):
            raise ValueError(f"factor '{self.name}' needs either low/high or levels")

    def n_levels(self) -> int:
        return len(self.levels) if self.levels is not None else 2

    def coded_to_natural(self, coded: float) -> Any:
        if self.levels is not None:
            return self.levels[int(coded)]
        center = (self.low + self.high) / 2  # type: ignore[operator]
        half_range = (self.high - self.low) / 2  # type: ignore[operator]
        return center + coded * half_range

    def unit_to_natural(self, unit: float) -> float:
        return self.low + unit * (self.high - self.low)  # type: ignore[operator]


def _resolution_from_alias_map(alias_map: list[str]) -> int | None:
    """The resolution is the shortest word length in the defining relation. For any
    alias group `term_a = term_b = ...`, the two shortest terms combine (their
    letters are disjoint by construction) into a defining-relation word of length
    len(term_a)+len(term_b); the minimum of that across all groups is the design's
    resolution."""
    best = None
    for line in alias_map:
        terms = [t.strip() for t in line.split("=")]
        if len(terms) < 2:
            continue
        lengths = sorted(len(t) for t in terms)
        combined = lengths[0] + lengths[1]
        if best is None or combined < best:
            best = combined
    return best


def generate_design(
    design_type: DesignType,
    factors: dict[str, FactorSpec],
    *,
    center_points: int = 0,
    replicates: int = 1,
    resolution: int | None = None,
    generators: str | None = None,
    ccd_alpha: Literal["orthogonal", "rotatable"] = "orthogonal",
    ccd_face: Literal["circumscribed", "inscribed", "faced"] = "circumscribed",
    lhs_samples: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    names = list(factors.keys())
    k = len(names)
    if k < 1:
        raise ValueError("at least one factor is required")

    alias_map: list[str] | None = None
    resolution_found: int | None = None
    is_unit_design = False

    if design_type == "full_factorial":
        if any(factors[n].levels is not None for n in names):
            coded = pyDOE3.fullfact([factors[n].n_levels() for n in names])
        else:
            coded = pyDOE3.ff2n(k)
    elif design_type == "fractional_factorial":
        if generators:
            coded = pyDOE3.fracfact(generators)
        elif resolution:
            coded = pyDOE3.fracfact_by_res(k, resolution)
        else:
            raise ValueError("fractional_factorial requires 'resolution' or 'generators'")
        alias_map, _ = pyDOE3.fracfact_aliasing(coded)
        resolution_found = _resolution_from_alias_map(alias_map)
    elif design_type == "plackett_burman":
        coded = pyDOE3.pbdesign(k)
    elif design_type == "ccd":
        coded = pyDOE3.ccdesign(
            k, center=(center_points or 4, center_points or 4), alpha=ccd_alpha, face=ccd_face
        )
    elif design_type == "box_behnken":
        coded = pyDOE3.bbdesign(k, center=center_points or 3)
    elif design_type == "lhs":
        coded = pyDOE3.lhs(k, samples=lhs_samples, random_state=seed)
        is_unit_design = True
    else:
        raise ValueError(f"unsupported design_type '{design_type}'")

    if replicates > 1:
        coded = np.tile(coded, (replicates, 1))

    def _to_natural(name: str, v: float) -> Any:
        factor = factors[name]
        return factor.unit_to_natural(v) if is_unit_design else factor.coded_to_natural(v)

    # Fitted effects models need a coded (-1/+1-scaled) column to make "effect =
    # 2*coefficient" correct -- the natural-units column alone isn't enough. LHS
    # coded values are the [0,1] sample rescaled to [-1,1]; factor-level (non
    # low/high) factors don't have a meaningful continuous coding, so are skipped.
    columns: dict[str, list[Any]] = {}
    for i, name in enumerate(names):
        factor = factors[name]
        columns[name] = [_to_natural(name, v) for v in coded[:, i]]
        if factor.levels is None:
            columns[f"{name}_coded"] = [
                (2 * v - 1) if is_unit_design else float(v) for v in coded[:, i]
            ]

    df = pd.DataFrame(columns)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "run_order", range(1, len(df) + 1))

    return {
        "design_type": design_type,
        "n_runs": len(df),
        "n_factors": k,
        "resolution": resolution_found,
        "alias_map": alias_map,
        "run_table": df,
    }


def evaluate_design(
    coded_matrix: np.ndarray, factor_names: list[str], sigma: float = 1.0
) -> dict[str, Any]:
    """Diagnostics for an arbitrary 2-level +/-1 coded design: orthogonality
    (max absolute pairwise column correlation), alias structure (via pyDOE3's
    generic column-aliasing check), and power for main effects at a given sigma."""
    corr = np.corrcoef(coded_matrix, rowvar=False)
    k = coded_matrix.shape[1]
    off_diag = corr[np.triu_indices(k, 1)] if k > 1 else np.array([0.0])
    max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0

    try:
        alias_map, _ = pyDOE3.fracfact_aliasing(coded_matrix)
    except Exception:
        alias_map = None

    n_runs = coded_matrix.shape[0]
    n_params = k + 1
    power_by_factor = {
        name: factorial_effect_power(n_runs, n_params, effect_size=2.0, sigma=sigma)
        for name in factor_names
    }

    return {
        "n_runs": n_runs,
        "n_factors": k,
        "max_abs_pairwise_correlation": max_abs_corr,
        "orthogonal": max_abs_corr < 1e-9,
        "alias_map": alias_map,
        "power_for_main_effects": power_by_factor,
    }


def factorial_effect_power(
    n_runs: int, n_params: int, effect_size: float, sigma: float, alpha: float = 0.05
) -> float:
    """Power to detect a main effect of the given size (full low-to-high swing, in
    response units) in a 2-level orthogonal coded design, via the noncentral-t
    distribution of the fitted regression coefficient (Var(b_i)=sigma^2/n_runs for
    an orthogonal +/-1 coded design)."""
    df = n_runs - n_params
    if df < 1:
        return float("nan")
    se = sigma / np.sqrt(n_runs)
    ncp = (effect_size / 2) / se
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
    power = 1 - sp_stats.nct.cdf(t_crit, df, ncp) + sp_stats.nct.cdf(-t_crit, df, ncp)
    return float(np.clip(power, 0.0, 1.0))
