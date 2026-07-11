from __future__ import annotations

from typing import Any, Literal

import numpy as np
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize

Alternative = Literal["two-sided", "less", "greater"]
TestFamily = Literal["one_sample_t", "two_sample_t", "paired_t", "proportion", "anova"]

_UNKNOWNS = ("effect_size", "n", "alpha", "power")


def _solve_for(
    effect_size: float | None, n: float | None, alpha: float | None, power: float | None
) -> str:
    given = {"effect_size": effect_size, "n": n, "alpha": alpha, "power": power}
    unknowns = [k for k in _UNKNOWNS if given[k] is None]
    if len(unknowns) != 1:
        raise ValueError(
            "exactly one of effect_size, n, alpha, power must be omitted (it is solved for); "
            f"got {len(unknowns)} omitted: {unknowns}"
        )
    return unknowns[0]


def solve_power_t_test(
    test_family: Literal["one_sample_t", "two_sample_t", "paired_t"],
    effect_size: float | None = None,
    n: float | None = None,
    alpha: float | None = None,
    power: float | None = None,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> dict[str, Any]:
    solve_for = _solve_for(effect_size, n, alpha, power)

    if test_family == "two_sample_t":
        solved = TTestIndPower().solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative=alternative,
        )
    else:
        solved = TTestPower().solve_power(
            effect_size=effect_size, nobs=n, alpha=alpha, power=power, alternative=alternative
        )

    result: dict[str, Any] = {"effect_size": effect_size, "n": n, "alpha": alpha, "power": power}
    result[solve_for] = float(solved)
    result["solved_for"] = solve_for
    result["test_family"] = test_family
    result["alternative"] = alternative
    return result


def solve_power_proportion(
    prop1: float,
    prop2: float,
    n: float | None = None,
    alpha: float | None = None,
    power: float | None = None,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
) -> dict[str, Any]:
    effect_size = proportion_effectsize(prop1, prop2)
    solve_for = _solve_for(effect_size, n, alpha, power)
    if solve_for == "effect_size":
        raise ValueError(
            "effect_size is derived from prop1/prop2 here; omit n, alpha, or power instead"
        )

    solved = NormalIndPower().solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    result: dict[str, Any] = {
        "prop1": prop1,
        "prop2": prop2,
        "effect_size_h": float(effect_size),
        "n": n,
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
    }
    result[solve_for] = float(solved)
    result["solved_for"] = solve_for
    return result


def solve_power_anova(
    k_groups: int,
    effect_size: float | None = None,
    n: float | None = None,
    alpha: float | None = None,
    power: float | None = None,
) -> dict[str, Any]:
    solve_for = _solve_for(effect_size, n, alpha, power)
    solved = FTestAnovaPower().solve_power(
        effect_size=effect_size, nobs=n, alpha=alpha, power=power, k_groups=k_groups
    )
    result: dict[str, Any] = {
        "k_groups": k_groups,
        "effect_size": effect_size,
        "n": n,
        "alpha": alpha,
        "power": power,
    }
    result[solve_for] = float(solved)
    result["solved_for"] = solve_for
    result["test_family"] = "anova"
    return result


def _power_at_n(
    test_family: TestFamily,
    n: float,
    effect_size: float,
    alpha: float,
    alternative: Alternative,
    ratio: float,
    k_groups: int,
) -> float:
    if test_family == "two_sample_t":
        power = TTestIndPower().power(
            effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio, alternative=alternative
        )
    elif test_family in ("one_sample_t", "paired_t"):
        power = TTestPower().power(
            effect_size=effect_size, nobs=n, alpha=alpha, alternative=alternative
        )
    elif test_family == "proportion":
        power = NormalIndPower().power(
            effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio, alternative=alternative
        )
    else:
        power = FTestAnovaPower().power(
            effect_size=effect_size, nobs=n, alpha=alpha, k_groups=k_groups
        )
    return float(power)


def power_curve_points(
    test_family: TestFamily,
    effect_size: float,
    alpha: float,
    n_values: list[float],
    *,
    alternative: Alternative = "two-sided",
    ratio: float = 1.0,
    k_groups: int = 2,
) -> list[dict[str, float]]:
    return [
        {
            "n": float(n),
            "power": float(
                np.clip(
                    _power_at_n(test_family, n, effect_size, alpha, alternative, ratio, k_groups),
                    0.0,
                    1.0,
                )
            ),
        }
        for n in n_values
    ]
