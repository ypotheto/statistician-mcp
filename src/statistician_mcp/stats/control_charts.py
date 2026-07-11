from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np
from scipy import integrate
from scipy import stats as sp_stats
from scipy.special import gammaln

ChartType = Literal["xbar_r", "xbar_s", "i_mr", "p", "np", "c", "u", "ewma", "cusum"]

# Control-chart constants (c4, d2, d3, A2, A3, B3, B4, D3, D4) are computed here
# from first principles rather than transcribed from a published table, to remove
# any risk of a mistyped digit in a 24-row reference table. c4 has an exact closed
# form; d2/d3 (mean/sd of the range of n iid standard normals) are computed via
# numerical integration of the range's CDF. Verified against the standard SPC
# textbook anchor points: n=2 (d2=1.128, D4=3.267), n=5 (d2=2.326, D4=2.114,
# A2=0.577) — see tests/stats/test_control_charts.py.


@lru_cache(maxsize=64)
def c4(n: int) -> float:
    """Exact bias-correction constant for the sample standard deviation."""
    return float(np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


@lru_cache(maxsize=64)
def d2(n: int) -> float:
    """Expected value of the range of n iid standard normal variables."""

    def integrand(x: float) -> float:
        return x * n * sp_stats.norm.cdf(x) ** (n - 1) * sp_stats.norm.pdf(x)

    value, _ = integrate.quad(integrand, -12, 12, limit=300)
    return float(2 * value)


@lru_cache(maxsize=64)
def d3(n: int) -> float:
    """Standard deviation of the range of n iid standard normal variables, via the
    range's survival function (numerically far more stable than the naive 2D
    integral over the joint density of the min/max, which was verified to converge
    to a confidently-wrong -- even negative -- variance for several n)."""

    def range_cdf(r: float) -> float:
        def integrand(x: float) -> float:
            spread = sp_stats.norm.cdf(x + r) - sp_stats.norm.cdf(x)
            return sp_stats.norm.pdf(x) * spread ** (n - 1)

        value, _ = integrate.quad(integrand, -12, 12, limit=300)
        return n * value

    def surv_times_2r(r: float) -> float:
        return 2 * r * (1 - range_cdf(r))

    e_r2, _ = integrate.quad(surv_times_2r, 0, 12, limit=300)
    variance = e_r2 - d2(n) ** 2
    return float(np.sqrt(max(variance, 0.0)))


def A2(n: int) -> float:
    return 3 / (d2(n) * np.sqrt(n))


def A3(n: int) -> float:
    return 3 / (c4(n) * np.sqrt(n))


def B3(n: int) -> float:
    return max(0.0, 1 - 3 * np.sqrt(1 - c4(n) ** 2) / c4(n))


def B4(n: int) -> float:
    return 1 + 3 * np.sqrt(1 - c4(n) ** 2) / c4(n)


def D3(n: int) -> float:
    return max(0.0, 1 - 3 * d3(n) / d2(n))


def D4(n: int) -> float:
    return 1 + 3 * d3(n) / d2(n)


def chart_constants(n: int) -> dict[str, float]:
    return {
        "c4": c4(n),
        "d2": d2(n),
        "d3": d3(n),
        "A2": A2(n),
        "A3": A3(n),
        "B3": B3(n),
        "B4": B4(n),
        "D3": D3(n),
        "D4": D4(n),
    }


def xbar_r_limits(subgroups: list[list[float]]) -> dict[str, Any]:
    n = len(subgroups[0])
    if any(len(sg) != n for sg in subgroups):
        raise ValueError("all subgroups must have the same size for an xbar-R chart")
    means = np.array([np.mean(sg) for sg in subgroups])
    ranges = np.array([np.max(sg) - np.min(sg) for sg in subgroups])
    grand_mean, mean_range = float(means.mean()), float(ranges.mean())
    a2 = A2(n)
    return {
        "xbar": {
            "points": means.tolist(),
            "cl": grand_mean,
            "ucl": grand_mean + a2 * mean_range,
            "lcl": grand_mean - a2 * mean_range,
            "sigma": a2 * mean_range / 3,
        },
        "r": {
            "points": ranges.tolist(),
            "cl": mean_range,
            "ucl": D4(n) * mean_range,
            "lcl": D3(n) * mean_range,
            "sigma": (D4(n) - 1) * mean_range / 3 if n > 1 else 0.0,
        },
        "n": n,
    }


def xbar_s_limits(subgroups: list[list[float]]) -> dict[str, Any]:
    n = len(subgroups[0])
    if any(len(sg) != n for sg in subgroups):
        raise ValueError("all subgroups must have the same size for an xbar-S chart")
    means = np.array([np.mean(sg) for sg in subgroups])
    sds = np.array([np.std(sg, ddof=1) for sg in subgroups])
    grand_mean, mean_sd = float(means.mean()), float(sds.mean())
    a3 = A3(n)
    return {
        "xbar": {
            "points": means.tolist(),
            "cl": grand_mean,
            "ucl": grand_mean + a3 * mean_sd,
            "lcl": grand_mean - a3 * mean_sd,
            "sigma": a3 * mean_sd / 3,
        },
        "s": {
            "points": sds.tolist(),
            "cl": mean_sd,
            "ucl": B4(n) * mean_sd,
            "lcl": B3(n) * mean_sd,
            "sigma": (B4(n) - 1) * mean_sd / 3,
        },
        "n": n,
    }


def i_mr_limits(values: list[float]) -> dict[str, Any]:
    x = np.array(values, dtype=float)
    moving_ranges = np.abs(np.diff(x))
    mr_bar = float(moving_ranges.mean())
    grand_mean = float(x.mean())
    factor = 3 / d2(2)
    return {
        "individuals": {
            "points": x.tolist(),
            "cl": grand_mean,
            "ucl": grand_mean + factor * mr_bar,
            "lcl": grand_mean - factor * mr_bar,
            "sigma": factor * mr_bar / 3,
        },
        "moving_range": {
            "points": moving_ranges.tolist(),
            "cl": mr_bar,
            "ucl": D4(2) * mr_bar,
            "lcl": D3(2) * mr_bar,
            "sigma": (D4(2) - 1) * mr_bar / 3,
        },
    }


def p_chart_limits(nonconforming: list[int], sample_sizes: list[int]) -> dict[str, Any]:
    nc, n = np.array(nonconforming, dtype=float), np.array(sample_sizes, dtype=float)
    p_bar = float(nc.sum() / n.sum())
    props = nc / n
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    return {
        "points": props.tolist(),
        "cl": p_bar,
        "ucl": np.clip(p_bar + 3 * sigma, 0, 1).tolist(),
        "lcl": np.clip(p_bar - 3 * sigma, 0, None).tolist(),
        "sigma": sigma.tolist(),
        "variable_limits": True,
    }


def np_chart_limits(nonconforming: list[int], sample_size: int) -> dict[str, Any]:
    nc = np.array(nonconforming, dtype=float)
    p_bar = float(nc.mean() / sample_size)
    cl = sample_size * p_bar
    sigma = float(np.sqrt(sample_size * p_bar * (1 - p_bar)))
    return {
        "points": nc.tolist(),
        "cl": cl,
        "ucl": cl + 3 * sigma,
        "lcl": max(0.0, cl - 3 * sigma),
        "sigma": sigma,
        "variable_limits": False,
    }


def c_chart_limits(counts: list[int]) -> dict[str, Any]:
    c_arr = np.array(counts, dtype=float)
    c_bar = float(c_arr.mean())
    sigma = float(np.sqrt(c_bar))
    return {
        "points": c_arr.tolist(),
        "cl": c_bar,
        "ucl": c_bar + 3 * sigma,
        "lcl": max(0.0, c_bar - 3 * sigma),
        "sigma": sigma,
        "variable_limits": False,
    }


def u_chart_limits(counts: list[int], units: list[float]) -> dict[str, Any]:
    c_arr, u_arr = np.array(counts, dtype=float), np.array(units, dtype=float)
    u_bar = float(c_arr.sum() / u_arr.sum())
    rates = c_arr / u_arr
    sigma = np.sqrt(u_bar / u_arr)
    return {
        "points": rates.tolist(),
        "cl": u_bar,
        "ucl": np.clip(u_bar + 3 * sigma, 0, None).tolist(),
        "lcl": np.clip(u_bar - 3 * sigma, 0, None).tolist(),
        "sigma": sigma.tolist(),
        "variable_limits": True,
    }


def ewma_chart(
    values: list[float],
    target: float | None = None,
    sigma: float | None = None,
    lam: float = 0.2,
    width_l: float = 3.0,
) -> dict[str, Any]:
    x = np.array(values, dtype=float)
    mu = target if target is not None else float(x.mean())
    s = sigma if sigma is not None else float(x.std(ddof=1))

    z = np.empty(len(x))
    prev = mu
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev

    i_arr = np.arange(1, len(x) + 1)
    factor = width_l * s * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * i_arr)))
    return {
        "points": z.tolist(),
        "cl": mu,
        "ucl": (mu + factor).tolist(),
        "lcl": (mu - factor).tolist(),
        "lambda_": lam,
        "l": width_l,
        "variable_limits": True,
    }


def cusum_chart(
    values: list[float],
    target: float | None = None,
    sigma: float | None = None,
    k: float = 0.5,
    h: float = 5.0,
) -> dict[str, Any]:
    x = np.array(values, dtype=float)
    mu = target if target is not None else float(x.mean())
    s = sigma if sigma is not None else float(x.std(ddof=1))

    c_plus = np.empty(len(x))
    c_minus = np.empty(len(x))
    prev_plus = prev_minus = 0.0
    for i, xi in enumerate(x):
        prev_plus = max(0.0, xi - (mu + k * s) + prev_plus)
        prev_minus = max(0.0, (mu - k * s) - xi + prev_minus)
        c_plus[i] = prev_plus
        c_minus[i] = prev_minus

    decision_interval = h * s
    return {
        "c_plus": c_plus.tolist(),
        "c_minus": c_minus.tolist(),
        "decision_interval": decision_interval,
        "k": k,
        "h": h,
        "target": mu,
        "sigma": s,
        "violations": [
            i
            for i in range(len(x))
            if c_plus[i] > decision_interval or c_minus[i] > decision_interval
        ],
    }


def nelson_rules(
    points: list[float], cl: float, sigma: float | list[float]
) -> dict[int, list[int]]:
    """The 8 Nelson/Western Electric rules, evaluated against per-point standardized
    deviations `z = (x - cl) / sigma` so a single implementation covers both
    constant-sigma charts (Xbar/I/R/S) and variable-limit charts (p/u)."""
    x = np.asarray(points, dtype=float)
    n = len(x)
    sigma_arr = (
        np.full(n, sigma, dtype=float) if np.isscalar(sigma) else np.asarray(sigma, dtype=float)
    )
    z = np.divide(x - cl, sigma_arr, out=np.zeros(n), where=sigma_arr != 0)

    violations: dict[int, list[int]] = {rule: [] for rule in range(1, 9)}

    for i in range(n):
        if abs(z[i]) > 3:
            violations[1].append(i)

    side = np.sign(z)
    run = 1
    for i in range(1, n):
        run = run + 1 if side[i] != 0 and side[i] == side[i - 1] else 1
        if run >= 9 and side[i] != 0:
            violations[2].append(i)

    run, direction = 1, 0
    for i in range(1, n):
        step = np.sign(x[i] - x[i - 1])
        run = run + 1 if step != 0 and step == direction else (2 if step != 0 else 1)
        direction = step if step != 0 else direction
        if run >= 6 and step != 0:
            violations[3].append(i)

    run, prev_step = 1, 0
    for i in range(1, n):
        step = np.sign(x[i] - x[i - 1])
        alternates = step != 0 and prev_step != 0 and step == -prev_step
        run = run + 1 if alternates else (2 if step != 0 else 1)
        if step != 0:
            prev_step = step
        if run >= 14:
            violations[4].append(i)

    for i in range(2, n):
        window = z[i - 2 : i + 1]
        if np.sum(window > 2) >= 2 or np.sum(window < -2) >= 2:
            violations[5].append(i)

    for i in range(4, n):
        window = z[i - 4 : i + 1]
        if np.sum(window > 1) >= 4 or np.sum(window < -1) >= 4:
            violations[6].append(i)

    run = 0
    for i in range(n):
        run = run + 1 if abs(z[i]) < 1 else 0
        if run >= 15:
            violations[7].append(i)

    run = 0
    for i in range(n):
        run = run + 1 if abs(z[i]) > 1 else 0
        if run >= 8:
            violations[8].append(i)

    return violations
