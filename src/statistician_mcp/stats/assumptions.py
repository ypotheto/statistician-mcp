from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as sp_stats

Status = Literal["pass", "warn", "fail"]


@dataclass
class AssumptionResult:
    check: str
    status: Status
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "status": self.status, "detail": self.detail}


def _status_for_p(p: float) -> Status:
    if p >= 0.05:
        return "pass"
    if p >= 0.01:
        return "warn"
    return "fail"


def check_normality(x: np.ndarray, label: str = "data") -> AssumptionResult:
    """Shapiro-Wilk for n<=5000 (scipy's own recommended cutoff), Anderson-Darling
    (via the Stephens 1974 p-value approximation for the normal case) above that."""
    values = np.asarray(x, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 3:
        return AssumptionResult(
            check=f"normality ({label})",
            status="warn",
            detail=f"n={n} is too small to test normality; treat parametric results with caution",
        )

    if n <= 5000:
        stat, p = sp_stats.shapiro(values)
        method = "Shapiro-Wilk"
    else:
        # scipy's own `method=` p-value option only interpolates to one of five fixed
        # table levels; the Stephens (1974) formula below gives a continuous estimate,
        # so we intentionally still read `.statistic` off the legacy code path.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = sp_stats.anderson(values, dist="norm")
        stat = float(result.statistic)
        p = _anderson_darling_p_value(stat, n)
        method = "Anderson-Darling"

    status = _status_for_p(p)
    detail = f"{method}: statistic={stat:.4f}, p={p:.4f} (n={n})"
    return AssumptionResult(check=f"normality ({label})", status=status, detail=detail)


def _anderson_darling_p_value(statistic: float, n: int) -> float:
    """Stephens (1974) approximation for the AD p-value, normal case (mean and
    variance estimated from the sample)."""
    a2 = statistic * (1 + 4 / n - 25 / n**2)
    if a2 >= 0.6:
        p = np.exp(1.2937 - 5.709 * a2 + 0.0186 * a2**2)
    elif a2 >= 0.34:
        p = np.exp(0.9177 - 4.279 * a2 - 1.38 * a2**2)
    elif a2 >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a2 - 59.938 * a2**2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a2 - 223.73 * a2**2)
    return float(np.clip(p, 0.0, 1.0))


def check_equal_variance(
    groups: list[np.ndarray], label: str = "data", method: Literal["levene", "bartlett"] = "levene"
) -> AssumptionResult:
    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[~np.isnan(g)] for g in cleaned]
    if any(len(g) < 2 for g in cleaned):
        return AssumptionResult(
            check=f"equal variance ({label})",
            status="warn",
            detail="a group has fewer than 2 values; variance equality cannot be tested",
        )

    if method == "levene":
        stat, p = sp_stats.levene(*cleaned, center="median")
        method_name = "Levene"
    else:
        stat, p = sp_stats.bartlett(*cleaned)
        method_name = "Bartlett"

    status = _status_for_p(p)
    detail = f"{method_name}: statistic={stat:.4f}, p={p:.4f}"
    return AssumptionResult(check=f"equal variance ({label})", status=status, detail=detail)


def check_sample_size(n: int, minimum: int, label: str = "data") -> AssumptionResult:
    if n >= minimum:
        status: Status = "pass"
        detail = f"n={n} meets the recommended minimum of {minimum}"
    elif n >= max(2, minimum // 2):
        status = "warn"
        detail = f"n={n} is below the recommended minimum of {minimum}; interpret with caution"
    else:
        status = "fail"
        detail = f"n={n} is far below the recommended minimum of {minimum}"
    return AssumptionResult(check=f"sample size ({label})", status=status, detail=detail)
