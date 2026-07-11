from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def new_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (6.0, 4.0)) -> Any:
    """Return `(fig, ax)` (or `(fig, axes_array)` for nrows*ncols > 1) via
    `plt.subplots`. Centralizing figure creation here keeps every other module from
    needing to import `matplotlib.pyplot` (and thus from racing to set the backend)."""
    return plt.subplots(nrows, ncols, figsize=figsize)


def render_png(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
