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


def new_3d_figure(figsize: tuple[float, float] = (6.5, 5.0)) -> tuple[Any, Any]:
    """Return `(fig, ax)` with `ax` a 3D-projection Axes, for surface/wireframe plots."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def render_png(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def close_all_open_figures() -> None:
    """Safety net for the case where a plotting tool raises between figure
    creation and `render_png` (which is the only place a figure normally gets
    closed) -- matplotlib keeps every created figure registered globally until
    explicitly closed, so an uncaught exception there would otherwise leak
    memory for the life of the server process. Called from envelope.tool's
    exception handlers, which run for every tool regardless of whether it plots.

    Using the global `plt.close("all")` (rather than closing one specific figure)
    is only safe because it can never affect a *different* in-flight request's
    figure: every tool this server registers is wrapped into an async function
    by envelope.tool, and FastMCP always awaits async tools directly in the
    current event loop rather than offloading them to a thread pool (verified
    against the installed mcp SDK's dispatch code) -- so a tool's synchronous
    body (which is where all plotting happens; there are no `await` points
    inside it) runs to completion or raises before any other tool call's code
    can run at all, on a single worker or many. If that dispatch model ever
    changes, this function's safety argument needs re-checking.
    """
    plt.close("all")
