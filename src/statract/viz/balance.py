"""Love plot of covariate balance, after ``cobalt::love.plot``.

Reads the table from ``PropensityWeights.balance()`` or ``balance_table``:
covariates on the y axis, the mean difference on the x axis, one point for
the unadjusted sample and one for the weighted sample. Pure matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .forest import _FOREST_RC

_UNADJUSTED_COLOR = "#E18727"
_ADJUSTED_COLOR = "#0073C2"
_ORDERS = {None, "unadjusted", "adjusted", "alphabetical"}


def plot_love(
    balance: pl.DataFrame,
    path: Path | str | None = None,
    *,
    absolute: bool = True,
    threshold: float | None = 0.1,
    order: str | None = None,
    drop_distance: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
    also_pdf: bool = False,
) -> Path | Any:
    """Draw a Love plot from a balance table.

    Args:
        balance: Table with ``term``, ``diff_unadjusted``, and ``diff_adjusted``,
            as returned by ``PropensityWeights.balance()``.
        path: If set, save a PNG and return that path; otherwise return the
            matplotlib ``Figure``.
        absolute: Plot ``|diff|`` (the common Love plot). ``False`` keeps the sign.
        threshold: Dashed reference line(s) at this value (and its negative when
            ``absolute=False``). ``None`` draws none.
        order: ``None`` keeps the table order (cobalt's default), or sort by
            ``"unadjusted"`` / ``"adjusted"`` (largest at the top) or
            ``"alphabetical"``.
        drop_distance: Drop the ``Distance`` row (``prop.score``).
        title: Optional title.
        xlabel: Axis label. The default names the statistic.
        figsize: Figure size; height scales with the number of rows when omitted.
        dpi: PNG resolution when ``path`` is set.
        also_pdf: Also write ``path`` with a ``.pdf`` suffix.

    Returns:
        ``Path`` when ``path`` is given, otherwise a matplotlib ``Figure``.
    """
    for col in ("term", "diff_unadjusted", "diff_adjusted"):
        if col not in balance.columns:
            raise ValueError(f"balance table needs a {col!r} column")
    if order not in _ORDERS:
        raise ValueError("order must be None, 'unadjusted', 'adjusted', or 'alphabetical'")
    table = balance
    if drop_distance and "type" in table.columns:
        table = table.filter(pl.col("type") != "Distance")
    if table.height == 0:
        raise ValueError("balance table is empty")
    labels = table["term"].to_list()
    un = np.asarray(table["diff_unadjusted"].to_list(), dtype=float)
    adj = np.asarray(table["diff_adjusted"].to_list(), dtype=float)
    if absolute:
        un, adj = np.abs(un), np.abs(adj)
    if order == "alphabetical":
        rank = np.argsort(np.asarray(labels, dtype=object), kind="stable")
    elif order in {"unadjusted", "adjusted"}:
        key = un if order == "unadjusted" else adj
        rank = np.argsort(-np.abs(key), kind="stable")
    else:
        rank = np.arange(len(labels))
    labels = [labels[i] for i in rank]
    un, adj = un[rank], adj[rank]
    y = np.arange(len(labels))[::-1]
    if xlabel is None:
        stat = "Mean Differences"
        if "type" in table.columns and set(table["type"].to_list()) <= {"Contin.", "Distance"}:
            stat = "Standardized Mean Differences"
        xlabel = f"Absolute {stat}" if absolute else stat
    with plt.rc_context(_FOREST_RC):
        if figsize is None:
            figsize = (5.6, max(1.8, 0.28 * len(labels) + 1.0))
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(0.0, color="#111111", linewidth=0.8)
        if threshold is not None:
            ax.axvline(float(threshold), color="#555555", linewidth=0.8, linestyle="--")
            if not absolute:
                ax.axvline(-float(threshold), color="#555555", linewidth=0.8, linestyle="--")
        for k in range(len(labels)):
            ax.plot([un[k], adj[k]], [y[k], y[k]], color="#cccccc", linewidth=0.8, zorder=1)
        ax.scatter(un, y, color=_UNADJUSTED_COLOR, label="Unadjusted", zorder=3, s=22)
        ax.scatter(adj, y, color=_ADJUSTED_COLOR, label="Adjusted", zorder=3, s=22)
        ax.set_yticks(y, labels)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_xlabel(xlabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(title="Sample", frameon=False, loc="best", fontsize=8, title_fontsize=8)
        fig.tight_layout()
    if path is None:
        return fig
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = ["plot_love"]
