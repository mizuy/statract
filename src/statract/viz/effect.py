"""Effect curve of a spline term, like ``plot(rms::Predict(...))``."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_LINE = "#0073C2"


def plot_spline_effect(
    effect: pl.DataFrame,
    variable: str | None = None,
    *,
    ax: Any = None,
    ratio: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = _LINE,
) -> Any:
    """Draw ``spline_effect`` output: the curve, its band, and the null line.

    ``ratio=True`` reads the table as odds or hazard ratios
    (``spline_effect(..., exponentiate=True)``), puts the null line at 1, and
    uses a log y axis. With ``ratio=False`` the null line is at 0.
    """
    name = variable or effect.columns[0]
    x = effect.get_column(name).to_numpy()
    order = np.argsort(x)
    x = x[order]
    est = effect.get_column("estimate").to_numpy()[order]
    low = effect.get_column("conf_low").to_numpy()[order]
    high = effect.get_column("conf_high").to_numpy()[order]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, est, color=color, linewidth=1.6)
    ax.axhline(1.0 if ratio else 0.0, color="#555555", linewidth=0.8, linestyle="--")
    if ratio:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or name)
    ax.set_ylabel(ylabel or ("Ratio (95% CI)" if ratio else "Difference in linear predictor"))
    ax.spines[["top", "right"]].set_visible(False)
    return ax
