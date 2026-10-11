"""plot_spline_effect draws the curve, the band, and the null line."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl

from statract import fit_glm, plot_spline_effect, spline_effect


def test_plot_spline_effect() -> None:
    rng = np.random.default_rng(3)
    age = rng.uniform(30, 80, 300)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.002 * (age - 55) ** 2))))
    data = pl.DataFrame({"age": age, "y": y})
    fit = fit_glm(data, "y ~ rcs(age, 4)", family="binomial")
    curve = spline_effect(fit, data, "age", at=np.linspace(35, 75, 21), reference=55, exponentiate=True)
    ax = plot_spline_effect(curve)
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 2
