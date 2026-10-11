"""Smoke tests for plot_love."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

from statract import plot_love, propensity_weights


def _weights():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(50, 10, n)
    grp = rng.choice(["a", "b", "c"], n)
    treat = (rng.random(n) < 1 / (1 + np.exp(-0.05 * (age - 50)))).astype(int)
    frame = pl.DataFrame({"treat": treat, "age": age, "grp": grp})
    return propensity_weights(frame, "treat ~ age + grp")


def test_love_plot_returns_figure() -> None:
    fig = _weights().love_plot(order="unadjusted", drop_distance=True)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert sorted(labels) == ["age", "grp_a", "grp_b", "grp_c"]


def test_plot_love_saves_png(tmp_path: Path) -> None:
    out = plot_love(_weights().balance(), tmp_path / "love.png", absolute=False, also_pdf=True)
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_love_needs_columns() -> None:
    with pytest.raises(ValueError, match="diff_adjusted"):
        plot_love(pl.DataFrame({"term": ["x"], "diff_unadjusted": [0.1]}))
