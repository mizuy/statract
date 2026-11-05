"""Regression tests for KM, forest, balance and tree plot fixes (audit 2026-11-05)."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import statract  # noqa: F401  (sets up the Japanese font)
from statract import plot_forest, plot_survival
from statract.viz.forest import _format_estimate
from statract.viz.tree import _format_cut


def _km_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 4.0, 7.0],
            "status": [True, False, True, False, True, True, False, False, True, False],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B", None, None],
        }
    )


def _curve_ends(ax) -> dict[str, float]:
    return {line.get_label(): float(np.max(line.get_xdata())) for line in ax.get_lines() if not line.get_label().startswith("_")}


def test_km_curve_stops_at_own_last_time() -> None:
    ax = plot_survival(_km_frame(), time="time", status="status", hue="group", at_risk_counts=False)
    ends = _curve_ends(ax)
    assert ends["A"] == 5.0
    assert ends["B"] == 50.0
    plt.close(ax.figure)


def test_km_null_hue_has_no_curve() -> None:
    ax = plot_survival(_km_frame(), time="time", status="status", hue="group", at_risk_counts=False)
    assert set(_curve_ends(ax)) == {"A", "B"}
    plt.close(ax.figure)


def _glyph_warnings(make_fig) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = make_fig()
        fig.canvas.draw()
    plt.close(fig)
    return [str(w.message) for w in caught if "Glyph" in str(w.message)]


def test_forest_japanese_labels_render() -> None:
    tidy = pl.DataFrame(
        {"term": ["年齢", "性別"], "estimate": [1.2, 0.8], "conf_low": [1.0, 0.6], "conf_high": [1.5, 1.1]}
    )
    assert _glyph_warnings(lambda: plot_forest(tidy, ratio=True)) == []


def test_love_plot_japanese_labels_render() -> None:
    from statract import plot_love

    balance = pl.DataFrame({"term": ["年齢", "性別"], "diff_unadjusted": [0.3, 0.2], "diff_adjusted": [0.05, 0.01]})
    assert _glyph_warnings(lambda: plot_love(balance)) == []


def test_forest_no_negative_zero() -> None:
    assert _format_estimate(-0.001, -0.1, 0.09) == "0.00 (-0.10–0.09)"
    assert _format_estimate(0.5, -0.004, 0.9) == "0.50 (0.00–0.90)"


def test_forest_null_panel_raises_clear_error() -> None:
    tidy = pl.DataFrame(
        {
            "term": ["a", "b"],
            "estimate": [1.2, 0.8],
            "conf_low": [1.0, 0.6],
            "conf_high": [1.5, 1.1],
            "model": ["M1", None],
        }
    )
    with pytest.raises(ValueError, match="panel column 'model' has null"):
        plot_forest(tidy, ratio=True, panel="model")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1234.5, "1234.5"), (0.12345, "0.12345"), (2.5, "2.5"), (3.0, "3"), (1e-7, "0.0000001")],
)
def test_tree_cut_label_is_exact(value: float, expected: str) -> None:
    assert _format_cut(value, 3) == expected
