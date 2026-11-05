"""Polars 2 Categorical columns as model factors."""

from __future__ import annotations

import numpy as np
import polars as pl

from statract.models.fit import fit_ols
from statract.report.artifacts import plotly_category_orders


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    g = ["b", "a", "c"] * 20
    y = rng.normal(size=60) + np.array([{"a": 0.0, "b": 1.0, "c": 2.0}[v] for v in g])
    return pl.DataFrame({"y": y, "g": g})


def test_categorical_predictor_matches_string() -> None:
    df = _frame()
    as_string = fit_ols(df, "y ~ g")
    as_cat = fit_ols(df.with_columns(pl.col("g").cast(pl.Categorical)), "y ~ g")
    assert as_cat.names == as_string.names
    np.testing.assert_allclose(as_cat.coefficients, as_string.coefficients)


def test_categorical_predictor_column_api() -> None:
    df = _frame().with_columns(pl.col("g").cast(pl.Categorical))
    fit = fit_ols(df, "y", ["g"])
    assert len(fit.coefficients) == 3


def test_plotly_category_orders() -> None:
    df = pl.DataFrame({
        "e": pl.Series(["x", "y"]).cast(pl.Enum(["y", "x"])),
        "c": pl.Series(["b", "a"]).cast(pl.Categorical),
    })
    orders = plotly_category_orders(df, ["e", "c"])
    assert orders == {"e": ["y", "x"], "c": ["a", "b"]}
