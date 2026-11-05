import numpy as np
import polars as pl
import pytest

from statract.models.gam import gam, smooth


def _sine(n: int = 200, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x, "y": y})


def test_k_index_is_low_for_too_small_basis_and_near_one_otherwise():
    df = _sine()
    small = gam(df, "y", [smooth("x", k=3)], method="ml").check()["k_index"][0]
    large = gam(df, "y", [smooth("x", k=20)], method="ml").check()["k_index"][0]
    assert small < 0.5
    assert 0.85 < large < 1.15


@pytest.mark.parametrize("method", ["reml", "ml"])
@pytest.mark.parametrize("bad", [None, float("nan")])
@pytest.mark.parametrize("column", ["y", "x"])
def test_missing_row_is_dropped(method, bad, column):
    df = _sine()
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(bad).otherwise(pl.col(column)).alias(column)
    )
    fit = gam(holed, "y", [smooth("x")], method=method)
    ref = gam(df.slice(1), "y", [smooth("x")], method=method)
    assert fit.n_obs == 199
    np.testing.assert_allclose(fit.coefficients, ref.coefficients, rtol=1e-6, atol=1e-8)


def test_nan_in_parametric_weight_and_by_columns_is_dropped():
    df = _sine().with_columns(
        z=pl.Series(np.linspace(0, 1, 200)),
        w=pl.lit(1.0),
        grp=pl.Series(["a", "b"] * 100),
    )
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(float("nan")).otherwise(pl.col("z")).alias("z"),
        pl.when(pl.int_range(pl.len()) == 1).then(float("nan")).otherwise(pl.col("w")).alias("w"),
        pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col("grp")).alias("grp"),
    )
    kwargs = {"predictors": ["z"], "weights": "w", "method": "ml"}
    fit = gam(holed, "y", [smooth("x", by="grp")], **kwargs)
    ref = gam(df.slice(3), "y", [smooth("x", by="grp")], **kwargs)
    assert fit.n_obs == 197
    np.testing.assert_allclose(fit.coefficients, ref.coefficients, rtol=1e-6, atol=1e-8)
