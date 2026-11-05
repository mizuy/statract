"""Regression tests for the 2026-11-05 audit of the prediction models."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract.models.diagnostic import diagnostic_accuracy
from statract.models.impute import impute_chained, pool
from statract.models.multinom import multinomial_regression
from statract.models.ordinal import ordinal_regression
from statract.models.tree import conditional_tree
from statract.models.validation import calibrate_cox


@pytest.fixture(scope="module")
def imputed():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.digitize(x + rng.logistic(size=n), [-1, 0.5, 2])
    frame = pl.DataFrame({"x": x, "z": z, "y": y})
    frame = frame.with_columns(
        pl.when(pl.Series(rng.random(n) < 0.2))
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    return impute_chained(frame, m=4, seed=3, columns=["x", "z", "y"])


def test_pool_ordinal_includes_thresholds(imputed):
    fits = imputed.fit(lambda d: ordinal_regression(d, "y ~ x + z"))
    table = pool(fits)
    first = fits[0]
    assert table["term"].to_list() == [*first.names, *first.threshold_names]
    expected = np.mean(
        [np.concatenate([f.coefficients, f.thresholds]) for f in fits], axis=0
    )
    np.testing.assert_allclose(table["estimate"].to_numpy(), expected)
    ubar = np.mean([np.diag(f.covariance) for f in fits], axis=0)
    np.testing.assert_allclose(table["ubar"].to_numpy(), ubar)
    assert table["dfcom"][0] == first.n_obs - first.n_params


def test_pool_multinomial_flattens_by_level(imputed):
    fits = imputed.fit(lambda d: multinomial_regression(d, "y ~ x + z"))
    table = pool(fits)
    first = fits[0]
    tidy = first.tidy()
    expected_terms = [
        f"{lev}:{term}" for lev, term in zip(tidy["y_level"], tidy["term"], strict=True)
    ]
    assert table["term"].to_list() == expected_terms
    expected = np.mean([f.coefficients.reshape(-1) for f in fits], axis=0)
    np.testing.assert_allclose(table["estimate"].to_numpy(), expected)
    # With one fit repeated, the pooled variance is that fit's tidy() SE squared.
    single = pool([first, first])
    np.testing.assert_allclose(
        single["std_error"].to_numpy(), tidy["std_error"].to_numpy()
    )


def test_pool_rejects_mismatched_covariance():
    class Broken:
        names = ["a", "b"]
        coefficients = np.zeros(2)
        covariance = np.eye(3)
        n_obs = 10

    with pytest.raises(ValueError, match="does not support"):
        pool([Broken(), Broken()])


def _tree_data(nan_at: int | None) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 3.0 * (x > 0) + rng.normal(size=n)
    if nan_at is not None:
        x[nan_at] = np.nan
    return pl.DataFrame({"x": x, "y": y})


def test_conditional_tree_drops_nan_predictor_rows():
    clean = conditional_tree(_tree_data(None)[1:], "y", ["x"])
    with_nan = conditional_tree(_tree_data(0), "y", ["x"])
    assert with_nan.n_terminal() == clean.n_terminal() == 2
    assert with_nan.root.n == 199
    assert with_nan.tests().height == 1
    np.testing.assert_allclose(
        with_nan.tests()["statistic"], clean.tests()["statistic"]
    )


def test_conditional_tree_drops_nan_outcome_rows():
    y = _tree_data(None)["y"].to_numpy().copy()
    y[5] = np.nan
    frame = _tree_data(None).with_columns(pl.Series("y", y))
    fit = conditional_tree(frame, "y", ["x"])
    assert fit.root.n == 199
    assert np.all(np.isfinite(fit.y))


def test_calibrate_cox_skips_resamples_ending_before_u():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.5 * x))
    c = rng.exponential(2, size=n)
    time = np.minimum(t, c)
    frame = pl.DataFrame({"time": time, "ev": (t <= c).astype(int), "x": x})
    ordered = np.sort(time)
    u = (ordered[-1] + ordered[-2]) / 2
    result = calibrate_cox(frame, "time", "ev", ["x"], u=u, B=20, seed=1, m=100)
    used = result.table["n"].max()
    assert 0 < used < 20


def test_calibrate_cox_raises_when_every_resample_fails():
    frame = pl.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ev": [1, 1, 0, 1, 1, 1],
            "x": [0.1, 0.5, -0.2, 0.3, 0.9, -0.4],
        }
    )
    # Every resample omits the subject with the last time, so none reaches u.
    indices = np.tile([0, 1, 2, 3, 4, 4], (3, 1))
    with pytest.raises(ValueError, match="every bootstrap repetition failed"):
        calibrate_cox(
            frame,
            "time",
            "ev",
            ["x"],
            u=5.5,
            cuts=np.array([0.0, 1.0]),
            indices=indices,
        )


def test_predictive_values_zero_denominator_is_nan():
    acc = diagnostic_accuracy(tp=0, fp=3, fn=0, tn=10)
    out = acc.predictive_values(0.1)
    assert np.isnan(out["ppv"][0])
    assert np.isnan(
        acc.tidy().filter(pl.col("statistic") == "sensitivity")["estimate"][0]
    )


def test_multinomial_rejects_offset():
    rng = np.random.default_rng(1)
    n = 120
    x = rng.normal(size=n)
    frame = pl.DataFrame(
        {
            "x": x,
            "o": rng.normal(size=n),
            "y": np.digitize(x + rng.logistic(size=n), [-1, 0.5]),
        }
    )
    with pytest.raises(ValueError, match="offset"):
        multinomial_regression(frame, "y ~ x + offset(o)")
