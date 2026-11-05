"""Zero prior weights: those rows are not observations, as in ``lm`` and ``glm``."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from statract.models.covariance import cluster_covariance, hc_covariance, newey_west_covariance
from statract.models.fit import fit_glm, fit_ols
from statract.models.risk import fit_risk_ratio


@pytest.fixture(scope="module")
def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(5)
    n = 30
    x = rng.normal(size=n)
    y = 1.0 + x + rng.normal(size=n)
    w = np.r_[np.zeros(5), rng.uniform(0.5, 2.0, n - 5)]
    full = pl.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "g": np.exp(y / 3.0),
            "b": (rng.uniform(size=n) < 0.4).astype(float),
            "c": rng.poisson(2.0, size=n).astype(float),
            "cl": np.repeat(np.arange(10), 3),
        }
    )
    return full, full[5:]


def _same(a, b, full, sub) -> None:
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.covariance, b.covariance, rtol=1e-6, atol=1e-12)
    assert a.n_obs == b.n_obs
    assert a.residual_df == b.residual_df
    for name in ("log_likelihood", "aic", "bic", "deviance", "scale"):
        assert np.isfinite(getattr(a, name)), name
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=1e-6), name
    for kind in ("HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5", "const"):
        np.testing.assert_allclose(hc_covariance(a, kind), hc_covariance(b, kind), rtol=1e-6, atol=1e-12)
    for kind in ("HC0", "HC1", "HC2", "HC3"):
        np.testing.assert_allclose(
            cluster_covariance(a, "cl", data=full, kind=kind),
            cluster_covariance(b, "cl", data=sub, kind=kind),
            rtol=1e-6,
            atol=1e-12,
        )
    np.testing.assert_allclose(newey_west_covariance(a), newey_west_covariance(b), rtol=1e-6, atol=1e-12)


def test_ols_zero_weights_match_dropped_rows(frames):
    full, sub = frames
    a = fit_ols(full, "y ~ x", weights="w")
    b = fit_ols(sub, "y ~ x", weights="w")
    assert a.residual_df == 23
    assert a.n_obs == 25
    _same(a, b, full, sub)
    for name in ("r_squared", "adj_r_squared", "f_statistic"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=1e-6)
    # Zero-weight rows keep their fitted values and residuals.
    assert a.working_residuals.shape == (30,)
    np.testing.assert_allclose(a.predict()[:5], a.x[:5] @ a.coefficients)
    assert a.glance()["n_obs"].item() == 25


def test_ols_column_api_zero_weights(frames):
    full, sub = frames
    a = fit_ols(full, "y", ["x"], weights="w")
    b = fit_ols(sub, "y", ["x"], weights="w")
    _same(a, b, full, sub)


@pytest.mark.parametrize(
    ("family", "outcome"),
    [
        ("gaussian", "y"),
        ("gamma", "g"),
        ("binomial", "b"),
        ("poisson", "c"),
        ("negative_binomial", "c"),
    ],
)
def test_glm_zero_weights_match_dropped_rows(frames, family, outcome):
    full, sub = frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = fit_glm(full, f"{outcome} ~ x", family=family, weights="w")
        b = fit_glm(sub, f"{outcome} ~ x", family=family, weights="w")
    assert a.n_obs == 25
    _same(a, b, full, sub)


@pytest.mark.parametrize("method", ["poisson", "log-binomial"])
def test_risk_ratio_zero_weights_match_dropped_rows(frames, method):
    full, sub = frames
    a = fit_risk_ratio(full, "b ~ x", method=method, weights="w")
    b = fit_risk_ratio(sub, "b ~ x", method=method, weights="w")
    np.testing.assert_allclose(a.covariance, b.covariance, rtol=1e-6, atol=1e-12)
    _same(a.fit, b.fit, full, sub)


def test_negative_weights_raise(frames):
    full, _ = frames
    bad = full.with_columns(pl.col("w") - 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        fit_ols(bad, "y ~ x", weights="w")
    with pytest.raises(ValueError, match="non-negative"):
        fit_glm(bad, "y ~ x", family="gaussian", weights="w")
    with pytest.raises(ValueError, match="non-negative"):
        fit_risk_ratio(bad, "b ~ x", method="log-binomial", weights="w")
