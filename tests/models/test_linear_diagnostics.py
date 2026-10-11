"""Durbin–Watson, Breusch–Pagan and Breusch–Godfrey on small hand-worked examples."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract import (
    breusch_godfrey_test,
    breusch_pagan_test,
    durbin_watson_test,
    fit_ols,
)
from statract.models.linear_tests import _durbin_watson_moments


def _intercept_only():
    # Residuals about the mean 2.5: (-1.5, 0.5, -0.5, 1.5).
    data = pl.DataFrame({"y": [1.0, 3.0, 2.0, 4.0], "one": [1.0, 1.0, 1.0, 1.0]})
    return fit_ols(data, "y ~ 1")


def test_durbin_watson_hand_example():
    fit = _intercept_only()
    # d = (2^2 + 1^2 + 2^2) / (2.25 + 0.25 + 0.25 + 2.25) = 9 / 5.
    # Regression on a constant only (Durbin & Watson 1950): E d = 2 and
    # Var d = 4 (n - 2) / (n^2 - 1) = 8 / 15 for n = 4.
    z = (1.8 - 2.0) / math.sqrt(8 / 15)
    got = durbin_watson_test(fit)
    assert got.statistic == pytest.approx(1.8, rel=1e-12)
    assert got.p_value == pytest.approx(stats.norm.cdf(z), rel=1e-12)
    assert got.df == (3.0,)
    assert durbin_watson_test(fit, alternative="less").p_value == pytest.approx(stats.norm.sf(z), rel=1e-12)
    assert durbin_watson_test(fit, alternative="two.sided").p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-12)
    with pytest.raises(ValueError, match="alternative"):
        durbin_watson_test(fit, alternative="bigger")


def test_durbin_watson_moments_match_explicit_traces():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(12), rng.normal(size=12), rng.normal(size=12)])
    n, k = x.shape
    diff = np.diff(np.eye(n), axis=0)
    a = diff.T @ diff
    m = np.eye(n) - x @ np.linalg.solve(x.T @ x, x.T)
    ma = m @ a
    mean = np.trace(ma) / (n - k)
    var = 2 * (np.trace(ma @ ma) - np.trace(ma) * mean) / ((n - k) * (n - k + 2))
    got_mean, got_var, rank = _durbin_watson_moments(x)
    assert rank == k
    assert got_mean == pytest.approx(mean, rel=1e-12)
    assert got_var == pytest.approx(var, rel=1e-12)


def test_breusch_pagan_hand_example():
    # Two groups of three; residuals about the group means (2, 5):
    # e = (-1, 0, 1, -3, 0, 3), e^2 = (1, 0, 1, 9, 0, 9), sigma2 = 20/6.
    data = pl.DataFrame({"y": [1.0, 2.0, 3.0, 2.0, 5.0, 8.0], "g": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
    fit = fit_ols(data, "y ~ g")
    # Original (Breusch & Pagan 1979): g_i = e^2 / sigma2 has group means 0.2 and 1.8
    # around 1, so ESS = 6 * 0.8^2 = 3.84 and the statistic is ESS / 2 = 1.92.
    original = breusch_pagan_test(fit, studentize=False)
    assert original.statistic == pytest.approx(1.92, rel=1e-12)
    assert original.df == (1.0,)
    assert original.p_value == pytest.approx(stats.chi2.sf(1.92, 1), rel=1e-12)
    # Koenker (1981): n R^2 of e^2 on g; ESS = 6 (8/3)^2 = 384/9, TSS = 876/9.
    koenker = breusch_pagan_test(fit)
    assert koenker.statistic == pytest.approx(6 * 384 / 876, rel=1e-12)
    # Koenker's ratio: original = studentized * var(e^2) / (2 sigma2^2), var with divisor n.
    e2 = np.array([1.0, 0, 1, 9, 0, 9])
    ratio = np.var(e2) / (2 * (20 / 6) ** 2)
    assert original.statistic == pytest.approx(koenker.statistic * ratio, rel=1e-12)


def test_breusch_godfrey_hand_example():
    fit = _intercept_only()
    # e = (-1.5, 0.5, -0.5, 1.5), lag = (0, -1.5, 0.5, -0.5). Regressing e on (1, lag):
    # Sxy = -1.75, Sxx = 2.1875, ESS = 1.4, RSS = 5 - 1.4 = 3.6.
    chi2 = breusch_godfrey_test(fit)
    assert chi2.statistic == pytest.approx(4 * 1.4 / 5, rel=1e-12)
    assert chi2.df == (1.0,)
    assert chi2.p_value == pytest.approx(stats.chi2.sf(1.12, 1), rel=1e-12)
    f = breusch_godfrey_test(fit, distribution="f")
    assert f.statistic == pytest.approx(1.4 / (3.6 / 2), rel=1e-12)
    assert f.df == (1.0, 2.0)
    assert f.p_value == pytest.approx(stats.f.sf(1.4 / 1.8, 1, 2), rel=1e-12)
    with pytest.raises(ValueError, match="distribution"):
        breusch_godfrey_test(fit, distribution="t")


def test_breusch_godfrey_f_is_squared_t_of_the_lag():
    rng = np.random.default_rng(11)
    x = rng.normal(size=30)
    y = 1 + 0.5 * x + np.cumsum(rng.normal(size=30)) * 0.3
    fit = fit_ols(pl.DataFrame({"y": y, "x": x}), "y ~ x")
    e = fit.working_residuals
    lag = np.concatenate([[0.0], e[:-1]])
    aux = fit_ols(pl.DataFrame({"e": e, "x": x, "lag": lag}), "e ~ x + lag")
    t_lag = float(aux.tidy().filter(pl.col("term") == "lag")["statistic"][0])
    assert breusch_godfrey_test(fit, distribution="f").statistic == pytest.approx(t_lag**2, rel=1e-10)
