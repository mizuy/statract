"""Poisson and negative binomial GLMMs against glmmTMB and glmer.

Fixture: ``fixtures/glmm_count.json`` from ``scripts/glmm_count.R``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import integrate, special

from statract import fit_mixed

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "glmm_count.json").read_text())
CASES = sorted(FIXTURE["cases"])
ARM = pl.Enum(["control", "low", "high"])


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv").with_columns(pl.col("arm").cast(ARM))


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_laplace_matches_glmmtmb(name: str, family: str) -> None:
    case = FIXTURE["cases"][name]
    want = case[family]
    fit = fit_mixed(_data(name), case["formula"], family=family)
    assert fit.engine == "laplace"
    assert fit.converged
    assert fit.names == want["terms"]
    # glmmTMB stops at its optimiser tolerance. The objective is flat there, so
    # coefficients agree to about 1e-5 while our log-likelihood is never lower.
    assert fit.log_likelihood >= want["log_likelihood"] - 1e-9
    np.testing.assert_allclose(fit.log_likelihood, want["log_likelihood"], atol=1e-6)
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-4, atol=2e-5)
    se = np.sqrt(np.diag(fit.covariance))
    np.testing.assert_allclose(se, want["std_errors"], rtol=1e-4)
    np.testing.assert_allclose(fit.group_covariance, np.asarray(want["group_covariance"]), rtol=1e-3, atol=1e-6)
    if family == "negative_binomial":
        np.testing.assert_allclose(fit.theta, want["theta"], rtol=1e-4)


@pytest.mark.parametrize("name", [n for n in CASES if FIXTURE["cases"][n]["agq"] is not None])
def test_adaptive_quadrature_matches_glmer(name: str) -> None:
    case = FIXTURE["cases"][name]
    want = case["agq"]
    fit = fit_mixed(_data(name), case["formula"], family="poisson", n_agq=9)
    # glmer stops at its own optimiser tolerance, about 1e-5 from the optimum.
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fit.group_covariance[0, 0], want["variance"], rtol=1e-3)


def test_column_offset_equals_formula_offset() -> None:
    data = _data("glmm_count_a").with_columns(pl.col("years").log().alias("log_years"))
    formula = fit_mixed(data, "y ~ x + arm + offset(log(years)) + (1 | site)", family="poisson")
    columns = fit_mixed(data, "y", ["x", "arm"], groups="site", offset="log_years", family="poisson")
    np.testing.assert_allclose(columns.coefficients, formula.coefficients, rtol=1e-8)
    np.testing.assert_allclose(columns.predict(data), formula.predict(data), rtol=1e-8)
    np.testing.assert_allclose(formula.predict(data), formula.predict(), rtol=1e-12)


def test_quadrature_integral_is_the_marginal_likelihood() -> None:
    """Adaptive Gauss-Hermite at the fit equals direct numerical integration."""
    data = _data("glmm_count_c")
    fit = fit_mixed(data, "y ~ x + offset(log(years)) + (1 | site)", family="poisson", n_agq=15)
    sd = float(np.sqrt(fit.group_covariance[0, 0]))
    eta = fit.x @ fit.coefficients + fit.offset
    y = data["y"].to_numpy().astype(float)
    site = data["site"].to_numpy()
    total = 0.0
    for label in np.unique(site):
        rows = site == label

        def integrand(b, rows=rows):
            mu = np.exp(eta[rows] + b)
            ll = np.sum(y[rows] * (eta[rows] + b) - mu - special.gammaln(y[rows] + 1))
            return np.exp(ll) * np.exp(-0.5 * (b / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

        value, _ = integrate.quad(integrand, -10 * sd, 10 * sd, epsabs=0, epsrel=1e-12, limit=200)
        total += np.log(value)
    np.testing.assert_allclose(fit.log_likelihood, total, rtol=1e-9)


def test_count_family_guards() -> None:
    data = _data("glmm_count_a")
    with pytest.raises(ValueError, match="laplace"):
        fit_mixed(data, "y ~ x + (1 | site)", family="negative_binomial", engine="lme")
    with pytest.raises(ValueError, match="offsets"):
        fit_mixed(data, "x ~ arm + offset(log(years)) + (1 | site)")
    with pytest.raises(ValueError, match="single random effect"):
        fit_mixed(data, "y ~ x + (1 + x | site)", family="poisson", n_agq=5)
