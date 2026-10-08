"""Binomial, zero-inflated, and hurdle GLMMs against glmmTMB and glmer.

Fixture: ``fixtures/glmm_binary_zero.json`` from ``scripts/glmm_binary_zero.R``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_mixed

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "glmm_binary_zero.json").read_text())
ARM = pl.Enum(["control", "low", "high"])


def _data(name: str) -> pl.DataFrame:
    frame = pl.read_csv(HERE / "data" / f"{name}.csv")
    if "arm" in frame.columns:
        frame = frame.with_columns(pl.col("arm").cast(ARM))
    return frame


def _check(fit, want) -> None:
    # glmmTMB stops at its optimiser tolerance; our log-likelihood is never lower.
    assert fit.converged
    assert fit.names == want["terms"]
    assert fit.log_likelihood >= want["log_likelihood"] - 1e-9
    np.testing.assert_allclose(fit.log_likelihood, want["log_likelihood"], atol=1e-6)
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), want["std_errors"], rtol=1e-4)
    np.testing.assert_allclose(fit.group_covariance, np.asarray(want["group_covariance"]), rtol=1e-3, atol=1e-6)
    if want.get("theta") is not None:
        np.testing.assert_allclose(fit.theta, want["theta"], rtol=1e-4)


@pytest.mark.parametrize("name", sorted(FIXTURE["binary"]))
def test_binomial_laplace_matches_glmmtmb(name: str) -> None:
    case = FIXTURE["binary"][name]
    fit = fit_mixed(_data(name), case["formula"], family="binomial")
    assert fit.engine == "laplace"
    _check(fit, case["laplace"])


@pytest.mark.parametrize("name", sorted(n for n, c in FIXTURE["binary"].items() if c["agq"] is not None))
def test_binomial_quadrature_matches_glmer(name: str) -> None:
    case = FIXTURE["binary"][name]
    fit = fit_mixed(_data(name), case["formula"], family="binomial", n_agq=9)
    np.testing.assert_allclose(fit.coefficients, case["agq"]["coefficients"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fit.group_covariance[0, 0], case["agq"]["variance"], rtol=1e-3)


@pytest.mark.parametrize("name", sorted(FIXTURE["zero"]))
def test_zero_parts_match_glmmtmb(name: str) -> None:
    case = FIXTURE["zero"][name]
    want = case["fit"]
    columns = case["zero_columns"]
    fit = fit_mixed(
        _data(case["sample"]),
        case["formula"],
        family=case["family"],
        zero_inflation=columns if columns else True,
        hurdle=case["zero"] == "hurdle",
    )
    _check(fit, want)
    table = fit.zero_table()
    # jsonlite writes a one-element vector as a scalar.
    assert table["term"].to_list() == list(np.atleast_1d(want["zero_terms"]))
    np.testing.assert_allclose(table["estimate"].to_numpy(), want["zero_coefficients"], rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(table["std_error"].to_numpy(), want["zero_std_errors"], rtol=1e-4)


def test_zero_inflated_mean_and_guards() -> None:
    data = _data("glmm_zero_a")
    fit = fit_mixed(data, "y ~ x + offset(log(years)) + (1 | site)", family="poisson", zero_inflation=["z"])
    eta = fit.x @ fit.coefficients + fit.offset
    pi = 1 / (1 + np.exp(-(fit.zero_coefficients[0] + fit.zero_coefficients[1] * data["z"].to_numpy())))
    np.testing.assert_allclose(fit.predict(), (1 - pi) * np.exp(eta), rtol=1e-12)
    np.testing.assert_allclose(fit.predict(data), fit.predict(), rtol=1e-12)
    with pytest.raises(ValueError, match="zero-inflated and hurdle"):
        fit_mixed(data, "y ~ x + (1 | site)", family="gaussian", zero_inflation=True)
    with pytest.raises(TypeError, match="no zero"):
        fit_mixed(data, "y ~ x + (1 | site)", family="poisson").zero_table()
    with pytest.raises(ValueError, match="0/1"):
        fit_mixed(data, "y ~ x + (1 | site)", family="binomial")
