"""Firth logistic regression against an independent R implementation.

Fixture: ``fixtures/firth.json`` from ``scripts/firth.R``. logistf is not
installable here, so that script fits Firth's model in base R from Firth
(1993) and Heinze and Schemper (2002): the modified-score Newton-Raphson with
the iteration controls described on ``logistf.control()``'s help page,
restricted fits that keep the full-design penalty, and profile
penalized-likelihood limits found with ``uniroot``. It shares no code with
statract. Both stop the main fit at the same 1e-5 rule, so the estimates
agree to rounding; ``logistf`` itself should agree to about 1e-5.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract import fit_firth, fit_glm

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "firth.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _fit(name: str):
    case = FIXTURE["cases"][name]
    return fit_firth(_data(case["data"]), case["formula"], weights=case["weights"], maxit=case["maxit"])


@pytest.mark.parametrize("name", CASES)
def test_estimates_and_covariance(name: str) -> None:
    want = FIXTURE["cases"][name]
    fit = _fit(name)
    assert fit.names == want["names"]
    assert fit.converged
    assert fit.iterations == want["iter"]
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fit.covariance, np.asarray(want["vcov"]), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(fit.log_likelihood, want["loglik"], rtol=1e-11)
    np.testing.assert_allclose(fit.log_likelihood_null, want["loglik_null"], rtol=1e-10)
    np.testing.assert_allclose(fit.predict(), want["fitted"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_profile_table(name: str) -> None:
    want = FIXTURE["cases"][name]
    fit = _fit(name)
    got = fit.tidy(level=want["level"])
    np.testing.assert_allclose(got["statistic"].to_numpy(), want["lr_stat"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got["p_value"].to_numpy(), want["prob"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got["conf_low"].to_numpy(), want["ci_lower"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(got["conf_high"].to_numpy(), want["ci_upper"], rtol=1e-7, atol=1e-8)
    odds = fit.tidy(level=want["level"], exponentiate=True)
    np.testing.assert_allclose(odds["conf_low"].to_numpy(), np.exp(want["ci_lower"]), rtol=1e-7)


@pytest.mark.parametrize("name", CASES)
def test_global_test(name: str) -> None:
    want = FIXTURE["cases"][name]
    row = _fit(name).glance().row(0, named=True)
    np.testing.assert_allclose(row["statistic"], want["global_stat"], rtol=1e-8)
    assert row["df"] == want["global_df"]
    np.testing.assert_allclose(row["p_value"], stats.chi2.sf(want["global_stat"], want["global_df"]), rtol=1e-6)


def test_wald_table() -> None:
    want = FIXTURE["cases"]["firth_a"]
    got = _fit("firth_a").tidy(method="wald")
    se = np.sqrt(np.diag(np.asarray(want["vcov"])))
    beta = np.asarray(want["coefficients"])
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(got["std_error"].to_numpy(), se, rtol=1e-8)
    np.testing.assert_allclose(got["conf_low"].to_numpy(), beta - z * se, rtol=1e-8)
    np.testing.assert_allclose(got["p_value"].to_numpy(), 2 * stats.norm.sf(np.abs(beta / se)), rtol=1e-6)


def test_separation_stays_finite() -> None:
    # glm runs off toward infinity; the Firth estimate and limits are finite.
    for name in ("firth_sep_quasi", "firth_sep_complete"):
        want = FIXTURE["cases"][name]
        case_data = _data(want["data"])
        glm = fit_glm(case_data, want["formula"], family="binomial")
        assert np.max(np.abs(glm.coefficients)) > 15
        table = _fit(name).tidy()
        assert np.all(np.isfinite(table["conf_low"].to_numpy()))
        assert np.all(np.isfinite(table["conf_high"].to_numpy()))


def test_default_maxit_warns_under_complete_separation() -> None:
    want = FIXTURE["cases"]["firth_sep_complete"]
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit_firth(_data(want["data"]), want["formula"])


def test_column_interface_and_new_data() -> None:
    data = _data("firth_a")
    by_formula = _fit("firth_a")
    by_columns = fit_firth(data, "y", ["arm", "x", "sex"])
    np.testing.assert_allclose(by_columns.coefficients, by_formula.coefficients, rtol=1e-12)
    np.testing.assert_allclose(by_formula.predict(data), by_formula.predict(), rtol=1e-12)
    np.testing.assert_allclose(
        by_formula.predict(data, kind="link"), np.log(by_formula.predict() / (1 - by_formula.predict())), rtol=1e-8
    )


def test_offset_column_matches_formula_offset() -> None:
    data = _data("firth_a").with_columns((0.3 * pl.col("x")).alias("o"))
    by_column = fit_firth(data, "y ~ arm + sex", offset="o")
    by_formula = fit_firth(data, "y ~ arm + sex + offset(o)")
    np.testing.assert_allclose(by_column.coefficients, by_formula.coefficients, rtol=1e-12)


def test_rejects_non_binary_outcome() -> None:
    data = _data("firth_a").with_columns((pl.col("y") * 2).alias("y"))
    with pytest.raises(ValueError, match="0/1"):
        fit_firth(data, "y ~ arm")
