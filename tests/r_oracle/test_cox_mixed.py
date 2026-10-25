"""Mixed-effects Cox models against coxme.

Fixture: ``fixtures/cox_mixed.json`` from ``scripts/cox_mixed.R``. Each case
has the fit with estimated variances and, for most, a refit with the
variances fixed at coxme's estimates (``vfixed``). The fixed fits are
compared tightly. With estimated variances, statract's integrated
log-likelihood must be at least coxme's, and the estimates agree to the
precision of coxme's variance search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cox_mixed

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "cox_mixed.json").read_text())
ARM = pl.Enum(["control", "low", "high"])


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv").with_columns(pl.col("arm").cast(ARM))


def _fit(case: dict, variances=None):
    return cox_mixed(_data(case["sample"]), case["ours"], ties=case["ties"], weights=case["weights"], variances=variances)


def _by_r_order(fit, case: dict) -> list[dict]:
    """statract's random terms in coxme's term order."""
    groups = list(case["groups"])
    ours = [term["group"] for term in fit.random_terms]
    if len(set(ours)) < len(ours):
        assert ours == groups
        return list(fit.random_terms)
    return [fit.random_terms[ours.index(group)] for group in groups]


def _variances(case: dict):
    variances = case["fixed"]["variances"]
    groups = list(case["groups"])
    if len(set(groups)) < len(groups):
        return variances
    return dict(zip(groups, variances, strict=True))


def _check_blups(fit, case: dict, want: list[dict], atol: float) -> None:
    for term, ref in zip(_by_r_order(fit, case), want, strict=True):
        position = {level: k for k, level in enumerate(term["levels"])}
        at = [position[level.replace("/", ":")] for level in ref["levels"]]
        values = np.asarray(ref["values"], dtype=float).reshape(len(at), -1)
        np.testing.assert_allclose(term["blups"][at], values, atol=atol)


FIXED = sorted(name for name, case in FIXTURE["cases"].items() if "fixed" in case)


@pytest.mark.parametrize("name", FIXED)
def test_fixed_variances_match_coxme(name: str) -> None:
    case = FIXTURE["cases"][name]
    want = case["fixed"]
    fit = _fit(case, _variances(case))
    assert fit.names == list(np.atleast_1d(case["terms"]))
    assert fit.n_events == case["n_events"]
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fit.covariance, np.atleast_2d(want["covariance"]), rtol=1e-5, atol=1e-10)
    got = [fit.log_likelihood_null, fit.log_likelihood, fit.log_likelihood_penalized]
    np.testing.assert_allclose(got, want["loglik"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit.df, want["df"][1], rtol=1e-6)
    assert want["df"][0] == len(fit.coefficients) + fit.n_variance_parameters
    _check_blups(fit, case, want["blups"], atol=1e-6)


@pytest.mark.parametrize("name", sorted(n for n in FIXTURE["cases"] if not n.endswith("_sparse")))
def test_estimated_variances_match_coxme(name: str) -> None:
    case = FIXTURE["cases"][name]
    want = case["estimated"]
    fit = _fit(case)
    assert fit.converged
    # coxme's search stops near the optimum (farther when a variance is near
    # zero, where the likelihood is flat); statract must not end lower.
    assert fit.log_likelihood >= want["loglik"][1] - 1e-6
    assert fit.log_likelihood - want["loglik"][1] < 1e-2
    for term, cov in zip(_by_r_order(fit, case), want["covariances"], strict=True):
        np.testing.assert_allclose(term["covariance"], np.atleast_2d(cov), rtol=2e-2, atol=3e-3)
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=0, atol=2e-3)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), np.sqrt(np.diag(np.atleast_2d(want["covariance"]))), rtol=1e-2)
    _check_blups(fit, case, want["blups"], atol=1e-2)


def test_sparse_coxme_differs_from_the_exact_fit() -> None:
    # coxme's default sparse approximation (more than 50 levels) moves the
    # variance; statract computes the full information and matches the exact fit.
    exact = FIXTURE["cases"]["many_levels"]["estimated"]
    approx = FIXTURE["cases"]["many_levels_sparse"]["estimated"]
    fit = _fit(FIXTURE["cases"]["many_levels_sparse"])
    variance = fit.random_terms[0]["covariance"][0, 0]
    assert abs(variance - exact["covariances"][0][0][0]) < 1e-3
    assert abs(approx["covariances"][0][0][0] - exact["covariances"][0][0][0]) > 3e-3


def test_likelihood_table_follows_the_coxme_print() -> None:
    case = FIXTURE["cases"]["intercept"]
    want = case["fixed"]
    fit = _fit(case, _variances(case))
    table = fit.likelihood_table()
    null, integrated, _penalized = want["loglik"]
    chisq = 2 * (integrated - null)
    row = table.row(0, named=True)
    np.testing.assert_allclose(row["chisq"], chisq, atol=1e-5)
    np.testing.assert_allclose(row["aic"], chisq - 2 * want["df"][0], atol=1e-5)
    np.testing.assert_allclose(row["bic"], chisq - np.log(case["n_events"]) * want["df"][0], atol=1e-5)
    # The penalized row uses the unpenalized partial likelihood at the estimates.
    fitted = table.row(1, named=True)
    np.testing.assert_allclose(fitted["chisq"], 2 * (fit.log_likelihood_fitted - null), atol=1e-8)
    np.testing.assert_allclose(fitted["df"], want["df"][1], rtol=1e-6)
