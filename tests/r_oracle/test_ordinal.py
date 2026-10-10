"""Cumulative-link models against ``ordinal::clm`` and ``MASS::polr``.

Fixture: ``fixtures/ordinal.json`` from ``scripts/ordinal.R``. ``clm`` has an
analytic Hessian, so its covariance is compared at rtol 1e-6. ``polr`` takes
its Hessian from finite differences in ``optim``, so its standard errors are
compared at rtol 2e-3. The Brant reference is a hand-coded R re-implementation
(no R package is available), labelled as such in the script.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import brant_test, ordinal_regression

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "ordinal.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _fit(name: str):
    case = FIXTURE["cases"][name]
    return ordinal_regression(
        _data(case["data"]),
        case["formula"],
        link=case["link"],
        weights=case["weights"],
        levels=case["levels"],
    )


@pytest.mark.parametrize("name", CASES)
def test_matches_clm(name: str) -> None:
    want = FIXTURE["cases"][name]["clm"]
    got = _fit(name)
    assert got.converged
    assert got.names == want["names"]
    assert got.threshold_names == want["threshold_names"]
    # clm counts the weights; n_obs counts rows.
    np.testing.assert_allclose(got.weights.sum(), want["n"], rtol=1e-12)
    np.testing.assert_allclose(got.coefficients, want["coefficients"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.thresholds, want["thresholds"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.covariance, np.asarray(want["vcov"]), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got.log_likelihood, want["loglik"], rtol=1e-10)
    np.testing.assert_allclose(got.aic, want["aic"], rtol=1e-10)
    np.testing.assert_allclose(got.fitted, np.asarray(want["fitted"]), rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("name", [n for n in CASES if "polr" in FIXTURE["cases"][n]])
def test_matches_polr(name: str) -> None:
    want = FIXTURE["cases"][name]["polr"]
    got = _fit(name)
    np.testing.assert_allclose(got.coefficients, want["coefficients"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got.thresholds, want["zeta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got.log_likelihood, want["loglik"], rtol=1e-10)
    np.testing.assert_allclose(got.deviance, want["deviance"], rtol=1e-10)
    np.testing.assert_allclose(got.aic, want["aic"], rtol=1e-10)
    se = np.sqrt(np.diag(got.covariance))
    # optim's finite-difference Hessian (ndeps 1e-3) is off by up to about 1e-3.
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.asarray(want["vcov"]))), rtol=2e-3)
    np.testing.assert_allclose(got.fitted[:20], np.asarray(want["fitted_head"]), rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("name", [n for n in CASES if "brant" in FIXTURE["cases"][n]])
def test_brant_matches_hand_coded_r(name: str) -> None:
    want = FIXTURE["cases"][name]["brant"]
    got = brant_test(_fit(name))
    assert got.table["term"].to_list() == want["term"]
    np.testing.assert_allclose(got.binary_coefficients, np.asarray(want["binary_coefficients"]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.table["statistic"].to_numpy(), want["statistic"], rtol=1e-6)
    np.testing.assert_array_equal(got.table["df"].to_numpy(), want["df"])
    np.testing.assert_allclose(got.table["p_value"].to_numpy(), want["p_value"], rtol=1e-5, atol=1e-12)


def test_tidy_and_predict() -> None:
    fit = _fit("a_logit")
    table = fit.tidy(exponentiate=True)
    assert table["coef_type"].to_list() == ["coefficient"] * 4 + ["threshold"] * 3
    np.testing.assert_allclose(table["std_error"].to_numpy(), np.sqrt(np.diag(fit.covariance)))
    np.testing.assert_allclose(table["exp_estimate"].to_numpy(), np.exp(table["estimate"].to_numpy()))
    data = _data("ordinal_a")
    probs = fit.predict(data.head(10))
    np.testing.assert_allclose(probs, fit.fitted[:10], rtol=1e-12)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert set(fit.predict(data.head(10), kind="class")) <= set(fit.levels)
    frame = fit.probabilities()
    assert frame.columns == ["none", "mild", "moderate", "severe"]


def test_levels_default_order() -> None:
    data = _data("ordinal_b")
    fit = ordinal_regression(data, "y ~ x + stage", weights="w")
    assert fit.levels == ["1", "2", "3", "4", "5"]
    assert "(Intercept)" not in fit.names


def test_cauchit_reaches_a_stationary_point() -> None:
    # polr and clm bound the linear predictor, so cauchit has no R reference.
    fit = ordinal_regression(_data("ordinal_a"), "y ~ x + stage", link="cauchit", levels=["none", "mild", "moderate", "severe"])
    assert fit.converged
    assert fit.gradient_max < 1e-8


def test_brant_needs_logit() -> None:
    with pytest.raises(ValueError, match="logit"):
        brant_test(_fit("a_probit"))
