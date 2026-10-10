"""Multinomial logistic regression against ``nnet::multinom``.

Fixture: ``fixtures/multinom.json`` from ``scripts/multinom.R``. The ``tight``
fit (``reltol = 1e-12``) is within about 1e-6 of the optimum and is
compared at rtol 2e-5.
The default fit stops BFGS at ``reltol = 1e-8`` and is compared loosely.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import multinomial_regression

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "multinom.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _fit(name: str):
    case = FIXTURE["cases"][name]
    return multinomial_regression(
        _data(case["data"]), case["formula"], weights=case["weights"], levels=case["levels"]
    )


@pytest.mark.parametrize("name", CASES)
def test_matches_tight_multinom(name: str) -> None:
    want = FIXTURE["cases"][name]["tight"]
    got = _fit(name)
    assert got.converged
    assert got.names == want["names"]
    assert got.levels == want["levels"]
    # Even with reltol 1e-12, BFGS stops about 1e-6 from the optimum, so the
    # coefficients are compared at rtol 2e-5 and the deviance must not be larger.
    np.testing.assert_allclose(got.coefficients, np.asarray(want["coefficients"]), rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(got.std_errors, np.asarray(want["std_errors"]), rtol=2e-5)
    np.testing.assert_allclose(got.covariance, np.asarray(want["vcov"]), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(got.deviance, want["deviance"], rtol=1e-11)
    assert got.deviance <= want["deviance"] + 1e-9
    np.testing.assert_allclose(got.log_likelihood, want["loglik"], rtol=1e-11)
    np.testing.assert_allclose(got.aic, want["aic"], rtol=1e-11)
    assert got.coefficients.size == want["edf"]
    fitted = np.asarray(want["fitted"])
    # With two levels, fitted(multinom) is the probability of the second.
    np.testing.assert_allclose(got.fitted[:, -fitted.shape[1] :], fitted, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("name", CASES)
def test_near_default_multinom(name: str) -> None:
    want = FIXTURE["cases"][name]["default"]
    got = _fit(name)
    # BFGS with reltol 1e-8 leaves the coefficients about 1e-4 from the optimum.
    np.testing.assert_allclose(got.coefficients, np.asarray(want["coefficients"]), rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(got.deviance, want["deviance"], rtol=1e-7)
    assert got.deviance <= want["deviance"] + 1e-9


def test_tidy_and_predict() -> None:
    fit = _fit("three")
    table = fit.tidy(exponentiate=True)
    assert table["y_level"].to_list() == ["B"] * 4 + ["C"] * 4
    assert table["term"].to_list()[:4] == ["(Intercept)", "x", "age", "sexM"]
    np.testing.assert_allclose(table["std_error"].to_numpy(), fit.std_errors.reshape(-1))
    data = _data("multinom_a")
    np.testing.assert_allclose(fit.predict(data.head(8)), fit.fitted[:8], rtol=1e-12)
    assert fit.probabilities().columns == ["A", "B", "C"]
    assert set(fit.predict(data.head(8), kind="class")) <= {"A", "B", "C"}


def test_empty_level_is_dropped() -> None:
    data = _data("multinom_a")
    with pytest.warns(UserWarning, match="empty"):
        fit = multinomial_regression(data, "y ~ x", levels=["A", "B", "C", "D"])
    assert fit.levels == ["A", "B", "C"]
