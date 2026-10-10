"""Risk ratios against ``glm`` with ``sandwich::vcovHC`` and ``vcovCL``.

Fixture: ``fixtures/risk_ratio.json`` from ``scripts/risk_ratio.R``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract import fit_risk_ratio

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "risk_ratio.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


def _fit(name: str, **kwargs):
    case = FIXTURE["cases"][name]
    return fit_risk_ratio(
        _data(case["data"]),
        case["formula"],
        method=case["method"],
        weights=case["weights"],
        start=case["start"],
        **kwargs,
    )


@pytest.mark.parametrize("name", CASES)
def test_coefficients_and_covariances(name: str) -> None:
    want = FIXTURE["cases"][name]
    robust = _fit(name, covariance="robust")
    model = _fit(name, covariance="model")
    fit = robust.fit
    assert fit.names == want["names"]
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(robust.covariance, np.asarray(want["vcov_hc0"]), rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(model.covariance, np.asarray(want["vcov_model"]), rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(fit.deviance, want["deviance"], rtol=1e-10)
    np.testing.assert_allclose(fit.log_likelihood, want["loglik"], rtol=1e-10)
    np.testing.assert_allclose(fit.aic, want["aic"], rtol=1e-10)
    np.testing.assert_allclose(robust.predict(), want["fitted"], rtol=1e-8)
    if want["cluster"] is not None:
        clustered = _fit(name, cluster=want["cluster"])
        assert clustered.covariance_kind == "cluster"
        np.testing.assert_allclose(clustered.covariance, np.asarray(want["vcov_cluster"]), rtol=1e-6, atol=1e-14)


@pytest.mark.parametrize("name", CASES)
def test_tidy_risk_ratio(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _fit(name, covariance="robust").tidy()
    se = np.sqrt(np.diag(np.asarray(want["vcov_hc0"])))
    beta = np.asarray(want["coefficients"])
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(got["estimate"].to_numpy(), np.exp(beta), rtol=1e-8)
    np.testing.assert_allclose(got["std_error"].to_numpy(), se, rtol=1e-6)
    np.testing.assert_allclose(got["conf_low"].to_numpy(), np.exp(beta - z * se), rtol=1e-6)
    np.testing.assert_allclose(got["conf_high"].to_numpy(), np.exp(beta + z * se), rtol=1e-6)
    np.testing.assert_allclose(got["p_value"].to_numpy(), 2 * stats.norm.sf(np.abs(beta / se)), rtol=1e-5, atol=1e-14)


def test_defaults() -> None:
    assert _fit("poisson_a").covariance_kind == "HC0"
    assert _fit("logbin_a").covariance_kind == "model"
    assert _fit("logbin_a", cluster="site").covariance_kind == "cluster"


def test_column_interface_matches_formula() -> None:
    data = _data("risk_a")
    by_formula = fit_risk_ratio(data, "y ~ arm + age + sex")
    by_columns = fit_risk_ratio(data, "y", ["arm", "age", "sex"])
    np.testing.assert_allclose(by_columns.coefficients, by_formula.coefficients, rtol=1e-10)
    np.testing.assert_allclose(by_columns.covariance, by_formula.covariance, rtol=1e-8)
    log_bin = fit_risk_ratio(data, "y", ["arm", "age", "sex"], method="log-binomial")
    np.testing.assert_allclose(log_bin.coefficients, FIXTURE["cases"]["logbin_a"]["coefficients"], rtol=1e-8)


def test_rejects_non_binary_outcome() -> None:
    data = _data("risk_a").with_columns((pl.col("y") * 2).alias("y"))
    with pytest.raises(ValueError, match="0/1"):
        fit_risk_ratio(data, "y ~ arm")


def test_log_binomial_bad_start() -> None:
    data = _data("risk_a")
    with pytest.raises(ValueError, match="valid starting values"):
        fit_risk_ratio(data, "y ~ arm + age + sex", method="log-binomial", start=[1.0, 0.0, 0.0, 0.0])
