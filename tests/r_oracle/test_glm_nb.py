"""Negative binomial GLM against ``MASS::glm.nb``.

Fixture: ``fixtures/glm_nb.json`` from ``scripts/glm_nb.R``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_glm
from statract.models.covariance import hc_covariance

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "glm_nb.json").read_text())
CASES = sorted(FIXTURE["cases"])
ARM = pl.Enum(["control", "low", "high"])


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv").with_columns(pl.col("arm").cast(ARM))


def _fit(case: dict) -> object:
    return fit_glm(_data(case["data"]), case["formula"], family="negative_binomial", weights=case.get("weights") or None)


@pytest.mark.parametrize("name", CASES)
def test_matches_glm_nb(name: str) -> None:
    want = FIXTURE["cases"][name]
    fit = _fit(want)
    assert fit.names == want["terms"]
    # Same algorithm and stopping rule as glm.nb, so the numbers agree closely.
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), want["std_errors"], rtol=1e-7)
    np.testing.assert_allclose(fit.theta, want["theta"], rtol=1e-7)
    np.testing.assert_allclose(fit.theta_std_error, want["theta_std_error"], rtol=1e-6)
    np.testing.assert_allclose(fit.log_likelihood, want["log_likelihood"], rtol=1e-10)
    np.testing.assert_allclose(fit.aic, want["aic"], rtol=1e-10)
    np.testing.assert_allclose(fit.bic, want["bic"], rtol=1e-10)
    np.testing.assert_allclose(fit.deviance, want["deviance"], rtol=1e-8)
    assert fit.residual_df == want["df_residual"]
    np.testing.assert_allclose(fit.predict()[:5], want["fitted"], rtol=1e-7)
    np.testing.assert_allclose(hc_covariance(fit, "HC0"), np.asarray(want["hc0"]), rtol=1e-6, atol=1e-12)


def test_column_offset_equals_formula_offset() -> None:
    data = _data("glmm_count_b").with_columns(pl.col("years").log().alias("log_years"))
    formula = fit_glm(data, "y ~ x + arm + offset(log(years))", family="negative_binomial")
    column = fit_glm(data, "y ~ x + arm", family="negative_binomial", offset="log_years")
    np.testing.assert_allclose(column.coefficients, formula.coefficients, rtol=1e-12)
    np.testing.assert_allclose(column.theta, formula.theta, rtol=1e-12)


def test_rejects_negative_counts() -> None:
    data = pl.DataFrame({"y": [1.0, -1.0, 3.0, 0.0], "x": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="non-negative"):
        fit_glm(data, "y ~ x", family="negative_binomial")


def test_refit_reestimates_theta() -> None:
    fit = fit_glm(_data("glmm_count_b"), "y ~ x + arm", family="negative_binomial")
    full = fit.refit_rows(np.arange(fit.n_obs))
    np.testing.assert_allclose(full.coefficients, fit.coefficients, rtol=1e-12)
    half = fit.refit_rows(np.arange(0, fit.n_obs, 2))
    assert half.theta is not None and half.theta != fit.theta
