"""Binomial GLMMs against glmer: Laplace, nAGQ = 25, MOR, and reference predictions.

Fixture: ``fixtures/glmm_binary_glmer.json`` from ``scripts/glmm_binary_glmer.R``.
glmer's default inner tolerance (``tolPwrss = 1e-7``) leaves its Laplace
log-likelihood about 1e-4 short of the exact value. Cases fitted with
``tolPwrss = 1e-13`` are compared tightly; the default-control cases get the
looser tolerance that this gap allows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_mixed, median_odds_ratio

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "glmm_binary_glmer.json").read_text())
ARM = pl.Enum(["control", "low", "high"])
REFERENCE = pl.DataFrame(FIXTURE["reference"]).with_columns(pl.col("arm").cast(ARM))

# coefficients, SE, group covariance (relative); predictions (absolute); MOR (relative)
TIGHT = {"coef": 2e-6, "se": 2e-5, "cov": 1e-4, "pred": 1e-7, "mor": 1e-6}
DEFAULT = {"coef": 1e-3, "se": 1e-2, "cov": 2e-2, "pred": 2e-4, "mor": 2e-3}


def _fit(case: dict):
    data = pl.read_csv(HERE / "data" / f"{case['sample']}.csv").with_columns(pl.col("arm").cast(ARM))
    return fit_mixed(data, case["formula"], family="binomial", n_agq=case["nagq"])


@pytest.mark.parametrize("name", sorted(FIXTURE["cases"]))
def test_binomial_glmm_matches_glmer(name: str) -> None:
    case = FIXTURE["cases"][name]
    tol = TIGHT if case["tight"] else DEFAULT
    fit = _fit(case)
    assert fit.engine == "laplace"
    assert fit.converged
    assert fit.names == case["terms"]
    if case["tight"]:
        np.testing.assert_allclose(fit.log_likelihood, case["log_likelihood"], atol=1e-7)
    else:
        assert fit.log_likelihood >= case["log_likelihood"] - 1e-9
    np.testing.assert_allclose(fit.coefficients, case["coefficients"], rtol=tol["coef"], atol=1e-7)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), case["std_errors"], rtol=tol["se"])
    np.testing.assert_allclose(
        fit.group_covariance, np.atleast_2d(case["group_covariance"]), rtol=tol["cov"], atol=1e-8
    )
    np.testing.assert_allclose(fit.predict(REFERENCE, kind="link"), case["reference_link"], atol=10 * tol["pred"])
    np.testing.assert_allclose(fit.predict(REFERENCE), case["reference_response"], atol=tol["pred"])
    if "median_odds_ratio" in case:
        mor = median_odds_ratio(fit)["median_odds_ratio"][0]
        np.testing.assert_allclose(mor, case["median_odds_ratio"], rtol=tol["mor"])
