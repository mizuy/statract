"""cox_pspline against ``coxph(Surv(time, status) ~ pspline(x, df))``.

Fixture: ``fixtures/cox_pspline.json`` from ``scripts/cox_pspline.R``. Checks
the spline coefficients, both variances, the effective df, the
log-likelihoods and AIC, the ``linear``/``nonlin`` rows of the summary, and
the ``termplot`` curve with its standard error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cox_pspline

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "cox_pspline.json").read_text())
CASES = sorted(FIXTURE["cases"])


@pytest.mark.parametrize("name", CASES)
def test_cox_pspline_matches_coxph(name: str) -> None:
    case = FIXTURE["cases"][name]
    df = pl.read_csv(HERE / "data" / f"{case['data']}.csv")
    fit = cox_pspline(df, "time", "status", "x", df=case["df_target"])
    np.testing.assert_allclose(fit.coefficients, case["coefficients"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(fit.variance, case["variance"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(fit.variance_sandwich, case["variance_sandwich"], rtol=1e-5, atol=1e-7)
    assert fit.df == pytest.approx(case["df"], rel=1e-6)
    assert fit.log_likelihood == pytest.approx(case["log_likelihood"], rel=1e-8)
    assert fit.log_likelihood_null == pytest.approx(case["log_likelihood_null"], rel=1e-10)
    assert fit.aic() == pytest.approx(case["aic"], rel=1e-8)
    tests = fit.tests()
    assert tests["coef"][0] == pytest.approx(case["linear_coef"], rel=1e-5)
    assert tests["se_coef"][0] == pytest.approx(case["linear_se"], rel=1e-5)
    np.testing.assert_allclose(tests["chisq"].to_numpy(), case["chisq"], rtol=1e-5)
    np.testing.assert_allclose(tests["df"].to_numpy(), case["test_df"], rtol=1e-6)
    np.testing.assert_allclose(tests["p_value"].to_numpy(), case["p_value"], rtol=1e-4)
    term = fit.term()
    np.testing.assert_allclose(term["x"].to_numpy(), case["term_x"], rtol=1e-12)
    np.testing.assert_allclose(term["y"].to_numpy(), case["term_y"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(term["se"].to_numpy(), case["term_se"], rtol=1e-5, atol=1e-7)
