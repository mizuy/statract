"""Linear-model diagnostic tests and the gamma GLM likelihood against R.

Fixture: ``fixtures/linear_tests.json`` from ``scripts/linear_tests.R``.

RESET, Breusch–Pagan, Durbin–Watson (normal approximation, ``exact =
FALSE``), and Breusch–Godfrey are compared with lmtest on an unweighted fit.
lmtest drops the weights and offset of an ``lm`` fit, so the weighted and
offset forms of these tests (see ``tests/models/test_matching_balance_fixes.py``)
have no R reference.

The gamma GLM checks ``logLik``, ``AIC``, ``BIC``, the deviance, and the
standard errors of ``glm(family = Gamma())``, with and without weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.models.fit import fit_glm, fit_ols
from statract.models.linear_tests import (
    breusch_godfrey_test,
    breusch_pagan_test,
    durbin_watson_test,
    ramsey_reset_test,
)

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "linear_tests.json").read_text())
DATA = pl.read_csv(HERE / "data" / "gam_a.csv")

TESTS = {
    "reset": lambda fit: ramsey_reset_test(fit),
    "reset_2": lambda fit: ramsey_reset_test(fit, power=(2,)),
    "bp": lambda fit: breusch_pagan_test(fit),
    "bp_raw": lambda fit: breusch_pagan_test(fit, studentize=False),
    "dw": lambda fit: durbin_watson_test(fit),
    "dw_two": lambda fit: durbin_watson_test(fit, alternative="two.sided"),
    "bg_chi2": lambda fit: breusch_godfrey_test(fit, order=2),
    "bg_f": lambda fit: breusch_godfrey_test(fit, order=2, distribution="f"),
}


@pytest.mark.parametrize("name", list(TESTS))
def test_matches_lmtest(name: str) -> None:
    want = FIXTURE["linear"][name]
    result = TESTS[name](fit_ols(DATA, FIXTURE["linear"]["formula"]))
    np.testing.assert_allclose(result.statistic, want["statistic"], rtol=1e-10)
    np.testing.assert_allclose(result.p_value, want["p_value"], rtol=1e-8)
    if not name.startswith("dw"):
        np.testing.assert_allclose(result.df, np.atleast_1d(want["df"]))


@pytest.mark.parametrize("name", ["plain", "weighted"])
def test_gamma_glm_matches_glm(name: str) -> None:
    want = FIXTURE["gamma"][name]
    fit = fit_glm(DATA, "cost ~ x + z", family="gamma", weights=want["weights"])
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-8)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), want["std_errors"], rtol=1e-6)
    for key in ("deviance", "log_likelihood", "aic", "bic"):
        np.testing.assert_allclose(getattr(fit, key), want[key], rtol=1e-10, err_msg=key)
    np.testing.assert_allclose(fit.scale, want["dispersion"], rtol=1e-6)
