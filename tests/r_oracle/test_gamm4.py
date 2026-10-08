"""Additive mixed models against gamm4.

Fixture: ``fixtures/gamm4.json`` from ``scripts/gamm4.R``. Point estimates
(coefficients, smooth fits, examiner variance, MOR, log-likelihood, BLUPs)
are compared with gamm4's output. gamm4 0.2-6 with Matrix >= 1.6 computes
Vp from a permuted Cholesky factor (see the R script), so the SEs and edf
are compared with the same formula on the unpivoted factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import gamm, median_odds_ratio, smooth

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "gamm4.json").read_text())
DATA = pl.read_csv(HERE / "data" / "gamm_binary_a.csv")
GRID = pl.DataFrame(FIXTURE["grid"])


def _fit(case: dict):
    smooths = [smooth(s["column"], k=s["k"], basis=s["basis"]) for s in case["smooths"]]
    outcome = "y" if case["family"] == "binomial" else "n"
    return gamm(
        DATA, outcome, smooths, random="(1 | examiner)", predictors=case["predictors"] or None, family=case["family"]
    )


@pytest.mark.parametrize("name", sorted(FIXTURE["cases"]))
def test_gamm_matches_gamm4(name: str) -> None:
    case = FIXTURE["cases"][name]
    fit = _fit(case)
    assert fit.converged
    # gamm4 stops at glmer's tolerance; our log-likelihood is never lower.
    assert fit.log_likelihood >= case["log_likelihood"] - 1e-9
    np.testing.assert_allclose(fit.log_likelihood, case["log_likelihood"], atol=1e-5)
    table = fit.tidy()
    assert table["term"].to_list() == list(np.atleast_1d(case["terms"]))
    np.testing.assert_allclose(table["estimate"].to_numpy(), np.atleast_1d(case["coefficients"]), atol=3e-5)
    variance = fit.variance_table()["variance"][0]
    np.testing.assert_allclose(variance, case["examiner_variance"], rtol=1e-4)
    np.testing.assert_allclose(median_odds_ratio(variance)["median_odds_ratio"][0], case["median_odds_ratio"], rtol=1e-4)
    np.testing.assert_allclose(fit.predict(GRID), case["link"], atol=3e-5)
    for want in case["smooth_fits"]:
        effect = fit.partial_effect(want["column"], GRID[want["column"]].to_numpy())
        np.testing.assert_allclose(effect["fit"].to_numpy(), want["fit"], atol=3e-5)
    blups = fit.random_effects()
    assert blups["level"].to_list() == case["blup_levels"]
    np.testing.assert_allclose(blups["blup"].to_numpy(), case["blup"], atol=1e-4)

    corrected = case["corrected"]
    np.testing.assert_allclose(table["std_error"].to_numpy(), np.atleast_1d(corrected["std_errors"]), rtol=1e-4)
    np.testing.assert_allclose(fit.edf, np.atleast_1d(corrected["edf"]), rtol=1e-4)
    for want, se in zip(case["smooth_fits"], corrected["smooth_std_errors"], strict=True):
        effect = fit.partial_effect(want["column"], GRID[want["column"]].to_numpy())
        np.testing.assert_allclose(effect["std_error"].to_numpy(), se, rtol=1e-4)


def test_gamm_guards() -> None:
    with pytest.raises(ValueError, match="family"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="tp")], random="(1 | examiner)", family="gaussian")
    with pytest.raises(ValueError, match="tp or cr"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="ps")], random="(1 | examiner)")
    with pytest.raises(ValueError, match="at least one term"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="tp")], random="1")
