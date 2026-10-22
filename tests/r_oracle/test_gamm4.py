"""Additive mixed models against gamm4.

Fixture: ``fixtures/gamm4.json`` from ``scripts/gamm4.R``. Point estimates
(coefficients, smooth fits, examiner variance, MOR, log-likelihood, BLUPs)
are compared with gamm4's output. With Matrix >= 1.6 the SEs and edf that
gamm4 0.2-6 reports differ by about 1% from the mixed-model formula, so they
are compared with the formula's values (see the R script).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import compare_gamms, gamm, median_odds_ratio, smooth

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "gamm4.json").read_text())
COMPARE = json.loads((HERE / "fixtures" / "gamm4_compare.json").read_text())
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


def test_gamm_predict_standard_errors() -> None:
    fit = gamm(DATA, "y", [smooth("pre_size_mm", k=10, basis="tp")], random="(1 | examiner)")
    eta, se = fit.predict(GRID, se=True)
    np.testing.assert_allclose(eta, fit.predict(GRID))
    # Intercept plus one smooth: var = var(b0) + var(f) + 2 cov(b0, f).
    basis = fit._smooths[0].rebuild(GRID["pre_size_mm"].to_numpy())
    sl = fit._slices[0]
    cov = fit.covariance
    smooth_var = np.sum((basis @ cov[sl, sl]) * basis, axis=1)
    cross = basis @ cov[sl, 0]
    np.testing.assert_allclose(se**2, cov[0, 0] + smooth_var + 2 * cross, rtol=1e-10)
    effect = fit.partial_effect("pre_size_mm", GRID["pre_size_mm"].to_numpy())
    np.testing.assert_allclose(smooth_var, effect["std_error"].to_numpy() ** 2, rtol=1e-10)
    mu, se_mu = fit.predict(GRID, kind="response", se=True)
    np.testing.assert_allclose(mu, 1 / (1 + np.exp(-eta)))
    np.testing.assert_allclose(se_mu, se * mu * (1 - mu))


def test_gamm_without_random_effects_is_on_the_same_likelihood() -> None:
    null = gamm(DATA, "y", [smooth("pre_size_mm", k=10, basis="tp")])
    full = gamm(DATA, "y", [smooth("pre_size_mm", k=10, basis="tp")], random="(1 | examiner)")
    assert null.converged
    assert null.variance_table().height == 0
    assert null.random_effects().height == 0
    for fit, want in zip((null, full), COMPARE["fits"], strict=True):
        row = fit.glance().row(0, named=True)
        assert row["n_parameters"] == want["n_parameters"]
        for key in ("log_likelihood", "aic", "bic"):
            np.testing.assert_allclose(row[key], want[key], atol=1e-4, err_msg=key)
    table = compare_gamms(null, full)
    row = table.row(1, named=True)
    assert row["df"] == COMPARE["df"]
    np.testing.assert_allclose(row["statistic"], COMPARE["statistic"], atol=2e-4)
    np.testing.assert_allclose(row["p_value"], COMPARE["p_value"], rtol=1e-3)
    np.testing.assert_allclose(row["p_value_boundary"], row["p_value"] / 2)


def test_gamm_variance_wald_interval() -> None:
    fit = gamm(DATA, "y", [smooth("pre_size_mm", k=10, basis="tp")], random="(1 | examiner)")
    row = fit.variance_table().row(0, named=True)
    assert row["conf_low"] < row["variance"] < row["conf_high"]
    # Symmetric on the log-SD scale.
    np.testing.assert_allclose(
        np.log(row["conf_high"]) - np.log(row["variance"]), np.log(row["variance"]) - np.log(row["conf_low"])
    )
    # Close to lme4's profile interval for the SD (Wald 0.502-1.068 here).
    low, high = COMPARE["profile_sd"]
    np.testing.assert_allclose(np.sqrt([row["conf_low"], row["conf_high"]]), [low, high], rtol=0.02)
    narrower = fit.variance_table(level=0.5).row(0, named=True)
    assert row["conf_low"] < narrower["conf_low"] < narrower["conf_high"] < row["conf_high"]


def test_gamm_guards() -> None:
    with pytest.raises(ValueError, match="family"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="tp")], random="(1 | examiner)", family="gaussian")
    with pytest.raises(ValueError, match="tp or cr"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="ps")], random="(1 | examiner)")
    with pytest.raises(ValueError, match="at least one term"):
        gamm(DATA, "y", [smooth("pre_size_mm", basis="tp")], random="1")
