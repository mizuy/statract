"""cumulative_incidence and fine_gray_regression against cmprsk::cuminc and cmprsk::crr."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.surv import cumulative_incidence, fine_gray_regression

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "cmprsk.json").read_text())
SAMPLES = {sample["name"]: sample for sample in FIXTURE["samples"]}


def _frame(sample: dict) -> pl.DataFrame:
    return pl.DataFrame(sample["data"])


def _cases(kind: str):
    for name, sample in SAMPLES.items():
        for index, case in enumerate(sample[kind]):
            yield pytest.param(name, index, id=f"{name}-{index}")


@pytest.mark.parametrize(("name", "index"), list(_cases("cuminc")))
def test_cuminc_curves_tests_and_timepoints(name: str, index: int) -> None:
    sample = SAMPLES[name]
    case = sample["cuminc"][index]
    fit = cumulative_incidence(
        _frame(sample), "time", "status", case["by"], strata=case["strata"], rho=case["rho"]
    )
    expected = case["fit"]
    table = fit.frame()
    for curve in expected["curves"]:
        group, cause = curve["name"].split(" ")
        part = table.filter((pl.col("group") == group) & (pl.col("cause") == cause))
        np.testing.assert_allclose(part["time"].to_numpy(), curve["time"], rtol=0, atol=0)
        np.testing.assert_allclose(part["estimate"].to_numpy(), curve["est"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(part["variance"].to_numpy(), curve["var"], rtol=1e-10, atol=1e-15)
    if expected["tests"] is None:
        assert fit.tests is None
    else:
        tests = fit.tests
        assert tests["cause"].to_list() == expected["tests"]["cause"]
        np.testing.assert_allclose(tests["statistic"].to_numpy(), expected["tests"]["stat"], rtol=1e-8)
        np.testing.assert_allclose(tests["p_value"].to_numpy(), expected["tests"]["pv"], rtol=0, atol=1e-8)
        assert tests["df"].to_list() == expected["tests"]["df"]
    at = fit.at(expected["grid"])
    grid = np.unique(expected["grid"])
    for row, label in enumerate(expected["tp_names"]):
        group, cause = label.split(" ")
        part = at.filter((pl.col("group") == group) & (pl.col("cause") == cause)).sort("time")
        assert part["time"].to_list() == grid.tolist()
        est = np.array([np.nan if v is None else v for v in expected["tp_est"][row]], dtype=float)
        var = np.array([np.nan if v is None else v for v in expected["tp_var"][row]], dtype=float)
        np.testing.assert_allclose(part["estimate"].fill_null(np.nan).to_numpy(), est, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(part["variance"].fill_null(np.nan).to_numpy(), var, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize(("name", "index"), list(_cases("crr")))
def test_crr_matches(name: str, index: int) -> None:
    sample = SAMPLES[name]
    case = sample["crr"][index]
    fit = fine_gray_regression(
        _frame(sample),
        f"Surv(time, status) ~ {case['rhs']}",
        cause=case["failcode"],
        censor_group=case["cengroup"],
    )
    assert fit.names == case["names"]
    assert fit.converged == case["converged"]
    np.testing.assert_allclose(fit.coefficients, case["coef"], rtol=1e-8)
    np.testing.assert_allclose(fit.covariance, np.array(case["var"]).reshape(len(fit.names), -1), rtol=1e-8)
    np.testing.assert_allclose(fit.information, np.array(case["inf"]).reshape(len(fit.names), -1), rtol=1e-8)
    np.testing.assert_allclose(fit.log_likelihood, case["loglik"], rtol=1e-10)
    np.testing.assert_allclose(fit.null_log_likelihood, case["loglik_null"], rtol=1e-10)
    np.testing.assert_allclose(fit.event_times, case["uftime"], rtol=0, atol=0)
    np.testing.assert_allclose(fit.baseline_jumps, case["bfitj"], rtol=1e-8)
    np.testing.assert_allclose(fit.score_residuals, np.array(case["res"]).reshape(len(case["uftime"]), -1), rtol=1e-7, atol=1e-10)
    pred = fit.predict(_frame(sample)[:3])
    expected = np.array(case["pred"]).reshape(len(case["pred_time"]), -1)
    for row in range(3):
        part = pred.filter(pl.col("row") == row)
        np.testing.assert_allclose(part["time"].to_numpy(), case["pred_time"])
        np.testing.assert_allclose(part["estimate"].to_numpy(), expected[:, row], rtol=1e-8)


def test_crr_columns_match_formula() -> None:
    sample = SAMPLES["balanced"]
    data = _frame(sample)
    by_formula = fine_gray_regression(data, "Surv(time, status) ~ x1 + x2", cause=1)
    by_columns = fine_gray_regression(data, "time", "status", ["x1", "x2"], cause=1)
    np.testing.assert_allclose(by_columns.coefficients, by_formula.coefficients)
    tidy = by_columns.tidy(exponentiate=True)
    assert tidy.columns[:7] == ["term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
    glance = by_columns.glance()
    assert glance["df"].item() == 2
