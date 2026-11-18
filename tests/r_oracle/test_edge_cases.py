"""Small edge-case data shapes against R.

Fixture: ``fixtures/edge_cases.json`` from ``scripts/edge_cases.R``; data:
``data/edge_*.csv`` from ``scripts/make_data.py``. The larger oracle samples
do not have these shapes:

* Kaplan-Meier with a group of one, a group whose events share one time, an
  event and a censoring tied at the last time, and an all-censored group.
* Cox with every event at one time; a stratum with no events and a stratum
  of one subject; delayed entry equal to other subjects' event times.
* Log-rank with a group of one and a stratum holding one group.
* Cumulative incidence and Fine-Gray with a cause that never occurs in one arm.
* ``lmer`` with singleton groups and a group variance at zero, and ``glmer``.
* ``polr`` with a one-row middle category. (Firth with complete separation,
  n = 12, is ``firth_edge_tiny`` in ``test_firth.py``.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import (
    cox_ph,
    cumulative_incidence,
    fine_gray_regression,
    fit_mixed,
    log_rank,
    ordinal_regression,
    survival_curve,
)

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "edge_cases.json").read_text())


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


def _values(raw) -> np.ndarray:
    raw = raw if isinstance(raw, list) else [raw]
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def _close(got, want, rtol: float = 1e-6, atol: float = 1e-8, msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(got, dtype=float), _values(want), rtol=rtol, atol=atol, err_msg=msg)


def _km_check(table: pl.DataFrame, want: dict) -> None:
    groups = want["group"]
    for group in dict.fromkeys(groups):
        idx = [i for i, g in enumerate(groups) if g == group]
        rows = table.filter(pl.col("group").cast(pl.String) == str(group)).sort("time")
        assert rows.height == len(idx), f"{group}: {rows.height} rows, R has {len(idx)}"
        for col, key in [
            ("time", "time"),
            ("n_risk", "n_risk"),
            ("n_event", "n_event"),
            ("estimate", "estimate"),
            ("std_error", "std_error"),
            ("conf_low", "conf_low"),
            ("conf_high", "conf_high"),
        ]:
            _close(rows[col].to_numpy(), [want[key][i] for i in idx], msg=f"{group} {col}")


# Kaplan-Meier


def test_km_tiny_and_tied_groups() -> None:
    want = FIXTURE["km"]
    curve = survival_curve(_data("edge_km"), "time", "status", by="arm")
    _km_check(curve.table, want)
    medians = curve.quantile(0.5)
    for i, group in enumerate(want["median_group"]):
        row = medians.filter(pl.col("group").cast(pl.String) == group)
        _close(row["quantile"], want["median"][i], msg=f"{group} median")
        _close(row["conf_low"], want["median_low"][i], msg=f"{group} median low")
        _close(row["conf_high"], want["median_high"][i], msg=f"{group} median high")


def test_km_stratum_without_events() -> None:
    _km_check(survival_curve(_data("edge_cox_strata"), "time", "status", by="stratum").table, FIXTURE["km_strata"])


def test_km_delayed_entry() -> None:
    curve = survival_curve(_data("edge_entry"), "time", "status", by="arm", entry="entry")
    _km_check(curve.table, FIXTURE["km_entry"])


# Cox


def _cox_check(data: pl.DataFrame, case: dict) -> None:
    fit = cox_ph(data, case["formula"], ties=case["ties"])
    terms = case["terms"]
    assert fit.names == (terms if isinstance(terms, list) else [terms])
    _close(fit.coefficients, case["coefficients"], rtol=1e-6, msg="coef")
    _close(np.sqrt(np.diag(fit.covariance)), case["std_error"], rtol=1e-5, msg="se")
    _close(fit.log_likelihood, case["loglik"][1], rtol=1e-8, msg="loglik")


@pytest.mark.parametrize("index", [0, 1], ids=["efron", "breslow"])
def test_cox_all_events_at_one_time(index: int) -> None:
    _cox_check(_data("edge_cox_ties"), FIXTURE["cox_ties"][index])


@pytest.mark.parametrize("index", [0, 1], ids=["efron", "breslow"])
def test_cox_empty_and_single_strata(index: int) -> None:
    _cox_check(_data("edge_cox_strata"), FIXTURE["cox_strata"][index])


@pytest.mark.parametrize("index", [0, 1], ids=["efron", "breslow"])
def test_cox_delayed_entry(index: int) -> None:
    _cox_check(_data("edge_entry"), FIXTURE["cox_entry"][index])


# Log-rank


@pytest.mark.parametrize("index", [0, 1], ids=["plain", "strata"])
def test_log_rank_small_groups(index: int) -> None:
    case = FIXTURE["logrank"][index]
    strata = "site" if "strata" in case["formula"] else None
    got = log_rank(_data("edge_logrank"), "time", "status", "arm", strata)
    assert [str(g) for g in got.groups] == case["group"]
    _close(got.statistic, case["statistic"], msg="chisq")
    assert got.df == case["df"]
    _close(got.observed, case["observed"], msg="observed")
    _close(got.expected, case["expected"], msg="expected")


# Competing risks


def test_cumulative_incidence_missing_cause() -> None:
    want = FIXTURE["cuminc"]
    fit = cumulative_incidence(_data("edge_cr"), "time", "cause", by="arm")
    for curve in want["curves"]:
        rows = fit.table.filter((pl.col("group") == curve["group"]) & (pl.col("cause") == curve["cause"]))
        label = f"{curve['group']} {curve['cause']}"
        _close(rows["time"], curve["time"], msg=f"{label} time")
        _close(rows["estimate"], curve["estimate"], msg=f"{label} estimate")
        _close(rows["variance"], curve["variance"], msg=f"{label} variance")
    tests = fit.tests.sort("cause")
    assert tests["cause"].to_list() == want["test_cause"]
    _close(tests["statistic"], want["test_statistic"], msg="Gray statistic")
    _close(tests["p_value"], want["test_p"], msg="Gray p")


def test_fine_gray_missing_cause() -> None:
    want = FIXTURE["crr"]
    data = _data("edge_cr").with_columns(armb=(pl.col("arm") == "b").cast(pl.Int64))
    fit = fine_gray_regression(data, "time", "cause", ["x", "armb"])
    _close(fit.coefficients, want["coefficients"], rtol=1e-5, msg="coef")
    _close(np.sqrt(np.diag(fit.covariance)), want["std_error"], rtol=1e-5, msg="se")


# Mixed models


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=["reml", "ml", "boundary-reml", "boundary-ml"])
def test_lmer_singletons_and_zero_variance(index: int) -> None:
    case = FIXTURE["lmer"][index]
    fit = fit_mixed(_data("edge_mixed"), case["formula"], method="reml" if case["reml"] else "ml")
    assert fit.names == case["terms"]
    _close(fit.coefficients, case["coefficients"], rtol=1e-5, msg="fixef")
    _close(np.sqrt(np.diag(fit.covariance)), case["std_error"], rtol=1e-4, msg="se")
    _close(fit.group_covariance[0, 0], case["group_variance"], rtol=1e-4, atol=1e-6, msg="group variance")
    _close(fit.residual_variance, case["residual_variance"], rtol=1e-5, msg="residual variance")
    _close(fit.log_likelihood, case["loglik"], rtol=1e-7, msg="logLik")


def test_glmer_singletons() -> None:
    # glmer is run with tolPwrss = 1e-13 (see test_glmm_binary_glmer.py).
    case = FIXTURE["glmer"]
    fit = fit_mixed(_data("edge_mixed"), case["formula"], family="binomial")
    assert fit.converged
    _close(fit.log_likelihood, case["loglik"], rtol=0, atol=1e-7, msg="logLik")
    # The likelihood is flat here (100 rows), so the optima agree to about 1e-5.
    _close(fit.coefficients, case["coefficients"], rtol=5e-5, atol=1e-7, msg="fixef")
    _close(np.sqrt(np.diag(fit.covariance)), case["std_error"], rtol=2e-5, msg="se")
    _close(fit.group_covariance[0, 0], case["group_variance"], rtol=1e-4, msg="group variance")


# Ordinal


def test_polr_one_row_category() -> None:
    want = FIXTURE["polr"]
    fit = ordinal_regression(_data("edge_ordinal"), want["formula"], levels=["low", "mid", "high"])
    table = fit.tidy()
    _close(table.filter(pl.col("coef_type") == "coefficient")["estimate"], want["coefficients"], rtol=1e-5)
    _close(table.filter(pl.col("coef_type") == "threshold")["estimate"], want["zeta"], rtol=1e-5)
    _close(table["std_error"], want["std_error"], rtol=1e-4, msg="se")
    _close(fit.log_likelihood, want["loglik"], rtol=1e-8, msg="logLik")
