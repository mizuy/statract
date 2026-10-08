"""Cox regression standardization against ``stdReg2::standardize_coxph``.

Fixture: ``fixtures/stdreg_cox.json`` from ``scripts/stdreg_cox.R``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.surv import standardize_cox

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "stdreg_cox.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _fit(name: str):
    case = FIXTURE["cases"][name]
    return standardize_cox(
        _data(case["data"]),
        case["formula"],
        values={case["exposure"]: list(np.atleast_1d(case["values"]))},
        times=list(np.atleast_1d(case["times"])),
        measure=case["measure"],
        cluster=case.get("cluster"),
    )


@pytest.mark.parametrize("name", CASES)
def test_estimates_and_covariance(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _fit(name)
    np.testing.assert_allclose(got.estimates, np.asarray(want["estimates"], dtype=float), rtol=1e-6)
    for time, cov_got, cov_want in zip(got.times, got.covariances, want["covariances"], strict=True):
        np.testing.assert_allclose(cov_got, np.asarray(cov_want, dtype=float), rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(got.covariance(time), cov_got)


@pytest.mark.parametrize(
    ("name", "spec"),
    [(name, k) for name in CASES for k in range(len(FIXTURE["cases"][name]["tables"]))],
)
def test_tidy(name: str, spec: int) -> None:
    case = FIXTURE["cases"][name]
    want = case["tables"][spec]
    table = _fit(name).tidy(
        contrast=want["contrast"],
        reference=case["reference"] if want["contrast"] else None,
        transform=want["transform"],
        ci=want["ci"],
    )
    np.testing.assert_allclose(table["time"].to_numpy(), np.atleast_1d(want["time"]))
    np.testing.assert_allclose(table[case["exposure"]].to_numpy(), np.atleast_1d(want["value"]))
    for column in ("estimate", "std_error", "conf_low", "conf_high"):
        np.testing.assert_allclose(table[column].to_numpy(), np.atleast_1d(want[column]), rtol=1e-6, atol=1e-12)


def test_survival_point_estimate_is_mean_of_breslow_cox() -> None:
    data = _data("stdreg_a")
    got = standardize_cox(data, "Surv(time, status) ~ ope * age + sex", values={"ope": [1]}, times=[2.0])
    fit = got.fits[0]
    assert fit.ties == "breslow"
    counterfactual = data.with_columns(pl.lit(1).alias("ope"))
    surv = fit.predict(counterfactual, kind="survival", times=np.full(data.height, 2.0))
    np.testing.assert_allclose(got.estimates[0, 0], surv.mean(), rtol=1e-12)


def test_time_zero_is_one_with_no_variance() -> None:
    # Time 0 passes the event check only when an event happens at 0.
    data = _data("stdreg_a").with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(pl.col("time")).alias("time"),
        pl.when(pl.int_range(pl.len()) == 0).then(1).otherwise(pl.col("status")).alias("status"),
    )
    got = standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [0, 1]}, times=[0.5, 0.0])
    np.testing.assert_allclose(got.estimates[1], [1.0, 1.0])
    np.testing.assert_allclose(got.covariances[1], np.zeros((2, 2)))


def test_needs_an_event_before_the_first_time() -> None:
    data = _data("stdreg_a")
    with pytest.raises(ValueError, match="No events before first value in times"):
        standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [0, 1]}, times=[0.0, 1.0])


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("Surv(time, status) ~ ope + age + strata(sex)", "strata"),
        ("Surv(time, status) ~ ope + age + cluster(site)", "cluster"),
    ],
)
def test_rejects_special_terms(formula: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        standardize_cox(_data("stdreg_a"), formula, values={"ope": [0, 1]}, times=[1.0])


def test_rejects_log_interval_on_a_difference() -> None:
    fit = _fit("survival_binary")
    with pytest.raises(ValueError, match="ci='log'"):
        fit.tidy(contrast="difference", reference=0, ci="log")


def test_rmean_rejects_what_r_cannot_estimate() -> None:
    data = _data("stdreg_c")
    with pytest.raises(ValueError, match="covariate besides the exposure"):
        standardize_cox(data, "Surv(time, status) ~ ope", values={"ope": [0, 1]}, times=3.0, measure="rmean")
    with pytest.raises(ValueError, match="single time"):
        standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [0, 1]}, times=[2.0, 3.0], measure="rmean")
    with pytest.raises(ValueError, match="cluster"):
        standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [0, 1]}, times=3.0, measure="rmean", cluster="site")
    with pytest.raises(ValueError, match="not available"):
        _fit("rmean_main").tidy(transform="logit")


def test_rmean_drops_only_terms_that_use_the_exposure() -> None:
    # stdReg2 drops every term whose label contains the exposure name, so
    # ``operation`` would go with ``ope``. Here it stays in both group models.
    data = _data("stdreg_c").with_columns((pl.col("age") / 10).alias("operation"))
    with_extra = standardize_cox(
        data, "Surv(time, status) ~ ope + operation", values={"ope": [0, 1]}, times=3.0, measure="rmean"
    )
    same = standardize_cox(
        data.rename({"operation": "decades"}),
        "Surv(time, status) ~ ope + decades",
        values={"ope": [0, 1]},
        times=3.0,
        measure="rmean",
    )
    np.testing.assert_allclose(with_extra.estimates, same.estimates)
    np.testing.assert_allclose(with_extra.covariances[0], same.covariances[0])


def test_rmean_value_order_follows_values() -> None:
    forward = _fit("rmean_main")
    data = _data("stdreg_c")
    backward = standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [1, 0]}, times=3.0, measure="rmean")
    np.testing.assert_allclose(backward.estimates[0], forward.estimates[0][::-1])
    np.testing.assert_allclose(backward.covariances[0], forward.covariances[0][::-1, ::-1])
