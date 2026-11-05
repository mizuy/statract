"""NaN follow-up values are dropped like nulls; Cox predictions on new rows stay aligned."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract.surv import (
    accelerated_failure,
    cox_mixed,
    cox_ph,
    cumulative_incidence,
    fine_gray,
    fine_gray_regression,
    log_rank,
    restricted_mean_survival,
    standardize_cox,
    survival_curve,
)


def _data(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    return pl.DataFrame(
        {
            "t": np.round(rng.exponential(5, n), 1) + 0.1,
            "s": rng.binomial(1, 0.7, n).astype(float),
            "e": rng.integers(0, 3, n).astype(float),
            "x": rng.normal(size=n),
            "g": rng.integers(0, 2, n),
            "h": rng.integers(0, 5, n),
            "ent": np.zeros(n),
            "off": rng.normal(size=n),
        }
    )


def _with_row(df: pl.DataFrame, **values) -> pl.DataFrame:
    row = {"t": 5.0, "s": 1.0, "e": 1.0, "x": 0.3, "g": 1, "h": 0, "ent": 0.0, "off": 0.0, **values}
    return pl.concat([df, pl.DataFrame(row, schema=df.schema)])


def _same(a, b) -> None:
    if isinstance(a, pl.DataFrame):
        assert a.equals(b)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


ESTIMATORS = {
    "km": lambda d: survival_curve(d, "t", "s").table,
    "km_entry": lambda d: survival_curve(d, "t", "s", entry="ent").table,
    "log_rank": lambda d: log_rank(d, "t", "s", "g").statistic,
    "rmst": lambda d: restricted_mean_survival(d, "t", "s", "g").arms,
    "cox_columns": lambda d: cox_ph(d, "t", "s", ["x"], entry="ent").coefficients,
    "cox_formula": lambda d: cox_ph(d, "Surv(ent, t, s) ~ x").coefficients,
    "aft_columns": lambda d: accelerated_failure(d, "t", "s", ["x"]).coefficients,
    "aft_formula": lambda d: accelerated_failure(d, "Surv(t, s) ~ x").coefficients,
    "cif": lambda d: cumulative_incidence(d, "t", "e").table,
    "crr_columns": lambda d: fine_gray_regression(d, "t", "e", ["x"], cause=1).coefficients,
    "crr_formula": lambda d: fine_gray_regression(d, "Surv(t, e) ~ x", cause=1).coefficients,
    "cox_mixed": lambda d: cox_mixed(d, "Surv(t, s) ~ x + (1 | h)").coefficients,
    "standardize": lambda d: standardize_cox(d, "Surv(t, s) ~ g + x", values={"g": [0, 1]}, times=[3.0]).tidy(),
}


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
@pytest.mark.parametrize("column", ["t", "s", "ent"])
def test_nan_row_is_dropped(name: str, column: str) -> None:
    df = _data()
    if column == "s" and name in {"cif", "crr_columns", "crr_formula"}:
        column = "e"
    if column == "ent" and name not in {"km_entry", "cox_columns", "cox_formula"}:
        pytest.skip("no entry time")
    estimate = ESTIMATORS[name]
    _same(estimate(df), estimate(_with_row(df, **{column: float("nan")})))


@pytest.mark.parametrize("entry", [None, "ent"])
def test_fine_gray_drops_nan_time(entry) -> None:
    df = _data()
    nan_time = _with_row(df, t=float("nan"))
    _same(fine_gray(df, "t", "e", 1, entry=entry), fine_gray(nan_time, "t", "e", 1, entry=entry))
    _same(fine_gray(df, "Surv(t, e) ~ x", 1), fine_gray(nan_time, "Surv(t, e) ~ x", 1))
    if entry is not None:
        _same(fine_gray(df, "t", "e", 1, entry=entry), fine_gray(_with_row(df, ent=float("nan")), "t", "e", 1, entry=entry))


def test_cox_predict_new_rows_follow_dropped_covariate() -> None:
    df = _data(30)
    fit = cox_ph(df, "t", "s", ["x"])
    new = df.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("x")).alias("x"))
    for kind in ("linear_predictor", "expected", "survival"):
        full = fit.predict(df, kind=kind)
        dropped = fit.predict(new, kind=kind)
        assert dropped.shape == (29,)
        np.testing.assert_allclose(dropped, full[1:])
    curve = fit.survival_curve(new)
    np.testing.assert_array_equal(curve["time"].to_numpy(), df["t"].to_numpy()[1:])
    np.testing.assert_allclose(curve["survival"].to_numpy(), fit.predict(df, kind="survival")[1:])


def test_cox_predict_strata_new_rows_aligned() -> None:
    df = _data(30).with_columns(st=pl.Series([0, 1] * 15))
    fit = cox_ph(df, "t", "s", ["x"], strata="st")
    new = df.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col("x")).alias("x"))
    full = fit.predict(df, kind="survival")
    np.testing.assert_allclose(fit.predict(new, kind="survival"), np.delete(full, 2))


def test_cox_predict_times_broadcast() -> None:
    df = _data()
    fit = cox_ph(df, "t", "s", ["x"])
    one = fit.predict(kind="survival", times=[3.0])
    assert one.shape == (60,)
    np.testing.assert_allclose(one, fit.predict(kind="survival", times=np.full(60, 3.0)))
    np.testing.assert_allclose(fit.predict(df, kind="expected", times=3.0), -np.log(one))
    with pytest.raises(ValueError, match="times"):
        fit.predict(kind="survival", times=[1.0, 2.0])


@pytest.mark.parametrize("how", ["formula", "keyword"])
def test_cox_predict_new_data_adds_offset(how: str) -> None:
    df = _data()
    if how == "formula":
        fit = cox_ph(df, "Surv(t, s) ~ x + offset(off)")
    else:
        fit = cox_ph(df, "t", "s", ["x"], offset="off")
    np.testing.assert_allclose(fit.predict(df), fit.predict())
    np.testing.assert_allclose(fit.predict(df, kind="expected", times=4.0), fit.predict(kind="expected", times=4.0))
    # Without the offset column the offset is left out.
    np.testing.assert_allclose(fit.predict(df.drop("off")), fit.predict() - df["off"].to_numpy())
    # A row with a missing offset is dropped, as in the fit.
    holed = df.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("off")).alias("off"))
    np.testing.assert_allclose(fit.predict(holed), fit.predict()[1:])


def test_cox_mixed_predict_new_data_adds_offset() -> None:
    df = _data()
    fit = cox_mixed(df, "Surv(t, s) ~ x + offset(off) + (1 | h)")
    np.testing.assert_allclose(fit.predict(df), fit.predict())
