"""Regression tests for survival fixes from the 2026-11-05 audit (medium and low items)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract.surv import conditional_logit, cox_ph, fine_gray, split_follow_up, survival_curve
from statract.surv import cox as cox_module


def _aml() -> pl.DataFrame:
    t = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161, 5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    s = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    g = ["Maintained"] * 11 + ["Nonmaintained"] * 12
    return pl.DataFrame({"t": t, "s": s, "g": g})


def test_km_quantile_upper_limit_is_first_time_upper_band_reaches_p() -> None:
    # print(survfit(Surv(time, status) ~ x, data = aml)): medians 31 and 23,
    # 0.95LCL 18 and 8, 0.95UCL NA and NA.
    q = survival_curve(_aml(), "t", "s", by="g").quantile(0.5)
    assert q["quantile"].to_list() == [31.0, 23.0]
    assert q["conf_low"].to_list() == [18.0, 8.0]
    assert np.isnan(q["conf_high"].to_numpy()).all()


def test_km_quantile_flat_at_p_takes_midpoint() -> None:
    df = pl.DataFrame({"t": np.arange(1, 41, dtype=float), "s": [1] * 40})
    curve = survival_curve(df, "t", "s")
    q = curve.quantile(0.5)
    table = curve.table
    assert q["quantile"][0] == 20.5
    assert q["conf_low"][0] == table.filter(pl.col("conf_low") <= 0.5)["time"][0]
    assert q["conf_high"][0] == table.filter(pl.col("conf_high") <= 0.5)["time"][0] == 28.0


def _na_data() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = 60
    return pl.DataFrame({"t": np.round(rng.exponential(5, n), 1) + 0.1, "s": rng.binomial(1, 0.7, n)})


def test_nelson_aalen_confidence_types_differ_and_follow_formulas() -> None:
    df = _na_data()
    z = stats.norm.ppf(0.975)
    tables = {conf: survival_curve(df, "t", "s", kind="nelson_aalen", confidence=conf).table for conf in ("log", "log-log", "plain")}
    base = tables["log"]
    h = -np.log(base["estimate"].to_numpy())
    se = base["std_error"].to_numpy()
    ok = h > 0
    np.testing.assert_allclose(base["conf_low"].to_numpy()[ok], np.exp(-(h + z * se))[ok], rtol=1e-10)
    np.testing.assert_allclose(base["conf_high"].to_numpy()[ok], np.exp(-np.clip(h - z * se, 0, None))[ok], rtol=1e-10)
    ll = tables["log-log"]
    np.testing.assert_allclose(ll["conf_low"].to_numpy()[ok], np.exp(-h * np.exp(z * se / h))[ok], rtol=1e-10)
    np.testing.assert_allclose(ll["conf_high"].to_numpy()[ok], np.exp(-h * np.exp(-z * se / h))[ok], rtol=1e-10)
    plain = tables["plain"]
    surv = np.exp(-h)
    np.testing.assert_allclose(plain["conf_low"].to_numpy()[ok], np.clip(surv - z * surv * se, 0, 1)[ok], rtol=1e-10)
    assert not np.allclose(ll["conf_low"].to_numpy()[ok], base["conf_low"].to_numpy()[ok])


def _competing() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "e": [1, 0, 2, 1, 0, 2, 1, 0, 1, 2],
        }
    )


def test_aalen_johansen_n_censor_counts_censored_rows() -> None:
    table = survival_curve(_competing(), "t", "e", kind="aalen_johansen").table
    one = table.filter(pl.col("state") == "1")
    assert one["n_censor"].to_list() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert one["n_censor"].sum() == 3.0


def test_aalen_johansen_quantile_needs_cause_with_several_causes() -> None:
    curve = survival_curve(_competing(), "t", "e", kind="aalen_johansen")
    with pytest.raises(ValueError, match="cause"):
        curve.quantile(0.2)
    with pytest.raises(ValueError, match="cause"):
        curve.quantile(0.2, cause=3)
    one = curve.quantile(0.2, cause=1)
    two = curve.quantile(0.2, cause=2)
    f1 = curve.table.filter(pl.col("state") == "1")
    f2 = curve.table.filter(pl.col("state") == "2")
    assert one["quantile"][0] == f1.filter(pl.col("estimate") >= 0.2)["time"][0]
    assert two["quantile"][0] == f2.filter(pl.col("estimate") >= 0.2)["time"][0]
    assert one["quantile"][0] != two["quantile"][0]


def test_cox_no_events_raises() -> None:
    df = pl.DataFrame({"t": [1.0, 2, 3, 4, 5, 6], "s": [0] * 6, "x": [1.0, 2, 3, 1, 2, 3]})
    with pytest.raises(ValueError, match="no events"):
        cox_ph(df, "t", "s", ["x"])
    with pytest.raises(ValueError, match="no events"):
        cox_ph(df, "Surv(t, s) ~ x")


def test_cox_warns_when_newton_does_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _na_data().with_columns(x=pl.Series(np.random.default_rng(1).normal(size=60)))
    monkeypatch.setattr(cox_module, "_MAX_ITER", 1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = cox_ph(df, "t", "s", ["x"])
    assert not fit.converged


def _matched(seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sets = np.repeat(np.arange(30), 3)
    y = np.tile([1, 0, 0], 30)
    x = rng.normal(size=90) + 0.5 * y
    return pl.DataFrame(
        {
            "set": sets,
            "y": y,
            "x": x,
            "w": rng.integers(1, 4, 90).astype(float),
            "off": rng.normal(size=90),
            "time": np.ones(90),
        }
    )


def test_clogit_integer_weights_use_model_based_variance_like_cox() -> None:
    df = _matched()
    cl = conditional_logit(df, "y ~ x + strata(set)", weights="w", method="breslow")
    cx = cox_ph(df, "Surv(time, y) ~ x + strata(set)", weights="w", ties="breslow")
    np.testing.assert_allclose(cl.coefficients, cx.coefficients, rtol=1e-10)
    np.testing.assert_allclose(cl.covariance, cx.covariance, rtol=1e-8)
    np.testing.assert_allclose(cl.covariance, np.linalg.pinv(cl.information), rtol=1e-10)
    frac = df.with_columns(w=pl.col("w") + 0.5)
    cl2 = conditional_logit(frac, "y ~ x + strata(set)", weights="w", method="breslow")
    cx2 = cox_ph(frac, "Surv(time, y) ~ x + strata(set)", weights="w", ties="breslow")
    np.testing.assert_allclose(cl2.covariance, cx2.covariance, rtol=1e-8)
    assert not np.allclose(cl2.covariance, np.linalg.pinv(cl2.information))


@pytest.mark.parametrize("method", ["exact", "efron"])
def test_clogit_predict_newdata_includes_offset(method: str) -> None:
    df = _matched()
    by_formula = conditional_logit(df, "y ~ x + offset(off) + strata(set)", method=method)
    np.testing.assert_allclose(by_formula.predict(df), by_formula.predict(), rtol=1e-12)
    by_column = conditional_logit(df, "y", ["x"], strata="set", offset="off", method=method)
    np.testing.assert_allclose(by_column.predict(df), by_column.predict(), rtol=1e-12)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_fine_gray_column_path_drops_missing_event(missing) -> None:
    d = pl.DataFrame({"t": [1.0, 2, 3, 4, 5, 6], "e": [1.0, 2, missing, 0, 1, 2], "z": [0.0, 1, 0, 1, 0, 1]})
    column = fine_gray(d, "t", "e", cause=1)
    formula = fine_gray(d, "Surv(t, e) ~ z", cause=1)
    dropped = fine_gray(d.filter(pl.col("e").is_not_null() & pl.col("e").is_not_nan()), "t", "e", cause=1)
    assert column.height == formula.height == dropped.height == 6
    np.testing.assert_allclose(column["fgwt"].to_numpy(), formula["fgwt"].to_numpy())
    np.testing.assert_allclose(column["fgwt"].to_numpy(), dropped["fgwt"].to_numpy())


def test_split_follow_up_drops_nan_rows_like_null() -> None:
    d = pl.DataFrame({"t": [1.0, None, float("nan"), 5.0], "e": [1.0, 1.0, 0.0, float("nan")], "id": [1, 2, 3, 4]})
    out = split_follow_up(d, "t", "e", [2.0, 4.0])
    assert out["id"].to_list() == [1]
    assert not out["t"].is_nan().any()
