"""statract.validation against rms 6.7-1 (fixture from scripts/rms.R; R not needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import (
    calibrate_cox,
    calibrate_logistic,
    plot_calibration_curve,
    somers_dxy,
    validate_cox,
    validate_logistic,
)
from statract._lowess_r import lowess_r
from statract.validation import _cox_data, _cox_fit, _logistic_data, _lrm_fit

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "rms.json").read_text())

_COLUMNS = [
    ("index_orig", "index.orig"),
    ("training", "training"),
    ("test", "test"),
    ("optimism", "optimism"),
    ("index_corrected", "index.corrected"),
    ("n", "n"),
]


def _arr(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _ids(key: str) -> list[str]:
    return [case["name"] for case in FIXTURE[key]]


def _assert_validation(result: pl.DataFrame, expected: dict) -> None:
    assert result["index"].to_list() == expected["index"]
    for ours, theirs in _COLUMNS:
        np.testing.assert_allclose(result[ours].to_numpy(), _arr(expected[theirs]), rtol=1e-6, atol=1e-12)


def test_fixture_versions() -> None:
    assert FIXTURE["versions"]["rms"] == "6.7.1"


@pytest.mark.parametrize("case", FIXTURE["validate_lrm"], ids=_ids("validate_lrm"))
def test_validate_logistic_matches_rms(case: dict) -> None:
    data = pl.DataFrame(case["data"])
    result = validate_logistic(data, case["formula"], indices=np.array(case["indices"]))
    _assert_validation(result, case["result"])
    x, y, _ = _logistic_data(data, case["formula"], None)
    np.testing.assert_allclose(_lrm_fit(x, y).coefficients, case["coefficients"], rtol=1e-9)


@pytest.mark.parametrize("case", FIXTURE["calibrate_lrm"], ids=_ids("calibrate_lrm"))
def test_calibrate_logistic_matches_rms(case: dict) -> None:
    data = pl.DataFrame(case["data"])
    result = calibrate_logistic(data, case["formula"], indices=np.array(case["indices"]))
    t = result.table
    for ours, theirs in [
        ("predy", "predy"),
        ("calibrated_orig", "calibrated_orig"),
        ("calibrated_corrected", "calibrated_corrected"),
        ("optimism", "optimism"),
        ("n", "n"),
    ]:
        np.testing.assert_allclose(t[ours].to_numpy(), _arr(case[theirs]), atol=1e-6, rtol=0)
    assert result.mean_absolute_error == pytest.approx(case["mean_absolute_error"], abs=1e-6)
    assert result.mean_squared_error == pytest.approx(case["mean_squared_error"], abs=1e-6)
    assert result.quantile_90 == pytest.approx(case["quantile_90"], abs=1e-6)


@pytest.mark.parametrize("case", FIXTURE["validate_cph"], ids=_ids("validate_cph"))
def test_validate_cox_matches_rms(case: dict) -> None:
    data = pl.DataFrame(case["data"])
    result = validate_cox(data, case["formula"], indices=np.array(case["indices"]))
    _assert_validation(result, case["result"])
    x, t, e, _ = _cox_data(data, case["formula"], None, None, "efron")
    np.testing.assert_allclose(_cox_fit(x, t, e).coefficients, case["coefficients"], rtol=1e-9)


@pytest.mark.parametrize("case", FIXTURE["calibrate_cph"], ids=_ids("calibrate_cph"))
def test_calibrate_cox_matches_rms(case: dict) -> None:
    data = pl.DataFrame(case["data"])
    result = calibrate_cox(data, case["formula"], u=case["u"], m=case["m"], indices=np.array(case["indices"]))
    np.testing.assert_allclose(result.predicted, _arr(case["predicted"]), atol=1e-6, rtol=0)
    for col in [
        "index_orig",
        "training",
        "test",
        "mean_optimism",
        "mean_corrected",
        "n",
        "mean_predicted",
        "KM",
        "KM_corrected",
        "std_err",
    ]:
        np.testing.assert_allclose(result.table[col].to_numpy(), _arr(case[col]), atol=1e-6, rtol=0)


def test_somers_dxy_matches_hmisc() -> None:
    lr = FIXTURE["concordance"]["logistic"]
    case = next(c for c in FIXTURE["validate_lrm"] if c["name"] == lr["sample"])
    data = pl.DataFrame(case["data"])
    x, y, _ = _logistic_data(data, case["formula"], None)
    fit = _lrm_fit(x, y)
    lp = fit.coefficients[0] + x @ fit.coefficients[1:]
    out = somers_dxy(lp, y)
    assert out["c_index"][0] == pytest.approx(lr["C"], rel=1e-9)
    assert out["dxy"][0] == pytest.approx(lr["Dxy"], rel=1e-9)

    sv = FIXTURE["concordance"]["survival"]
    case = next(c for c in FIXTURE["validate_cph"] if c["name"] == sv["sample"])
    x, t, e, _ = _cox_data(pl.DataFrame(case["data"]), case["formula"], None, None, "efron")
    out = somers_dxy(x @ _cox_fit(x, t, e).coefficients, t, e)
    assert out["dxy"][0] == pytest.approx(sv["Dxy"], rel=1e-9)
    assert out["c_index"][0] == pytest.approx((sv["Dxy"] + 1) / 2, rel=1e-9)


def test_lowess_matches_r() -> None:
    lw = FIXTURE["lowess"]
    x, y = np.array(lw["x"]), np.array(lw["y"])
    sx, s3 = lowess_r(x, y)
    np.testing.assert_allclose(sx, lw["sorted_x"], rtol=0, atol=0)
    np.testing.assert_allclose(s3, lw["iter3"], atol=1e-10, rtol=0)
    _, s0 = lowess_r(x, y, iter=0)
    np.testing.assert_allclose(s0, lw["iter0"], atol=1e-10, rtol=0)
    _, s2 = lowess_r(x, y, f=0.2, delta=0.0)
    np.testing.assert_allclose(s2, lw["f02_delta0"], atol=1e-10, rtol=0)


# ---- seed-only runs (no R resamples) ----------------------------------------


def _logistic_frame() -> pl.DataFrame:
    case = FIXTURE["validate_lrm"][0]
    return pl.DataFrame(case["data"])


def test_validate_logistic_seed_only() -> None:
    data = _logistic_frame()
    a = validate_logistic(data, "y", ["x1", "x2"], B=25, seed=1)
    b = validate_logistic(data, "y ~ x1 + x2", B=25, seed=np.random.default_rng(1))
    assert a.shape == (11, 7)
    assert a.equals(b)
    row = {r["index"]: r for r in a.iter_rows(named=True)}
    assert abs(row["Dxy"]["optimism"]) < 0.05
    assert row["R2"]["training"] > row["R2"]["test"]
    assert 0 < row["Slope"]["index_corrected"] < 1.05
    assert row["Intercept"]["index_orig"] == 0.0
    assert row["Emax"]["index_corrected"] >= 0
    assert (a["n"] == 25).all()
    rest = a.filter(pl.col("index") != "Emax")
    np.testing.assert_allclose(rest["index_corrected"].to_numpy(), (rest["index_orig"] - rest["optimism"]).to_numpy())


def test_calibrate_logistic_seed_only(tmp_path: Path) -> None:
    data = _logistic_frame()
    cal = calibrate_logistic(data, "y ~ x1 + x2", B=20, seed=3)
    assert cal.table.height == 50
    assert cal.table.columns == ["predy", "calibrated_orig", "calibrated_corrected", "optimism", "n"]
    assert 0 <= cal.mean_absolute_error < 0.2
    assert cal.quantile_90 >= cal.mean_absolute_error
    path = plot_calibration_curve(cal, tmp_path / "cal.png")
    assert path.exists() and path.stat().st_size > 0


def test_validate_and_calibrate_cox_seed_only() -> None:
    case = FIXTURE["validate_cph"][0]
    data = pl.DataFrame(case["data"])
    v = validate_cox(data, "time", "status", ["x1", "x2"], B=15, seed=5)
    assert v["index"].to_list() == ["Dxy", "R2", "Slope", "D", "U", "Q", "g"]
    row = {r["index"]: r for r in v.iter_rows(named=True)}
    assert 0 < row["Dxy"]["index_corrected"] < row["Dxy"]["index_orig"] + 0.1
    assert 0.5 < row["Slope"]["index_corrected"] < 1.2
    c = calibrate_cox(data, case["formula"], u=5, m=50, B=10, seed=5)
    assert c.table.height == 3
    assert np.all((c.predicted > 0) & (c.predicted < 1))


def test_bad_indices() -> None:
    data = _logistic_frame()
    with pytest.raises(ValueError):
        validate_logistic(data, "y ~ x1", indices=np.zeros((3, 5), dtype=int))
    with pytest.raises(ValueError):
        validate_logistic(data, "y ~ x1", indices=np.full((3, data.height), data.height))
