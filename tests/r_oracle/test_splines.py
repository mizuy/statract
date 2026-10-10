"""ns / bs / rcs formula terms against R's splines and rms.

Fixture: ``fixtures/splines.json`` from ``scripts/splines.R``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cox_ph, fit_glm, fit_ols, model_matrix, spline_effect, spline_test
from statract.models.design import build_design

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "splines.json").read_text())
MATRICES = FIXTURE["matrices"]
FITS = FIXTURE["fits"]


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@pytest.mark.parametrize("case", MATRICES, ids=[c["formula"] + "|" + c["data"] for c in MATRICES])
def test_model_matrix(case: dict) -> None:
    data = _data(case["data"])
    built = model_matrix(case["formula"], data)
    assert built.design.names == case["names"]
    np.testing.assert_array_equal(built.design.row_index, case["rows"])
    np.testing.assert_allclose(built.design.x, np.asarray(case["x"]), rtol=1e-10, atol=1e-12)
    fit = fit_ols(data, case["formula"])
    np.testing.assert_allclose(fit.coefficients, case["coef"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("case", MATRICES, ids=[c["formula"] + "|" + c["data"] for c in MATRICES])
def test_new_rows_keep_knots(case: dict) -> None:
    data = _data(case["data"])
    built = model_matrix(case["formula"], data)
    first = {name: data.get_column(name).drop_nulls()[0] for name in data.columns}
    new = pl.DataFrame({name: [first[name]] * len(case["at"]) for name in data.columns}, schema=data.schema)
    new = new.with_columns(pl.Series(case["new"], [float(v) for v in case["at"]]).cast(data.schema[case["new"]]))
    rebuilt = build_design(new, built.design)
    np.testing.assert_allclose(rebuilt.x, np.asarray(case["new_x"]), rtol=1e-10, atol=1e-12)


def test_glm_with_splines() -> None:
    want = FITS["glm"]
    fit = fit_glm(_data("splines_a"), want["formula"], family="binomial")
    assert fit.names == want["names"]
    np.testing.assert_allclose(fit.coefficients, want["coef"], rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), want["se"], rtol=1e-6)


def test_cox_with_splines() -> None:
    want = FITS["cox"]
    fit = cox_ph(_data("splines_a"), want["formula"])
    assert fit.names == want["names"]
    np.testing.assert_allclose(fit.coefficients, want["coef"], rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), want["se"], rtol=1e-6)


def _rms_fit(key: str):
    want = FITS[key]
    data = _data("splines_a")
    if key == "lrm":
        return want, data, fit_glm(data, want["formula"], family="binomial")
    return want, data, cox_ph(data, want["formula"])


@pytest.mark.parametrize("key", ["lrm", "cph"])
def test_anova_rows(key: str) -> None:
    want, _data_a, fit = _rms_fit(key)
    np.testing.assert_allclose(fit.coefficients, want["coef"], rtol=1e-6)
    table = spline_test(fit, want["term"])
    assert table.get_column("part").to_list() == ["overall", "nonlinear"]
    for part in ("overall", "nonlinear"):
        row = table.filter(pl.col("part") == part).row(0, named=True)
        chi2, df, p = want["anova"][part]
        assert row["df"] == df
        np.testing.assert_allclose(row["statistic"], chi2, rtol=1e-6)
        np.testing.assert_allclose(row["p_value"], p, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("key", ["lrm", "cph"])
def test_contrast(key: str) -> None:
    want, data, fit = _rms_fit(key)
    table = spline_effect(fit, data, "age", at=want["grid"], reference=want["reference"])
    # lrm and cph stop on their own criteria, a little short of the optimum.
    for column in ("estimate", "std_error", "conf_low", "conf_high"):
        np.testing.assert_allclose(table.get_column(column).to_numpy(), want["contrast"][column], rtol=1e-5)
    ratio = spline_effect(fit, data, "age", at=want["grid"], reference=want["reference"], exponentiate=True)
    np.testing.assert_allclose(ratio.get_column("estimate").to_numpy(), np.exp(want["contrast"]["estimate"]), rtol=1e-6)


def test_spline_errors() -> None:
    data = _data("splines_a")
    with pytest.raises(ValueError, match="no argument"):
        model_matrix("y ~ ns(age, degree = 2)", data)
    with pytest.raises(ValueError, match="literal"):
        model_matrix("y ~ ns(age, df = bmi)", data)
    with pytest.raises(ValueError, match="factor"):
        model_matrix("y ~ rcs(sex, 3)", data)
    fit = fit_glm(data, "y ~ rcs(age, 4) + sex", family="binomial")
    with pytest.raises(ValueError, match="not a spline term"):
        spline_test(fit, "age")
    ns_fit = fit_glm(data, "y ~ ns(age, 3)", family="binomial")
    assert spline_test(ns_fit, "ns(age, 3)").get_column("part").to_list() == ["overall"]
