"""Cox residuals, cox.zph, concordance, basehaz, and robust variance against survival.

Fixture: ``fixtures/cox_inference.json`` from ``scripts/cox_inference.R``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.surv import cox_ph, proportional_hazards_test

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "cox_inference.json").read_text())
CASES = sorted(FIXTURE["cases"])
STAGE = pl.Enum(["I", "II", "III"])


def _data(name: str) -> pl.DataFrame:
    frame = pl.read_csv(HERE / "data" / f"{name}.csv")
    if "stage" in frame.columns:
        frame = frame.with_columns(pl.col("stage").cast(STAGE))
    return frame


@cache
def _fit(case_name: str):
    case = FIXTURE["cases"][case_name]
    data = _data(case_name.split("_")[0] + "_" + case_name.split("_")[1])
    return cox_ph(data, case["formula"], ties=case["ties"], weights=case.get("weights"))


def _matrix(values) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return out.reshape(len(out), -1)


@pytest.mark.parametrize("name", CASES)
def test_coefficients_and_covariance(name: str) -> None:
    want = FIXTURE["cases"][name]
    fit = _fit(name)
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-6)
    np.testing.assert_allclose(fit.log_likelihood, want["log_likelihood"], rtol=1e-9)
    np.testing.assert_allclose(fit.information_inverse(), np.asarray(want["naive_covariance"]), rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, np.asarray(want["covariance"]), rtol=1e-6)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("kind", ["martingale", "deviance", "score", "schoenfeld", "scaled_schoenfeld", "dfbeta", "dfbetas"])
def test_residuals(name: str, kind: str) -> None:
    want = FIXTURE["cases"][name][kind]
    got = _fit(name).residuals(kind)
    if got.ndim == 1:
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
    else:
        np.testing.assert_allclose(got, _matrix(want), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name", CASES)
def test_proportional_hazards_test(name: str) -> None:
    want = FIXTURE["cases"][name]
    table = proportional_hazards_test(_fit(name))
    assert table["term"].to_list()[-1] == "global"
    assert table["term"].to_list()[:-1] == [t for t in want["zph_terms"] if t != "GLOBAL"]
    np.testing.assert_allclose(table["statistic"].to_numpy(), want["zph_chisq"], rtol=1e-6)
    np.testing.assert_allclose(table["df"].to_numpy(), want["zph_df"])


@pytest.mark.parametrize("name", CASES)
def test_concordance(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _fit(name).concordance()
    np.testing.assert_allclose(got["concordance"][0], want["concordance"], rtol=1e-9)
    np.testing.assert_allclose(got["std_error"][0], want["concordance_se"], rtol=1e-6)


@pytest.mark.parametrize("name", CASES)
def test_baseline_hazard_and_expected(name: str) -> None:
    want = FIXTURE["cases"][name]
    fit = _fit(name)
    base = fit.baseline_hazard()
    np.testing.assert_allclose(base["time"].to_numpy(), want["basehaz_time"])
    np.testing.assert_allclose(base["hazard"].to_numpy(), want["basehaz_hazard"], rtol=1e-6)
    np.testing.assert_allclose(fit.predict(kind="expected"), want["expected"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(fit.predict(kind="linear_predictor"), want["lp"], rtol=1e-6, atol=1e-12)


def test_fine_gray_uses_the_row_robust_variance() -> None:
    """Fractional Fine–Gray weights make coxph report the Lin–Wei variance per row."""
    from statract.surv import fine_gray

    want = FIXTURE["fine_gray"]
    data = _data("fg_a")
    expanded = fine_gray(data, "Surv(time, status) ~ x + stage", cause=1)
    assert expanded.height == want["n_rows"]
    fit = cox_ph(expanded, "Surv(fgstart, fgstop, fgstatus) ~ x + stage", weights="fgwt")
    assert fit.names == want["terms"]
    np.testing.assert_allclose(fit.coefficients, want["coefficients"], rtol=1e-6)
    np.testing.assert_allclose(fit.log_likelihood, want["log_likelihood"], rtol=1e-9)
    np.testing.assert_allclose(fit.information_inverse(), np.asarray(want["naive_covariance"]), rtol=1e-6)
    np.testing.assert_allclose(fit.covariance, np.asarray(want["covariance"]), rtol=1e-6)
