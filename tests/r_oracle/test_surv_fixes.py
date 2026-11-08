"""Survival fixes from the 2026-11 audit against ``survival``.

Fixture: ``fixtures/surv_fixes.json`` from ``scripts/surv_fixes.R``.

* ``SurvivalCurve.quantile`` against ``quantile(survfit(...))`` for each
  confidence transform. ``quantile.survfit`` takes the cumulative
  probability ``1 - S``; statract takes the survival probability ``S``, so
  R's ``probs = 0.25`` is statract's ``p = 0.75``.
* Nelson–Aalen curves against ``survfit(stype = 2, ctype = 1)``.
* ``conditional_logit`` against ``clogit`` with an offset and with whole-number
  and fractional weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.surv import conditional_logit, survival_curve

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "surv_fixes.json").read_text())


def _values(raw: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv").with_columns(pl.col("arm").cast(pl.String))


@pytest.mark.parametrize("name", sorted(FIXTURE["quantile"]))
def test_km_quantile_matches_quantile_survfit(name: str) -> None:
    case = FIXTURE["quantile"][name]
    curve = survival_curve(_data(case["data"]), "time", "status", by="arm", confidence=case["confidence"])
    survival_p = [1 - p for p in case["probability"]]
    got = curve.quantile(survival_p)
    for index, group in enumerate(case["group"]):
        rows = got.filter(pl.col("group") == group)
        np.testing.assert_allclose(rows["probability"].to_numpy(), survival_p)
        for column in ("quantile", "conf_low", "conf_high"):
            np.testing.assert_allclose(
                rows[column].to_numpy(), _values(case[column][index]), rtol=1e-12, err_msg=f"{group} {column}"
            )


@pytest.mark.parametrize("name", sorted(FIXTURE["nelson_aalen"]))
def test_nelson_aalen_matches_survfit(name: str) -> None:
    case = FIXTURE["nelson_aalen"][name]
    table = survival_curve(
        _data(case["data"]), "time", "status", kind="nelson_aalen", confidence=case["confidence"]
    ).table
    np.testing.assert_allclose(table["time"].to_numpy(), case["time"])
    np.testing.assert_array_equal(table["n_risk"].to_numpy(), case["n_risk"])
    for column in ("estimate", "std_error", "conf_low", "conf_high"):
        np.testing.assert_allclose(
            table[column].to_numpy(), _values(case[column]), rtol=1e-10, atol=1e-15, err_msg=column
        )


@pytest.mark.parametrize("name", sorted(FIXTURE["clogit"]))
def test_conditional_logit_matches_clogit(name: str) -> None:
    case = FIXTURE["clogit"][name]
    data = pl.read_csv(HERE / "data" / "clogit_a.csv").with_columns(pl.col("z").cast(pl.Enum(["lo", "mid", "hi"])))
    kwargs = {"method": case["method"]}
    formula = case["formula"]
    if "offset(o)" in formula:
        formula = formula.replace(" + offset(o)", "")
        kwargs["offset"] = "o"
    if case["weights"] is not None:
        kwargs["weights"] = case["weights"]
    fit = conditional_logit(data, formula, **kwargs)
    tidy = fit.tidy()
    assert tidy["term"].to_list() == ["x", "zmid", "zhi"]
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), case["coefficients"], rtol=1e-8)
    np.testing.assert_allclose(tidy["std_error"].to_numpy(), case["std_errors"], rtol=1e-8)
    np.testing.assert_allclose(fit.log_likelihood, case["log_likelihood"], rtol=1e-10)
