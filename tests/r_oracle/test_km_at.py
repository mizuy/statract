"""SurvivalCurve.at() against ``summary(survfit(...), times=)``.

Fixture: ``fixtures/km_at.json`` from ``scripts/km_at.R``. The grid covers
times before the first time, on tied times, between times, and past the last
time, with and without groups and delayed entry. The Aalen–Johansen cases
also cover case weights, and check the infinitesimal-jackknife standard
error of each cause.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract.surv.curve import survival_curve

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "km_at.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    df = pl.read_csv(HERE / "data" / f"{name}.csv")
    return df.with_columns(entry=(pl.int_range(pl.len()) % 3) * 0.5)


@pytest.mark.parametrize("name", CASES)
def test_at_matches_summary_survfit(name: str) -> None:
    case = FIXTURE["cases"][name]
    df = _data(case["data"]).with_columns(pl.col("arm").cast(pl.String))
    if case["entry"]:
        df = df.filter(pl.col("time") > pl.col("entry"))
    curve = survival_curve(df, "time", "status", "arm" if case["by"] else None, entry="entry" if case["entry"] else None)
    got = curve.at(case["grid"])
    if case["by"]:
        expected = pl.DataFrame({"group": case["group"], "time": case["time"]}, schema={"group": pl.String, "time": pl.Float64})
        got = expected.join(got, on=["group", "time"], how="left", maintain_order="left")
    np.testing.assert_array_equal(got["n_risk"].to_numpy(), case["n_risk"])
    np.testing.assert_allclose(got["estimate"].to_numpy(), case["estimate"], rtol=1e-12)
    # R gives NaN for the standard error once the curve reaches 0.
    se = np.array([np.nan if v is None else v for v in case["std_error"]], dtype=float)
    np.testing.assert_allclose(got["std_error"].to_numpy(), se, rtol=1e-10, atol=1e-15)
    for column in ("conf_low", "conf_high"):
        bound = np.array([np.nan if v is None else v for v in case[column]], dtype=float)
        np.testing.assert_allclose(got[column].to_numpy(), bound, rtol=1e-10, atol=1e-15, err_msg=column)


AJ_CASES = sorted(FIXTURE["aj"])


def _values(raw: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


@pytest.mark.parametrize("name", AJ_CASES)
def test_aalen_johansen_at_matches_summary_survfit(name: str) -> None:
    case = FIXTURE["aj"][name]
    df = _data("fg_a").with_columns(w=0.5 + (pl.int_range(pl.len()) % 4) * 0.5)
    if case["entry"]:
        df = df.filter(pl.col("time") > pl.col("entry"))
    curve = survival_curve(
        df,
        "time",
        "status",
        "stage" if case["by"] else None,
        kind="aalen_johansen",
        weights="w" if case["weighted"] else None,
        entry="entry" if case["entry"] else None,
    )
    got = curve.at(case["grid"])
    expected = pl.DataFrame(
        {"group": case["group"] or [None] * len(case["time"]), "time": case["time"]},
        schema={"group": pl.String, "time": pl.Float64},
    )
    for state in ("1", "2"):
        part = got.filter(pl.col("state") == state)
        if case["by"]:
            part = expected.join(part, on=["group", "time"], how="left", maintain_order="left")
        if not case["weighted"]:
            # Multi-state survfit counts n.risk without the weights; statract
            # keeps the weighted risk set, as its Kaplan-Meier does.
            np.testing.assert_allclose(part["n_risk"].to_numpy(), case["n_risk"], rtol=1e-12)
        np.testing.assert_allclose(part["estimate"].to_numpy(), case["estimate"][state], rtol=1e-12, atol=1e-15)
        for column in ("std_error", "conf_low", "conf_high"):
            np.testing.assert_allclose(
                part[column].to_numpy(), _values(case[column][state]), rtol=1e-9, atol=1e-14, err_msg=f"{state} {column}"
            )
