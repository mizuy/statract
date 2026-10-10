"""IPTW weights and balance against WeightIt / cobalt formulas.

Fixture: ``fixtures/iptw.json`` from ``scripts/iptw.R``. That script is
hand-coded from the WeightIt 1.x / cobalt 4.x formulas with ``glm()`` and base
R, because the packages are not installable here. ``scripts/iptw_weightit.R``
writes the same JSON with the packages themselves.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import balance_table, propensity_weights

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "iptw.json").read_text())
CASES = sorted(FIXTURE["cases"])
NUMERIC = [
    "mean_treated_unadjusted",
    "mean_control_unadjusted",
    "diff_unadjusted",
    "variance_ratio_unadjusted",
    "mean_treated_adjusted",
    "mean_control_adjusted",
    "diff_adjusted",
    "variance_ratio_adjusted",
]


def _num(values) -> np.ndarray:
    return np.asarray([np.nan if v in ("NA", None) else v for v in values], dtype=float)


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _weights(name: str):
    case = FIXTURE["cases"][name]
    return propensity_weights(
        _data(case["data"]),
        case["formula"],
        estimand=case["estimand"],
        stabilize=case["stabilize"],
        trim=case["trim"],
        trim_lower=case["trim_lower"],
        sampling_weights=case["s_weights"],
    )


@pytest.mark.parametrize("name", CASES)
def test_ps_and_weights(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _weights(name)
    rows = np.asarray(want["rows"], dtype=int)
    np.testing.assert_array_equal(got.row_index, rows)
    np.testing.assert_allclose(got.ps[rows], want["ps"], rtol=1e-8)
    np.testing.assert_allclose(got.weights[rows], want["weights"], rtol=1e-8)
    dropped = np.setdiff1d(np.arange(got.data.height), rows)
    assert np.all(np.isnan(got.weights[dropped]))


@pytest.mark.parametrize("name", CASES)
def test_balance(name: str) -> None:
    want = FIXTURE["cases"][name]
    table = _weights(name).balance(binary=want["binary"])
    bal = want["balance"]
    assert table["term"].to_list() == bal["term"]
    assert table["type"].to_list() == bal["type"]
    for col in NUMERIC:
        np.testing.assert_allclose(
            table[col].cast(pl.Float64).fill_null(np.nan).to_numpy(), _num(bal[col]), rtol=1e-7, atol=1e-12, err_msg=col
        )


@pytest.mark.parametrize("name", CASES)
def test_effective_sample_size(name: str) -> None:
    want = FIXTURE["cases"][name]["ess"]
    ess = _weights(name).effective_sample_size()
    np.testing.assert_allclose(ess["control"].to_numpy(), want["control"], rtol=1e-8)
    np.testing.assert_allclose(ess["treated"].to_numpy(), want["treated"], rtol=1e-8)


def test_balance_table_matches_method() -> None:
    name = "b_ate_sw_stab"
    case = FIXTURE["cases"][name]
    got = _weights(name)
    frame = got.frame()
    table = balance_table(
        frame,
        case["formula"],
        weights="weights",
        sampling_weights=case["s_weights"],
        distance="ps",
        estimand=case["estimand"],
    )
    assert table.equals(got.balance())
