"""statract.models.htest against R 4.3 stats (fixture from tests/r_oracle/scripts/htest.R)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import (
    binom_test,
    mcnemar_test,
    p_adjust,
    prop_test,
    proportion_ci,
    t_test,
    wilcox_test,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "htest.json").read_text())
FUNCS = {
    "t_test": t_test,
    "wilcox_test": wilcox_test,
    "mcnemar_test": mcnemar_test,
    "binom_test": binom_test,
    "prop_test": prop_test,
}
RTOL = 1e-10


def _num(v):
    """Decode jsonlite ``na="string"`` values."""
    if isinstance(v, list):
        return [_num(x) for x in v]
    if isinstance(v, str):
        return {"NA": np.nan, "Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan}.get(v, v)
    return v


def _args(case: dict) -> dict:
    args = {k: _num(v) for k, v in case["args"].items()}
    if case["fn"] == "mcnemar_test":
        x = args["x"]
        if isinstance(x, list) and isinstance(x[0], list):
            # jsonlite writes an R matrix row by row.
            args["x"] = np.asarray(x, dtype=float)
        else:
            args["x"] = pl.Series([None if isinstance(v, float) and np.isnan(v) else v for v in x])
            args["y"] = pl.Series([None if isinstance(v, float) and np.isnan(v) else v for v in args["y"]])
    return args


def _case_id(case: dict) -> str:
    keys = ",".join(f"{k}={v}" for k, v in case["args"].items() if not isinstance(v, list))
    return f"{case['fn']}({keys})"


CASES = FIXTURE["cases"]


def test_versions():
    assert FIXTURE["r_version"].startswith("R version 4.3")


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-{_case_id(c)}" for i, c in enumerate(CASES)])
def test_case(case):
    exp = case["expect"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = FUNCS[case["fn"]](**_args(case))
    assert got.method == exp["method"].strip()  # R pastes a leading space into " Two Sample t-test"
    assert got.alternative == exp["alternative"]
    np.testing.assert_allclose(got.statistic, _num(exp["statistic"]), rtol=RTOL)
    np.testing.assert_allclose(got.p_value, _num(exp["p_value"]), rtol=1e-9, atol=1e-14)
    if exp["parameter"] is None:
        assert got.parameter is None
    else:
        np.testing.assert_allclose(got.parameter, _num(exp["parameter"]), rtol=RTOL)
    if exp["conf_int"] is None:
        assert got.conf_int is None
    else:
        np.testing.assert_allclose(got.conf_int, _num(exp["conf_int"]), rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(got.conf_level, exp["conf_level"], rtol=RTOL)
    if exp["estimates"] is None:
        assert not got.estimates
    else:
        assert list(got.estimates) == list(exp["estimates"])
        np.testing.assert_allclose(
            list(got.estimates.values()), [_num(v) for v in exp["estimates"].values()], rtol=RTOL, atol=1e-12
        )


@pytest.mark.parametrize("entry", FIXTURE["p_adjust"]["results"], ids=lambda e: f"{e['method']}-n{e['n']}")
def test_p_adjust(entry):
    p = _num(FIXTURE["p_adjust"]["p"]) if len(entry["value"]) == len(FIXTURE["p_adjust"]["p"]) else None
    if p is None:
        p = {2: [0.03, 0.04], 3: [0.04, 0.01, 0.04]}[len(entry["value"])]
    got = p_adjust(p, entry["method"], n=entry["n"])
    np.testing.assert_allclose(got, _num(entry["value"]), rtol=RTOL)


@pytest.mark.parametrize("entry", FIXTURE["proportion_ci"], ids=lambda e: f"{e['x']}of{e['n']}")
@pytest.mark.parametrize("method", ["wilson", "clopper-pearson"])
def test_proportion_ci(entry, method):
    frame = pl.DataFrame({"y": [1] * entry["x"] + [0] * (entry["n"] - entry["x"])})
    ci = frame.select(proportion_ci(pl.col("y"), alpha=entry["alpha"], method=method).alias("ci")).unnest("ci")
    exp = entry["wilson" if method == "wilson" else "clopper_pearson"]
    np.testing.assert_allclose([ci["lo"][0], ci["hi"][0]], exp, rtol=RTOL, atol=1e-14)


def test_frame_shape():
    frame = t_test([1.0, 2.0, 4.0], [2.0, 5.0, 6.0, 9.0]).frame()
    assert frame.columns == ["estimate", "statistic", "p_value", "parameter", "conf_low", "conf_high", "method", "alternative"]
    assert frame["estimate"][0] == pytest.approx(7 / 3 - 5.5)
