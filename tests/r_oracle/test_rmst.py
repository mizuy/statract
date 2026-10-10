"""Restricted mean survival time against ``survfit`` and a survRM2 re-implementation.

Fixture: ``fixtures/rmst.json`` from ``scripts/rmst.R``. survRM2 is not
installable in the build environment, so the contrasts are checked against a
hand-coded R re-implementation of ``rmst2`` (labelled in the script). The
per-arm RMST and its standard error are also checked against
``summary(survfit(...), rmean = tau)``, which the script confirms agrees with
the re-implementation.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import restricted_mean_survival

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "rmst.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


def _fit(name: str):
    case = FIXTURE["cases"][name]
    return restricted_mean_survival(
        _data(case["data"]), "time", "status", "arm", tau=case["tau"], level=case["level"], reference=case["arms"][0]
    )


@pytest.mark.parametrize("name", CASES)
def test_arms(name: str) -> None:
    case = FIXTURE["cases"][name]
    got = _fit(name)
    assert got.tau == pytest.approx(case["tau_used"], rel=1e-15)
    assert got.groups == case["arms"]
    arms = got.arms
    np.testing.assert_allclose(arms["rmst"].to_numpy(), case["survfit"]["rmean"], rtol=1e-10)
    np.testing.assert_allclose(arms["std_error"].to_numpy(), case["survfit"]["se"], rtol=1e-10)
    for column, values in case["reimpl"]["arms"].items():
        np.testing.assert_allclose(arms[column].to_numpy(), values, rtol=1e-10, err_msg=column)


@pytest.mark.parametrize("name", CASES)
def test_contrasts(name: str) -> None:
    want = FIXTURE["cases"][name]["reimpl"]["contrasts"]
    got = _fit(name).contrasts
    assert got is not None
    assert got["contrast"].to_list() == ["RMST difference", "RMST ratio", "RMTL ratio"]
    for column in ("estimate", "conf_low", "conf_high", "p_value"):
        np.testing.assert_allclose(got[column].to_numpy(), want[column], rtol=1e-9, err_msg=column)


def test_reference_flips_the_contrast() -> None:
    forward = _fit("a_default").contrasts
    backward = restricted_mean_survival(_data("rmst_a"), "time", "status", "arm", reference=1).contrasts
    np.testing.assert_allclose(backward["estimate"][0], -forward["estimate"][0], rtol=1e-12)
    np.testing.assert_allclose(backward["estimate"][1], 1 / forward["estimate"][1], rtol=1e-12)
    np.testing.assert_allclose(backward["p_value"].to_numpy(), forward["p_value"].to_numpy(), rtol=1e-12)


def test_single_group_and_tau_limits() -> None:
    data = _data("rmst_a").filter(pl.col("arm") == 0)
    one = restricted_mean_survival(data, "time", "status", tau=4.0)
    assert one.contrasts is None
    assert one.arms.height == 1
    # The shorter arm of rmst_a ends censored, so tau cannot pass it.
    with pytest.raises(ValueError, match="tau must be at most"):
        restricted_mean_survival(_data("rmst_a"), "time", "status", "arm", tau=100.0)
