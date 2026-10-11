"""IPTW outcome models against a hand-coded ``WeightIt::glm_weightit``.

Fixture: ``fixtures/iptw_outcome.json`` from ``scripts/iptw_outcome.R``. WeightIt
is not installable here, so the script codes its M-estimation (``vcov="asympt"``)
with ``glm()``: the outcome score stacked on the propensity score's logistic
score, with the analytic Jacobian checked against central differences. ``HC0``
is the outcome block alone; the script also writes ``sandwich::vcovHC`` /
``vcovCL`` (``type="HC0"``), which differs only by ``glm``'s convergence error.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import propensity_weights, weighted_effect, weighted_glm

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "iptw_outcome.json").read_text())
GLM_CASES = sorted(FIXTURE["glm"])
EFFECT_CASES = sorted(FIXTURE["effect"])


@cache
def _data() -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / "iptw_outcome_a.csv")


def _weights(case: dict):
    return propensity_weights(
        _data(),
        case["ps_formula"],
        estimand=case["estimand"],
        stabilize=case["stabilize"],
        sampling_weights=case["s_weights"],
    )


@pytest.mark.parametrize("name", GLM_CASES)
def test_weighted_glm(name: str) -> None:
    case = FIXTURE["glm"][name]
    pw = _weights(case)
    asympt = weighted_glm(pw, case["outcome"], family=case["family"], cluster=case["cluster"])
    assert asympt.names == case["terms"]
    np.testing.assert_allclose(asympt.coefficients, case["coefficients"], rtol=1e-8)
    np.testing.assert_allclose(asympt.covariance, case["vcov_asympt"], rtol=1e-7, atol=1e-14)
    hc0 = weighted_glm(pw, case["outcome"], family=case["family"], vcov="HC0", cluster=case["cluster"])
    np.testing.assert_allclose(hc0.covariance, case["vcov_hc0"], rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(hc0.covariance, case["vcov_hc0_sandwich"], rtol=1e-4, atol=1e-10)
    table = asympt.tidy()
    np.testing.assert_allclose(table["std_error"].to_numpy(), np.sqrt(np.diag(case["vcov_asympt"])), rtol=1e-7)


@pytest.mark.parametrize("name", EFFECT_CASES)
def test_effect(name: str) -> None:
    case = FIXTURE["effect"][name]
    pw = _weights(case)
    for vcov, key in (("asympt", "se_asympt"), ("HC0", "se_hc0")):
        got = pw.effect(case["outcome"], measure=case["measure"], vcov=vcov, cluster=case["cluster"])
        assert got["measure"].to_list() == case["measure"]
        np.testing.assert_allclose(got["mean_treated"].to_numpy(), case["mean_treated"], rtol=1e-8)
        np.testing.assert_allclose(got["mean_control"].to_numpy(), case["mean_control"], rtol=1e-8)
        np.testing.assert_allclose(got["estimate"].to_numpy(), case["estimate"], rtol=1e-8)
        np.testing.assert_allclose(got["std_error"].to_numpy(), case[key], rtol=1e-7, err_msg=vcov)


def test_effect_function_matches_method() -> None:
    case = FIXTURE["effect"]["att"]
    pw = _weights(case)
    assert weighted_effect(pw, "ybin", measure="ratio").equals(pw.effect("ybin", measure="ratio"))
