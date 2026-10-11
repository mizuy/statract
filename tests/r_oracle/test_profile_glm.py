"""Profile-likelihood intervals for ``fit_glm`` against ``confint.glm``.

Fixture: ``fixtures/profile_glm.json`` from ``scripts/profile_glm.R``, with
``confint()`` (MASS's ``confint.glm``; ``stats::confint.glm`` from R 4.4 is the
same code) and the exact crossing solved with ``uniroot``. ``confint``
interpolates a spline through the profile, so ``profile_interval`` matches it
to 1e-3 of a standard error (seen: at most 2.5e-4) and the exact crossing to
1e-7.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_glm, profile_interval

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "profile_glm.json").read_text())
CASES = sorted(FIXTURE["cases"])


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _interval(name: str) -> pl.DataFrame:
    case = FIXTURE["cases"][name]
    fit = fit_glm(_data(case["data"]), case["formula"], family=case["family"], weights=case["weights"])
    np.testing.assert_allclose(fit.coefficients, case["coefficients"], rtol=1e-6, atol=1e-9)
    return profile_interval(fit, level=case["level"])


@pytest.mark.parametrize("name", CASES)
def test_exact_crossing(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _interval(name)
    assert got["term"].to_list() == want["names"]
    se = np.asarray(want["std_error"])
    np.testing.assert_allclose(got["conf_low"].to_numpy(), want["exact_lower"], rtol=0, atol=1e-7 * se.max())
    np.testing.assert_allclose(got["conf_high"].to_numpy(), want["exact_upper"], rtol=0, atol=1e-7 * se.max())


@pytest.mark.parametrize("name", CASES)
def test_confint(name: str) -> None:
    want = FIXTURE["cases"][name]
    got = _interval(name)
    se = np.asarray(want["std_error"])
    low_gap = np.abs(got["conf_low"].to_numpy() - want["confint_lower"]) / se
    high_gap = np.abs(got["conf_high"].to_numpy() - want["confint_upper"]) / se
    assert low_gap.max() < 1e-3
    assert high_gap.max() < 1e-3


def test_rejects_ols() -> None:
    from statract import fit_ols

    fit = fit_ols(_data("profile_a"), "score ~ x")
    with pytest.raises(ValueError, match="fit_glm"):
        profile_interval(fit)


def test_separation_gives_infinite_limit() -> None:
    fit = fit_glm(_data("firth_sep"), "y ~ g + z", family="binomial")
    table = profile_interval(fit)
    high = table.filter(pl.col("term") == "g")["conf_high"].item()
    assert high == np.inf
