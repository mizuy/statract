"""``link="linear.logit"`` matches on the logit of the propensity score."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import fit_glm, match_sample


def _frame(n: int = 400, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    smoke = rng.binomial(1, 0.4, n)
    lp = -0.8 + 0.08 * (age - 60) + 1.2 * smoke
    treat = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return pl.DataFrame({"treat": treat, "age": age, "smoke": smoke})


def test_linear_logit_equals_numeric_logit_distance() -> None:
    data = _frame()
    lps = fit_glm(data, "treat ~ age + smoke", family="binomial").predict(kind="link")
    by_link = match_sample(
        data, "treat", ["age", "smoke"], link="linear.logit", caliper=0.2
    )
    by_vector = match_sample(
        data, "treat", ["age", "smoke"], distance=np.asarray(lps), caliper=0.2
    )
    w_link = by_link.frame()["weights"].to_numpy()
    w_vec = by_vector.frame()["weights"].to_numpy()
    np.testing.assert_array_equal(w_link, w_vec)


def test_linear_logit_differs_from_probability_scale() -> None:
    data = _frame()
    prob = (
        match_sample(data, "treat", ["age", "smoke"], caliper=0.05)
        .frame()["weights"]
        .to_numpy()
    )
    logit = (
        match_sample(data, "treat", ["age", "smoke"], link="linear.logit", caliper=0.05)
        .frame()["weights"]
        .to_numpy()
    )
    assert not np.array_equal(prob, logit)


def test_linear_probit_runs() -> None:
    data = _frame()
    matched = match_sample(
        data, "treat", ["age", "smoke"], link="linear.probit", caliper=0.2
    )
    assert matched.frame()["weights"].sum() > 0


def test_unknown_link_raises() -> None:
    with pytest.raises(ValueError, match="link"):
        match_sample(_frame(), "treat", ["age", "smoke"], link="cloglog")
