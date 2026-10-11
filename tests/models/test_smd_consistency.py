"""One SMD definition across tableone, matching, and weighting balance."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import match_sample, propensity_weights
from statract.tableone.stat import standardized_difference


def _frame(n: int = 300, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    smoke = rng.binomial(1, 0.4, n)
    lp = -0.5 + 0.05 * (age - 60) + 0.8 * smoke
    treat = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return pl.DataFrame({"treat": treat, "age": age, "smoke": smoke})


def _row(table: pl.DataFrame, term: str) -> dict:
    return table.filter(pl.col("term") == term).row(0, named=True)


def test_unadjusted_smd_matches_tableone() -> None:
    data = _frame()
    groups = data["treat"]
    want_age = standardized_difference(data["age"], groups)
    want_smoke = standardized_difference(data["smoke"], groups, categorical=True)

    matched = match_sample(data, "treat", ["age", "smoke"]).balance()
    assert abs(_row(matched, "age")["smd_all"]) == pytest.approx(want_age, rel=1e-10)
    assert abs(_row(matched, "smoke")["smd_all"]) == pytest.approx(want_smoke, rel=1e-10)

    for estimand in ("ATE", "ATT", "ATC", "ATO"):
        weighted = propensity_weights(data, "treat ~ age + smoke", estimand=estimand).balance()
        assert abs(_row(weighted, "age")["diff_unadjusted"]) == pytest.approx(want_age, rel=1e-10)
        assert abs(_row(weighted, "smoke")["diff_unadjusted"]) == pytest.approx(want_smoke, rel=1e-10)


def test_binary_uses_p_one_minus_p() -> None:
    data = _frame()
    t = data["treat"].to_numpy() == 1
    x = data["smoke"].to_numpy().astype(float)
    p1, p0 = x[t].mean(), x[~t].mean()
    want = (p1 - p0) / np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    matched = match_sample(data, "treat", ["age", "smoke"]).balance()
    assert _row(matched, "smoke")["smd_all"] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("sd_denominator", ["pooled", "treated", "control"])
def test_denominator_is_fixed_before_and_after_matching(sd_denominator: str) -> None:
    data = _frame()
    # A caliper drops treated units, so a recomputed denominator would move.
    table = match_sample(data, "treat", ["age", "smoke"], caliper=0.05).balance(sd_denominator=sd_denominator)
    for term in ("age", "smoke"):
        row = _row(table, term)
        before = (row["mean_treated_all"] - row["mean_control_all"]) / row["smd_all"]
        after = (row["mean_treated_matched"] - row["mean_control_matched"]) / row["smd_matched"]
        assert after == pytest.approx(before, rel=1e-10)


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATO"])
@pytest.mark.parametrize("sd_denominator", ["pooled", "treated", "weighted"])
def test_denominator_is_fixed_before_and_after_weighting(estimand: str, sd_denominator: str) -> None:
    table = propensity_weights(_frame(), "treat ~ age + smoke", estimand=estimand).balance(
        sd_denominator=sd_denominator
    )
    for term in ("age", "smoke"):
        row = _row(table, term)
        before = (row["mean_treated_unadjusted"] - row["mean_control_unadjusted"]) / row["diff_unadjusted"]
        after = (row["mean_treated_adjusted"] - row["mean_control_adjusted"]) / row["diff_adjusted"]
        assert after == pytest.approx(before, rel=1e-10)


def test_matchit_style_treated_denominator() -> None:
    data = _frame()
    t = data["treat"].to_numpy() == 1
    age = data["age"].to_numpy()
    table = match_sample(data, "treat", ["age", "smoke"]).balance(sd_denominator="treated")
    row = _row(table, "age")
    sd_t = np.std(age[t], ddof=1)
    assert row["smd_all"] == pytest.approx((age[t].mean() - age[~t].mean()) / sd_t, rel=1e-12)
    assert row["smd_matched"] == pytest.approx((row["mean_treated_matched"] - row["mean_control_matched"]) / sd_t)
    raw = match_sample(data, "treat", ["age", "smoke"]).balance(binary="raw")
    smoke = data["smoke"].to_numpy()
    assert _row(raw, "smoke")["smd_all"] == pytest.approx(smoke[t].mean() - smoke[~t].mean(), rel=1e-12)


def test_cobalt_defaults_with_options() -> None:
    data = _frame()
    t = data["treat"].to_numpy() == 1
    age = data["age"].to_numpy()
    smoke = data["smoke"].to_numpy()
    table = propensity_weights(data, "treat ~ age + smoke", estimand="ATT").balance(
        binary="raw", sd_denominator="treated"
    )
    assert _row(table, "age")["diff_unadjusted"] == pytest.approx(
        (age[t].mean() - age[~t].mean()) / np.std(age[t], ddof=1), rel=1e-10
    )
    assert _row(table, "smoke")["diff_unadjusted"] == pytest.approx(smoke[t].mean() - smoke[~t].mean(), rel=1e-12)


def test_matching_balance_rejects_unknown_options() -> None:
    matched = match_sample(_frame(), "treat", ["age", "smoke"])
    with pytest.raises(ValueError, match="sd_denominator"):
        matched.balance(sd_denominator="weighted")
    with pytest.raises(ValueError, match="binary"):
        matched.balance(binary="x")
