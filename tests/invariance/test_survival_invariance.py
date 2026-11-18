"""Survival estimators must not change under transformations that leave the model unchanged.

Each test changes the data in a way that should not change the answer (row
order, labels, time units, a location shift of a covariate, follow-up split
into counting-process rows, integer weights versus repeated rows) and checks
that the estimate, its standard error and the test statistics stay the same.
These checks need no R and catch bugs that only show on a data shape that the
R oracle fixtures do not have (the cox_mixed split follow-up bug was one).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import (
    accelerated_failure,
    conditional_logit,
    cox_mixed,
    cox_ph,
    cumulative_incidence,
    fine_gray_regression,
    log_rank,
    restricted_mean_survival,
    standardize_cox,
    survival_curve,
)


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 240
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n)
    # Rounded times give many ties.
    time = np.round(rng.exponential(np.exp(-0.5 * x - 0.3 * g)), 1) + 0.1
    cause = rng.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25])
    return pl.DataFrame(
        {
            "id": np.arange(n),
            "t": time,
            "e": (cause > 0).astype(np.int64),
            "c": cause.astype(np.int64),
            "x": x,
            "g": g,
            "cl": rng.integers(0, 12, n),
            "strat": rng.integers(0, 3, n),
            "w": rng.integers(1, 4, n),
        }
    )


def permute(df: pl.DataFrame) -> pl.DataFrame:
    return df.sample(fraction=1.0, shuffle=True, seed=11)


def repeat_by_weight(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.all().repeat_by(pl.col("w"))).explode(pl.all())


def split_follow_up_at(df: pl.DataFrame, cut: float) -> pl.DataFrame:
    """Each row with t > cut becomes (0, cut] censored and (cut, t] with the outcome."""
    df = df.with_columns(s=pl.lit(0.0))
    later = df.filter(pl.col("t") > cut)
    early = later.with_columns(t=pl.lit(cut), e=pl.lit(0, dtype=pl.Int64), c=pl.lit(0, dtype=pl.Int64))
    return pl.concat([df.filter(pl.col("t") <= cut), early, later.with_columns(s=pl.lit(cut))])


def estimates(fit) -> np.ndarray:
    return np.concatenate([fit.coefficients, np.sqrt(np.diag(fit.covariance))])


def numeric(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(pl.selectors.numeric()).to_numpy().astype(float)


def assert_same(a, b, rtol: float = 1e-8) -> None:
    np.testing.assert_allclose(np.asarray(b, float), np.asarray(a, float), rtol=rtol, atol=rtol)


# Kaplan-Meier


def test_km_row_order(data):
    a = survival_curve(data, "t", "e", "g").table.sort("group", "time")
    b = survival_curve(permute(data), "t", "e", "g").table.sort("group", "time")
    assert_same(numeric(a), numeric(b))


def test_km_time_unit(data):
    a = survival_curve(data, "t", "e", "g").table
    b = survival_curve(data.with_columns(pl.col("t") * 365.25), "t", "e", "g").table
    assert_same(a["estimate"], b["estimate"])
    assert_same(a["std_error"], b["std_error"])
    assert_same(a["time"] * 365.25, b["time"])


def test_km_split_follow_up(data):
    times = [0.3, 0.6, 1.0, 2.0]
    a = survival_curve(data, "t", "e", "g").at(times)
    b = survival_curve(split_follow_up_at(data, 0.55), "t", "e", "g", entry="s").at(times)
    assert_same(numeric(a), numeric(b))


def test_km_weights_equal_repeated_rows(data):
    a = survival_curve(data, "t", "e", "g", weights="w").table.sort("group", "time")
    b = survival_curve(repeat_by_weight(data), "t", "e", "g").table.sort("group", "time")
    assert_same(a["estimate"], b["estimate"])
    assert_same(a["n_risk"], b["n_risk"])


# Log-rank, cumulative incidence, RMST


@pytest.mark.parametrize("strata", [None, "strat"])
def test_log_rank_row_order_and_time_unit(data, strata):
    base = log_rank(data, "t", "e", "g", strata)
    for df in (permute(data), data.with_columns(pl.col("t") * 1000)):
        other = log_rank(df, "t", "e", "g", strata)
        assert_same([base.statistic, base.p_value], [other.statistic, other.p_value])


def test_log_rank_strata_labels(data):
    a = log_rank(data, "t", "e", "g", "strat").statistic
    b = log_rank(
        data.with_columns(strat=pl.format("site-{}", pl.col("strat"))),
        "t",
        "e",
        "g",
        "strat",
    ).statistic
    assert_same(a, b)


def test_cumulative_incidence_row_order_and_time_unit(data):
    base = cumulative_incidence(data, "t", "c", "g")
    other = cumulative_incidence(permute(data), "t", "c", "g")
    assert_same(numeric(base.table), numeric(other.table))
    assert_same(numeric(base.tests), numeric(other.tests))
    scaled = cumulative_incidence(data.with_columns(pl.col("t") * 1000), "t", "c", "g")
    assert_same(base.table["estimate"], scaled.table["estimate"])
    assert_same(base.table["variance"], scaled.table["variance"])


def test_rmst_row_order_and_time_unit(data):
    base = restricted_mean_survival(data, "t", "e", "g", tau=1.5)
    assert_same(
        numeric(base.arms),
        numeric(restricted_mean_survival(permute(data), "t", "e", "g", tau=1.5).arms),
    )
    scaled = restricted_mean_survival(data.with_columns(pl.col("t") * 10), "t", "e", "g", tau=15.0)
    assert_same(base.arms["rmst"] * 10, scaled.arms["rmst"])
    assert_same(base.arms["std_error"] * 10, scaled.arms["std_error"])
    ratios = base.contrasts.filter(pl.col("contrast") != "RMST difference")
    scaled_ratios = scaled.contrasts.filter(pl.col("contrast") != "RMST difference")
    assert_same(numeric(ratios), numeric(scaled_ratios))


# Cox models


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize(
    "change",
    [
        "row_order",
        "shift_x",
        "time_unit",
    ],
)
def test_cox_ph_simple_changes(data, ties, change):
    base = estimates(cox_ph(data, "t", "e", ["x", "g"], ties=ties))
    df = {
        "row_order": permute(data),
        "shift_x": data.with_columns(pl.col("x") + 1000),
        "time_unit": data.with_columns(pl.col("t") * 1000),
    }[change]
    assert_same(base, estimates(cox_ph(df, "t", "e", ["x", "g"], ties=ties)), rtol=1e-7)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_ph_split_follow_up(data, ties):
    base = estimates(cox_ph(data, "t", "e", ["x", "g"], strata="strat", ties=ties))
    split = split_follow_up_at(data, 0.55)
    assert_same(
        base,
        estimates(cox_ph(split, "t", "e", ["x", "g"], strata="strat", entry="s", ties=ties)),
    )


def test_cox_ph_split_follow_up_robust_se(data):
    base = estimates(cox_ph(data, "t", "e", ["x", "g"], cluster="id"))
    split = split_follow_up_at(data, 0.55)
    assert_same(base, estimates(cox_ph(split, "t", "e", ["x", "g"], cluster="id", entry="s")))


def test_cox_ph_breslow_weights_equal_repeated_rows(data):
    # Efron's tie correction treats a weighted row and repeated rows differently
    # (in R as well), so only Breslow is exact here.
    a = cox_ph(data, "t", "e", ["x", "g"], weights="w", ties="breslow")
    b = cox_ph(repeat_by_weight(data), "t", "e", ["x", "g"], ties="breslow")
    assert_same(a.coefficients, b.coefficients)


def test_cox_ph_strata_and_cluster_labels(data):
    base = estimates(cox_ph(data, "t", "e", ["x", "g"], strata="strat", cluster="cl"))
    relabelled = data.with_columns(
        strat=pl.format("s{}", pl.col("strat")),
        cl=pl.col("cl") * 13 + 5,
    )
    assert_same(
        base,
        estimates(cox_ph(relabelled, "t", "e", ["x", "g"], strata="strat", cluster="cl")),
    )
    assert_same(
        base,
        estimates(cox_ph(permute(data), "t", "e", ["x", "g"], strata="strat", cluster="cl")),
    )


def test_cox_ph_offset_moves_coefficient(data):
    base = cox_ph(data, "t", "e", ["x", "g"])
    shifted = cox_ph(
        data.with_columns(o=0.5 * pl.col("g").cast(pl.Float64)),
        "t",
        "e",
        ["x", "g"],
        offset="o",
    )
    assert_same(base.coefficients - np.array([0.0, 0.5]), shifted.coefficients)
    assert_same(base.covariance, shifted.covariance)


FORMULA = "Surv(t, e) ~ x + g + (1 | cl)"


@pytest.mark.parametrize("change", ["row_order", "shift_x", "time_unit", "cluster_labels"])
def test_cox_mixed_simple_changes(data, change):
    base = cox_mixed(data, FORMULA)
    df = {
        "row_order": permute(data),
        "shift_x": data.with_columns(pl.col("x") + 1000),
        "time_unit": data.with_columns(pl.col("t") * 1000),
        "cluster_labels": data.with_columns(cl=pl.format("hospital-{}", pl.col("cl") * 7 + 3)),
    }[change]
    other = cox_mixed(df, FORMULA)
    assert_same(estimates(base), estimates(other), rtol=1e-6)
    assert_same(np.ravel(base.theta), np.ravel(other.theta), rtol=1e-5)


def test_cox_mixed_breslow_weights_equal_repeated_rows(data):
    a = cox_mixed(data, FORMULA, weights="w", ties="breslow")
    b = cox_mixed(repeat_by_weight(data), FORMULA, ties="breslow")
    assert_same(a.coefficients, b.coefficients, rtol=1e-6)


@pytest.mark.parametrize("change", ["row_order", "shift_x", "time_unit"])
def test_standardize_cox(data, change):
    k = 1000.0 if change == "time_unit" else 1.0
    df = {
        "row_order": permute(data),
        "shift_x": data.with_columns(pl.col("x") + 1000),
        "time_unit": data.with_columns(pl.col("t") * k),
    }[change]

    def run(frame, scale):
        return standardize_cox(
            frame,
            "Surv(t, e) ~ g + x",
            values={"g": [0, 1]},
            times=[0.5 * scale, 1.0 * scale],
        )

    a, b = run(data, 1.0), run(df, k)
    assert_same(a.estimates, b.estimates)
    assert_same(np.asarray(a.covariances), np.asarray(b.covariances))


# Competing risks


@pytest.mark.parametrize("change", ["row_order", "shift_x", "time_unit", "cause_labels"])
def test_fine_gray_regression(data, change):
    base = estimates(fine_gray_regression(data, "t", "c", ["x", "g"]))
    if change == "cause_labels":
        other = fine_gray_regression(
            data.with_columns(c=pl.col("c").replace({1: 5, 2: 9})),
            "t",
            "c",
            ["x", "g"],
            cause=5,
        )
    else:
        df = {
            "row_order": permute(data),
            "shift_x": data.with_columns(pl.col("x") + 1000),
            "time_unit": data.with_columns(pl.col("t") * 1000),
        }[change]
        other = fine_gray_regression(df, "t", "c", ["x", "g"])
    assert_same(base, estimates(other), rtol=1e-7)


# Parametric and conditional models


def test_aft_time_unit_moves_only_the_intercept(data):
    base = accelerated_failure(data, "t", "e", ["x", "g"])
    scaled = accelerated_failure(data.with_columns(pl.col("t") * 1000), "t", "e", ["x", "g"])
    shift = np.zeros(len(base.coefficients))
    shift[0] = np.log(1000)
    assert_same(base.coefficients + shift, scaled.coefficients)
    assert_same(base.covariance, scaled.covariance, rtol=1e-6)
    assert_same(
        estimates(base),
        estimates(accelerated_failure(permute(data), "t", "e", ["x", "g"])),
    )


def test_aft_weights_equal_repeated_rows(data):
    a = accelerated_failure(data, "t", "e", ["x", "g"], weights="w")
    b = accelerated_failure(repeat_by_weight(data), "t", "e", ["x", "g"])
    assert_same(a.coefficients, b.coefficients, rtol=1e-7)


def test_conditional_logit(data):
    df = data.with_columns(y=pl.col("e"))
    base = estimates(conditional_logit(df, "y", ["x"], strata="cl"))
    for other in (
        permute(df),
        df.with_columns(cl=pl.format("set-{}", pl.col("cl"))),
        df.with_columns(pl.col("x") + 1000),
    ):
        assert_same(
            base,
            estimates(conditional_logit(other, "y", ["x"], strata="cl")),
            rtol=1e-7,
        )
