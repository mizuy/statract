"""weighted_glm and PropensityWeights.effect: bootstrap, options, and errors."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import fit_glm, hc_covariance, propensity_weights, weighted_glm


def _frame(n: int = 400, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    smoke = rng.integers(0, 2, n)
    site = rng.integers(0, 20, n)
    lp = -0.2 + 0.05 * (age - 50) + 0.7 * smoke
    treat = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.8 + 0.6 * treat + 0.04 * (age - 50) + 0.5 * smoke)))).astype(int)
    return pl.DataFrame({"treat": treat, "age": age, "smoke": smoke, "site": site, "y": y})


FORMULA = "treat ~ age + smoke"


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _stacked_rd(df: pl.DataFrame, estimand: str) -> tuple[float, float]:
    """The hand-written estimator of the theory text: numeric Jacobian of the stack."""
    x = np.column_stack([np.ones(df.height), df["age"].to_numpy(), df["smoke"].to_numpy()]).astype(float)
    a = df["treat"].to_numpy().astype(float)
    y = df["y"].to_numpy().astype(float)
    fit = fit_glm(df, FORMULA, family="binomial")

    def weights(e):
        if estimand == "ATE":
            return a / e + (1 - a) / (1 - e)
        return a + (1 - a) * e / (1 - e)

    def psi(theta):
        e = _expit(x @ theta[:3])
        w = weights(e)
        m1, m0 = theta[3:]
        return np.column_stack([x * (a - e)[:, None], w * a * (y - m1), w * (1 - a) * (y - m0)])

    w = weights(_expit(x @ fit.coefficients))
    m1 = np.sum(w * a * y) / np.sum(w * a)
    m0 = np.sum(w * (1 - a) * y) / np.sum(w * (1 - a))
    theta = np.r_[fit.coefficients, m1, m0]
    k = theta.size
    bread = np.zeros((k, k))
    for j in range(k):
        d = np.zeros(k)
        d[j] = 1e-6 * max(1.0, abs(theta[j]))
        bread[:, j] = (psi(theta + d).sum(0) - psi(theta - d).sum(0)) / (2 * d[j])
    p = psi(theta)
    inv = np.linalg.inv(bread)
    v = inv @ (p.T @ p) @ inv.T
    g = np.zeros(k)
    g[-2:] = [1.0, -1.0]
    return float(m1 - m0), float(np.sqrt(g @ v @ g))


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
def test_effect_matches_hand_written_stack(estimand: str) -> None:
    df = _frame()
    got = propensity_weights(df, FORMULA, estimand=estimand).effect("y")
    rd, se = _stacked_rd(df, estimand)
    assert got["estimate"][0] == pytest.approx(rd, rel=1e-10)
    assert got["std_error"][0] == pytest.approx(se, rel=1e-6)


def test_hc0_effect_is_weighted_linear_model() -> None:
    df = _frame()
    pw = propensity_weights(df, FORMULA)
    lin = fit_glm(pw.frame(), "y ~ treat", weights="weights")
    got = pw.effect("y", vcov="HC0")
    assert got["estimate"][0] == pytest.approx(lin.coefficients[1], rel=1e-10)
    assert got["std_error"][0] == pytest.approx(np.sqrt(hc_covariance(lin, "HC0")[1, 1]), rel=1e-8)
    model = weighted_glm(pw, "y ~ treat", vcov="HC0")
    np.testing.assert_allclose(model.covariance, hc_covariance(lin, "HC0"), rtol=1e-8)


def test_bootstrap_is_seeded_and_close_to_asympt() -> None:
    pw = propensity_weights(_frame(), FORMULA)
    measures = ["difference", "ratio", "odds_ratio"]
    boot = pw.effect("y", measure=measures, vcov="bootstrap", n_boot=400, seed=1)
    again = pw.effect("y", measure=measures, vcov="bootstrap", n_boot=400, seed=1)
    assert boot.equals(again)
    asympt = pw.effect("y", measure=measures)
    np.testing.assert_allclose(boot["estimate"].to_numpy(), asympt["estimate"].to_numpy())
    np.testing.assert_allclose(boot["std_error"].to_numpy(), asympt["std_error"].to_numpy(), rtol=0.2)
    assert (boot["conf_low"] < boot["estimate"]).all() and (boot["estimate"] < boot["conf_high"]).all()


def test_weighted_glm_bootstrap_and_cluster() -> None:
    pw = propensity_weights(_frame(), FORMULA)
    asympt = weighted_glm(pw, "y ~ treat", family="binomial", cluster="site")
    boot = weighted_glm(pw, "y ~ treat", family="binomial", vcov="bootstrap", cluster="site", n_boot=300, seed=2)
    assert boot.replicates.shape == (300, 2)
    np.testing.assert_allclose(boot.coefficients, asympt.coefficients)
    table = boot.tidy()
    low, high = np.quantile(boot.replicates, [0.025, 0.975], axis=0)
    np.testing.assert_allclose(table["conf_low"].to_numpy(), low)
    np.testing.assert_allclose(table["conf_high"].to_numpy(), high)
    np.testing.assert_allclose(table["std_error"].to_numpy(), boot.replicates.std(axis=0, ddof=1))
    assert np.all(np.isfinite(np.diag(asympt.covariance)))


def test_ate_asympt_is_smaller_than_hc0() -> None:
    # For ATE, ignoring the propensity score's estimation is conservative.
    pw = propensity_weights(_frame(), FORMULA)
    asympt = pw.effect("y")["std_error"][0]
    hc0 = pw.effect("y", vcov="HC0")["std_error"][0]
    assert asympt < hc0


def test_trimmed_weights() -> None:
    pw = propensity_weights(_frame(), FORMULA, trim=0.95, trim_lower=True)
    assert pw.trim_lower
    with pytest.raises(ValueError, match="untrimmed"):
        pw.effect("y")
    with pytest.raises(ValueError, match="untrimmed"):
        weighted_glm(pw, "y ~ treat")
    assert np.isfinite(pw.effect("y", vcov="HC0")["std_error"][0])
    assert np.isfinite(pw.effect("y", vcov="bootstrap", n_boot=50, seed=3)["std_error"][0])


def test_errors() -> None:
    df = _frame()
    pw = propensity_weights(df, FORMULA)
    with pytest.raises(ValueError, match="vcov"):
        pw.effect("y", vcov="HC3")
    with pytest.raises(ValueError, match="measure"):
        pw.effect("y", measure="hazard")
    with pytest.raises(ValueError, match="family"):
        weighted_glm(pw, "y ~ treat", family="probit")
    with pytest.raises(ValueError, match="between 0 and 1"):
        pw.effect("age", measure="odds_ratio")
    holes = df.with_columns(pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("y")).alias("y"))
    pw_holes = propensity_weights(holes, FORMULA)
    with pytest.raises(ValueError, match="missing"):
        pw_holes.effect("y")
    with pytest.raises(ValueError, match="drops rows"):
        weighted_glm(pw_holes, "y ~ treat")


def test_rows_dropped_by_the_propensity_model() -> None:
    df = _frame().with_columns(pl.when(pl.int_range(pl.len()) == 5).then(None).otherwise(pl.col("age")).alias("age"))
    pw = propensity_weights(df, FORMULA)
    got = pw.effect("y")
    model = weighted_glm(pw, "y ~ treat")
    assert model.fit.n_obs == df.height - 1
    assert got["estimate"][0] == pytest.approx(model.coefficients[1], rel=1e-10)
    assert got["std_error"][0] == pytest.approx(np.sqrt(model.covariance[1, 1]), rel=1e-8)
