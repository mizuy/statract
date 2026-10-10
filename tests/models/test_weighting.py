"""propensity_weights: options, errors, and the hand-off to outcome models."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import cox_ph, fit_glm, hc_covariance, propensity_weights


def _frame(n: int = 300, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    grp = rng.choice(["a", "b", "c"], n)
    smoke = rng.integers(0, 2, n)
    lp = -0.3 + 0.04 * (age - 50) + 0.6 * (grp == "b") + 0.5 * smoke
    treat = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.5 * treat + 0.02 * (age - 50))))).astype(int)
    time = rng.exponential(5 * np.exp(-0.3 * treat), n)
    return pl.DataFrame(
        {"treat": treat, "age": age, "grp": grp, "smoke": smoke, "y": y, "time": time, "event": np.ones(n, dtype=int)}
    )


FORMULA = "treat ~ age + grp + smoke"


def test_ate_weights_are_inverse_probabilities() -> None:
    res = propensity_weights(_frame(), FORMULA)
    t = res.data["treat"].to_numpy() == 1
    np.testing.assert_allclose(res.weights[t], 1 / res.ps[t])
    np.testing.assert_allclose(res.weights[~t], 1 / (1 - res.ps[~t]))


def test_ato_balances_means_exactly() -> None:
    table = propensity_weights(_frame(), FORMULA, estimand="ATO").balance()
    covs = table.filter(pl.col("type") != "Distance")
    np.testing.assert_allclose(covs["diff_adjusted"].to_numpy(), 0.0, atol=1e-8)


def test_trim_bounds_leave_focal_arm() -> None:
    res = propensity_weights(_frame(), FORMULA, estimand="ATT", trim=(0.05, 1.5))
    t = res.data["treat"].to_numpy() == 1
    assert np.all(res.weights[t] == 1.0)
    assert res.weights[~t].max() <= 1.5
    assert res.weights[~t].min() >= 0.05


def test_trim_quantile_caps_top() -> None:
    base = propensity_weights(_frame(), FORMULA)
    res = propensity_weights(_frame(), FORMULA, trim=0.9)
    cap = np.quantile(base.weights, 0.9)
    np.testing.assert_allclose(res.weights.max(), cap)
    np.testing.assert_allclose(res.weights, np.minimum(base.weights, cap))
    low = propensity_weights(_frame(), FORMULA, trim=0.1)
    np.testing.assert_allclose(low.weights, res.weights)


def test_missing_rows_are_null_in_frame() -> None:
    frame = _frame().with_columns(pl.when(pl.int_range(pl.len()) == 4).then(None).otherwise(pl.col("age")).alias("age"))
    res = propensity_weights(frame, FORMULA)
    out = res.frame()
    assert out["weights"][4] is None
    assert out["ps"][4] is None
    assert out["weights"].null_count() == 1


def test_errors() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="estimand"):
        propensity_weights(frame, FORMULA, estimand="ATM")
    with pytest.raises(ValueError, match="stabilize"):
        propensity_weights(frame, FORMULA, estimand="ATT", stabilize=True)
    with pytest.raises(ValueError, match="0/1"):
        propensity_weights(frame.with_columns(pl.col("treat") * 2), FORMULA)
    with pytest.raises(ValueError, match="whole number"):
        propensity_weights(frame, FORMULA, trim=2.5)
    with pytest.raises(ValueError, match="binary"):
        propensity_weights(frame, FORMULA).balance(binary="x")


def test_balance_threshold_and_ess() -> None:
    res = propensity_weights(_frame(), FORMULA)
    table = res.balance(threshold=0.1)
    assert table["term"].to_list() == ["prop.score", "age", "grp_a", "grp_b", "grp_c", "smoke"]
    assert table["balanced"].dtype == pl.Boolean
    ess = res.effective_sample_size()
    assert ess["sample"].to_list() == ["unadjusted", "adjusted"]
    summary = res.summary()
    np.testing.assert_allclose(summary["ess"].to_numpy(), ess.row(1)[1:])


def test_weights_plug_into_outcome_models() -> None:
    res = propensity_weights(_frame(), FORMULA, stabilize=True)
    frame = res.frame()
    fit = fit_glm(frame, "y ~ treat", family="binomial", weights="weights")
    se = np.sqrt(np.diag(hc_covariance(fit, "HC0")))
    assert np.all(np.isfinite(se))
    cox = cox_ph(frame, "Surv(time, event) ~ treat", weights="weights")
    assert np.isfinite(cox.tidy()["std_error"][0])
