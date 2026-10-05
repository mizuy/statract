"""Conditional logit against an enumerated likelihood and against cox_ph."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize

from statract import conditional_logit, cox_ph


def _matched(*, double: bool) -> pl.DataFrame:
    rows = [
        (1, 1, 0.2, -0.4, 0.1),
        (1, 0, -0.3, 0.2, 0.0),
        (1, 0, 0.5, 0.1, -0.2),
        (2, 1, -0.1, 0.6, 0.3),
        (2, 0, 0.4, -0.2, 0.1),
        (2, 1 if double else 0, 0.0, 0.3, -0.4),
        (3, 0, 0.7, -0.5, 0.2),
        (3, 0, -0.2, 0.1, 0.0),
        (3, 0, 0.3, 0.4, -0.1),
        (4, 1, 0.6, 0.2, 0.5),
        (4, 0, -0.4, -0.1, 0.2),
        (4, 0, 0.1, 0.5, -0.3),
    ]
    return pl.DataFrame(
        {
            "set": [row[0] for row in rows],
            "y": [row[1] for row in rows],
            "x": [row[2] for row in rows],
            "z": [row[3] for row in rows],
            "off": [row[4] for row in rows],
        }
    )


def _enumerated_ll(x: np.ndarray, y: np.ndarray, strata: np.ndarray, beta: np.ndarray, offset: np.ndarray) -> float:
    eta = x @ beta + offset
    total = 0.0
    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        yy = y[idx]
        k = int(yy.sum())
        if k == 0 or k == len(idx):
            continue
        ee = eta[idx]
        total += float(ee[yy == 1].sum())
        logs = [float(ee[list(comb)].sum()) for comb in combinations(range(len(idx)), k)]
        pivot = max(logs)
        total -= pivot + float(np.log(np.exp(np.asarray(logs) - pivot).sum()))
    return total


def test_exact_matches_the_enumerated_likelihood():
    data = _matched(double=True)
    fit = conditional_logit(data, "y ~ x + z + offset(off) + strata(set)")
    x = np.column_stack([data["x"].to_numpy(), data["z"].to_numpy()])
    y = data["y"].to_numpy().astype(float)
    strata = data["set"].to_numpy()
    offset = data["off"].to_numpy()
    assert fit.log_likelihood == pytest.approx(_enumerated_ll(x, y, strata, fit.coefficients, offset))
    for index in range(2):
        step = np.zeros(2)
        step[index] = 1e-5
        upper = _enumerated_ll(x, y, strata, fit.coefficients + step, offset)
        lower = _enumerated_ll(x, y, strata, fit.coefficients - step, offset)
        assert (upper - lower) / 2e-5 == pytest.approx(0.0, abs=1e-4)
    solved = minimize(
        lambda beta: -_enumerated_ll(x, y, strata, beta, offset),
        np.zeros(2),
        method="BFGS",
    )
    assert solved.success
    np.testing.assert_allclose(fit.coefficients, solved.x, atol=1e-5)
    assert fit.converged
    assert fit.family == "conditional_logit"
    assert fit.ties == "exact"
    assert fit.names == ["x", "z"]


def test_one_event_per_set_matches_cox():
    data = _matched(double=False).with_columns(pl.lit(1.0).alias("time"))
    exact = conditional_logit(data, "y ~ x + z + strata(set)")
    efron = cox_ph(data, "Surv(time, y) ~ x + z + strata(set)", ties="efron")
    breslow = cox_ph(data, "Surv(time, y) ~ x + z + strata(set)", ties="breslow")
    np.testing.assert_allclose(exact.coefficients, efron.coefficients, atol=1e-8)
    np.testing.assert_allclose(exact.coefficients, breslow.coefficients, atol=1e-8)
    np.testing.assert_allclose(exact.covariance, efron.covariance, atol=1e-8)
    assert exact.log_likelihood == pytest.approx(efron.log_likelihood)


def test_efron_and_approximate_match_cox():
    data = _matched(double=True).with_columns(pl.lit(1.0).alias("time"))
    efron = conditional_logit(data, "y ~ x + strata(set)", method="efron")
    cox = cox_ph(data, "Surv(time, y) ~ x + strata(set)", ties="efron")
    np.testing.assert_allclose(efron.coefficients, cox.coefficients)
    np.testing.assert_allclose(efron.covariance, cox.covariance)
    assert efron.log_likelihood == pytest.approx(cox.log_likelihood)
    approximate = conditional_logit(data, "y ~ x + strata(set)", method="approximate")
    breslow = cox_ph(data, "Surv(time, y) ~ x + strata(set)", ties="breslow")
    np.testing.assert_allclose(approximate.coefficients, breslow.coefficients)
    assert approximate.ties == "breslow"
    exact = conditional_logit(data, "y ~ x + strata(set)")
    assert not np.allclose(exact.coefficients, efron.coefficients)
    np.testing.assert_allclose(efron.predict(kind="survival"), cox.predict(kind="survival"))


def test_column_interface_matches_the_formula():
    data = _matched(double=True).with_columns(pl.Series("arm", ["a", "b", "a"] * 4))
    formula = conditional_logit(data, "y ~ x + arm + strata(set)")
    columns = conditional_logit(data, "y", ["x", "arm"], strata="set")
    assert formula.names == columns.names == ["x", "armb"]
    np.testing.assert_allclose(formula.coefficients, columns.coefficients)
    np.testing.assert_allclose(formula.predict(), formula.x @ formula.coefficients + formula.offset)
    fresh = data.with_columns(pl.Series("arm", ["b", "a"] * 6))
    assert formula.predict(fresh, kind="risk").shape == (fresh.height,)


def test_null_rows_and_uninformative_sets_are_dropped_from_the_likelihood():
    data = _matched(double=True)
    base = conditional_logit(data, "y ~ x + strata(set)")
    extra = pl.concat(
        [
            data,
            pl.DataFrame(
                {
                    "set": [9, 9, 10],
                    "y": [0, 0, 1],
                    "x": [0.2, None, 0.4],
                    "z": [0.0, 0.0, 0.0],
                    "off": [0.0, 0.0, 0.0],
                }
            ),
        ],
        how="diagonal",
    )
    # The null x drops one row of set 9, leaving a set with no event. Set 10 has one row.
    fitted = conditional_logit(extra, "y ~ x + z + strata(set)")
    kept = conditional_logit(data, "y ~ x + z + strata(set)")
    np.testing.assert_allclose(fitted.coefficients, kept.coefficients)
    assert fitted.n_obs == data.height + 2
    assert base.names == ["x"]


def test_exact_ignores_weights_and_rejects_cluster():
    data = _matched(double=True).with_columns(pl.Series("w", [1.0, 2.0] * 6), pl.Series("id", [1, 1, 2, 2, 3, 3] * 2))
    plain = conditional_logit(data, "y ~ x + strata(set)")
    with pytest.warns(UserWarning, match="weights ignored"):
        weighted = conditional_logit(data, "y ~ x + strata(set)", weights="w")
    np.testing.assert_allclose(weighted.coefficients, plain.coefficients)
    with pytest.raises(ValueError, match="cluster"):
        conditional_logit(data, "y ~ x + cluster(id) + strata(set)")
    with pytest.raises(ValueError, match="cluster"):
        conditional_logit(data, "y", ["x"], strata="set", cluster="id")


def test_efron_cluster_and_weights_match_cox():
    data = _matched(double=True).with_columns(
        pl.lit(1.0).alias("time"),
        pl.Series("w", [1.0, 1.2, 0.8] * 4),
        pl.Series("id", [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
    )
    clustered = conditional_logit(data, "y ~ x + strata(set) + cluster(id)", method="efron")
    cox = cox_ph(data, "Surv(time, y) ~ x + strata(set) + cluster(id)", ties="efron")
    np.testing.assert_allclose(clustered.coefficients, cox.coefficients)
    np.testing.assert_allclose(clustered.covariance, cox.covariance)
    weighted = conditional_logit(data, "y ~ x + strata(set)", weights="w", method="breslow")
    cox_w = cox_ph(data, "Surv(time, y) ~ x + strata(set)", weights="w", ties="breslow")
    np.testing.assert_allclose(weighted.coefficients, cox_w.coefficients)
    np.testing.assert_allclose(weighted.covariance, cox_w.covariance)


def test_rejects_bad_input():
    data = _matched(double=False)
    with pytest.raises(ValueError, match="0 or 1"):
        conditional_logit(data.with_columns(pl.Series("y", [0, 2] * 6)), "y ~ x + strata(set)")
    with pytest.raises(ValueError, match="method"):
        conditional_logit(data, "y ~ x + strata(set)", method="discrete")
    with pytest.raises(ValueError, match="both an event and a non-event"):
        conditional_logit(data.with_columns(pl.Series("y", [1] * data.height)), "y ~ x + strata(set)")
    with pytest.raises(ValueError, match="not both"):
        conditional_logit(data, "y ~ x + strata(set)", ["x"])
    with pytest.raises(ValueError, match="baseline hazard"):
        conditional_logit(data, "y ~ x + strata(set)").predict(kind="survival")
    with pytest.raises(ValueError, match="residuals"):
        conditional_logit(data, "y ~ x + strata(set)").residuals()


def test_exact_matches_survival_clogit():
    # survival 3.8.6: clogit(y ~ x + z + offset(off) + strata(set), method="exact")
    fit = conditional_logit(_matched(double=True), "y ~ x + z + offset(off) + strata(set)")
    np.testing.assert_allclose(fit.coefficients, [0.8065595334473922, 0.8007649347605327], atol=1e-8)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), [1.9100720111876677, 2.3972788176307462], atol=1e-8)
    assert fit.log_likelihood == pytest.approx(-2.8694408296285134)


def test_tidy_exponentiates():
    fit = conditional_logit(_matched(double=True), "y ~ x + z + strata(set)")
    tidy = fit.tidy(exponentiate=True)
    assert tidy["term"].to_list() == ["x", "z"]
    np.testing.assert_allclose(tidy["exp_estimate"].to_numpy(), np.exp(fit.coefficients))
