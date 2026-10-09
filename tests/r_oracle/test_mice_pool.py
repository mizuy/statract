"""Rubin's rules against mice::pool, and the chained-equations imputer.

Fixture: ``fixtures/mice_pool.json`` from ``scripts/mice_pool.R``. R imputes
the sample with mice; the fixture holds the completed data sets, so pooling
is compared digit by digit. Our own draws use another generator, so
``impute_chained`` is checked for behaviour, not for equal values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import MultipleImputation, cox_ph, fit_glm, fit_ols, impute_chained, pool

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "mice_pool.json").read_text())
LEVELS = pl.Enum(["a", "b", "c"])


def _data() -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / "mi_a.csv").with_columns(pl.col("g").cast(LEVELS))


def _completed() -> list[pl.DataFrame]:
    data = _data()
    return [
        data.with_columns(
            pl.Series("x2", c["x2"], dtype=pl.Float64),
            pl.Series("g", c["g"]).cast(LEVELS),
            pl.Series("b", c["b"], dtype=pl.Int64),
        )
        for c in FIXTURE["completed"]
    ]


MODELS = {
    "ols": lambda d: fit_ols(d, "y ~ x1 + x2 + g"),
    "logit": lambda d: fit_glm(d, "b ~ x1 + x2 + y", family="binomial"),
    "cox": lambda d: cox_ph(d, "Surv(time, status) ~ x1 + x2 + b"),
}
COLUMNS = ["estimate", "ubar", "b", "t", "df", "riv", "lambda", "fmi", "std_error", "statistic", "p_value", "conf_low", "conf_high"]


@pytest.mark.parametrize("model", sorted(MODELS))
def test_pool_matches_mice(model: str) -> None:
    want = FIXTURE[model]
    table = pool([MODELS[model](d) for d in _completed()])
    assert table["term"].to_list() == want["terms"]
    assert table["dfcom"][0] == want["dfcom"]
    for column in COLUMNS:
        np.testing.assert_allclose(table[column].to_numpy(), want[column], rtol=1e-6, err_msg=column)


def test_pool_options() -> None:
    fits = [MODELS["logit"](d) for d in _completed()]
    plain = pool(fits)
    wide = pool(fits, exponentiate=True, level=0.9)
    np.testing.assert_allclose(wide["exp_estimate"].to_numpy(), np.exp(plain["estimate"].to_numpy()))
    assert np.all(wide["conf_low"].to_numpy() > plain["conf_low"].to_numpy())
    # An infinite complete-data df gives the classic Rubin df.
    rubin = pool(fits, dfcom=float("inf"))
    lam = rubin["lambda"].to_numpy()
    np.testing.assert_allclose(rubin["df"].to_numpy(), 4 / lam**2)
    with pytest.raises(ValueError, match="at least two"):
        pool(fits[:1])
    with pytest.raises(ValueError, match="same terms"):
        pool([fits[0], MODELS["ols"](_completed()[0])])


def test_impute_chained_fills_every_gap_and_keeps_types() -> None:
    data = _data().with_columns(pl.col("b").cast(pl.Boolean))
    mi = impute_chained(data, m=3, n_iter=5, seed=1)
    assert isinstance(mi, MultipleImputation)
    assert mi.m == 3
    assert mi.methods == {"x2": "pmm", "g": "polyreg", "b": "logreg"}
    assert mi.missing == {name: data[name].null_count() for name in ("x2", "g", "b")}
    for frame in mi.datasets:
        assert frame.schema == data.schema
        assert frame.null_count().sum_horizontal()[0] == 0
        # Observed values never change; pmm only uses observed donors.
        observed = data["x2"].is_not_null()
        np.testing.assert_array_equal(frame["x2"].filter(observed), data["x2"].filter(observed))
        assert set(frame["x2"].to_list()) <= set(data["x2"].drop_nulls().to_list())
    assert not mi.complete(0).equals(mi.complete(1))
    assert mi.long().height == 3 * data.height
    again = impute_chained(data, m=3, n_iter=5, seed=1)
    assert all(a.equals(b) for a, b in zip(mi.datasets, again.datasets, strict=True))


def test_impute_chained_recovers_estimates_under_mar() -> None:
    """Complete-case analysis is biased when x2 is missing by y; MICE is not."""
    rng = np.random.default_rng(7)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(size=n)
    y = 1.0 + 1.0 * x1 - 0.7 * x2 + rng.normal(size=n)
    gone = rng.uniform(size=n) < 0.6 / (1 + np.exp(-2 * (y - 1)))
    data = pl.DataFrame({"y": y, "x1": x1, "x2": np.where(gone, np.nan, x2)}).fill_nan(None)
    model = lambda d: fit_ols(d, "y ~ x1 + x2")
    truth = fit_ols(pl.DataFrame({"y": y, "x1": x1, "x2": x2}), "y ~ x1 + x2").coefficients
    complete_case = model(data.drop_nulls()).coefficients
    table = impute_chained(data, m=10, n_iter=10, seed=3).pool(model)
    assert np.max(np.abs(complete_case - truth)) > 0.1
    np.testing.assert_allclose(table["estimate"].to_numpy(), truth, atol=0.05)
    assert np.all(table["fmi"].to_numpy() > 0.05)


def test_impute_chained_guards() -> None:
    data = _data()
    with pytest.raises(ValueError, match="pmm needs a numeric"):
        impute_chained(data, methods={"g": "pmm"})
    with pytest.raises(ValueError, match="unknown method"):
        impute_chained(data, methods={"x2": "cart"})
    with pytest.raises(ValueError, match="not in data"):
        impute_chained(data, columns=["nope"])
    skipped = impute_chained(data, m=2, n_iter=2, methods={"g": "none"}, seed=0)
    assert "g" not in skipped.methods
    assert skipped.complete(0)["g"].null_count() == data["g"].null_count()


def test_polyreg_fit_reaches_the_optimum() -> None:
    from scipy import special

    from statract.impute import _RIDGE, _multinomial

    rng = np.random.default_rng(5)
    n, k = 800, 4
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3)), rng.integers(0, 2, n)])
    truth = rng.normal(scale=1.5, size=(x.shape[1], k))
    logits = x @ truth
    prob = np.exp(logits - special.logsumexp(logits, axis=1, keepdims=True))
    y = (rng.uniform(size=(n, 1)) > np.cumsum(prob, axis=1)).sum(axis=1)
    coef = _multinomial(y, k, x)
    fitted = np.column_stack([np.zeros(n), x @ coef])
    fitted = np.exp(fitted - special.logsumexp(fitted, axis=1, keepdims=True))
    onehot = np.eye(k)[y]
    grad = x.T @ (fitted - onehot)[:, 1:] + _RIDGE * coef
    assert np.max(np.abs(grad)) < 1e-6


def test_pmm_donors_are_among_the_nearest() -> None:
    from statract.impute import _N_DONORS, _impute_pmm, _norm_draw

    for seed in range(50):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 300))
        x_obs = np.column_stack([np.ones(n), rng.normal(size=n)])
        # Rounded outcomes give tied predictions, which the matcher must handle.
        y_obs = np.round(x_obs[:, 1] + rng.normal(size=n), int(rng.integers(0, 3)))
        x_obs[:, 1] = np.round(x_obs[:, 1], 1)
        x_mis = np.column_stack([np.ones(40), rng.normal(size=40)])
        donors = _impute_pmm(y_obs, x_obs, x_mis, np.random.default_rng(seed))
        # Replay the same draws to recover the predictions the matcher used.
        beta_hat, beta_star = _norm_draw(y_obs, x_obs, np.random.default_rng(seed))
        yhat_obs, yhat_mis = x_obs @ beta_hat, x_mis @ beta_star
        k = min(_N_DONORS, n)
        for target, value in zip(yhat_mis, donors, strict=True):
            dist = np.abs(yhat_obs - target)
            cutoff = np.partition(dist, k - 1)[k - 1]
            assert np.any((y_obs == value) & (dist <= cutoff + 1e-12))
