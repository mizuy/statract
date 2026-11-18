"""Regression models must not change under transformations that leave the model unchanged.

See ``test_survival_invariance.py``. Each test changes the data in a way that
should not change the answer (row order, group labels, a location shift of a
covariate, integer weights versus repeated rows) and checks the estimates,
standard errors and tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import (
    fit_firth,
    fit_glm,
    fit_mixed,
    fit_risk_ratio,
    gam,
    gamm,
    multinomial_regression,
    ordinal_regression,
    penalized_regression,
    propensity_weights,
    smooth,
    standardize_glm,
)


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    z = rng.uniform(0, 1, n)
    g = rng.integers(0, 15, n)
    a = rng.integers(0, 2, n)
    u = rng.normal(0, 0.7, 15)[g]
    return pl.DataFrame(
        {
            "x": x,
            "z": z,
            "g": g,
            "a": a,
            "w": rng.integers(1, 4, n),
            # yb and yc do not depend on z: their smooth of z is null.
            "yb": (rng.random(n) < 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.5 * a + u)))).astype(np.int64),
            "yc": rng.poisson(np.exp(0.2 + 0.4 * x + u)).astype(float),
            "yg": 1 + x + np.sin(5 * z) + u + rng.normal(size=n),
            "yo": np.digitize(x + 0.5 * a + rng.logistic(size=n), [-1, 0.5, 2]),
        }
    )


OUTCOME = {"gaussian": "yg", "binomial": "yb", "poisson": "yc"}


def permute(df: pl.DataFrame) -> pl.DataFrame:
    return df.sample(fraction=1.0, shuffle=True, seed=11)


def repeat_by_weight(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.all().repeat_by(pl.col("w"))).explode(pl.all())


def numeric(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(pl.selectors.numeric()).to_numpy().astype(float)


def rows(fit, terms: list[str]) -> np.ndarray:
    """Numeric tidy columns for the named terms (the intercept moves under a shift)."""
    table = fit.tidy()
    return numeric(table.filter(pl.col("term").is_in(terms)))


def assert_same(a, b, rtol: float = 1e-8) -> None:
    np.testing.assert_allclose(np.asarray(b, float), np.asarray(a, float), rtol=rtol, atol=rtol)


# GLM


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_glm(data, family):
    y = OUTCOME[family]
    base = fit_glm(data, y, ["x", "a"], family=family, weights="w")
    assert_same(
        numeric(base.tidy()),
        numeric(fit_glm(permute(data), y, ["x", "a"], family=family, weights="w").tidy()),
    )
    dup = fit_glm(repeat_by_weight(data), y, ["x", "a"], family=family)
    assert_same(base.coefficients, dup.coefficients, rtol=1e-7)
    shifted = fit_glm(data.with_columns(pl.col("x") + 140), y, ["x", "a"], family=family, weights="w")
    assert_same(rows(base, ["x", "a"]), rows(shifted, ["x", "a"]), rtol=1e-6)
    zero = data.with_columns(pl.col("w").cast(pl.Float64))
    padded = pl.concat([zero, zero.head(5).with_columns(w=pl.lit(0.0))])
    assert_same(
        numeric(base.tidy()),
        numeric(fit_glm(padded, y, ["x", "a"], family=family, weights="w").tidy()),
    )


# Mixed models


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_mixed(data, family):
    y = OUTCOME[family]

    def run(df):
        return fit_mixed(df, y, ["x", "a"], groups="g", family=family)

    base = run(data)
    assert_same(numeric(base.tidy()), numeric(run(permute(data)).tidy()), rtol=1e-6)
    relabelled = data.with_columns(g=pl.format("ward-{}", pl.col("g") * 7 + 2))
    assert_same(numeric(base.tidy()), numeric(run(relabelled).tidy()), rtol=1e-6)
    # A realistic location (age, sodium). A shift of 1000 changes the GLMM
    # standard errors by up to 80 %; see the PR for that limit.
    assert_same(
        rows(base, ["x", "a"]),
        rows(run(data.with_columns(pl.col("x") + 140)), ["x", "a"]),
        rtol=1e-4,
    )


def test_mixed_random_slope_row_order(data):
    def run(df):
        return fit_mixed(df, "yg", ["x", "a"], groups="g", slopes=["x"])

    assert_same(numeric(run(data).tidy()), numeric(run(permute(data)).tidy()), rtol=1e-6)


# GAM and GAMM


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_gam(data, family):
    y = OUTCOME[family]

    def run(df, **kw):
        return gam(df, y, [smooth("z")], predictors=["x"], family=family, **kw)

    base = run(data)
    table = numeric(base.smooth_table())
    assert_same(numeric(base.tidy()), numeric(run(permute(data)).tidy()), rtol=1e-5)
    assert_same(table, numeric(run(permute(data)).smooth_table()), rtol=1e-4)
    assert_same(
        table,
        numeric(run(data.with_columns(pl.col("z") * 1000)).smooth_table()),
        rtol=1e-4,
    )
    assert_same(
        table,
        numeric(run(data.with_columns(pl.col("z") + 1000)).smooth_table()),
        rtol=1e-4,
    )
    if family != "gaussian":
        # With a fixed scale, prior weights equal repeated rows. (The Gaussian
        # scale is estimated from n, so REML picks a different penalty there.)
        assert_same(
            rows(run(data, weights="w"), ["x"]),
            rows(run(repeat_by_weight(data)), ["x"]),
            rtol=1e-4,
        )


@pytest.mark.parametrize("basis", ["tp", "cr", "ps", "cc"])
def test_gam_bases(data, basis):
    def run(df):
        return gam(df, "yg", [smooth("z", basis=basis)], predictors=["x"])

    base = numeric(run(data).smooth_table())
    assert_same(base, numeric(run(permute(data)).smooth_table()), rtol=1e-4)
    assert_same(
        base,
        numeric(run(data.with_columns(pl.col("z") * 1000)).smooth_table()),
        rtol=1e-4,
    )


@pytest.fixture(scope="module")
def null_smooth_data() -> pl.DataFrame:
    """Data where gamm's REML puts the smooth variance of z at zero."""
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    z = rng.uniform(0, 1, n)
    g = rng.integers(0, 15, n)
    a = rng.integers(0, 2, n)
    rng.integers(1, 4, n)
    u = rng.normal(0, 0.7, 15)[g]
    yb = (rng.random(n) < 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.5 * a + u)))).astype(np.int64)
    return pl.DataFrame({"x": x, "z": z, "g": g, "a": a, "yb": yb})


@pytest.mark.parametrize("predictors", [None, ["x"], ["a"]])
def test_gamm_null_smooth(null_smooth_data, predictors):
    """A smooth penalized to its linear part has edf 1, not 0, and no false signal.

    This used to raise a singular-matrix error with a parametric term, and to
    report edf near 0 with p near 0 without one.
    """
    fit = gamm(
        null_smooth_data,
        "yb",
        [smooth("z", basis="cr")],
        random="(1 | g)",
        predictors=predictors,
    )
    row = fit.smooth_table().row(0, named=True)
    assert row["variance"] < 1e-8
    assert row["edf"] == pytest.approx(1.0, abs=1e-3)
    assert row["ref_df"] == pytest.approx(1.0, abs=1e-3)
    assert 0.05 < row["p_value"] < 1.0


def test_gamm_labels_and_row_order(data):
    def run(df):
        return gamm(df, "yb", [smooth("z", basis="cr")], random="(1 | g)", predictors=["x"])

    base = run(data)
    for other in (
        permute(data),
        data.with_columns(g=pl.format("ward-{}", pl.col("g"))),
    ):
        fit = run(other)
        assert_same(numeric(base.tidy()), numeric(fit.tidy()), rtol=1e-5)
        assert_same(numeric(base.smooth_table()), numeric(fit.smooth_table()), rtol=1e-5)


# Firth, multinomial, ordinal, penalized


def test_firth(data):
    base = fit_firth(data, "yb", ["x", "a"])
    assert_same(
        numeric(base.tidy()),
        numeric(fit_firth(permute(data), "yb", ["x", "a"]).tidy()),
        rtol=1e-7,
    )
    assert_same(
        numeric(fit_firth(data, "yb", ["x", "a"], weights="w").tidy()),
        numeric(fit_firth(repeat_by_weight(data), "yb", ["x", "a"]).tidy()),
        rtol=1e-6,
    )


@pytest.mark.parametrize("location", [140.0, 2020.0])
def test_firth_covariate_far_from_zero(data, location):
    """Sodium (~140 mmol/L) or calendar year: tidy() used to fail with a singular matrix."""
    base = fit_firth(data, "yb", ["x", "a"])
    shifted = fit_firth(data.with_columns(pl.col("x") + location), "yb", ["x", "a"])
    assert shifted.converged
    assert_same(rows(base, ["x", "a"]), rows(shifted, ["x", "a"]), rtol=1e-6)


def test_multinomial(data):
    def run(df, **kw):
        return numeric(multinomial_regression(df, "yo ~ x + a", **kw).tidy())

    base = run(data)
    assert_same(base, run(permute(data)), rtol=1e-7)
    assert_same(base, run(data.with_columns(yo=pl.format("grade {}", pl.col("yo")))), rtol=1e-7)
    assert_same(run(data, weights="w"), run(repeat_by_weight(data)), rtol=1e-7)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_ordinal(data, link):
    def run(df, **kw):
        return ordinal_regression(df, "yo ~ x + a", link=link, **kw)

    base = run(data)
    assert_same(numeric(base.tidy()), numeric(run(permute(data)).tidy()), rtol=1e-7)
    assert_same(
        numeric(run(data, weights="w").tidy()),
        numeric(run(repeat_by_weight(data)).tidy()),
        rtol=1e-7,
    )
    assert_same(
        rows(base, ["x", "a"]),
        rows(run(data.with_columns(pl.col("x") + 1000)), ["x", "a"]),
        rtol=1e-6,
    )


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_penalized(data, family):
    y = OUTCOME[family]

    def run(df, **kw):
        path = penalized_regression(df, f"{y} ~ x + a + z", family=family, lambdas=[0.2, 0.05, 0.01], **kw)
        return numeric(path.tidy(0.05))

    assert_same(run(data), run(permute(data)), rtol=1e-6)
    assert_same(run(data, weights="w"), run(repeat_by_weight(data)), rtol=1e-6)


# Causal estimators


def test_standardize_glm(data):
    def run(df, **kw):
        return standardize_glm(df, "yb ~ a + x", values={"a": [0, 1]}, **kw)

    base = run(data)
    for other in (run(permute(data)), run(data.with_columns(pl.col("x") + 1000))):
        assert_same(base.estimates, other.estimates, rtol=1e-7)
        assert_same(base.covariance, other.covariance, rtol=1e-6)
    assert_same(
        run(data, weights="w").estimates,
        run(repeat_by_weight(data)).estimates,
        rtol=1e-7,
    )


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATO"])
def test_propensity_weights_row_order(data, estimand):
    a = propensity_weights(data, "a ~ x + z", estimand=estimand)
    b = propensity_weights(permute(data), "a ~ x + z", estimand=estimand)
    assert_same(np.sort(a.weights), np.sort(b.weights))
    assert_same(np.sort(a.ps), np.sort(b.ps))


def test_risk_ratio(data):
    base = fit_risk_ratio(data, "yb", ["a", "x"])
    assert_same(
        numeric(base.tidy()),
        numeric(fit_risk_ratio(permute(data), "yb", ["a", "x"]).tidy()),
        rtol=1e-7,
    )
    assert_same(
        rows(base, ["a", "x"]),
        rows(
            fit_risk_ratio(data.with_columns(pl.col("x") + 1000), "yb", ["a", "x"]),
            ["a", "x"],
        ),
        rtol=1e-6,
    )
