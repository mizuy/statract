"""Regressions for the models-core audit: formula precedence, unused levels,
weighted GLM degrees of freedom, IRLS warnings, offsets in predict, and AIC."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract.models.covariance import bootstrap_covariance
from statract.models.fit import fit_glm, fit_ols
from statract.models.formula import model_matrix
from statract.models.risk import fit_risk_ratio

# M1: unary minus binds looser than ``^``.


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("I(-x^2)", lambda x: -(x**2)),
        ("I((-x)^2)", lambda x: x**2),
        ("I(x^-1)", lambda x: 1.0 / x),
        ("I(2^x^2)", lambda x: 2.0 ** (x**2)),
        ("I(-x * 3)", lambda x: -3.0 * x),
    ],
)
def test_unary_minus_precedence(term, expected) -> None:
    x = np.array([0.5, 1.0, 2.0, 3.0])
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": x})
    built = model_matrix(f"y ~ {term}", df)
    np.testing.assert_allclose(built.design.x[:, 1], expected(x))


def test_unary_minus_label_round_trips() -> None:
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    assert model_matrix("y ~ I(-x^2)", df).design.names[1] == "I(-x^2)"
    assert model_matrix("y ~ I((-x)^2)", df).design.names[1] == "I((-x)^2)"


# M2: unused levels are dropped, as drop.unused.levels.


def test_enum_unused_level_dropped() -> None:
    rng = np.random.default_rng(1)
    g = pl.Series(["b", "c"] * 30).cast(pl.Enum(["a", "b", "c"]))
    df = pl.DataFrame({"y": rng.normal(size=60), "g": g})
    assert fit_ols(df, "y", ["g"]).names == ["(Intercept)", "gc"]
    assert fit_ols(df, "y ~ g").names == ["(Intercept)", "gc"]


def test_level_only_in_dropped_rows_is_dropped() -> None:
    rng = np.random.default_rng(2)
    g = ["a", "b", "c"] * 20
    z = [None if level == "c" else float(v) for level, v in zip(g, rng.normal(size=60), strict=True)]
    df = pl.DataFrame({"y": rng.normal(size=60), "g": g, "z": z})
    column = fit_ols(df, "y", ["g", "z"])
    formula = fit_ols(df, "y ~ g + z")
    assert column.names == formula.names == ["(Intercept)", "gb", "z"]
    np.testing.assert_allclose(column.coefficients, formula.coefficients)
    # The Enum order is kept; the reference is the first remaining level.
    enum = df.with_columns(pl.col("g").cast(pl.Enum(["c", "b", "a"])))
    assert fit_ols(enum, "y", ["g", "z"]).names == ["(Intercept)", "ga", "z"]
    assert fit_ols(enum, "y ~ g + z").names == ["(Intercept)", "ga", "z"]


def test_single_used_level_raises() -> None:
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "g": pl.Series(["a"] * 3).cast(pl.Enum(["a", "b"]))})
    with pytest.raises(ValueError, match="at least two levels"):
        fit_ols(df, "y", ["g"])
    with pytest.raises(ValueError, match="at least two levels"):
        fit_ols(df, "y ~ g")


def test_constant_boolean_in_kept_rows_raises() -> None:
    df = pl.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [True] * 5 + [False], "z": [1.0, 2.0, 3.0, 4.0, 5.0, None]}
    )
    with pytest.raises(ValueError, match="at least two levels"):
        model_matrix("y ~ b + z", df)


# M3: weighted binomial and poisson follow glm.


def test_aggregated_binomial_matches_glm() -> None:
    rng = np.random.default_rng(3)
    n = 20
    x = rng.normal(size=n)
    m = rng.integers(5, 30, size=n)
    k = rng.binomial(m, 1 / (1 + np.exp(-(0.3 + 0.5 * x))))
    df = pl.DataFrame({"p": k / m, "x": x, "m": m.astype(float)})
    fit = fit_glm(df, "p ~ x", family="binomial", weights="m")
    ll = float(stats.binom.logpmf(k, m, fit.predict()).sum())
    assert fit.residual_df == n - 2
    assert fit.log_likelihood == pytest.approx(ll, rel=1e-12)
    assert fit.aic == pytest.approx(-2 * ll + 4, rel=1e-12)
    assert fit.bic == pytest.approx(-2 * ll + 2 * np.log(n), rel=1e-12)


def test_frequency_weights_binomial_loglik_unchanged() -> None:
    rng = np.random.default_rng(4)
    n = 40
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    w = rng.integers(1, 4, size=n).astype(float)
    df = pl.DataFrame({"y": y, "x": x, "w": w})
    fit = fit_glm(df, "y ~ x", family="binomial", weights="w")
    mu = fit.predict()
    bernoulli = float(np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
    assert fit.log_likelihood == pytest.approx(bernoulli, rel=1e-10)
    assert fit.residual_df == n - 2


def test_weighted_poisson_df_and_bic() -> None:
    rng = np.random.default_rng(5)
    n = 20
    df = pl.DataFrame({"y": rng.poisson(2, size=n), "x": rng.normal(size=n), "w": rng.uniform(0.2, 0.6, n)})
    fit = fit_glm(df, "y ~ x", family="poisson", weights="w")
    assert fit.residual_df == n - 2
    assert fit.bic == pytest.approx(-2 * fit.log_likelihood + 2 * np.log(n), rel=1e-12)


def test_log_binomial_df_matches_poisson_rule() -> None:
    rng = np.random.default_rng(6)
    n = 60
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.2 * np.exp(0.2 * x)).astype(float)
    w = rng.integers(1, 4, size=n).astype(float)
    df = pl.DataFrame({"y": y, "x": x, "w": w})
    fit = fit_risk_ratio(df, "y ~ x", method="log-binomial", weights="w").fit
    assert fit.residual_df == n - 2
    assert fit.bic == pytest.approx(-2 * fit.log_likelihood + 2 * np.log(n), rel=1e-12)


# M4: IRLS warnings.


def test_separation_warns() -> None:
    x = np.arange(20.0)
    df = pl.DataFrame({"y": (x > 9).astype(float), "x": x})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_glm(df, "y ~ x", family="binomial")
    messages = [str(w.message) for w in caught]
    assert "fitted probabilities numerically 0 or 1 occurred" in messages
    assert all(issubclass(w.category, RuntimeWarning) for w in caught)


def test_ordinary_logistic_fit_does_not_warn() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=100)
    y = (rng.uniform(size=100) < 1 / (1 + np.exp(-x))).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_glm(pl.DataFrame({"y": y, "x": x}), "y ~ x", family="binomial")


def test_non_convergence_warns(monkeypatch) -> None:
    from statract.models import fit as fit_module

    original = fit_module._glm_irls

    def one_step(y, x, family, prior, offset, **kwargs):
        kwargs["maxit"] = 1
        return original(y, x, family, prior, offset, **kwargs)

    monkeypatch.setattr(fit_module, "_glm_irls", one_step)
    rng = np.random.default_rng(8)
    df = pl.DataFrame({"y": rng.poisson(3, size=50), "x": rng.normal(size=50)})
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit_glm(df, "y ~ x", family="poisson")


# M5: predict(data) adds the offset when its columns are present.


@pytest.fixture(scope="module")
def exposure_frame() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    t = rng.uniform(1, 5, size=n)
    y = rng.poisson(np.exp(0.2 + 0.3 * x) * t)
    return pl.DataFrame({"y": y, "x": x, "lt": np.log(t), "t": t})


def test_predict_new_data_adds_offset_argument(exposure_frame) -> None:
    fit = fit_glm(exposure_frame, "y", ["x"], family="poisson", offset="lt")
    np.testing.assert_allclose(fit.predict(exposure_frame), fit.predict())
    without = exposure_frame.drop("lt")
    np.testing.assert_allclose(fit.predict(without, kind="link"), fit.x @ fit.coefficients)


def test_predict_new_data_adds_formula_offset(exposure_frame) -> None:
    fit = fit_glm(exposure_frame, "y ~ x + offset(log(t))", family="poisson")
    np.testing.assert_allclose(fit.predict(exposure_frame), fit.predict())
    ols = fit_ols(exposure_frame, "lt ~ x + offset(x)")
    np.testing.assert_allclose(ols.predict(exposure_frame), ols.predict())


def test_predict_drops_rows_with_missing_offset(exposure_frame) -> None:
    fit = fit_glm(exposure_frame, "y", ["x"], family="poisson", offset="lt")
    new = pl.DataFrame({"x": [0.0, 1.0, 2.0], "lt": [0.0, None, 1.0]})
    got = fit.predict(new, kind="link")
    b = fit.coefficients
    np.testing.assert_allclose(got, [b[0], b[0] + 2 * b[1] + 1.0])


# Low: the wild bootstrap is for linear models.


def test_wild_bootstrap_rejects_glm() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=50)
    df = pl.DataFrame({"y": (rng.uniform(size=50) < 0.5).astype(float), "x": x})
    fit = fit_glm(df, "y ~ x", family="binomial")
    with pytest.raises(ValueError, match="linear models"):
        bootstrap_covariance(fit, kind="wild", n=5)


def test_wild_bootstrap_gaussian_offset_matches_ols() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=50)
    off = rng.normal(size=50)
    df = pl.DataFrame({"y": 1 + x + off + rng.normal(size=50), "x": x, "off": off})
    ols = bootstrap_covariance(fit_ols(df, "y", ["x"], offset="off"), kind="wild", n=50)
    glm = bootstrap_covariance(fit_glm(df, "y", ["x"], offset="off"), kind="wild", n=50)
    np.testing.assert_allclose(glm, ols, rtol=1e-8)


# Low: AIC and BIC count the dispersion for OLS, gaussian, and gamma.


def test_ols_aic_counts_sigma() -> None:
    rng = np.random.default_rng(11)
    n = 40
    x = rng.normal(size=n)
    y = 1 + x + rng.normal(size=n)
    fit = fit_ols(pl.DataFrame({"y": y, "x": x}), "y ~ x")
    rss = float(np.sum((y - fit.x @ fit.coefficients) ** 2))
    ll = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    assert fit.log_likelihood == pytest.approx(ll, rel=1e-12)
    assert fit.aic == pytest.approx(-2 * ll + 2 * 3, rel=1e-12)
    assert fit.bic == pytest.approx(-2 * ll + np.log(n) * 3, rel=1e-12)
    glm = fit_glm(pl.DataFrame({"y": y, "x": x}), "y ~ x", family="gaussian")
    assert glm.aic == pytest.approx(fit.aic, rel=1e-12)
    assert glm.bic == pytest.approx(fit.bic, rel=1e-12)


def test_gamma_aic_counts_dispersion() -> None:
    rng = np.random.default_rng(12)
    n = 40
    x = rng.normal(size=n)
    y = rng.gamma(2.0, np.exp(0.5 + 0.2 * x) / 2.0)
    fit = fit_glm(pl.DataFrame({"y": y, "x": x}), "y ~ x", family="gamma")
    assert fit.aic == pytest.approx(-2 * fit.log_likelihood + 2 * 3, rel=1e-12)
    assert fit.bic == pytest.approx(-2 * fit.log_likelihood + np.log(n) * 3, rel=1e-12)
