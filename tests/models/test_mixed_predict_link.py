import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract.models.mixed import fit_mixed


def _data(seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 400
    g = rng.integers(0, 20, n)
    x = rng.normal(size=n)
    b = rng.normal(0, 0.5, 20)
    return pl.DataFrame({"x": x, "g": g}), b[g]


def test_probit_predict_uses_probit_with_and_without_data():
    df, b = _data()
    rng = np.random.default_rng(1)
    eta = -0.3 + 0.8 * df["x"].to_numpy() + b
    df = df.with_columns(y=pl.Series((rng.uniform(size=len(eta)) < stats.norm.cdf(eta)).astype(int)))
    fit = fit_mixed(df, "y ~ x + (1|g)", family="binomial", link="probit")
    expected = stats.norm.cdf(fit.x @ fit.coefficients)
    np.testing.assert_allclose(fit.predict(kind="response"), expected, rtol=1e-10)
    np.testing.assert_allclose(fit.predict(df, kind="response"), expected, rtol=1e-8)


@pytest.mark.parametrize(("link", "inverse"), [("log", np.exp), (None, lambda eta: 1 / eta)])
def test_gamma_predict_uses_the_fitted_link(link, inverse):
    df, b = _data()
    rng = np.random.default_rng(2)
    x = df["x"].to_numpy()
    mu = np.exp(0.5 + 0.3 * x + 0.5 * b) if link == "log" else 1 / (1 + 0.2 * x**2)
    df = df.with_columns(y=pl.Series(rng.gamma(2.0, mu / 2.0)))
    fit = fit_mixed(df, "y ~ x + (1|g)", family="gamma", link=link)
    expected = inverse(fit.x @ fit.coefficients)
    np.testing.assert_allclose(fit.predict(kind="response"), expected, rtol=1e-10)
    np.testing.assert_allclose(fit.predict(df, kind="response"), expected, rtol=1e-8)
