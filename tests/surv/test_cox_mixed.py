"""cox_mixed checks that do not need R (the coxme comparison is in tests/r_oracle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cox_mixed, cox_ph, fit_mixed
from statract.models.formula import model_matrix

DATA = Path(__file__).resolve().parents[1] / "r_oracle" / "data"
ARM = pl.Enum(["control", "low", "high"])


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.read_csv(DATA / "coxme_a.csv").with_columns(pl.col("arm").cast(ARM))


@pytest.fixture(scope="module")
def fit(data):
    return cox_mixed(data, "Surv(time, status) ~ x + arm + (1 | hosp)")


def test_nested_groups_expand_like_lme4(data) -> None:
    built = model_matrix("Surv(time, status) ~ x + (1 | hosp/ward)", data)
    assert [effect.group for effect in built.random_effects] == ["hosp", "hosp:ward"]
    built = model_matrix("Surv(time, status) ~ x + (1 | hosp:ward)", data)
    assert [effect.group for effect in built.random_effects] == ["hosp:ward"]


def test_zero_variance_is_the_cox_model(data) -> None:
    mixed = cox_mixed(data, "Surv(time, status) ~ x + arm + (1 | hosp)", variances={"hosp": 0.0})
    plain = cox_ph(data, "Surv(time, status) ~ x + arm")
    np.testing.assert_allclose(mixed.coefficients, plain.coefficients, rtol=1e-8)
    np.testing.assert_allclose(mixed.covariance, plain.covariance, rtol=1e-6)
    np.testing.assert_allclose(mixed.log_likelihood, plain.log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(mixed.df, len(plain.coefficients), rtol=1e-8)
    assert np.all(mixed.random_terms[0]["blups"] == 0)


def test_estimate_maximizes_the_integrated_likelihood(data, fit) -> None:
    variance = fit.random_terms[0]["covariance"][0, 0]
    for factor in (0.9, 1.1):
        other = cox_mixed(data, "Surv(time, status) ~ x + arm + (1 | hosp)", variances={"hosp": variance * factor})
        assert other.log_likelihood < fit.log_likelihood


def test_predict_adds_the_blups(data, fit) -> None:
    fixed = fit.x @ fit.coefficients
    np.testing.assert_allclose(fit.predict(random=False), fixed)
    np.testing.assert_allclose(fit.predict(data), fit.predict())
    blups = dict(zip(fit.random_terms[0]["levels"], fit.random_terms[0]["blups"][:, 0], strict=True))
    want = fixed + np.array([blups[h] for h in data["hosp"].to_list()])
    np.testing.assert_allclose(fit.predict(), want)
    new = data.head(3).with_columns(pl.lit("unseen").alias("hosp"))
    np.testing.assert_allclose(fit.predict(new), fixed[:3])
    np.testing.assert_allclose(fit.predict(kind="risk"), np.exp(fit.predict()))


def test_tables(fit) -> None:
    table = fit.tidy(exponentiate=True)
    assert table["term"].to_list() == ["x", "armlow", "armhigh"]
    np.testing.assert_allclose(table["estimate"].to_numpy(), np.exp(fit.coefficients))
    variance = fit.variance_table()
    assert variance["group"].to_list() == ["hosp"]
    assert variance["term"].to_list() == ["(Intercept)"]
    ranef = fit.random_effects()
    assert ranef.columns == ["group", "level", "term", "blup"]
    assert ranef.height == 30
    glance = fit.glance()
    np.testing.assert_allclose(glance["aic"][0], -2 * fit.log_likelihood + 2 * 4)


def test_correlated_slope_covariance_is_positive_semidefinite(data) -> None:
    slope = cox_mixed(data, "Surv(time, status) ~ x + arm + (1 + x | hosp)")
    cov = slope.random_terms[0]["covariance"]
    assert cov.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)
    table = slope.variance_table()
    assert table["term"].to_list() == ["(Intercept)", "x"]
    assert -1 <= table["correlation"][1] <= 1
    assert slope.random_effects().height == 60


def test_errors(data) -> None:
    with pytest.raises(ValueError, match="needs a random effect"):
        cox_mixed(data, "Surv(time, status) ~ x")
    with pytest.raises(ValueError, match=r"\|\|"):
        cox_mixed(data, "Surv(time, status) ~ x + (1 + x || hosp)")
    with pytest.raises(ValueError, match="ties"):
        cox_mixed(data, "Surv(time, status) ~ x + (1 | hosp)", ties="exact")
    with pytest.raises(ValueError, match="one entry for each group"):
        cox_mixed(data, "Surv(time, status) ~ x + (1 | hosp)", variances={"ward": 0.1})
    with pytest.raises(ValueError, match="2 x 2"):
        cox_mixed(data, "Surv(time, status) ~ x + (1 + x | hosp)", variances=[0.1])
    with pytest.raises(ValueError, match="cox_mixed only"):
        fit_mixed(data.with_columns(y=pl.col("status")), "y ~ x + (1 | hosp/ward)", family="binomial")
