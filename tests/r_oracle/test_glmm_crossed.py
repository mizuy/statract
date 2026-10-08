"""Crossed random intercepts and ar1() GLMMs against glmmTMB.

Fixture: ``fixtures/glmm_crossed.json`` from ``scripts/glmm_crossed.R``.
Patients are crossed with examiners; the ar1 cases give each examiner an
effect per year with ``sigma^2 rho^|i - j|`` covariance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_mixed

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "glmm_crossed.json").read_text())
FAMILY = {"nbinom2": "negative_binomial", "poisson": "poisson", "binomial": "binomial"}
DATA = pl.read_csv(HERE / "data" / "glmm_crossed_a.csv").with_columns(
    pl.col("arm").cast(pl.Enum(["control", "low", "high"]))
)
# binomial_ar1 sits on the rho = 1 boundary, where glmmTMB stops a little short.
BOUNDARY = {"binomial_ar1"}


@pytest.mark.parametrize("name", sorted(FIXTURE["cases"]))
def test_crossed_and_ar1_match_glmmtmb(name: str) -> None:
    case = FIXTURE["cases"][name]
    fit = fit_mixed(DATA, case["formula"], family=FAMILY[case["family"]])
    assert fit.engine == "laplace"
    assert fit.converged
    assert fit.names == case["terms"]
    assert fit.log_likelihood >= case["log_likelihood"] - 1e-9
    np.testing.assert_allclose(fit.log_likelihood, case["log_likelihood"], atol=1e-5 if name in BOUNDARY else 1e-6)
    np.testing.assert_allclose(fit.coefficients, case["coefficients"], rtol=1e-4, atol=1e-5)
    se_tol = 1e-3 if name in BOUNDARY else 1e-4
    np.testing.assert_allclose(np.sqrt(np.diag(fit.covariance)), case["std_errors"], rtol=se_tol)
    if case["theta"] is not None:
        np.testing.assert_allclose(fit.theta, case["theta"], rtol=1e-4)
    blups = fit.random_effects()
    assert [t["group"] for t in fit.random_terms] == [t["group"] for t in case["random"]]
    for term, want in zip(fit.random_terms, case["random"], strict=True):
        names = list(np.atleast_1d(want["names"]))
        assert term["names"] == names
        np.testing.assert_allclose(term["covariance"], np.atleast_2d(want["covariance"]), rtol=1e-3, atol=1e-6)
        if term["structure"] == "ar1":
            np.testing.assert_allclose(term["correlation"], want["correlation"], atol=1e-4)
        rows = blups.filter(pl.col("group") == term["group"])
        ours = np.column_stack([rows.filter(pl.col("term") == n)["blup"].to_numpy() for n in names])
        assert rows.filter(pl.col("term") == names[0])["level"].to_list() == want["levels"]
        np.testing.assert_allclose(ours, np.asarray(want["blup"]).reshape(ours.shape), atol=1e-4)


def test_variance_table_lists_every_term() -> None:
    case = FIXTURE["cases"]["nb_ar1"]
    fit = fit_mixed(DATA, case["formula"], family="negative_binomial")
    table = fit.variance_table()
    assert table["group"].unique(maintain_order=True).to_list() == ["id_patient", "e_examiner"]
    ar1 = table.filter(pl.col("structure") == "ar1")
    assert ar1.height == 6
    np.testing.assert_allclose(ar1["variance"].to_numpy(), ar1["variance"][0])
    np.testing.assert_allclose(ar1["correlation"].to_numpy(), fit.random_terms[1]["correlation"])


def test_one_term_through_the_sparse_engine_equals_the_block_engine() -> None:
    from statract._laplace import RandomTerm, fit_laplace_terms

    single = fit_mixed(DATA, "y ~ x + arm + offset(log(exposure)) + (1 | e_examiner)", family="negative_binomial")
    _, codes = np.unique(DATA["e_examiner"].to_numpy(), return_inverse=True)
    term = RandomTerm(np.ones((DATA.height, 1)), codes, int(codes.max()) + 1)
    result = fit_laplace_terms(
        DATA["y"].to_numpy(), single.x, [term], np.log(DATA["exposure"].to_numpy()), family="negative_binomial"
    )
    np.testing.assert_allclose(result.log_likelihood, single.log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(result.coefficients, single.coefficients, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.modes[0][:, 0], single.random_effects().sort("group")["blup"].to_numpy(), atol=1e-8)


def test_guards() -> None:
    with pytest.raises(ValueError, match="engine='laplace'"):
        fit_mixed(DATA, "x ~ arm + (1 | id_patient) + (1 | e_examiner)")
    with pytest.raises(ValueError, match="n_agq"):
        fit_mixed(DATA, "y ~ x + (1 | id_patient) + (1 | e_examiner)", family="poisson", n_agq=5)
    with pytest.raises(ValueError, match="time factor without an intercept"):
        fit_mixed(DATA, "y ~ x + ar1(year | e_examiner)", family="poisson")
    with pytest.raises(ValueError, match="takes a random-effect term"):
        fit_mixed(DATA, "y ~ x + ar1(year) + (1 | e_examiner)", family="poisson")
