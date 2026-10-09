"""Exact Laplace gradient, dense inner solve, and Newton polish in ``_laplace``.

The exact gradient must agree with finite differences of the log-likelihood,
models it does not cover must keep the finite-difference path, and fits must
match the finite-difference path and the values from before the change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import _laplace as L
from statract import fit_mixed, gamm, smooth

DATA = Path(__file__).resolve().parent / "r_oracle" / "data"
ARM = pl.Enum(["control", "low", "high"])
FAMILIES = ["poisson", "negative_binomial", "binomial"]


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(DATA / f"{name}.csv").with_columns(pl.col("arm").cast(ARM))


def _outcome(family: str, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if family == "binomial":
        return rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    if family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    return rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta))).astype(float)


def _fd_gradient(prob, phi):
    return L._gradient(prob.loglik, phi, 1e-5 * np.maximum(1.0, np.abs(phi)))


@pytest.mark.parametrize("slope", [False, True])
@pytest.mark.parametrize("family", FAMILIES)
def test_one_factor_gradient_matches_finite_differences(family: str, slope: bool) -> None:
    rng = np.random.default_rng(1)
    n, m = 400, 12
    group = rng.integers(0, m, n)
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)])
    offset = rng.normal(0, 0.2, n)
    z = rng.normal(1.0, 0.5, (n, 1)) if slope else np.ones((n, 1))
    eta = x @ [0.3, 0.5, -0.4] + rng.normal(0, 0.7, m)[group] * z[:, 0] + offset
    prob = L._Problem(_outcome(family, eta, rng), x, z, group, offset, family, 1, None, None)
    assert prob.has_gradient()
    phi = L._start(prob) + 0.1
    value, grad = prob.loglik_grad(phi)
    np.testing.assert_allclose(value, prob.loglik(phi), rtol=1e-14)
    np.testing.assert_allclose(grad, _fd_gradient(prob, phi), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("family", FAMILIES)
def test_dense_terms_gradient_matches_finite_differences(family: str) -> None:
    """A smooth's iid term plus two crossed factors, one with a slope column."""
    rng = np.random.default_rng(2)
    n = 500
    g1, g2 = rng.integers(0, 8, n), rng.integers(0, 5, n)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    zs = rng.normal(size=(n, 4))
    offset = rng.normal(0, 0.2, n)
    eta = x @ [0.2, 0.4] + rng.normal(0, 0.6, 8)[g1] + rng.normal(0, 0.4, 5)[g2] + zs @ [0.3, -0.2, 0.1, 0.0] + offset
    terms = [
        L.RandomTerm(zs, np.zeros(n, dtype=np.int64), 1, "iid"),
        L.RandomTerm(np.ones((n, 1)), g1, 8),
        L.RandomTerm(rng.normal(1.0, 0.3, (n, 1)), g2, 5),
    ]
    prob = L._SparseProblem(_outcome(family, eta, rng), x, terms, offset, family, None, None)
    assert prob.use_dense and prob.has_gradient()
    phi = L._start(prob) + 0.1
    _, grad = prob.loglik_grad(phi)
    np.testing.assert_allclose(grad, _fd_gradient(prob, phi), rtol=1e-7, atol=1e-7)


def test_uncovered_models_keep_finite_differences() -> None:
    rng = np.random.default_rng(3)
    n = 200
    y = rng.poisson(2.0, n).astype(float)
    x = np.ones((n, 1))
    group = rng.integers(0, 10, n)
    one = np.ones((n, 1))
    assert L._Problem(y, x, one, group, None, "poisson", 1, None, None).has_gradient()
    # adaptive quadrature, zero part, correlated slopes
    assert not L._Problem(y, x, one, group, None, "poisson", 9, None, None).has_gradient()
    assert not L._Problem(y, x, one, group, None, "poisson", 1, "inflated", one).has_gradient()
    assert not L._Problem(y, x, np.column_stack([one, rng.normal(size=n)]), group, None, "poisson", 1, None, None).has_gradient()
    # ar1, a zero part, and a system too large for the dense solve
    years = rng.normal(size=(n, 3))
    assert not L._SparseProblem(y, x, [L.RandomTerm(years, group, 10, "ar1")], None, "poisson", None, None).has_gradient()
    term = L.RandomTerm(one, group, 10)
    assert not L._SparseProblem(y, x, [term], None, "poisson", "hurdle", one).has_gradient()
    big = L.RandomTerm(one, np.arange(n), n)
    assert not L._SparseProblem(y, x, [big, term], None, "poisson", None, None).has_gradient()


def test_dense_and_sparse_inner_solves_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    n = 300
    g1, g2 = rng.integers(0, 12, n), rng.integers(0, 4, n)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    eta = 0.5 * x[:, 1] + rng.normal(0, 0.5, 12)[g1] + rng.normal(0, 0.5, 4)[g2]
    y = rng.poisson(np.exp(eta)).astype(float)
    terms = [L.RandomTerm(np.ones((n, 1)), g1, 12), L.RandomTerm(rng.normal(size=(n, 2)), g2, 4)]
    phi = np.array([0.1, 0.4, -0.3, -0.5, 0.2, 0.1])
    values = {}
    for mode in ["matmul", "bincount", "blocks"]:
        if mode == "blocks":
            monkeypatch.setattr(L, "_DENSE_MAX", 0)
        prob = L._SparseProblem(y, x, terms, None, "poisson", None, None)
        if mode == "bincount":
            prob.dense_matmul = False
            prob.dense_key = (prob.cols[:, :, None] * prob.n_u + prob.cols[:, None, :]).ravel()
        assert prob.use_dense == (mode != "blocks")
        values[mode] = (prob.loglik(phi), prob.u.copy())
    for mode in ["bincount", "blocks"]:
        np.testing.assert_allclose(values[mode][0], values["matmul"][0], rtol=1e-13)
        np.testing.assert_allclose(values[mode][1], values["matmul"][1], atol=1e-10)


def _summary(fit) -> dict:
    return {
        "coef": np.asarray(fit.coefficients),
        "se": np.sqrt(np.diag(fit.covariance)),
        "ll": fit.log_likelihood,
        "re": np.ravel(fit.group_covariance),
        "theta": fit.theta,
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_and_finite_difference_paths_agree(family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data("glmm_binary_a" if family == "binomial" else "glmm_count_a")
    formula = "y ~ x + arm + (1 | site)" if family == "binomial" else "y ~ x + arm + offset(log(years)) + (1 | site)"
    exact = _summary(fit_mixed(data, formula, family=family))
    monkeypatch.setattr(L._Problem, "has_gradient", lambda self: False)
    fd = _summary(fit_mixed(data, formula, family=family))
    np.testing.assert_allclose(exact["coef"], fd["coef"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(exact["ll"], fd["ll"], atol=1e-9)
    np.testing.assert_allclose(exact["re"], fd["re"], rtol=1e-6)
    np.testing.assert_allclose(exact["se"], fd["se"], rtol=1e-5)
    if family == "negative_binomial":
        np.testing.assert_allclose(exact["theta"], fd["theta"], rtol=1e-5)


# Fits before the exact gradient, dense solve, and polish stop (statract 2ec4525).
BEFORE = {
    "poisson": {
        "coef": [0.34602925594182465, 0.3731945708019772, 0.2544137806003867, -0.4334682535045605],
        "ll": -1080.7555370670325,
        "re": [0.3026591790200915],
        "se": [0.10971010663908068, 0.02493407331838778, 0.05661146424943165, 0.06449776010524813],
    },
    "negative_binomial": {
        "coef": [0.3452297398778599, 0.3742838833090242, 0.25510205831227567, -0.4328787756219927],
        "ll": -1080.6223502372175,
        "re": [0.3020352753904081],
        "se": [0.10986927632050834, 0.02554246960659448, 0.05769045291725541, 0.06537132275137639],
        "theta": 136.97433362300478,
    },
    "slope": {
        "coef": [0.5152702663438112, 0.3289894457482519, 0.42641387106554846, -0.2897842031476983],
        "ll": -1755.2589296624953,
        "re": [0.1810595049046581, 0.03721231055248078, 0.03721231055248078, 0.11394031527338905],
        "se": [0.10517513027437772, 0.07789666364100849, 0.08537517630056093, 0.09289784580919512],
        "theta": 1.734952134322654,
    },
    "binomial": {
        "coef": [-0.1951074916486517, 0.5003030380309524, 0.5377340962063221, -0.8057179127686304],
        "ll": -443.8493171542068,
        "re": [0.4866502527297475],
        "se": [0.18238250390475733, 0.09079466094464918, 0.20995796591465254, 0.20689803899422565],
    },
}


@pytest.mark.parametrize("case", sorted(BEFORE))
def test_glmm_fits_unchanged(case: str) -> None:
    want = BEFORE[case]
    if case == "binomial":
        fit = fit_mixed(_data("glmm_binary_a"), "y ~ x + arm + (1 | site)", family="binomial")
    elif case == "slope":
        fit = fit_mixed(_data("glmm_count_b"), "y ~ x + arm + offset(log(years)) + (1 + x | site)", family="negative_binomial")
    else:
        fit = fit_mixed(_data("glmm_count_a"), "y ~ x + arm + offset(log(years)) + (1 | site)", family=case)
    got = _summary(fit)
    np.testing.assert_allclose(got["coef"], want["coef"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["ll"], want["ll"], atol=1e-9)
    np.testing.assert_allclose(got["re"], want["re"], rtol=1e-6)
    np.testing.assert_allclose(got["se"], want["se"], rtol=1e-5)
    if "theta" in want:
        # theta near 137 is close to the Poisson limit, where the likelihood is flat in it.
        np.testing.assert_allclose(got["theta"], want["theta"], rtol=1e-5)


def test_gamm_fit_unchanged() -> None:
    data = pl.read_csv(DATA / "gamm_binary_a.csv")
    fit = gamm(data, "y", [smooth("pre_size_mm", k=10, basis="tp"), smooth("age", k=6, basis="cr")], random="(1 | examiner)")
    coef = [
        0.18816186087423942, 0.08307939831547369, 0.2431318333844602, 0.06193222004390551, 0.34610860482906813,
        0.09539063356147401, 0.45488330873118, -0.09860932269527879, 0.8842919841810278, -0.46079490841815296,
        0.028537982666267083, 0.1945930570855951, 0.31338802620091505, 0.37228125624762576, 0.27701992308654355,
    ]  # fmt: skip
    np.testing.assert_allclose(fit.coefficients, coef, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.log_likelihood, -515.2258842287554, atol=1e-9)
    np.testing.assert_allclose(fit.variance_table()["variance"][0], 0.5645224316912025, rtol=1e-6)
    se = np.sqrt(np.diag(fit.covariance))
    assert se.shape == (15,)
    np.testing.assert_allclose(se[[0, 8, 14]], [0.1688686305688735, 0.5940835695627047, 0.39045116137305924], rtol=1e-5)
