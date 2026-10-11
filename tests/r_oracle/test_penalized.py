"""Penalized regression against ``glmnet`` 4.1-8 and ``cv.glmnet``.

Fixture: ``fixtures/penalized.json`` from ``scripts/penalized.R``.

statract's solver was written from the papers, not from glmnet's code, so the
two agree only to glmnet's own convergence error. glmnet's default
``thresh = 1e-7`` stops coordinate descent once no update moves the objective
by more than 1e-7 of the null deviance, which leaves standardized
coefficients up to about ``sqrt(1e-7) = 3e-4`` from the optimum. statract
solves each problem 1e-4 times tighter. So the coefficients are compared
relative to the largest coefficient at the same lambda, at 3e-3, and
``test_objective_not_worse_than_glmnet`` checks the other direction: at
every lambda, statract's penalized objective is no larger than the objective
at glmnet's coefficients. On the wide samples (more columns than rows)
glmnet's error is much larger (its fit there can have more nonzero
coefficients than rows, which no lasso solution has), so those cases are
compared at 0.1.

glmnet's rule for ending the automatic path early is not documented in
detail. statract stops when the fraction of deviance explained grows by less
than 1e-5 of itself, which gives glmnet's path length for gaussian (within
one lambda) and a longer path for the other families. Paths are compared on
the lambdas they share.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cv_penalized, penalized_regression

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "penalized.json").read_text())
PATHS = sorted(FIXTURE["paths"])
CVS = sorted(FIXTURE["cv"])

COEF_TOL = 3e-3  # relative to the largest coefficient at each lambda
WIDE_TOL = 0.1
DEV_RATIO_ATOL = 1e-4
WIDE_DEV_RATIO_ATOL = 1e-3


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


def _wide(case: dict) -> bool:
    return case["data"] == "penalized_b"


def _kwargs(case: dict) -> dict:
    out = {"family": case["family"]}
    if case.get("alpha") is not None:
        out["alpha"] = case["alpha"]
    if case.get("weights") is not None:
        out["weights"] = case["weights"]
    if case.get("penalty") is not None:
        out["penalty_factor"] = case["penalty"]
    if case.get("standardize") is not None:
        out["standardize"] = case["standardize"]
    return out


@cache
def _auto(name: str):
    case = FIXTURE["paths"][name]
    return penalized_regression(_data(case["data"]), case["formula"], **_kwargs(case))


def _lambda_rtol(case: dict) -> float:
    # lambda_max is exact for gaussian. For the other families it comes from
    # the score at the null fit (intercept and unpenalized columns), which
    # glmnet solves iteratively, so its lambdas carry that error (up to 4e-6).
    return 1e-8 if case["family"] == "gaussian" else 1e-5


def _close_by_column(got: np.ndarray, want: np.ndarray, tol: float) -> None:
    scale = np.maximum(np.abs(want).max(axis=0, keepdims=True), 1e-2)
    np.testing.assert_array_less(np.abs(got - want) / scale, tol)


def _check_path(fit, want: dict, case: dict, *, automatic: bool) -> None:
    lam = np.asarray(want["lambda"])
    family = case["family"]
    if automatic:
        if family == "gaussian":
            assert abs(fit.lambdas.size - lam.size) <= 1
        else:
            assert fit.lambdas.size >= lam.size
    else:
        assert fit.lambdas.size == lam.size
    m = min(fit.lambdas.size, lam.size)
    np.testing.assert_allclose(fit.lambdas[:m], lam[:m], rtol=_lambda_rtol(case))
    beta = np.asarray(want["beta"], dtype=float).reshape(-1, lam.size)[:, :m]
    wide = _wide(case)
    tol = WIDE_TOL if wide else COEF_TOL
    _close_by_column(fit.coefs[:, :m], beta, tol)
    if family != "cox":
        scale = np.maximum(np.abs(beta).max(axis=0), 1.0)
        np.testing.assert_array_less(np.abs(fit.intercepts[:m] - np.asarray(want["a0"])[:m]) / scale, tol)
    df = np.asarray(want["df"])[:m]
    if automatic and case.get("alpha") == 0:
        # glmnet reports its first ridge lambda with coefficients of 1e-37,
        # counted as nonzero; statract reports the null model there (zeros).
        assert fit.df[0] == 0
        df, ours = df[1:], fit.df[1:m]
    else:
        ours = fit.df[:m]
    if wide:
        # glmnet's wide fits are far from converged; compare the support size loosely.
        assert np.all(np.abs(ours - df) <= 8)
    else:
        np.testing.assert_array_equal(ours, df)
    atol = WIDE_DEV_RATIO_ATOL if wide else DEV_RATIO_ATOL
    np.testing.assert_allclose(fit.dev_ratio[:m], np.asarray(want["dev_ratio"])[:m], rtol=0, atol=atol)
    np.testing.assert_allclose(fit.null_deviance, want["nulldev"], rtol=1e-10)


@pytest.mark.parametrize("name", PATHS)
def test_automatic_path(name: str) -> None:
    case = FIXTURE["paths"][name]
    fit = _auto(name)
    assert fit.names == case["names"]
    _check_path(fit, case["auto"], case, automatic=True)


@pytest.mark.parametrize("name", PATHS)
def test_user_lambdas(name: str) -> None:
    case = FIXTURE["paths"][name]
    fit = penalized_regression(_data(case["data"]), case["formula"], lambdas=case["grid"], **_kwargs(case))
    _check_path(fit, case["user"], case, automatic=False)


@pytest.mark.parametrize("name", PATHS)
def test_r_lambdas_passed_back(name: str) -> None:
    """R's automatic lambdas given back as ``lambda=`` (no early stop).

    The first automatic lambda is ``lambda_max`` itself, where the top
    coefficient is 0 only up to rounding, so both sides start from the
    second one.
    """
    case = FIXTURE["paths"][name]
    lambdas = case["auto"]["lambda"][1:]
    fit = penalized_regression(_data(case["data"]), case["formula"], lambdas=lambdas, **_kwargs(case))
    _check_path(fit, case["refit"], case, automatic=False)


def _objective(case: dict, lam: float, a0: float, beta: np.ndarray) -> float:
    """glmnet's documented objective, written out here independently of the solver."""
    data = _data(case["data"])
    fit = _auto_names(case)
    d = fit._data
    x, y, off = d.x, d.y, (np.zeros(d.y.size) if d.offset is None else d.offset)
    w = d.w / d.w.sum()
    alpha = 1.0 if case.get("alpha") is None else case["alpha"]
    intercept = fit.intercept
    if case.get("standardize") is not False:
        mean = w @ x
        s = np.sqrt(w @ (x - mean) ** 2) if intercept else np.sqrt(w @ x**2 - mean**2)
    else:
        s = np.ones(x.shape[1])
    vp = d.pen_factor * x.shape[1] / d.pen_factor.sum()
    eta = x @ beta + (a0 if intercept else 0.0) + off
    family = case["family"]
    gam = beta * s
    if family == "gaussian":
        t = y - off
        ys = np.sqrt(w @ (t - w @ t) ** 2) if intercept else np.sqrt(w @ t**2)
        loss = 0.5 * float(w @ (y - eta) ** 2)
        pen = lam * float(vp @ ((1 - alpha) / 2 * gam**2 / ys + alpha * np.abs(gam)))
        return loss + pen
    if family == "binomial":
        loss = float(w @ (np.logaddexp(0.0, eta) - y * eta))
    elif family == "poisson":
        loss = float(w @ (np.exp(eta) - y * eta))
    else:
        time = np.asarray(data[case["formula"].split("(")[1].split(",")[0]].to_numpy(), dtype=float)[
            fit.design.row_index
        ]
        loss = 0.0
        for t in np.unique(time[y == 1]):
            dead = (time == t) & (y == 1)
            risk = time >= t
            loss -= float(w[dead] @ eta[dead]) - float(w[dead].sum()) * np.log(float(w[risk] @ np.exp(eta[risk])))
    return loss + lam * float(vp @ ((1 - alpha) / 2 * gam**2 + alpha * np.abs(gam)))


def _auto_names(case: dict):
    name = next(k for k, v in FIXTURE["paths"].items() if v is case)
    return _auto(name)


@pytest.mark.parametrize("name", PATHS)
def test_objective_not_worse_than_glmnet(name: str) -> None:
    """At each of R's lambdas, statract's solution is at least as good as glmnet's."""
    case = FIXTURE["paths"][name]
    want = case["refit"]
    lam = np.asarray(want["lambda"])
    fit = penalized_regression(_data(case["data"]), case["formula"], lambdas=lam, **_kwargs(case))
    beta_r = np.asarray(want["beta"], dtype=float).reshape(-1, lam.size)
    a0_r = np.asarray(want["a0"]) if case["family"] != "cox" else np.zeros(lam.size)
    for j in range(lam.size):
        ours = _objective(case, lam[j], fit.intercepts[j], fit.coefs[:, j])
        theirs = _objective(case, lam[j], a0_r[j], beta_r[:, j])
        assert ours <= theirs + 1e-10 * abs(theirs)


@pytest.mark.parametrize("name", PATHS)
def test_coef_and_predict_between_lambdas(name: str) -> None:
    case = FIXTURE["paths"][name]
    fit = _auto(name)
    s = np.asarray(case["s"])
    coef = fit.coef(s)
    want = np.asarray(case["coef_s"], dtype=float).reshape(-1, s.size)
    if case["family"] != "cox" and not fit.intercept:
        # R keeps a zero "(Intercept)" row when intercept = FALSE.
        assert np.all(want[0] == 0.0)
        want = want[1:]
    # The last s is below R's path end, where R clamps to its last lambda.
    # statract's path ends at the same lambda for gaussian but runs further
    # for the other families, so that column is compared only for gaussian.
    end = np.asarray(case["auto"]["lambda"])[-1]
    inside = (s >= end) | (fit.lambdas[-1] >= end * (1 - 1e-8))
    tol = WIDE_TOL if _wide(case) else COEF_TOL
    if case["family"] != "cox" and fit.intercept:
        # The intercept is compared on the scale of the slopes or 1.
        scale = np.maximum(np.abs(want).max(axis=0), 1.0)
        np.testing.assert_array_less((np.abs(coef - want) / scale)[:, inside], tol)
    else:
        _close_by_column(coef[:, inside], want[:, inside], tol)
    link = fit.predict(None, s, kind="link")[:20]
    want_link = np.asarray(case["link"]).reshape(20, -1)
    link_scale = np.maximum(np.abs(want_link).max(), 1.0)
    np.testing.assert_array_less((np.abs(link - want_link) / link_scale)[:, inside], 5 * tol)
    response = fit.predict(None, s, kind="response")[:20]
    want_resp = np.asarray(case["response"]).reshape(20, -1)
    np.testing.assert_allclose(
        response[:, inside], want_resp[:, inside], rtol=5 * tol, atol=5 * tol * np.abs(want_resp).max()
    )
    if case["family"] == "binomial":
        cls = fit.predict(None, s, kind="class")[:20]
        want_cls = np.asarray(case["class"]).reshape(20, -1)
        # A class can flip only where the link is within the tolerance of zero.
        flipped = (cls != want_cls) & inside
        assert np.all(np.abs(want_link[flipped]) < 5 * tol)
    # Below the path's end the coefficients are clamped to it.
    np.testing.assert_array_equal(fit.coef(fit.lambdas[-1] / 3), fit.coef(fit.lambdas[-1]))
    if "offset(" not in case["formula"]:
        # New rows rebuild the design, factors included.
        new = fit.predict(_data(case["data"]).head(20), s, kind="link")
        np.testing.assert_allclose(new, link, rtol=1e-12, atol=1e-12)
    # A scalar lambda gives a vector.
    assert fit.coef(float(s[1])).shape == (coef.shape[0],)


@pytest.mark.parametrize("name", CVS)
def test_cross_validation(name: str) -> None:
    case = FIXTURE["cv"][name]
    data = _data(case["data"])
    kwargs = _kwargs(case)
    if name == "cv_binomial_mse_lambda":
        ref = penalized_regression(data, case["formula"], **kwargs)
        kwargs["lambdas"] = ref.lambdas[0] * np.exp(np.linspace(np.log(0.95), np.log(0.005), 30))
    cv = cv_penalized(data, case["formula"], foldid=case["fold"], measure=case["measure"], **kwargs)
    lam = np.asarray(case["lambda"])
    if case["family"] == "gaussian":
        assert abs(cv.lambdas.size - lam.size) <= 1
    else:
        assert cv.lambdas.size >= lam.size
    m = min(cv.lambdas.size, lam.size)
    np.testing.assert_allclose(cv.lambdas[:m], lam[:m], rtol=_lambda_rtol(case))
    cvm, cvsd = np.asarray(case["cvm"])[:m], np.asarray(case["cvsd"])[:m]
    if case["measure"] in ("class", "auc"):
        # Both are step functions of the fitted values: a coefficient
        # difference of 1e-4 can move one row across 0.5 (class) or swap one
        # pair of tied-looking predictions (AUC) at a few lambdas.
        same = np.isclose(cv.cvm[:m], cvm, rtol=1e-8, atol=1e-10)
        assert same.mean() >= 0.75
        np.testing.assert_allclose(cv.cvm[:m], cvm, rtol=0, atol=0.01)
    elif _wide(case):
        np.testing.assert_allclose(cv.cvm[:m], cvm, rtol=0.05)
        np.testing.assert_allclose(cv.cvsd[:m], cvsd, rtol=0.25)
    else:
        np.testing.assert_allclose(cv.cvm[:m], cvm, rtol=5e-4)
        np.testing.assert_allclose(cv.cvsd[:m], cvsd, rtol=5e-3)
    nzero = np.asarray(case["nzero"])[:m]
    if _wide(case):
        assert np.all(np.abs(cv.nzero[:m] - nzero) <= 8)
    else:
        np.testing.assert_array_equal(cv.nzero[:m], nzero)
    np.testing.assert_allclose(cv.lambda_min, case["lambda_min"], rtol=_lambda_rtol(case))
    np.testing.assert_allclose(cv.lambda_1se, case["lambda_1se"], rtol=_lambda_rtol(case))


def test_cv_accessors() -> None:
    case = FIXTURE["cv"]["cv_binomial_deviance"]
    data = _data(case["data"])
    cv = cv_penalized(data, case["formula"], family="binomial", foldid="fold")
    assert cv.measure == "deviance"
    np.testing.assert_allclose(cv.coef("min"), cv.fit.coef(cv.lambda_min))
    np.testing.assert_allclose(cv.coef(), cv.fit.coef(cv.lambda_1se))
    prob = cv.predict(data, kind="response")
    assert prob.shape == (data.height,)
    assert np.all((prob > 0) & (prob < 1))
    table = cv.table()
    assert table.columns == ["lambda", "cvm", "cvsd", "cvup", "cvlo", "nzero"]
    tidy = cv.tidy("min")
    assert tidy["term"].to_list() == cv.fit.coef_names


def test_seeded_folds_and_path_table() -> None:
    data = _data("penalized_a")
    a = cv_penalized(data, "y ~ x1 + x2 + x3 + grp", n_folds=5, seed=3)
    b = cv_penalized(data, "y ~ x1 + x2 + x3 + grp", n_folds=5, seed=3)
    np.testing.assert_array_equal(a.foldid, b.foldid)
    np.testing.assert_allclose(a.cvm, b.cvm)
    table = a.fit.path()
    assert table.columns == ["step", "lambda", "df", "dev_ratio"]
    assert table.height == a.fit.lambdas.size


def test_penalty_factor_sequence_and_exclusion() -> None:
    data = _data("penalized_a")
    names = ["x1", "x2", "x3", "grpb", "grpc"]
    by_name = penalized_regression(data, "y ~ x1 + x2 + x3 + grp", penalty_factor={"x1": 0.0, "x3": float("inf")})
    by_list = penalized_regression(data, "y ~ x1 + x2 + x3 + grp", penalty_factor=[0.0, 1.0, float("inf"), 1.0, 1.0])
    assert by_name.names == names
    np.testing.assert_allclose(by_name.coefs, by_list.coefs)
    assert np.all(by_name.coefs[2] == 0.0)
    assert np.all(by_name.coefs[0, 1:] != 0.0)
    with pytest.raises(ValueError, match="not in the design"):
        penalized_regression(data, "y ~ x1 + x2", penalty_factor={"x9": 0.0})


def test_kkt_conditions_hold() -> None:
    """The lasso optimality conditions, checked on the original scale."""
    data = _data("penalized_a")
    fit = penalized_regression(data, "yb ~ x1 + x2 + x3 + x4 + x5 + x6 + grp + flag", family="binomial")
    d = fit._data
    w = d.w / d.w.sum()
    s = np.sqrt(w @ (d.x - w @ d.x) ** 2)
    for j in range(5, fit.lambdas.size, 7):
        eta = d.x @ fit.coefs[:, j] + fit.intercepts[j]
        score = (d.x / s).T @ (w * (d.y - 1 / (1 + np.exp(-eta))))
        lam = fit.lambdas[j]
        active = fit.coefs[:, j] != 0
        np.testing.assert_allclose(score[active], lam * np.sign(fit.coefs[active, j]), rtol=1e-3, atol=1e-6)
        assert np.all(np.abs(score[~active]) <= lam * (1 + 1e-6))
        assert abs(float(w @ (d.y - 1 / (1 + np.exp(-eta))))) < 1e-6


def test_input_errors() -> None:
    data = _data("penalized_a")
    with pytest.raises(ValueError, match="family"):
        penalized_regression(data, "y ~ x1 + x2", family="gamma")
    with pytest.raises(ValueError, match="0/1"):
        penalized_regression(data, "yc ~ x1 + x2", family="binomial")
    with pytest.raises(ValueError, match="Surv"):
        penalized_regression(data, "time ~ x1 + x2", family="cox")
    fit = penalized_regression(data, "yc ~ x1 + grp + offset(log(exposure))", family="poisson")
    with pytest.raises(ValueError, match="offset"):
        fit.predict(data.head(5), 0.1)
    with pytest.raises(ValueError, match="class"):
        fit.predict(None, 0.1, kind="class")
