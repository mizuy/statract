"""Cox model with one penalized spline term, as ``coxph(Surv(t, d) ~ pspline(x, df))``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats
from scipy.interpolate import BSpline

from ..models.design import ColumnRef, column_series
from .cox import _partial

__all__ = ["CoxPsplineFit", "cox_pspline", "pspline_basis"]


@dataclass
class CoxPsplineFit:
    """Penalized-spline Cox fit.

    ``coefficients`` are the spline coefficients (``ps(x)3`` ... in R).
    ``variance`` is the inverse penalized information (``fit$var``);
    ``variance_sandwich`` is ``var %*% I %*% var`` (``fit$var2``). ``df`` is the
    effective degrees of freedom of the term, ``trace(I var)``.
    ``log_likelihood`` is ``fit$loglik[2]``, the partial log-likelihood at the
    penalized estimate, and ``log_likelihood_penalized`` subtracts the penalty.
    :meth:`aic` uses ``df``, as ``AIC()`` on a ``coxph.penal``.
    """

    coefficients: np.ndarray
    variance: np.ndarray
    variance_sandwich: np.ndarray
    df: float
    theta: float
    log_likelihood: float
    log_likelihood_penalized: float
    log_likelihood_null: float
    knots: np.ndarray
    boundary: tuple[float, float]
    centers: np.ndarray
    x: np.ndarray
    basis_means: np.ndarray
    n_obs: int
    n_events: int

    def aic(self) -> float:
        return float(-2 * self.log_likelihood + 2 * self.df)

    def tests(self) -> pl.DataFrame:
        """``linear`` and ``nonlin`` rows of ``summary(fit)$coefficients``.

        ``linear`` is the Wald test of the generalized least-squares slope of
        the spline coefficients on the knot centres; ``nonlin`` is the rest of
        the overall Wald test, on ``df - 1`` degrees of freedom.
        """
        b, v, v2 = self.coefficients, self.variance, self.variance_sandwich
        test1 = float(b @ np.linalg.solve(v, b))
        xmat = np.column_stack([np.ones(len(b)), self.centers])
        xsig = np.linalg.solve(v, xmat)
        cmat = np.linalg.solve(xmat.T @ xsig, xsig.T)[1]
        linear = float(cmat @ b)
        lvar1 = float(cmat @ v @ cmat)
        lvar2 = float(cmat @ v2 @ cmat)
        test2 = linear**2 / lvar1
        return pl.DataFrame(
            {
                "term": ["linear", "nonlin"],
                "coef": [linear, None],
                "se_coef": [np.sqrt(lvar1), None],
                "se2": [np.sqrt(lvar2), None],
                "chisq": [test2, test1 - test2],
                "df": [1.0, self.df - 1],
                "p_value": [
                    float(stats.chi2.sf(test2, 1)),
                    float(stats.chi2.sf(test1 - test2, max(0.5, self.df - 1))),
                ],
            }
        )

    def term(self, values: np.ndarray | None = None) -> pl.DataFrame:
        """Centred term and its standard error, as ``termplot(fit, se=TRUE)``.

        Without ``values`` the curve is drawn at the sorted unique data values.
        """
        x = np.unique(self.x) if values is None else np.asarray(values, dtype=float)
        basis = pspline_basis(x, self.knots) - self.basis_means
        fit = basis @ self.coefficients
        se = np.sqrt(np.einsum("ij,jk,ik->i", basis, self.variance, basis))
        return pl.DataFrame({"x": x, "y": fit, "se": se})


def _knots(x: np.ndarray, nterm: int, degree: int = 3) -> tuple[np.ndarray, tuple[float, float]]:
    lo, hi = float(np.min(x)), float(np.max(x))
    dx = (hi - lo) / nterm
    knots = np.concatenate([lo + dx * np.arange(-degree, nterm), hi + dx * np.arange(0, degree + 1)])
    return knots, (lo, hi)


def pspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """B-spline basis of ``pspline`` without its first column (``intercept=FALSE``)."""
    full = BSpline.design_matrix(np.asarray(x, dtype=float), knots, degree, extrapolate=True).toarray()
    return full[:, 1:]


def _penalty(nvar: int) -> np.ndarray:
    d = np.diff(np.eye(nvar), n=2, axis=0)
    return (d.T @ d)[1:, 1:]


def _fit_theta(x, time, event, pmat, theta, init):
    lam = 0.0 if theta <= 0 else theta / (1 - theta)
    n = len(time)
    zeros = np.zeros(n)
    ones = np.ones(n)
    strata = np.zeros(n, dtype=int)
    beta = init.copy()

    def score(b):
        ll, g, h = _partial(b, x, time, event, ones, zeros, zeros, strata, "efron")
        pen = lam / 2 * b @ pmat @ b
        return ll, ll - pen, g - lam * pmat @ b, -h + lam * pmat

    ll, pll, g, info = score(beta)
    for _ in range(50):
        step = np.linalg.solve(info, g)
        t = 1.0
        while t > 1e-8:
            trial = beta + t * step
            ll_t, pll_t, g_t, info_t = score(trial)
            if np.isfinite(pll_t) and pll_t >= pll - 1e-10:
                break
            t /= 2
        done = abs(pll_t - pll) <= 1e-9 * (abs(pll) + 1e-9)
        beta, ll, pll, g, info = trial, ll_t, pll_t, g_t, info_t
        if done:
            break
    hinv = np.linalg.inv(info)
    imat = info - lam * pmat
    df = float(np.sum(imat * hinv))
    return beta, ll, pll, hinv, hinv @ imat @ hinv, df


def _control_df(target: float, history: list[tuple[float, float]], theta: float, df: float, it: int, half: int, eps: float):
    """``survival:::frailty.controldf``: next theta toward the target df."""
    thetas = [h[0] for h in history] + [theta]
    dfs = [h[1] for h in history] + [df]
    nx = len(thetas)
    done = it > 1 and abs(dfs[-1] - target) < eps
    x = np.asarray(thetas, dtype=float)
    y = np.asarray(dfs, dtype=float)
    tgt = target
    doing_well = abs((y[-1] - tgt) / (y[-2] - tgt)) <= 0.6
    order = np.argsort(x, kind="stable")
    if (x[0] - x[1]) * (y[0] - y[1]) > 0:
        y = y[order]
    else:
        y = -y[order]
        tgt = -tgt
    x = x[order]
    new_history = list(zip(thetas, dfs, strict=True))
    if np.all(y > tgt):
        b1 = 0
    elif np.all(y < tgt):
        b1 = nx - 3
    else:
        b1 = int(np.max(np.flatnonzero(y <= tgt)))
        if not doing_well and half < 2:
            return float(np.mean(x[b1 : b1 + 2])), done, new_history, half + 1
        if b1 + 2 == nx or (b1 > 0 and (tgt - y[b1]) < (y[b1 + 1] - tgt)):
            b1 -= 1
    b2 = [b1 + 1, b1 + 2]
    xx = np.log(x[b2] - x[b1])
    yy = np.log(y[b2] - y[b1])
    power = (yy[1] - yy[0]) / (xx[1] - xx[0])
    a = yy[0] - power * xx[0]
    newx = (np.log(tgt - y[b1]) - a) / power
    return float(x[b1] + np.exp(newx)), done, new_history, 0


def cox_pspline(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    x: ColumnRef,
    *,
    df: float = 4,
    eps: float = 0.1,
    outer_max: int = 10,
) -> CoxPsplineFit:
    """Cox model with one penalized spline, as ``coxph(Surv(time, event) ~ pspline(x, df))``.

    The basis, penalty, and the search for the penalty that gives ``df``
    follow ``survival::pspline`` (``nterm = round(2.5 df)`` cubic B-splines
    on equally spaced knots, second-difference penalty, and
    ``frailty.controldf`` with tolerance ``eps``). Ties use Efron.
    Rows with a missing value are dropped.
    """
    frame = pl.DataFrame(
        {
            "t": column_series(data, time).cast(pl.Float64),
            "d": column_series(data, event).cast(pl.Float64),
            "x": column_series(data, x).cast(pl.Float64),
        }
    ).drop_nulls().filter(pl.all_horizontal(pl.all().is_not_nan()))
    t = frame["t"].to_numpy()
    d = frame["d"].to_numpy()
    xv = frame["x"].to_numpy()
    nterm = round(2.5 * df)
    if df <= 1 or df > nterm:
        raise ValueError("df must be above 1 and below 2.5 df")
    knots, boundary = _knots(xv, nterm)
    basis = pspline_basis(xv, knots)
    means = basis.mean(axis=0)
    xc = basis - means
    pmat = _penalty(basis.shape[1] + 1)

    history = [(1.0, 1.0), (0.0, float(nterm))]
    theta = 1 - df / nterm
    half = 0
    init = np.zeros(basis.shape[1])
    saved: list[tuple[float, np.ndarray]] = []
    for it in range(1, outer_max + 1):
        beta, ll, pll, var, var2, dfit = _fit_theta(xc, t, d, pmat, theta, init)
        new_theta, done, history, half = _control_df(df, history, theta, dfit, it, half, eps)
        if done:
            break
        saved.append((theta, beta))
        # Start the next fit from the saved fit whose theta is closest, as coxpenal.fit does.
        init = min(saved, key=lambda s: (s[0] - new_theta) ** 2)[1]
        theta = new_theta
    ll0 = _partial(np.zeros(basis.shape[1]), xc, t, d, np.ones(len(t)), np.zeros(len(t)), np.zeros(len(t)),
                   np.zeros(len(t), dtype=int), "efron")[0]
    nvar = basis.shape[1] + 1
    return CoxPsplineFit(
        coefficients=beta,
        variance=var,
        variance_sandwich=var2,
        df=dfit,
        theta=theta,
        log_likelihood=ll,
        log_likelihood_penalized=pll,
        log_likelihood_null=ll0,
        knots=knots,
        boundary=boundary,
        centers=knots[1:nvar] + (boundary[0] - knots[0]),
        x=xv,
        basis_means=means,
        n_obs=len(t),
        n_events=int(d.sum()),
    )
