"""Multinomial (baseline-category) logistic regression, matching ``nnet::multinom``.

The first response level is the baseline. Each other level ``k`` has a row of
coefficients with ``log(P(Y = k) / P(Y = baseline)) = x'beta_k``. The fit is
Newton's method with the exact Hessian, run to a tight tolerance. ``multinom``
uses BFGS and stops at ``reltol = 1e-8``, so its estimates sit within about
1e-5 of the optimum unless refitted with a smaller ``reltol``. The covariance
is the inverse of the exact Hessian, as ``vcov.multinom`` computes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import special, stats

from ._categorical import categorical_data
from .design import ColumnRef, Design, build_design


@dataclass
class MultinomialFit:
    """Fitted baseline-category logit model.

    ``coefficients`` is ``(K - 1) x p``, one row per non-baseline level.
    ``covariance`` follows ``vcov(multinom)``: level by level, terms within a level.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    levels: list[str]
    log_likelihood: float
    deviance: float
    aic: float
    bic: float
    n_obs: int
    fitted: np.ndarray
    x: np.ndarray
    codes: np.ndarray
    weights: np.ndarray
    offset: np.ndarray
    row_index: np.ndarray
    design: Design
    response: str
    iterations: int
    converged: bool

    @property
    def std_errors(self) -> np.ndarray:
        """Standard errors in the shape of ``coefficients``, as ``summary(multinom)``."""
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        return se.reshape(self.coefficients.shape)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """One row per level and term, with Wald z tests and intervals.

        ``exponentiate`` adds relative-risk-ratio columns ``exp_estimate``,
        ``exp_conf_low``, and ``exp_conf_high``.
        """
        estimate = self.coefficients.reshape(-1)
        se = self.std_errors.reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = estimate / se
        crit = float(stats.norm.ppf(0.5 + level / 2))
        k1, p = self.coefficients.shape
        frame = pl.DataFrame(
            {
                "y_level": [lev for lev in self.levels[1:] for _ in range(p)],
                "term": self.names * k1,
                "estimate": estimate,
                "std_error": se,
                "statistic": stat,
                "p_value": 2 * stats.norm.sf(np.abs(stat)),
                "conf_low": estimate - crit * se,
                "conf_high": estimate + crit * se,
            }
        )
        if exponentiate:
            frame = frame.with_columns(
                pl.col("estimate").exp().alias("exp_estimate"),
                pl.col("conf_low").exp().alias("exp_conf_low"),
                pl.col("conf_high").exp().alias("exp_conf_high"),
            )
        return frame

    def glance(self) -> pl.DataFrame:
        """One-row summary: n, edf, log-likelihood, deviance, AIC, BIC."""
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "edf": [int(self.coefficients.size)],
                "log_likelihood": [self.log_likelihood],
                "deviance": [self.deviance],
                "aic": [self.aic],
                "bic": [self.bic],
                "converged": [self.converged],
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "probs") -> np.ndarray:
        """``kind="probs"`` gives an n by K matrix, ``"class"`` the most likely level."""
        if kind not in {"probs", "class"}:
            raise ValueError("kind must be 'probs' or 'class'")
        if data is None:
            probs = self.fitted
        else:
            design = build_design(data, self.design)
            if design.x.shape[1] != self.coefficients.shape[1]:
                raise ValueError("new data produced a different number of columns")
            probs = _probs(design.x, self.coefficients, np.zeros(design.n_obs))
        if kind == "probs":
            return probs
        return np.asarray(self.levels, dtype=object)[np.argmax(probs, axis=1)]

    def probabilities(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """Class probabilities as a frame with one column per level."""
        probs = self.predict(data, kind="probs")
        return pl.DataFrame({level: probs[:, k] for k, level in enumerate(self.levels)})


def _probs(x: np.ndarray, coef: np.ndarray, off: np.ndarray) -> np.ndarray:
    eta = np.column_stack([np.zeros(x.shape[0]), x @ coef.T]) + off[:, None]
    return special.softmax(eta, axis=1)


def _loglik(x, coef, off, y, w) -> float:
    eta = np.column_stack([np.zeros(x.shape[0]), x @ coef.T]) + off[:, None]
    logp = eta - special.logsumexp(eta, axis=1, keepdims=True)
    return float(np.sum(w * logp[np.arange(x.shape[0]), y]))


def _gradient_hessian(x, coef, off, y, w, k1):
    n, p = x.shape
    probs = _probs(x, coef, off)
    onehot = np.zeros((n, k1 + 1))
    onehot[np.arange(n), y] = 1.0
    resid = (onehot - probs)[:, 1:]
    grad = ((resid * w[:, None]).T @ x).reshape(-1)
    pk = probs[:, 1:]
    hess = np.zeros((k1 * p, k1 * p))
    for j in range(k1):
        for l in range(j, k1):
            wt = w * pk[:, j] * ((1.0 if j == l else 0.0) - pk[:, l])
            block = -(x * wt[:, None]).T @ x
            hess[j * p : (j + 1) * p, l * p : (l + 1) * p] = block
            if l != j:
                hess[l * p : (l + 1) * p, j * p : (j + 1) * p] = block.T
    return grad, hess


def multinomial_regression(
    data: pl.DataFrame,
    formula: str,
    *,
    weights: ColumnRef | None = None,
    levels: Sequence[object] | None = None,
    maxit: int = 100,
    tol: float = 1e-10,
) -> MultinomialFit:
    """Baseline-category logistic regression, as ``nnet::multinom``.

    ``formula`` is ``"y ~ x + stage"``. The first of the response levels is the
    baseline; the order is ``levels=``, then a ``pl.Enum``'s order, then sorted
    values (numbers numerically). Empty levels are dropped with a warning, as
    ``multinom`` does. ``weights`` are case weights.
    """
    prepared = categorical_data(data, formula, weights=weights, levels=levels, drop_intercept=False, min_levels=2)
    x = prepared.design.x
    y = prepared.codes
    w = prepared.weights
    off = prepared.offset
    n, p = x.shape
    k1 = len(prepared.levels) - 1
    coef = np.zeros((k1, p))
    ll = _loglik(x, coef, off, y, w)
    converged = False
    it = 0
    for it in range(1, maxit + 1):
        grad, hess = _gradient_hessian(x, coef, off, y, w, k1)
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        t = 1.0
        while True:
            cand = coef + t * step.reshape(k1, p)
            ll_new = _loglik(x, cand, off, y, w)
            if ll_new >= ll - 1e-12 * abs(ll) or t < 1e-10:
                break
            t *= 0.5
        coef = cand
        change = abs(ll_new - ll)
        ll = ll_new
        if np.max(np.abs(t * step)) < tol * (1.0 + np.max(np.abs(coef))) or change < 1e-15 * (1 + abs(ll)):
            converged = True
            break
    grad, hess = _gradient_hessian(x, coef, off, y, w, k1)
    info = -hess
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    k = k1 * p
    n_eff = float(np.sum(w))
    return MultinomialFit(
        coefficients=coef,
        covariance=cov,
        names=list(prepared.design.names),
        levels=prepared.levels,
        log_likelihood=ll,
        deviance=-2.0 * ll,
        aic=-2.0 * ll + 2.0 * k,
        bic=-2.0 * ll + np.log(n_eff) * k,
        n_obs=int(n),
        fitted=_probs(x, coef, off),
        x=x,
        codes=y,
        weights=w,
        offset=off,
        row_index=prepared.design.row_index,
        design=prepared.design,
        response=prepared.response,
        iterations=it,
        converged=converged,
    )
