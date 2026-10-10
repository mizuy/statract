"""Accelerated failure-time models in the ``survreg`` parameterization.

The linear predictor is an intercept plus the covariates. Weibull, exponential,
lognormal, and log-logistic are fit on the log-time scale. Gaussian and
logistic are fit on the time scale. The last coefficient, when the scale is
free, is ``Log(scale)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ..models.design import ColumnRef, column_series, design_matrix
from ..models.fit import Fit
from ..models.formula import is_formula
from .spec import parse_survival_formula

_LOG_TIME = {"weibull", "exponential", "lognormal", "loglogistic"}
_DISTRIBUTIONS = _LOG_TIME | {"gaussian", "logistic"}
_MAX_ITER = 50
_EPS = 1e-10


@dataclass
class AftFit:
    """Accelerated failure-time fit. ``tidy`` matches the coefficient table of a ``survreg``."""

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    log_likelihood: float | None
    residual_df: int | None
    family: str
    x: np.ndarray
    y: np.ndarray
    row_index: np.ndarray
    design: object
    weights: np.ndarray
    offset: np.ndarray
    distribution: str
    scale: float
    converged: bool

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        return Fit.tidy(self, level=level, exponentiate=exponentiate, df=np.inf)


def accelerated_failure(
    data: pl.DataFrame,
    time: ColumnRef | None = None,
    event: ColumnRef | None = None,
    predictors: list[ColumnRef] | None = None,
    *,
    distribution: str = "weibull",
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
) -> AftFit:
    """Fit an accelerated failure-time model.

    ``time`` may be a Wilkinson formula such as ``"Surv(time, status) ~ age + sex"``.
    """
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {sorted(_DISTRIBUTIONS)}")
    if is_formula(time):
        if event is not None or predictors is not None:
            raise ValueError("a Surv formula already names the time, the event, and the predictors")
        parsed = parse_survival_formula(
            data,
            str(time),
            weights=weights,
            offset=offset,
            drop_intercept=False,
            allow_counting=False,
            allow_strata=False,
            allow_cluster=False,
        )
        design = parsed.design
        time_a = parsed.time
        event_a = np.asarray(parsed.event, dtype=float) > 0
        offset_a = parsed.offset
        weight_a = np.ones(design.n_obs) if weights is None else np.asarray(
            column_series(data, weights).gather(design.row_index.tolist()).to_numpy(), dtype=float
        )
    else:
        if time is None or event is None or predictors is None:
            raise ValueError("time, event, and predictors are required when time is a column")
        extra: list[ColumnRef] = [time, event]
        for ref in (weights, offset):
            if ref is not None:
                extra.append(ref)
        design = design_matrix(data, predictors, extra=extra, intercept=True)
        idx = design.row_index
        time_a = np.asarray(column_series(data, time).gather(idx.tolist()).to_numpy(), dtype=float)
        event_a = np.asarray(column_series(data, event).gather(idx.tolist()).to_numpy(), dtype=float) > 0
    if distribution in _LOG_TIME and np.any(time_a <= 0):
        raise ValueError("weibull, exponential, lognormal, and loglogistic need positive follow-up times")
    y = np.log(time_a) if distribution in _LOG_TIME else time_a
    if not is_formula(time):
        weight_a = np.ones(design.n_obs) if weights is None else np.asarray(
            column_series(data, weights).gather(design.row_index.tolist()).to_numpy(), dtype=float
        )
        offset_a = np.zeros(design.n_obs) if offset is None else np.asarray(
            column_series(data, offset).gather(design.row_index.tolist()).to_numpy(), dtype=float
        )
    free_scale = distribution != "exponential"
    beta0 = np.linalg.lstsq(design.x, y - offset_a, rcond=None)[0]
    resid = y - offset_a - design.x @ beta0
    sigma0 = float(np.sqrt(np.average(resid**2, weights=weight_a)))
    theta0 = np.concatenate([beta0, [np.log(max(sigma0, 1e-3))]]) if free_scale else beta0
    theta, ll, hess, converged = _newton(theta0, design.x, y, event_a, weight_a, offset_a, distribution, free_scale)
    cov = np.linalg.pinv(-hess)
    scale = 1.0 if not free_scale else float(np.exp(theta[-1]))
    names = list(design.names)
    if free_scale:
        names = names + ["Log(scale)"]
    return AftFit(
        coefficients=theta,
        covariance=cov,
        names=names,
        n_obs=design.n_obs,
        log_likelihood=ll,
        residual_df=None,
        family=distribution,
        x=design.x,
        y=y,
        row_index=design.row_index,
        design=design,
        weights=weight_a,
        offset=offset_a,
        distribution=distribution,
        scale=scale,
        converged=converged,
    )


def _newton(theta, x, y, event, weights, offset, distribution, free_scale):
    ll, grad, hess = _objective(theta, x, y, event, weights, offset, distribution, free_scale)
    converged = False
    for _ in range(_MAX_ITER):
        step = _ascent_step(hess, grad)
        lam = 1.0
        accepted = False
        while lam > 1e-10:
            trial = theta + lam * step
            ll_t, grad_t, hess_t = _objective(trial, x, y, event, weights, offset, distribution, free_scale)
            if np.isfinite(ll_t) and ll_t >= ll - 1e-8:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        theta, ll, grad, hess = trial, ll_t, grad_t, hess_t
        if np.max(np.abs(grad)) < 1e-8:
            converged = True
            break
    return theta, ll, hess, converged


def _ascent_step(hess: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Newton step, damped (Levenberg–Marquardt) where the Hessian is not negative definite.

    Far from the optimum, for example with heavy censoring, the Newton
    direction can point downhill, and the line search then stops at the start.
    """
    neg = -hess
    try:
        step = np.linalg.solve(neg, grad)
        np.linalg.cholesky(neg)
        return step
    except np.linalg.LinAlgError:
        pass
    mu = max(1e-6, float(np.max(np.abs(np.diag(neg)))) * 1e-3)
    eye = np.eye(neg.shape[0])
    for _ in range(60):
        try:
            np.linalg.cholesky(neg + mu * eye)
            return np.linalg.solve(neg + mu * eye, grad)
        except np.linalg.LinAlgError:
            mu *= 10.0
    return grad


def _objective(theta, x, y, event, weights, offset, distribution, free_scale):
    beta = theta[:-1] if free_scale else theta
    scale = float(np.exp(theta[-1])) if free_scale else 1.0
    u = (y - offset - x @ beta) / scale
    survival, density, dlogf, d2f = _distribution(u, distribution)
    live = event
    cens = ~event
    d1 = np.empty_like(u)
    d2 = np.empty_like(u)
    d1[live] = dlogf[live]
    d2[live] = d2f[live] - dlogf[live] ** 2
    hazard_ratio = np.zeros_like(u)
    hazard_ratio[cens] = density[cens] / np.clip(survival[cens], 1e-300, None)
    d1[cens] = -hazard_ratio[cens]
    d2[cens] = -dlogf[cens] * hazard_ratio[cens] - hazard_ratio[cens] ** 2
    ll = np.zeros_like(u)
    ll[live] = np.log(np.clip(density[live], 1e-300, None)) - np.log(scale)
    ll[cens] = np.log(np.clip(survival[cens], 1e-300, None))
    # Log-time families are optimized on y = log(t). survreg reports the
    # likelihood of t, which adds the Jacobian -log(t) at each event.
    if distribution in _LOG_TIME:
        ll[live] -= y[live]
    ll_sum = float(np.sum(weights * ll))
    # Derivatives with respect to beta and log-scale.
    # du/dbeta = -x/scale, du/dlogscale = -u.
    p = beta.shape[0]
    grad = np.zeros(p + (1 if free_scale else 0))
    grad[:p] = (weights * d1 * (-1.0 / scale)) @ x
    if free_scale:
        d_g = d1 * (-u)
        d_g[live] -= 1.0
        grad[-1] = float(np.sum(weights * d_g))
    # Observed Hessian.
    hess = np.zeros((grad.shape[0], grad.shape[0]))
    # d2 ll / dbeta_j dbeta_k = d2 * (x_j/scale) * (x_k/scale)   because du/db = -x/s, product positive
    w2 = weights * d2 / scale**2
    hess[:p, :p] = x.T @ (x * w2[:, None])
    if free_scale:
        # d/dlogscale of (d1 * -1/scale) needs the chain rule through u and scale.
        # dll/dbeta = sum w * d1 * (-x/scale)
        # Differentiate d1(u) * (-1/scale) w.r.t g=log scale.
        # d(d1)/dg = d2 * du/dg = d2 * (-u)
        # d(-1/scale)/dg = -1/scale * d(scale^{-1})/dg wait scale=e^g, 1/scale = e^{-g}, d(e^{-g})/dg = -e^{-g} = -1/scale
        # d/dg [d1 * (-1/scale)] = (d2*(-u))*(-1/scale) + d1 * (1/scale)
        # = d2 * u / scale + d1 / scale
        cross = weights * (d2 * u + d1) / scale
        hess[:p, -1] = x.T @ (-cross)  # the - from du/dbeta's remaining sign is already in (-1/scale) derivative
        # Recheck: dll/db_j = w * d1 * (-x_j/scale)
        # d/dg of that factor d1*(-1/scale) = d2*(-u)*(-1/scale) + d1*(d(-1/scale)/dg)
        # d(-1/scale)/dg = d(-e^{-g})/dg = e^{-g} = 1/scale
        # so = d2 * u / scale + d1 / scale
        # times x_j? dll/db_j has * x_j as well: factor is d1 * (-x_j/scale) = x_j * [d1 * (-1/scale)]
        # d/dg = x_j * (d2*u/scale + d1/scale)
        # I put hess[:p,-1] = x.T @ (-cross) which has an extra minus. Fix below.
        hess[:p, -1] = (x * (weights * (d2 * u + d1) / scale)[:, None]).sum(axis=0)
        hess[-1, :p] = hess[:p, -1]
        # d2 ll / dg^2
        # dll/dg = w * (d1 * (-u) - event)
        # d/dg [d1*(-u)] = (d2*(-u))*(-u) + d1*(- du/dg) = d2*u^2 + d1*u
        # event term has derivative 0
        hess[-1, -1] = float(np.sum(weights * (d2 * u**2 + d1 * u)))
    return ll_sum, grad, hess


def _distribution(z: np.ndarray, distribution: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return survival, density, d(log f)/dz, and f''/f."""
    if distribution in {"weibull", "exponential"}:
        w = np.exp(np.clip(z, -50, 50))
        surv = np.exp(-w)
        dens = w * surv
        dlogf = 1.0 - w
        d2f = w * (w - 3.0) + 1.0
        return surv, dens, dlogf, d2f
    if distribution in {"loglogistic", "logistic"}:
        w = np.exp(np.clip(z, -50, 50))
        surv = 1.0 / (1.0 + w)
        dens = w / (1.0 + w) ** 2
        dlogf = (1.0 - w) / (1.0 + w)
        d2f = (w * (w - 4.0) + 1.0) / (1.0 + w) ** 2
        return surv, dens, dlogf, d2f
    # gaussian / lognormal
    surv = stats.norm.sf(z)
    dens = stats.norm.pdf(z)
    dlogf = -z
    d2f = z**2 - 1.0
    return surv, dens, dlogf, d2f
