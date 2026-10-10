"""Risk ratios for a binary outcome: modified Poisson and log-binomial.

``method="poisson"`` is Zou's (2004) modified Poisson regression: a Poisson
GLM on the 0/1 outcome with the HC0 sandwich, as
``glm(family = poisson)`` with ``sandwich::vcovHC(type = "HC0")``. A cluster
column gives ``vcovCL(type = "HC0")`` instead (Zou and Donner 2013).

``method="log-binomial"`` is ``glm(family = binomial(link = "log"))``. The
iteration copies ``glm.fit``: the start is ``glm``'s ``mustart`` or ``start=``,
and a step that leaves ``0 < mu < 1`` is halved toward the previous one. As in
R, an invalid first step raises and asks for ``start``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from .covariance import cluster_covariance, hc_covariance
from .design import ColumnRef, column_series, design_matrix
from .fit import (
    Fit,
    _from_formula,
    _hat,
    _optional_numeric,
    _outcome_array,
    _take_aligned,
    _xtwx_inv,
    fit_glm,
)
from .formula import is_formula

_METHODS = ("poisson", "log-binomial")


@dataclass
class RiskRatioFit:
    """Risk-ratio model. ``fit`` is the underlying GLM and ``covariance`` the one used by ``tidy``.

    ``covariance_kind`` is ``"HC0"``, ``"cluster"`` (``vcovCL`` HC0), or ``"model"``.
    """

    fit: Fit
    covariance: np.ndarray
    method: str
    covariance_kind: str

    @property
    def names(self) -> list[str]:
        return self.fit.names

    @property
    def coefficients(self) -> np.ndarray:
        return self.fit.coefficients

    def tidy(self, *, level: float = 0.95, exponentiate: bool = True) -> pl.DataFrame:
        """Coefficient table. With ``exponentiate`` the estimate and interval are risk ratios.

        ``std_error`` and ``statistic`` stay on the log scale.
        """
        est = self.fit.coefficients
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = est / se
        crit = float(stats.norm.ppf(0.5 + level / 2))
        low = est - crit * se
        high = est + crit * se
        if exponentiate:
            est, low, high = np.exp(est), np.exp(low), np.exp(high)
        return pl.DataFrame(
            {
                "term": self.fit.names,
                "estimate": est,
                "std_error": se,
                "statistic": stat,
                "p_value": 2 * stats.norm.sf(np.abs(stat)),
                "conf_low": low,
                "conf_high": high,
            }
        )

    def glance(self) -> pl.DataFrame:
        return self.fit.glance()

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        """Risk (``kind="response"``) or log risk (``"link"``)."""
        return self.fit.predict(data, kind=kind)


def fit_risk_ratio(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    method: str = "poisson",
    weights: ColumnRef | None = None,
    cluster: ColumnRef | Sequence[ColumnRef] | None = None,
    covariance: str | None = None,
    start: Sequence[float] | None = None,
) -> RiskRatioFit:
    """Risk ratios for a 0/1 outcome by modified Poisson or log-binomial regression.

    ``outcome`` is a column or a Wilkinson formula such as ``"y ~ arm + age"``.
    ``covariance`` is ``"robust"`` (HC0, or cluster-robust with ``cluster``)
    or ``"model"``. The default is ``"robust"`` for ``method="poisson"`` and
    ``"model"`` for ``"log-binomial"`` (``summary.glm``). ``start`` gives
    log-binomial starting coefficients, as ``glm(start=)``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if covariance is None:
        covariance = "robust" if (method == "poisson" or cluster is not None) else "model"
    if covariance not in {"robust", "model"}:
        raise ValueError("covariance must be 'robust' or 'model'")
    if cluster is not None and covariance != "robust":
        raise ValueError("cluster needs covariance='robust'")
    if method == "poisson":
        if start is not None:
            raise ValueError("start is only for method='log-binomial'")
        fit = fit_glm(data, outcome, predictors, family="poisson", weights=weights)
    else:
        fit = _log_binomial(data, outcome, predictors, weights, start)
    y = fit.y
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("the outcome must be 0/1")
    if covariance == "model":
        cov, kind = fit.covariance, "model"
    elif cluster is None:
        cov, kind = hc_covariance(fit, "HC0"), "HC0"
    else:
        cov, kind = cluster_covariance(fit, cluster, data=data, kind="HC0"), "cluster"
    return RiskRatioFit(fit=fit, covariance=cov, method=method, covariance_kind=kind)


def _log_binomial(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None,
    weights: ColumnRef | None,
    start: Sequence[float] | None,
) -> Fit:
    if is_formula(outcome):
        if predictors is not None:
            raise ValueError("pass a Wilkinson formula or predictors, not both")
        design, y, w, off = _from_formula(data, str(outcome), weights, None)
    else:
        if predictors is None:
            raise ValueError("predictors are required when outcome is a column")
        extra: list[ColumnRef] = [outcome]
        if weights is not None:
            extra.append(weights)
        design = design_matrix(data, predictors, extra=extra)
        y = _outcome_array(_take_aligned(column_series(data, outcome), design.row_index))
        w = _optional_numeric(data, weights, design.row_index)
        off = np.zeros(design.n_obs)
    w = np.ones(design.n_obs) if w is None else np.asarray(w, dtype=float)
    off = np.asarray(off, dtype=float)
    x = design.x
    coef, working_w, mu, deviance = _log_binomial_irls(y, x, w, off, start)
    n, rank = x.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        working_resid = (y - mu) / mu
    log_likelihood = _r_binomial_loglik(y, mu, w)
    wnobs = float(np.sum(w))
    return Fit(
        coefficients=coef,
        covariance=_xtwx_inv(x, working_w),
        names=list(design.names),
        n_obs=int(n),
        log_likelihood=log_likelihood,
        residual_df=int(wnobs - rank),
        family="binomial_log",
        x=x,
        y=y,
        row_index=design.row_index,
        design=design,
        weights=w,
        offset=off,
        working_residuals=working_resid,
        working_weights=working_w,
        hat_values=_hat(x, working_w),
        dispersion=1.0,
        deviance=deviance,
        scale=1.0,
        aic=-2.0 * log_likelihood + 2.0 * rank,
        bic=-2.0 * log_likelihood + rank * np.log(wnobs),
    )


def _r_binomial_loglik(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    """``logLik`` of a binomial ``glm``: weights are trial counts, rounded as ``binomial()$aic`` does.

    Integer weights give the frequency-weighted log-likelihood.
    """
    keep = w > 0
    trials = np.round(w[keep])
    successes = np.round(w[keep] * y[keep])
    return float(np.sum(stats.binom.logpmf(successes, trials, mu[keep])))


def _binomial_deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) + np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
    return float(np.sum(2.0 * w * term))


def _log_binomial_irls(
    y: np.ndarray,
    x: np.ndarray,
    prior: np.ndarray,
    offset: np.ndarray,
    start: Sequence[float] | None,
    *,
    eps: float = 1e-8,
    maxit: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``glm.fit`` for ``binomial(link = "log")``: coefficients, working weights, mean, deviance."""
    tiny = np.finfo(float).eps

    def linkinv(eta: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(eta), tiny)

    def valid(mu: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(mu)) and np.all((mu > 0) & (mu < 1)))

    if start is None:
        mustart = (prior * y + 0.5) / (prior + 1.0)
        eta = np.log(mustart)
    else:
        start_arr = np.asarray(start, dtype=float)
        if start_arr.shape != (x.shape[1],):
            raise ValueError(f"start must have {x.shape[1]} values")
        eta = x @ start_arr + offset
    mu = linkinv(eta)
    if not valid(mu):
        raise ValueError("cannot find valid starting values: please specify some (start=)")
    dev_old = _binomial_deviance(y, mu, prior)
    coef_old: np.ndarray | None = None
    coef = np.zeros(x.shape[1])
    weights = np.zeros(y.shape[0])
    dev = dev_old
    converged = False
    for _ in range(maxit):
        mu_eta = np.maximum(np.exp(eta), tiny)
        good = (prior > 0) & (mu_eta != 0)
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta[good]
        w_good = np.sqrt(prior[good] * mu_eta[good] ** 2 / (mu[good] * (1 - mu[good])))
        xw = x[good] * w_good[:, None]
        coef, *_ = np.linalg.lstsq(xw, z * w_good, rcond=None)
        weights = np.zeros(y.shape[0])
        weights[good] = w_good**2
        eta = x @ coef + offset
        mu = linkinv(eta)
        dev = _binomial_deviance(y, mu, prior)
        if not np.isfinite(dev):
            if coef_old is None:
                raise ValueError("no valid set of coefficients has been found: please supply starting values (start=)")
            for _inner in range(maxit):
                coef = (coef + coef_old) / 2
                eta = x @ coef + offset
                mu = linkinv(eta)
                dev = _binomial_deviance(y, mu, prior)
                if np.isfinite(dev):
                    break
            else:
                raise ValueError("inner loop 1; cannot correct step size")
        if not valid(mu):
            if coef_old is None:
                raise ValueError("no valid set of coefficients has been found: please supply starting values (start=)")
            for _inner in range(maxit):
                coef = (coef + coef_old) / 2
                eta = x @ coef + offset
                mu = linkinv(eta)
                if valid(mu):
                    break
            else:
                raise ValueError("inner loop 2; cannot correct step size")
            dev = _binomial_deviance(y, mu, prior)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < eps:
            converged = True
            break
        dev_old = dev
        coef_old = coef
    if not converged:
        import warnings

        warnings.warn("log-binomial IRLS did not converge", RuntimeWarning, stacklevel=3)
    return coef, weights, mu, dev
