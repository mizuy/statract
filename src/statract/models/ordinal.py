"""Cumulative-link (proportional odds) models, matching ``MASS::polr`` and ``ordinal::clm``.

The model is ``F^{-1}(P(Y <= k)) = zeta_k - x'beta`` with no intercept in
``x``. Positive ``beta`` moves the response toward higher levels, as in
``polr`` and ``clm``. The fit is Newton's method with the exact Hessian, and
the covariance is the inverse of the observed information. ``clm`` reports the
same; ``polr`` uses a finite-difference Hessian from ``optim``.

``brant_test`` is Brant's (1990) Wald test of proportional odds from separate
binary logistic fits at each cutpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import special, stats

from ._categorical import categorical_data
from .design import ColumnRef, Design, build_design

_LINKS = {
    "logit": "logit",
    "logistic": "logit",
    "probit": "probit",
    "cloglog": "cloglog",
    "loglog": "loglog",
    "cauchit": "cauchit",
}


def _cdf(link: str, z: np.ndarray) -> np.ndarray:
    if link == "logit":
        return special.expit(z)
    if link == "probit":
        return special.ndtr(z)
    if link == "cloglog":
        return -np.expm1(-np.exp(z))
    if link == "loglog":
        return np.exp(-np.exp(-z))
    return 0.5 + np.arctan(z) / np.pi


def _pdf(link: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density and its derivative. Infinite arguments give zero."""
    with np.errstate(over="ignore", invalid="ignore"):
        if link == "logit":
            f_cdf = special.expit(z)
            f = f_cdf * (1.0 - f_cdf)
            df = f * (1.0 - 2.0 * f_cdf)
        elif link == "probit":
            f = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
            df = -z * f
        elif link == "cloglog":
            ez = np.exp(z)
            f = np.exp(z - ez)
            df = f * (1.0 - ez)
        elif link == "loglog":
            emz = np.exp(-z)
            f = np.exp(-z - emz)
            df = f * (emz - 1.0)
        else:
            f = 1.0 / (np.pi * (1.0 + z * z))
            df = -2.0 * z / (np.pi * (1.0 + z * z) ** 2)
    f = np.where(np.isfinite(z), f, 0.0)
    df = np.where(np.isfinite(z), df, 0.0)
    return np.nan_to_num(f), np.nan_to_num(df)


def _quantile(link: str, p: np.ndarray) -> np.ndarray:
    if link == "logit":
        return special.logit(p)
    if link == "probit":
        return special.ndtri(p)
    if link == "cloglog":
        return np.log(-np.log1p(-p))
    if link == "loglog":
        return -np.log(-np.log(p))
    return np.tan(np.pi * (p - 0.5))


@dataclass
class OrdinalFit:
    """Fitted cumulative-link model.

    ``covariance`` covers ``coefficients`` then ``thresholds``, in the order of
    ``vcov(polr)``. ``fitted`` holds the class probabilities of the fitted rows.
    """

    coefficients: np.ndarray
    thresholds: np.ndarray
    names: list[str]
    threshold_names: list[str]
    covariance: np.ndarray
    levels: list[str]
    link: str
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
    gradient_max: float = field(default=float("nan"))

    @property
    def n_params(self) -> int:
        return len(self.coefficients) + len(self.thresholds)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Coefficients then thresholds, with Wald z tests and intervals.

        With ``exponentiate`` the estimate and interval are exponentiated; for
        the coefficients these are cumulative odds ratios under the logit link.
        ``std_error`` and ``statistic`` stay on the log scale.
        """
        estimate = np.concatenate([self.coefficients, self.thresholds])
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = estimate / se
        crit = float(stats.norm.ppf(0.5 + level / 2))
        frame = pl.DataFrame(
            {
                "term": [*self.names, *self.threshold_names],
                "coef_type": ["coefficient"] * len(self.names) + ["threshold"] * len(self.threshold_names),
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
                pl.col("estimate", "conf_low", "conf_high").exp(),
            )
        return frame

    def glance(self) -> pl.DataFrame:
        """One-row summary: n, edf, log-likelihood, deviance, AIC, BIC."""
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "edf": [self.n_params],
                "log_likelihood": [self.log_likelihood],
                "deviance": [self.deviance],
                "aic": [self.aic],
                "bic": [self.bic],
                "converged": [self.converged],
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "probs") -> np.ndarray:
        """``kind="probs"`` gives an n by K matrix, ``"class"`` the most likely level, ``"link"`` ``x'beta``.

        New data uses the fitted design without the offset, as ``predict.polr``.
        """
        if kind not in {"probs", "class", "link"}:
            raise ValueError("kind must be 'probs', 'class', or 'link'")
        if data is None:
            eta = self.x @ self.coefficients + self.offset
        else:
            design = build_design(data, self.design)
            if design.x.shape[1] != len(self.coefficients):
                raise ValueError("new data produced a different number of columns")
            eta = design.x @ self.coefficients
        if kind == "link":
            return eta
        probs = _class_probs(self.link, self.thresholds, eta)
        if kind == "probs":
            return probs
        return np.asarray(self.levels, dtype=object)[np.argmax(probs, axis=1)]

    def probabilities(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """Class probabilities as a frame with one column per level."""
        probs = self.predict(data, kind="probs")
        return pl.DataFrame({level: probs[:, k] for k, level in enumerate(self.levels)})


def _class_probs(link: str, zeta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    cuts = np.concatenate([[-np.inf], zeta, [np.inf]])
    cum = _cdf(link, cuts[None, :] - eta[:, None])
    return np.diff(cum, axis=1)


def _loglik(theta, x, y, w, off, link, p):
    beta = theta[:p]
    zeta = theta[p:]
    if np.any(np.diff(zeta) <= 0):
        return -np.inf
    cuts = np.concatenate([[-np.inf], zeta, [np.inf]])
    eta = x @ beta + off
    prob = _cdf(link, cuts[y + 1] - eta) - _cdf(link, cuts[y] - eta)
    if np.any(prob[w > 0] <= 0):
        return -np.inf
    with np.errstate(divide="ignore"):
        return float(np.sum(w * np.where(w > 0, np.log(np.where(prob > 0, prob, 1.0)), 0.0)))


def _derivatives(theta, x, y, w, off, link, p, q):
    """Gradient and Hessian of the log-likelihood."""
    n = x.shape[0]
    beta = theta[:p]
    zeta = theta[p:]
    cuts = np.concatenate([[-np.inf], zeta, [np.inf]])
    eta = x @ beta + off
    a = cuts[y + 1] - eta
    b = cuts[y] - eta
    prob = _cdf(link, a) - _cdf(link, b)
    fa, dfa = _pdf(link, a)
    fb, dfb = _pdf(link, b)
    # u = d a / d theta, v = d b / d theta.
    u = np.zeros((n, p + q))
    v = np.zeros((n, p + q))
    u[:, :p] = -x
    v[:, :p] = -x
    rows = np.arange(n)
    upper = y < q
    u[rows[upper], p + y[upper]] = 1.0
    lower = y > 0
    v[rows[lower], p + y[lower] - 1] = 1.0
    ca = fa / prob
    cb = -fb / prob
    g_i = ca[:, None] * u + cb[:, None] * v
    grad = g_i.T @ w
    hess = (u * (w * dfa / prob)[:, None]).T @ u - (v * (w * dfb / prob)[:, None]).T @ v
    hess -= (g_i * w[:, None]).T @ g_i
    return grad, hess


def _newton(x, y, w, off, link, q, *, tol=1e-10, maxit=100):
    p = x.shape[1]
    total = np.bincount(y, weights=w, minlength=q + 1)
    cum = np.cumsum(total)[:-1] / total.sum()
    theta = np.concatenate([np.zeros(p), _quantile(link, cum)])
    ll = _loglik(theta, x, y, w, off, link, p)
    converged = False
    grad = np.full(p + q, np.nan)
    it = 0
    for it in range(1, maxit + 1):
        grad, hess = _derivatives(theta, x, y, w, off, link, p, q)
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)) or float(grad @ step) <= 0:
            step = grad / max(1.0, float(np.max(np.abs(grad))))
        t = 1.0
        while True:
            cand = theta + t * step
            ll_new = _loglik(cand, x, y, w, off, link, p)
            if ll_new >= ll - 1e-12 * abs(ll) or t < 1e-10:
                break
            t *= 0.5
        theta = cand
        change = abs(ll_new - ll)
        ll = ll_new
        if np.max(np.abs(t * step)) < tol * (1.0 + np.max(np.abs(theta))) or change < 1e-15 * (1 + abs(ll)):
            grad, hess = _derivatives(theta, x, y, w, off, link, p, q)
            converged = bool(np.max(np.abs(grad)) < 1e-6 * max(1.0, float(np.sum(w))))
            break
    return theta, ll, grad, hess, it, converged


def ordinal_regression(
    data: pl.DataFrame,
    formula: str,
    *,
    link: str = "logit",
    weights: ColumnRef | None = None,
    levels: Sequence[object] | None = None,
) -> OrdinalFit:
    """Proportional odds (cumulative link) model, as ``polr`` and ``clm``.

    ``formula`` is ``"y ~ x + stage"``. The response is ordered by ``levels=``,
    then the order of a ``pl.Enum``, then sorted values (numbers numerically).
    An intercept in the formula is dropped, since the thresholds take its place.
    ``link`` is ``"logit"`` (``polr``'s ``logistic``), ``"probit"``,
    ``"cloglog"``, ``"loglog"``, or ``"cauchit"``, with ``clm``'s meanings.
    ``weights`` are case weights.
    """
    if link not in _LINKS:
        raise ValueError(f"link must be one of {sorted(set(_LINKS))}")
    link = _LINKS[link]
    prepared = categorical_data(data, formula, weights=weights, levels=levels, drop_intercept=True, min_levels=2)
    x = prepared.design.x
    y = prepared.codes
    w = prepared.weights
    off = prepared.offset
    q = len(prepared.levels) - 1
    p = x.shape[1]
    theta, ll, grad, hess, it, converged = _newton(x, y, w, off, link, q)
    info = -hess
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    beta = theta[:p]
    zeta = theta[p:]
    eta = x @ beta + off
    fitted = _class_probs(link, zeta, eta)
    k = p + q
    n_eff = float(np.sum(w))
    names = list(prepared.design.names)
    lev = prepared.levels
    return OrdinalFit(
        coefficients=beta,
        thresholds=zeta,
        names=names,
        threshold_names=[f"{lev[i]}|{lev[i + 1]}" for i in range(q)],
        covariance=cov,
        levels=lev,
        link=link,
        log_likelihood=ll,
        deviance=-2.0 * ll,
        aic=-2.0 * ll + 2.0 * k,
        bic=-2.0 * ll + np.log(n_eff) * k,
        n_obs=int(x.shape[0]),
        fitted=fitted,
        x=x,
        codes=y,
        weights=w,
        offset=off,
        row_index=prepared.design.row_index,
        design=prepared.design,
        response=prepared.response,
        iterations=it,
        converged=converged,
        gradient_max=float(np.max(np.abs(grad))) if grad.size else 0.0,
    )


@dataclass
class BrantTest:
    """Brant's test of proportional odds. ``table`` has the omnibus row first."""

    table: pl.DataFrame
    binary_coefficients: np.ndarray
    cutpoints: list[str]
    names: list[str]


def brant_test(fit: OrdinalFit) -> BrantTest:
    """Brant (1990, Biometrics 46:1171) Wald test that the slopes are equal across cutpoints.

    For each cutpoint k a binary logistic regression of ``1{Y > k}`` on the
    same covariates is fitted (weights are frequency weights, the model offset
    is kept). Brant's sec. 3 gives the large-sample covariance between the
    k-th and l-th binary estimates (k <= l) as
    ``(X'W_kk X)^{-1} X'W_kl X (X'W_ll X)^{-1}`` with
    ``W_kl = diag(w_i (pi_il - pi_ik pi_il))``, pi_ik = P(Y_i > k); the
    diagonal blocks reduce to the usual inverse information. The Wald
    statistic tests that every cutpoint's slopes (intercepts excluded) equal
    the first cutpoint's: the omnibus test has ``p (K - 2)`` degrees of
    freedom, each covariate ``K - 2``. Only the logit link has this test.
    """
    if fit.link != "logit":
        raise ValueError("brant_test needs a proportional odds model with the logit link")
    n_levels = len(fit.levels)
    if n_levels < 3:
        raise ValueError("brant_test needs at least three response levels")
    x = np.asarray(fit.x, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("brant_test needs at least one covariate")
    n, p = x.shape
    codes = np.asarray(fit.codes)
    weights = np.ones(n) if fit.weights is None else np.asarray(fit.weights, dtype=float)
    offset = np.zeros(n) if fit.offset is None else np.asarray(fit.offset, dtype=float)
    design = np.column_stack([np.ones(n), x])
    n_cut = n_levels - 1
    width = p + 1

    coefficients = np.empty((n_cut, width))
    probabilities = np.empty((n_cut, n))
    inverse_information = []
    for k in range(n_cut):
        above = (codes > k).astype(float)
        beta, prob, inverse = _weighted_logistic(design, above, weights, offset)
        coefficients[k] = beta
        probabilities[k] = prob
        inverse_information.append(inverse)

    # Joint covariance of the stacked binary estimates (Brant 1990, sec. 3).
    joint = np.zeros((n_cut * width, n_cut * width))
    for k in range(n_cut):
        for l in range(k, n_cut):
            if k == l:
                block = inverse_information[k]
            else:
                cross_weight = weights * (probabilities[l] - probabilities[k] * probabilities[l])
                middle = design.T @ (design * cross_weight[:, None])
                block = inverse_information[k] @ middle @ inverse_information[l]
            joint[k * width : (k + 1) * width, l * width : (l + 1) * width] = block
            joint[l * width : (l + 1) * width, k * width : (k + 1) * width] = block.T

    # Contrast: slopes of cutpoint k (k >= 2) minus slopes of cutpoint 1.
    n_contrast = n_cut - 1
    contrast = np.zeros((n_contrast * p, n_cut * width))
    for j in range(n_contrast):
        for c in range(p):
            row = j * p + c
            contrast[row, 1 + c] = -1.0
            contrast[row, (j + 1) * width + 1 + c] = 1.0
    difference = contrast @ coefficients.reshape(-1)
    covariance = contrast @ joint @ contrast.T

    def wald(rows: np.ndarray) -> float:
        d = difference[rows]
        return float(d @ np.linalg.solve(covariance[np.ix_(rows, rows)], d))

    terms = ["Omnibus", *fit.names]
    statistics = [wald(np.arange(n_contrast * p))]
    dfs = [p * n_contrast]
    for c in range(p):
        statistics.append(wald(np.arange(n_contrast) * p + c))
        dfs.append(n_contrast)
    statistic = np.asarray(statistics)
    df = np.asarray(dfs, dtype=np.int64)
    table = pl.DataFrame(
        {"term": terms, "statistic": statistic, "df": df, "p_value": stats.chi2.sf(statistic, df)}
    )
    cutpoints = [f"{fit.levels[k]}|{fit.levels[k + 1]}" for k in range(n_cut)]
    return BrantTest(
        table=table,
        binary_coefficients=coefficients,
        cutpoints=cutpoints,
        names=["(Intercept)", *fit.names],
    )


def _weighted_logistic(
    design: np.ndarray, outcome: np.ndarray, weights: np.ndarray, offset: np.ndarray, *, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-weighted logistic regression by Newton's method.

    Returns the estimates, fitted probabilities, and the inverse Fisher
    information ``(X' diag(w pi (1 - pi)) X)^{-1}`` at the optimum. Stops when
    the largest relative change of the deviance is below 1e-14 or the Newton
    step is negligible.
    """
    beta = np.zeros(design.shape[1])
    share = np.sum(weights * outcome) / np.sum(weights)
    share = min(max(share, 1e-6), 1 - 1e-6)
    beta[0] = np.log(share / (1 - share)) - np.average(offset, weights=weights)
    for _ in range(max_iter):
        eta = design @ beta + offset
        prob = special.expit(eta)
        info = design.T @ (design * (weights * prob * (1 - prob))[:, None])
        score = design.T @ (weights * (outcome - prob))
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.max(np.abs(step)) <= 1e-12 * (1.0 + np.max(np.abs(beta))):
            break
    eta = design @ beta + offset
    prob = special.expit(eta)
    info = design.T @ (design * (weights * prob * (1 - prob))[:, None])
    return beta, prob, np.linalg.inv(info)
