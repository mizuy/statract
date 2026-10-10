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

        ``exponentiate`` adds ``exp_estimate``, ``exp_conf_low``, and
        ``exp_conf_high``. For the coefficients these are cumulative odds ratios
        under the logit link.
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


def _binary_logit(x: np.ndarray, z: np.ndarray, w: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton's method: coefficients and fitted probabilities."""
    from .fit import _glm_irls

    coef, _weights, _mu, _dev, _work = _glm_irls(z, x, "binomial", w, off, eps=1e-12, maxit=100)
    mu = special.expit(x @ coef + off)
    return coef, mu


@dataclass
class BrantTest:
    """Brant's test of proportional odds. ``table`` has the omnibus row first."""

    table: pl.DataFrame
    binary_coefficients: np.ndarray
    cutpoints: list[str]
    names: list[str]


def brant_test(fit: OrdinalFit) -> BrantTest:
    """Brant (1990) Wald test that the slopes are equal across cutpoints.

    One logistic regression of ``Y > k`` on the same covariates is fitted for
    each cutpoint ``k``. The covariance across fits is
    ``(X'W_k X)^{-1} X'W_{kl} X (X'W_l X)^{-1}`` with ``W_{kl} = pi_l (1 - pi_k)``
    for ``k <= l``, as in the ``brant`` R package. The statistic compares every
    cutpoint's slopes with the first. The omnibus test has ``p (K - 2)``
    degrees of freedom, each coefficient ``K - 2``. Weights count as frequency
    weights. Only the logit link has this test.
    """
    if fit.link != "logit":
        raise ValueError("the Brant test needs the logit link")
    q = len(fit.thresholds)
    p = len(fit.coefficients)
    if q < 2:
        raise ValueError("the Brant test needs at least three response levels")
    if p == 0:
        raise ValueError("the Brant test needs at least one covariate")
    x1 = np.column_stack([np.ones(fit.x.shape[0]), fit.x])
    w = fit.weights
    k1 = p + 1
    coefs = np.zeros((q, k1))
    pis = np.zeros((fit.x.shape[0], q))
    for m in range(q):
        z = (fit.codes > m).astype(float)
        coefs[m], pis[:, m] = _binary_logit(x1, z, w, fit.offset)
    inv = [np.linalg.inv((x1 * (w * pis[:, m] * (1 - pis[:, m]))[:, None]).T @ x1) for m in range(q)]
    var = np.zeros((q * k1, q * k1))
    for m in range(q):
        var[m * k1 : (m + 1) * k1, m * k1 : (m + 1) * k1] = inv[m]
        for l in range(m + 1, q):
            wml = w * (pis[:, l] - pis[:, m] * pis[:, l])
            block = inv[m] @ ((x1 * wml[:, None]).T @ x1) @ inv[l]
            var[m * k1 : (m + 1) * k1, l * k1 : (l + 1) * k1] = block
            var[l * k1 : (l + 1) * k1, m * k1 : (m + 1) * k1] = block.T
    slope_idx = np.concatenate([np.arange(m * k1 + 1, (m + 1) * k1) for m in range(q)])
    beta_star = coefs[:, 1:].reshape(-1)
    var_star = var[np.ix_(slope_idx, slope_idx)]

    def wald(columns: np.ndarray) -> tuple[float, int]:
        rows = []
        for m in range(1, q):
            for j in columns:
                r = np.zeros(q * p)
                r[j] = 1.0
                r[m * p + j] = -1.0
                rows.append(r)
        d = np.asarray(rows)
        diff = d @ beta_star
        stat = float(diff @ np.linalg.solve(d @ var_star @ d.T, diff))
        return stat, d.shape[0]

    terms = ["Omnibus", *fit.names]
    stat_list = []
    df_list = []
    stat, df = wald(np.arange(p))
    stat_list.append(stat)
    df_list.append(df)
    for j in range(p):
        stat, df = wald(np.array([j]))
        stat_list.append(stat)
        df_list.append(df)
    table = pl.DataFrame(
        {
            "term": terms,
            "statistic": stat_list,
            "df": df_list,
            "p_value": stats.chi2.sf(np.asarray(stat_list), np.asarray(df_list)),
        }
    )
    lev = fit.levels
    return BrantTest(
        table=table,
        binary_coefficients=coefs,
        cutpoints=[f"{lev[m]}|{lev[m + 1]}" for m in range(q)],
        names=["(Intercept)", *fit.names],
    )
