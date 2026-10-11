"""Ordinary and generalized linear models on Polars frames.

``fit_ols`` follows ``lm`` (residual scale, t intervals). ``fit_glm`` follows
``glm`` for gaussian, binomial, poisson, and gamma, with the same default links
(identity, logit, log, inverse), and ``MASS::glm.nb`` for the negative binomial
(NB2, log link). ``tidy`` uses z intervals for GLMs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
from numba import njit
from scipy import linalg, special, stats

from .design import ColumnRef, Design, build_design, column_series, design_matrix
from .formula import is_formula, model_matrix, reject_survival_syntax

_FAMILIES = ("gaussian", "binomial", "poisson", "gamma", "negative_binomial")
_LINKS = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
    "negative_binomial": "log",
}


def _link_from_eta(family: str, eta: np.ndarray) -> np.ndarray:
    if family == "gaussian":
        return eta
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    if family in {"poisson", "negative_binomial"}:
        return np.exp(eta)
    if family == "gamma":
        return 1.0 / eta
    if family == "binomial_log":
        # The log-binomial model of ``fit_risk_ratio(method="log-binomial")``.
        return np.exp(eta)
    raise ValueError(family)


@dataclass
class Fit:
    """Fitted linear or generalized linear model.

    ``covariance`` is the model-based covariance. Sandwich estimators live in
    ``covariance.py`` and read ``score_contributions`` and ``bread``.
    The negative binomial stores ``theta`` (variance ``mu + mu^2 / theta``)
    and its standard error. Its coefficient covariance treats ``theta`` as
    known, as ``summary.negbin`` does.
    """

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
    design: Design
    weights: np.ndarray
    offset: np.ndarray
    working_residuals: np.ndarray
    working_weights: np.ndarray
    hat_values: np.ndarray
    dispersion: float
    deviance: float | None = None
    scale: float | None = None
    aic: float | None = None
    bic: float | None = None
    r_squared: float | None = None
    adj_r_squared: float | None = None
    f_statistic: float | None = None
    f_p_value: float | None = None
    theta: float | None = None
    theta_std_error: float | None = None

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False, df: float | None = None) -> pl.DataFrame:
        """Coefficient table.

        OLS uses a t reference distribution with ``residual_df``. GLM uses a
        normal reference, matching ``coeftest`` on a ``glm`` object.
        """
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        estimate = self.coefficients
        use_df = self.residual_df if df is None else df
        if self.family == "ols" and use_df is not None and np.isfinite(use_df) and use_df > 0:
            dist_df: float | None = float(use_df)
        else:
            dist_df = None
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = estimate / se
        if dist_df is None:
            p_value = 2 * stats.norm.sf(np.abs(stat))
            crit = float(stats.norm.ppf(0.5 + level / 2))
        else:
            p_value = 2 * stats.t.sf(np.abs(stat), dist_df)
            crit = float(stats.t.ppf(0.5 + level / 2, dist_df))
        low = estimate - crit * se
        high = estimate + crit * se
        frame = pl.DataFrame(
            {
                "term": self.names,
                "estimate": estimate,
                "std_error": se,
                "statistic": stat,
                "p_value": p_value,
                "conf_low": low,
                "conf_high": high,
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
        """One-row model summary."""
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "residual_df": [self.residual_df],
                "log_likelihood": [self.log_likelihood],
                "aic": [self.aic],
                "bic": [self.bic],
                "deviance": [self.deviance],
                "sigma": [None if self.scale is None else float(np.sqrt(self.scale))],
                "r_squared": [self.r_squared],
                "adj_r_squared": [self.adj_r_squared],
                "statistic": [self.f_statistic],
                "p_value": [self.f_p_value],
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        """Linear predictor or mean response.

        ``kind`` is ``"link"`` or ``"response"``. With ``data=None`` the
        fitted rows are used.
        """
        if kind not in {"link", "response"}:
            raise ValueError("kind must be 'link' or 'response'")
        if data is None:
            eta = self.x @ self.coefficients + self.offset
        else:
            design = build_design(data, self.design)
            if design.x.shape[1] != len(self.coefficients):
                raise ValueError("new data produced a different number of columns")
            eta = design.x @ self.coefficients
        if kind == "link" or self.family == "ols":
            return eta
        family = "gaussian" if self.family == "ols" else self.family
        return _link_from_eta(family, eta)

    def score_contributions(self) -> np.ndarray:
        """Per-observation score, ``n`` by ``p``.

        OLS matches ``sandwich::estfun.lm``. GLM matches ``estfun.glm``.
        """
        if self.family == "ols":
            return self.working_residuals[:, None] * self.weights[:, None] * self.x
        wres = self.working_residuals * self.working_weights
        return wres[:, None] * self.x / self.dispersion

    def bread(self) -> np.ndarray:
        """Sandwich bread matrix.

        OLS bread is ``n (X'WX)^{-1}`` with no residual variance. GLM bread is
        ``n (X'WX)^{-1}`` times the estimating-equation dispersion (1 for
        binomial and poisson).
        """
        inv = _xtwx_inv(self.x, self.working_weights if self.family != "ols" else self.weights)
        if self.family == "ols":
            return self.n_obs * inv
        return self.n_obs * inv * self.dispersion

    def refit_rows(self, index: np.ndarray, *, y: np.ndarray | None = None) -> Fit:
        """Refit on a subset or a replaced response. Used by the bootstrap."""
        if self._refit is None:
            raise TypeError(f"refit is not available for family {self.family!r}")
        return self._refit(index, y)


def _xtwx_inv(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(X'WX)^{-1}`` from a column-scaled QR, as ``lm`` and ``glm`` get it.

    Inverting the Gram matrix directly loses cond(X)^2 * eps, which a design
    with a column such as birth year pushes past the sandwich tolerance. A
    rank-deficient or non-positive weighting keeps the pseudo-inverse.
    """
    w = np.asarray(weights, dtype=float)
    n, p = x.shape
    if p and n >= p and np.all(w > 0.0) and np.all(np.isfinite(w)):
        xw = x * np.sqrt(w)[:, None]
        column_norm = np.linalg.norm(xw, axis=0)
        if np.all(column_norm > 0.0) and np.all(np.isfinite(column_norm)):
            r = np.linalg.qr(xw / column_norm, mode="r")
            diagonal = np.abs(np.diag(r))
            if diagonal.min() > diagonal.max() * max(n, p) * np.finfo(float).eps:
                r_inv = linalg.solve_triangular(r, np.eye(p))
                return (r_inv @ r_inv.T) / column_norm[:, None] / column_norm[None, :]
    xtw = x.T * w
    return np.linalg.pinv(xtw @ x)


def _hat(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    inv = _xtwx_inv(x, w)
    # h_i = w_i x_i (X'WX)^{-1} x_i'
    cooked = x @ inv
    return np.sum(cooked * x, axis=1) * w


def _complete_vectors(
    data: pl.DataFrame,
    columns: Sequence[ColumnRef | None],
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Return a mask and the raw numpy columns. Nulls are not yet dropped."""
    arrays: list[np.ndarray | None] = []
    mask = np.ones(data.height, dtype=bool)
    for ref in columns:
        if ref is None:
            arrays.append(None)
            continue
        series = column_series(data, ref)
        mask &= series.is_not_null().to_numpy()
        arrays.append(series.to_numpy())
    return mask, arrays


def _outcome_array(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.Boolean:
        return series.to_numpy().astype(float)
    return np.asarray(series.to_numpy(), dtype=float)


def _k_constant(x: np.ndarray) -> int:
    """Whether the column space contains a constant, as ``statsmodels`` counts it.

    An explicit constant column is enough. A full set of dummies is an implicit
    constant and needs a rank check, which the explicit-intercept path skips.
    """
    if x.size == 0:
        return 0
    constant = np.flatnonzero(np.max(x, axis=0) == np.min(x, axis=0))
    if constant.size == 1 and float(x[:, constant[0]].mean()) != 0.0:
        return 1
    if constant.size > 1:
        means = x[:, constant].mean(axis=0)
        if np.any(means == 1.0) or np.any(means != 0.0):
            return 1
    augmented = np.column_stack((np.ones(x.shape[0]), x))
    return int(np.linalg.matrix_rank(x) == np.linalg.matrix_rank(augmented))


def _ols_via_chol(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    unity: bool,
    xw: np.ndarray,
    yw: np.ndarray,
) -> dict[str, Any] | None:
    """Column-scaled normal equations. Returns None when the Gram matrix is ill-conditioned."""
    column_norm = np.linalg.norm(xw, axis=0)
    if column_norm.shape[0] != x.shape[1] or np.any(column_norm == 0.0):
        return None
    xs = xw / column_norm
    try:
        chol = np.linalg.cholesky(xs.T @ xs)
    except np.linalg.LinAlgError:
        return None
    diagonal = np.diag(chol)
    smallest = float(diagonal.min())
    # The normal equations lose cond(X)^2 * eps. Past a pivot ratio of 100
    # (cond of the scaled design near 400) QR keeps the sandwich tolerance.
    if smallest <= 0.0 or float(diagonal.max()) / smallest > 1e2:
        return None
    beta_s = np.linalg.solve(chol.T, np.linalg.solve(chol, xs.T @ yw))
    beta = beta_s / column_norm
    inv_chol = np.linalg.inv(chol)
    gram_inv = (inv_chol.T @ inv_chol) / column_norm[:, None] / column_norm[None, :]
    resid = y - x @ beta
    rss = float(resid @ resid) if unity else float(w @ (resid * resid))
    if not np.isfinite(rss) or rss <= 0.0:
        return None
    # Diagonal of the weighted hat matrix is the squared row norm of X_s L^{-T}.
    projected = xs @ inv_chol.T
    hat = np.sum(projected * projected, axis=1)
    return _ols_fields(y, x, w, unity, beta, resid, rss, hat, gram_inv)


def _ols_fields(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    unity: bool,
    beta: np.ndarray,
    resid: np.ndarray,
    rss: float,
    hat: np.ndarray,
    xtwx_inv: np.ndarray,
    rank: int | None = None,
) -> dict[str, Any]:
    n, p = x.shape
    if rank is None:
        rank = p
    df_resid = n - rank
    sigma2 = rss / df_resid if df_resid else np.nan
    nobs2 = n / 2.0
    log_likelihood = -np.log(rss) * nobs2 - (1.0 + np.log(np.pi / nobs2)) * nobs2
    if not unity:
        log_likelihood += 0.5 * float(np.sum(np.log(w)))
    k_constant = _k_constant(x)
    if k_constant:
        center = float(np.average(y, weights=None if unity else w))
        total = float(np.sum((y - center) ** 2)) if unity else float(np.sum(w * (y - center) ** 2))
    else:
        total = float(y @ y) if unity else float(np.sum(w * y * y))
    r_squared = 1.0 - rss / total if total else np.nan
    if df_resid:
        adj_r_squared = 1.0 - ((n - k_constant) / df_resid) * (1.0 - r_squared)
    else:
        adj_r_squared = np.nan
    df_model = float(rank - k_constant)
    if df_model == 0.0 or df_resid == 0 or not np.isfinite(sigma2) or sigma2 == 0.0 or not np.isfinite(total):
        f_statistic = np.nan
        f_p_value = np.nan
    else:
        f_statistic = ((total - rss) / df_model) / (rss / df_resid)
        f_p_value = float(stats.f.sf(f_statistic, df_model, df_resid))
    rank = float(rank)
    return {
        "coefficients": beta,
        "covariance": xtwx_inv * sigma2,
        "residuals": resid,
        "hat": hat,
        "log_likelihood": float(log_likelihood),
        "residual_df": int(df_resid),
        "rss": rss,
        "sigma2": float(sigma2),
        "aic": float(-2.0 * log_likelihood + 2.0 * rank),
        "bic": float(-2.0 * log_likelihood + np.log(n) * rank),
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
    }


def _ols_via_qr(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> dict[str, Any] | None:
    """Full-rank weighted least squares by QR.

    Returns None when the weighted design is rank-deficient or a weight is not
    positive, so the caller can keep the singular-value result.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n, p = x.shape
    if n < p or p == 0 or np.any(w <= 0.0) or not np.all(np.isfinite(w)):
        return None
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        return None
    unity = bool(np.all(w == 1.0))
    if unity:
        xw = x
        yw = y
    else:
        scale = np.sqrt(w)
        xw = x * scale[:, None]
        yw = y * scale
    solved = _ols_via_chol(y, x, w, unity, xw, yw)
    if solved is not None:
        return solved
    q, r = np.linalg.qr(xw, mode="reduced")
    if r.shape != (p, p):
        return None
    diagonal = np.abs(np.diag(r))
    pivot = float(diagonal.max()) if diagonal.size else 0.0
    tol = pivot * max(n, p) * np.finfo(float).eps
    if pivot == 0.0 or np.any(diagonal <= tol):
        return None
    beta = np.linalg.solve(r, q.T @ yw)
    resid = y - x @ beta
    rss = float(resid @ resid) if unity else float(w @ (resid * resid))
    if not np.isfinite(rss) or rss <= 0.0:
        return None
    hat = np.sum(q * q, axis=1)
    factor = np.linalg.inv(r)
    return _ols_fields(y, x, w, unity, beta, resid, rss, hat, factor @ factor.T)


def _pinv_extended(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-inverse and singular values, with ``statsmodels``' cutoff of 1e-15."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    cutoff = 1e-15 * float(s.max()) if s.size else 0.0
    s_inv = np.zeros_like(s)
    keep = s > cutoff
    s_inv[keep] = 1.0 / s[keep]
    return (vt.T * s_inv) @ u.T, s


def _ols_via_pinv(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Singular or non-positive-weight least squares. Same fields as the QR fit.

    The coefficients are the minimum-norm solution, as ``statsmodels`` returns.
    The rank sets the residual degrees of freedom.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    unity = bool(np.all(w == 1.0))
    if unity:
        xw = x
        yw = y
    else:
        scale = np.sqrt(w)
        xw = x * scale[:, None]
        yw = y * scale
    pinv, singular = _pinv_extended(xw)
    beta = pinv @ yw
    p = x.shape[1]
    rank = int(np.sum(singular > singular.max() * p * np.finfo(float).eps)) if singular.size else 0
    resid = y - x @ beta
    rss = float(resid @ resid) if unity else float(w @ (resid * resid))
    return _ols_fields(y, x, w, unity, beta, resid, rss, _hat(x, w), pinv @ pinv.T, rank)


def _solve_ols(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    solved = _ols_via_qr(y, x, w)
    if solved is None:
        return _ols_via_pinv(y, x, w)
    return solved


def fit_ols(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
) -> Fit:
    """Ordinary or weighted least squares, matching ``lm``.

    ``outcome`` may be a column or a Wilkinson formula such as
    ``"y ~ x * stage"``. A formula already names the predictors.
    Full-rank problems use a QR decomposition. A rank-deficient design falls
    back to a singular-value fit.
    """
    if is_formula(outcome):
        if predictors is not None:
            raise ValueError("pass a Wilkinson formula or predictors, not both")
        design, y, w, off = _from_formula(data, str(outcome), weights, offset)
    else:
        if predictors is None:
            raise ValueError("predictors are required when outcome is a column")
        y_series = column_series(data, outcome)
        extra: list[ColumnRef] = [outcome]
        if weights is not None:
            extra.append(weights)
        if offset is not None:
            extra.append(offset)
        design = design_matrix(data, predictors, extra=extra)
        y = _outcome_array(_take_aligned(y_series, design.row_index))
        w = _optional_numeric(data, weights, design.row_index)
        off = _optional_numeric(data, offset, design.row_index)
        if off is None:
            off = np.zeros(design.n_obs)
    response = y - off
    prior_w = w
    if w is None:
        w = np.ones(design.n_obs)
    solved = _solve_ols(response, design.x, np.asarray(w, dtype=float))
    working_weights = np.asarray(w, dtype=float)
    n = design.n_obs
    rss = float(solved["rss"])
    fit = Fit(
        coefficients=solved["coefficients"],
        covariance=solved["covariance"],
        names=list(design.names),
        n_obs=n,
        log_likelihood=solved["log_likelihood"],
        residual_df=solved["residual_df"],
        family="ols",
        x=design.x,
        y=y,
        row_index=design.row_index,
        design=design,
        weights=working_weights,
        offset=off,
        working_residuals=solved["residuals"],
        working_weights=working_weights,
        hat_values=solved["hat"],
        dispersion=rss / n,
        deviance=rss,
        scale=solved["sigma2"],
        aic=solved["aic"],
        bic=solved["bic"],
        r_squared=solved["r_squared"],
        adj_r_squared=solved["adj_r_squared"],
        f_statistic=solved["f_statistic"],
        f_p_value=solved["f_p_value"],
    )
    fit._refit = _ols_refit(y, off, None if prior_w is None else np.asarray(prior_w, dtype=float), design)
    return fit


def fit_glm(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    family: str = "gaussian",
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
) -> Fit:
    """GLM with the same default link as ``R``'s ``glm``.

    ``outcome`` may be a column or a Wilkinson formula such as
    ``"y ~ x + stage"``. ``family="negative_binomial"`` is NB2 with a log
    link, variance ``mu + mu^2 / theta``. It estimates ``theta`` by maximum
    likelihood, alternating with the coefficients as ``MASS::glm.nb`` does.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
    if is_formula(outcome):
        if predictors is not None:
            raise ValueError("pass a Wilkinson formula or predictors, not both")
        design, y, w, off = _from_formula(data, str(outcome), weights, offset)
        if w is None:
            w = np.ones(design.n_obs)
    else:
        if predictors is None:
            raise ValueError("predictors are required when outcome is a column")
        y_series = column_series(data, outcome)
        extra: list[ColumnRef] = [outcome]
        if weights is not None:
            extra.append(weights)
        if offset is not None:
            extra.append(offset)
        design = design_matrix(data, predictors, extra=extra)
        y = _outcome_array(_take_aligned(y_series, design.row_index))
        w = _optional_numeric(data, weights, design.row_index)
        off = _optional_numeric(data, offset, design.row_index)
        if w is None:
            w = np.ones(design.n_obs)
        if off is None:
            off = np.zeros(design.n_obs)
    fit = _fit_glm_arrays(
        y,
        design.x,
        family,
        np.asarray(w, dtype=float),
        np.asarray(off, dtype=float),
        list(design.names),
        design.row_index,
        design,
    )
    stored_w = None if weights is None else np.asarray(w, dtype=float)
    fit._refit = _glm_refit(y, np.asarray(off, dtype=float), stored_w, family, design)
    return fit


def _fit_glm_arrays(
    y: np.ndarray,
    x: np.ndarray,
    family: str,
    w: np.ndarray,
    off: np.ndarray,
    names: list[str],
    row_index: np.ndarray,
    design: Design,
) -> Fit:
    """Fit one GLM on arrays that are already aligned."""
    n = int(x.shape[0])
    if family == "negative_binomial":
        return _fit_negbin(y, x, w, off, names, row_index, design)
    if family == "gaussian":
        coef, log_likelihood, deviance, scale, rank = _gaussian_glm(y, x, w, off)
        working_resid = y - (x @ coef + off)
        df_resid = n - rank
        working_w, dispersion, cov, hat = _glm_sandwich_pieces(
            y, x, family, w, off, working_resid, df_resid
        )
        aic = -2.0 * log_likelihood + 2.0 * rank
        bic = -2.0 * log_likelihood + rank * np.log(n)
    else:
        # One IRLS supplies the coefficients and the weights R keeps from the
        # converging least-squares step. A second pass repeats that work and
        # stops at a different point under separation.
        coef, working_w, mu, deviance, working_resid = _glm_irls(y, x, family, w, off)
        rank = int(x.shape[1])
        counted = family in {"binomial", "poisson"}
        df_resid_f = float(np.sum(w) - rank) if counted else float(n - rank)
        df_resid = int(df_resid_f)
        if counted:
            dispersion = 1.0
            model_dispersion = 1.0
            scale = 1.0
        else:
            total = float(np.sum(working_w))
            dispersion = float(np.sum((working_resid * working_w) ** 2) / total) if total else np.nan
            model_dispersion = (
                float(np.sum(working_w * working_resid**2) / df_resid_f) if df_resid_f else np.nan
            )
            scale = model_dispersion
        inv = _xtwx_inv(x, working_w)
        cov = inv * model_dispersion
        hat = _hat(x, working_w)
        log_likelihood = _glm_loglik(family, y, mu, w, scale)
        wnobs = float(np.sum(w)) if counted else float(n)
        aic = -2.0 * log_likelihood + 2.0 * rank
        bic = -2.0 * log_likelihood + rank * np.log(wnobs) if wnobs > 0 else float("nan")
    return Fit(
        coefficients=coef,
        covariance=cov,
        names=names,
        n_obs=n,
        log_likelihood=log_likelihood,
        residual_df=df_resid,
        family=family,
        x=x,
        y=y,
        row_index=row_index,
        design=design,
        weights=np.asarray(w, dtype=float),
        offset=np.asarray(off, dtype=float),
        working_residuals=working_resid,
        working_weights=working_w,
        hat_values=hat,
        dispersion=dispersion,
        deviance=deviance,
        scale=scale,
        aic=aic,
        bic=bic,
    )


def _gaussian_glm(
    y: np.ndarray, x: np.ndarray, w: np.ndarray, off: np.ndarray
) -> tuple[np.ndarray, float, float, float, int]:
    """Identity-link Gaussian GLM: coefficients, log-likelihood, deviance, scale, rank.

    The log-likelihood is concentrated over the scale, as ``GLMResults.llf``
    reports it. The scale is the Pearson statistic over the residual degrees
    of freedom.
    """
    n = y.shape[0]
    sw = np.sqrt(w)
    pinv, _singular = _pinv_extended(x * sw[:, None])
    coef = pinv @ ((y - off) * sw)
    rank = int(np.linalg.matrix_rank(x))
    resid = y - (x @ coef + off)
    deviance = float(np.sum(w * resid**2))
    df_resid = n - rank
    scale = deviance / df_resid if df_resid else np.nan
    profile = deviance / n
    with np.errstate(divide="ignore"):
        ll = -0.5 * (w * resid**2 / profile + np.log(profile / w) + np.log(2.0 * np.pi))
    return coef, float(np.sum(ll)), deviance, float(scale), rank


def _glm_sandwich_pieces(
    y: np.ndarray,
    x: np.ndarray,
    family: str,
    prior: np.ndarray,
    offset: np.ndarray,
    working_resid: np.ndarray,
    df_resid: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Working weights, sandwich dispersion, model covariance, and hat values.

    The weights are those from the weighted least squares step that met
    ``glm``'s convergence test. The working residual is evaluated at the
    final mean. That is the pair ``sandwich::estfun.glm`` multiplies.
    """
    weights = _glm_irls_weights(y, x, family, prior, offset)
    if family in {"binomial", "poisson"}:
        sandwich_dispersion = 1.0
        model_dispersion = 1.0
    else:
        total = float(np.sum(weights))
        sandwich_dispersion = float(np.sum((working_resid * weights) ** 2) / total) if total else np.nan
        model_dispersion = (
            float(np.sum(weights * working_resid**2) / df_resid) if df_resid else np.nan
        )
    inv = _xtwx_inv(x, weights)
    return weights, sandwich_dispersion, inv * model_dispersion, _hat(x, weights)


def _glm_irls_weights(
    y: np.ndarray,
    x: np.ndarray,
    family: str,
    prior: np.ndarray,
    offset: np.ndarray,
    *,
    eps: float = 1e-8,
    maxit: int = 25,
) -> np.ndarray:
    """Working weights of the IRLS step that produced the final coefficients."""
    _coef, weights, _mu, _deviance, _resid = _glm_irls(
        y, x, family, prior, offset, eps=eps, maxit=maxit
    )
    return weights


def _glm_irls(
    y: np.ndarray,
    x: np.ndarray,
    family: str,
    prior: np.ndarray,
    offset: np.ndarray,
    *,
    eps: float = 1e-8,
    maxit: int = 25,
    theta: float | None = None,
    etastart: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Coefficients, weights, mean, deviance, and working residual.

    The working residual is evaluated at the final mean; the weights are
    those of the least-squares step that produced the final coefficients.
    The negative binomial needs ``theta``.
    ``etastart`` replaces the family's starting linear predictor.
    """
    y = np.asarray(y, dtype=float)
    prior = np.asarray(prior, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if family == "binomial":
        return _binomial_irls(y, x, prior, offset, eps, maxit)
    return _glm_irls_python(y, x, family, prior, offset, eps=eps, maxit=maxit, theta=theta, etastart=etastart)


def _glm_loglik(
    family: str,
    y: np.ndarray,
    mu: np.ndarray,
    prior: np.ndarray,
    scale: float,
) -> float:
    """Log-likelihood at the fitted mean, matching ``GLMResults.llf``."""
    if family == "binomial":
        # One trial per row. ``prior`` is the frequency weight.
        ll = (
            special.gammaln(2.0)
            - special.gammaln(y + 1.0)
            - special.gammaln(2.0 - y)
            + y * np.log((mu + 1e-20) / (1.0 - mu + 1e-20))
            + np.log(1.0 - mu + 1e-20)
        )
        return float(np.sum(ll * prior))
    if family == "poisson":
        ll = y * np.log(mu) - mu - special.gammaln(y + 1.0)
        return float(np.sum(prior * ll / scale))
    if family == "gamma":
        endog_mu = np.clip(y / mu, np.finfo(float).eps, np.inf)
        weight_scale = prior / scale
        ll = (
            weight_scale * np.log(weight_scale * endog_mu)
            - weight_scale * endog_mu
            - special.gammaln(weight_scale)
            - np.log(y)
        )
        return float(np.sum(ll))
    raise ValueError(family)


def _glm_start(y: np.ndarray, family: str, prior: np.ndarray, theta: float | None = None):
    """Starting linear predictor and family maps ``(linkinv, mu_eta, variance, deviance, eta)``.

    Each family uses its canonical or default link (McCullagh & Nelder 1989,
    Table 2.1): identity for gaussian, logit for binomial, log for poisson
    and negative binomial, inverse for gamma. ``deviance(y, mu, wt)`` is the
    weighted sum of unit deviances ``sum wt d(y, mu)`` with ``y log(y/mu) = 0``
    at ``y = 0``. The start follows McCullagh & Nelder (1989, sec. 2.5): put
    ``mu = y`` and move it off the edge of the mean space so that the link is
    finite there. Binomial proportions start at ``(prior y + 1/2) / (prior + 1)``
    (an empirical logit), counts at ``y + 0.1``; gaussian and gamma start at
    ``y`` itself. The IRLS stopping rule is on the deviance, so the point where
    it stops depends a little on the start; the count offset 0.1 was chosen
    because it reproduces R's ``glm()`` fixtures at rtol 1e-10 (offsets 0.05,
    0.25 and 0.5 stop up to about 4e-6 away in derived quantities).
    """
    y = np.asarray(y, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if family == "gaussian":
        def linkinv(eta):
            return np.asarray(eta, dtype=float)

        def mu_eta(eta):
            return np.ones_like(np.asarray(eta, dtype=float))

        def variance(mu):
            return np.ones_like(np.asarray(mu, dtype=float))

        def deviance(y, mu, wt):
            return float(np.sum(wt * (y - mu) ** 2))

        return linkinv, mu_eta, variance, deviance, y.copy()

    if family == "binomial":
        if np.any((y < 0) | (y > 1)):
            raise ValueError("binomial outcomes must be proportions in [0, 1]")

        def linkinv(eta):
            # Keep the mean strictly inside (0, 1) so the deviance stays finite.
            return np.clip(special.expit(eta), _MEAN_EDGE, 1.0 - _MEAN_EDGE)

        def mu_eta(eta):
            mu = special.expit(eta)
            return np.maximum(mu * (1.0 - mu), _MEAN_EDGE)

        def variance(mu):
            return mu * (1.0 - mu)

        def deviance(y, mu, wt):
            unit = special.xlogy(y, y / mu) + special.xlogy(1.0 - y, (1.0 - y) / (1.0 - mu))
            return float(np.sum(2.0 * wt * unit))

        start = (prior * y + 0.5) / (prior + 1.0)
        return linkinv, mu_eta, variance, deviance, np.log(start / (1.0 - start))

    if family in ("poisson", "negative_binomial"):
        if np.any(y < 0):
            raise ValueError(f"{family} outcomes must be non-negative counts")

        def linkinv(eta):
            # exp() is capped below overflow and floored above zero.
            return np.maximum(np.exp(np.minimum(eta, _LOG_MAX)), _MEAN_EDGE)

        mu_eta = linkinv
        if family == "poisson":
            def variance(mu):
                return mu

            def deviance(y, mu, wt):
                return float(np.sum(2.0 * wt * (special.xlogy(y, y / mu) - (y - mu))))
        else:
            if theta is None:
                raise ValueError("the negative binomial family needs theta")
            theta = float(theta)

            def variance(mu):
                return mu + mu * mu / theta

            def deviance(y, mu, wt):
                return float(np.sum(_negbin_dev_resids(y, mu, wt, theta)))

        return linkinv, mu_eta, variance, deviance, np.log(y + 0.1)

    if family == "gamma":
        if np.any(y <= 0):
            raise ValueError("gamma outcomes must be positive")

        def linkinv(eta):
            return 1.0 / eta

        def mu_eta(eta):
            return -1.0 / (eta * eta)

        def variance(mu):
            return mu * mu

        def deviance(y, mu, wt):
            return float(np.sum(-2.0 * wt * (np.log(y / mu) - (y - mu) / mu)))

        return linkinv, mu_eta, variance, deviance, 1.0 / y

    raise ValueError(f"unknown GLM family {family!r}")


# Smallest mean kept away from the edge of the mean space (binomial 0 and 1,
# log-link 0), and the largest exponent passed to exp().
_MEAN_EDGE = float(np.finfo(float).eps)
_LOG_MAX = float(np.log(np.finfo(float).max)) - 1.0


def _negbin_dev_resids(y: np.ndarray, mu: np.ndarray, wt: np.ndarray, theta: float) -> np.ndarray:
    """Deviance contributions of the NB2 family at fixed ``theta``.

    ``2 wt [y log(y/mu) - (y + theta) log((y + theta)/(mu + theta))]`` with
    ``y log(y/mu) = 0`` at ``y = 0``: twice the log-likelihood ratio of the
    saturated model (``mu = y``) against ``mu`` (Lawless 1987, sec. 2;
    Hilbe 2011, sec. 8.2).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(theta)
    return 2.0 * wt * (special.xlogy(y, y / mu) - (y + theta) * np.log((y + theta) / (mu + theta)))


def _negbin_loglik(theta: float, mu: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Negative binomial (NB2) log-likelihood at ``theta`` and means ``mu``.

    Sum over rows of ``w_i log f(y_i)`` with the NB2 mass function
    ``f(y) = Gamma(theta + y) / (Gamma(theta) y!) (theta / (theta + mu))^theta
    (mu / (theta + mu))^y`` (Lawless 1987, eq. 2.1; Hilbe 2011, eq. 8.6). The
    two power terms are expanded as ``theta log theta + y log mu -
    (theta + y) log(theta + mu)``; ``y log mu`` is taken as 0 when ``y = 0``.
    """
    theta = float(theta)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    per_row = (
        special.gammaln(theta + y)
        - special.gammaln(theta)
        - special.gammaln(y + 1.0)
        + theta * np.log(theta)
        + special.xlogy(y, mu)
        - (theta + y) * np.log(theta + mu)
    )
    return float(np.sum(w * per_row))


def _theta_score_information(
    theta: float, y: np.ndarray, mu: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    """Score and observed information of the NB2 log-likelihood in ``theta``.

    Differentiating the log-likelihood of :func:`_negbin_loglik` with the means
    held fixed (Lawless 1987, sec. 3; Hilbe 2011, sec. 8.2):

    ``U(theta) = sum w [psi(theta + y) - psi(theta) + log theta + 1
    - log(theta + mu) - (theta + y) / (theta + mu)]``

    ``I(theta) = -dU/dtheta = sum w [psi'(theta) - psi'(theta + y) - 1/theta
    + 2 / (theta + mu) - (theta + y) / (theta + mu)^2]``.
    """
    shifted = theta + mu
    score = np.sum(
        w
        * (
            special.digamma(theta + y)
            - special.digamma(theta)
            + np.log(theta)
            + 1.0
            - np.log(shifted)
            - (theta + y) / shifted
        )
    )
    information = np.sum(
        w
        * (
            special.polygamma(1, theta)
            - special.polygamma(1, theta + y)
            - 1.0 / theta
            + 2.0 / shifted
            - (theta + y) / shifted**2
        )
    )
    return float(score), float(information)


def _theta_ml(y: np.ndarray, mu: np.ndarray, w: np.ndarray, limit: int = 25) -> tuple[float, float]:
    """Maximum-likelihood negative binomial theta given fitted means.

    Returns ``(theta, standard_error)``.

    Newton-Raphson on the score in ``theta`` with the means fixed (Lawless
    1987, sec. 3), started at the moment estimate ``n / sum w (y/mu - 1)^2``
    with ``n = sum w`` (Hilbe 2011, sec. 8.2). Each update ``theta + U / I``
    is folded back to positive values; iteration stops once the update is no
    larger than ``eps^(1/4)`` or after ``limit`` updates. Weights act as
    frequency weights. The standard error is ``I(theta)^(-1/2)``.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    total_weight = float(np.sum(w))
    theta = total_weight / float(np.sum(w * (y / mu - 1.0) ** 2))
    tolerance = np.finfo(float).eps ** 0.25
    updates = 0
    step = np.inf
    information = float("nan")
    while updates < limit and abs(step) > tolerance:
        theta = abs(theta)
        score, information = _theta_score_information(theta, y, mu, w)
        step = score / information
        theta = theta + step
        updates += 1
    if theta < 0.0:
        theta = 0.0
        warnings.warn("negative binomial theta fell below zero and was set to 0", RuntimeWarning, stacklevel=2)
    if updates == limit and abs(step) > tolerance:
        warnings.warn(f"negative binomial theta did not converge in {limit} Newton steps", RuntimeWarning, stacklevel=2)
    # The information is the one of the last Newton step; the final update is
    # below eps^(1/4), so it differs from the information at the returned
    # theta by a relative amount of that order.
    standard_error = float(np.sqrt(1.0 / information)) if information > 0.0 else float("nan")
    return float(theta), standard_error


def _fit_negbin(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    off: np.ndarray,
    names: list[str],
    row_index: np.ndarray,
    design: Design,
    *,
    eps: float = 1e-8,
    maxit: int = 25,
) -> Fit:
    """NB2 (log link) regression with theta estimated by maximum likelihood. Returns :class:`Fit`.

    Alternating maximisation (Lawless 1987, sec. 3; Venables & Ripley 2002,
    sec. 7.4; Hilbe 2011, sec. 8.2): start from a Poisson GLM and the moment
    estimate of ``theta`` at its means, then repeat

    1. ``theta`` by :func:`_theta_ml` at the current means;
    2. IRLS for the coefficients at that ``theta``, started from the current
       means (``etastart = log mu``);
    3. the log-likelihood at that ``theta`` and the new means.

    The returned coefficients, working weights and deviance are therefore
    those of an IRLS at the returned ``theta``.

    A round settles the fit when the change in log-likelihood, in units of
    the standard deviation of a chi-square on the residual degrees of
    freedom (``sqrt(2 max(1, df))``), plus the absolute change in ``theta``
    is at most ``eps``. At most ``maxit`` rounds are run. The coefficient covariance treats
    ``theta`` as known.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    off = np.asarray(off, dtype=float)
    if np.any(y < 0):
        raise ValueError("negative binomial outcomes must be non-negative counts")
    n = int(x.shape[0])
    n_ok = int(np.count_nonzero(w))
    rank = int(np.linalg.matrix_rank(x[w != 0])) if n_ok else 0
    residual_df = n_ok - rank
    # Log-likelihood changes are measured in standard deviations of a
    # chi-square on the residual degrees of freedom.
    chisq_sd = float(np.sqrt(2.0 * max(1, residual_df)))

    def round_change(ll_before: float, ll_after: float, theta_step: float) -> float:
        return abs(ll_before - ll_after) / chisq_sd + abs(theta_step)

    coef, working_w, mu, deviance, working_resid = _glm_irls(
        y, x, "poisson", w, off, eps=eps, maxit=maxit
    )
    # Starting theta: the moment estimate at the Poisson means (?glm.nb).
    theta = float(np.sum(w)) / float(np.sum(w * (y / mu - 1.0) ** 2))
    theta_se = float("nan")
    log_likelihood = _negbin_loglik(theta, mu, y, w)
    settled = False
    for _round in range(maxit):
        theta_before, ll_before = theta, log_likelihood
        theta, theta_se = _theta_ml(y, mu, w, limit=maxit)
        coef, working_w, mu, deviance, working_resid = _glm_irls(
            y, x, "negative_binomial", w, off, eps=eps, maxit=maxit, theta=theta, etastart=np.log(mu)
        )
        log_likelihood = _negbin_loglik(theta, mu, y, w)
        if round_change(ll_before, log_likelihood, theta_before - theta) <= eps:
            settled = True
            break
    if not settled:
        warnings.warn(
            f"negative binomial fit: theta and the coefficients did not settle in {maxit} rounds",
            RuntimeWarning,
            stacklevel=2,
        )

    covariance = _xtwx_inv(x, working_w)
    hat = _hat(x, working_w)
    n_params = rank + 1
    return Fit(
        coefficients=coef,
        covariance=covariance,
        names=names,
        n_obs=n,
        log_likelihood=log_likelihood,
        residual_df=residual_df,
        family="negative_binomial",
        x=x,
        y=y,
        row_index=row_index,
        design=design,
        weights=np.asarray(w, dtype=float),
        offset=np.asarray(off, dtype=float),
        working_residuals=working_resid,
        working_weights=working_w,
        hat_values=hat,
        dispersion=1.0,
        deviance=float(deviance),
        scale=1.0,
        aic=float(-2.0 * log_likelihood + 2.0 * n_params),
        bic=float(-2.0 * log_likelihood + n_params * np.log(n_ok)),
        theta=float(theta),
        theta_std_error=float(theta_se),
    )


def _binomial_irls(y, x, prior, offset, eps, maxit):
    """Logit IRLS in one compiled loop. The returned pieces match ``_glm_irls``."""
    try:
        coef, weights, mu, deviance, working, eta = _binomial_irls_numba(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(prior, dtype=np.float64),
            np.ascontiguousarray(offset, dtype=np.float64),
            float(eps),
            int(maxit),
        )
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return _glm_irls_python(y, x, "binomial", prior, offset, eps=eps, maxit=maxit)
    if not np.all(np.isfinite(coef)):
        # A singular step falls back to the interpreted loop, which uses lstsq.
        return _glm_irls_python(y, x, "binomial", prior, offset, eps=eps, maxit=maxit)
    return coef, weights, mu, float(deviance), working


# Relative pivot below which a Cholesky factor of X'WX is treated as singular
# and the weighted least-squares step is redone by a minimum-norm solve.
_GRAM_PIVOT_TOL = 1e-12


def _weighted_ls_step(x: np.ndarray, z: np.ndarray, root_w: np.ndarray) -> np.ndarray:
    """Weighted least squares ``argmin ||root_w * (z - X b)||``.

    Solves the normal equations ``X'WX b = X'Wz`` by Cholesky; when the Gram
    matrix is singular to working precision, or the result is not finite,
    falls back to an SVD least-squares solve, which returns the minimum-norm
    solution for a rank-deficient design.
    """
    xw = x * root_w[:, None]
    zw = z * root_w
    gram = xw.T @ xw
    rhs = xw.T @ zw
    p = gram.shape[0]
    if p == 0:
        return np.zeros(0)
    diagonal = np.diag(gram).copy()
    if np.all(np.isfinite(gram)) and np.all(diagonal > 0.0):
        try:
            factor = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            factor = None
        if factor is not None and np.all(np.diag(factor) ** 2 > _GRAM_PIVOT_TOL * diagonal):
            half = linalg.solve_triangular(factor, rhs, lower=True)
            coef = linalg.solve_triangular(factor.T, half, lower=False)
            if np.all(np.isfinite(coef)):
                return coef
    coef, *_ = np.linalg.lstsq(xw, zw, rcond=None)
    return coef


def _glm_irls_python(y, x, family, prior, offset, *, eps, maxit, theta=None, etastart=None):
    """IRLS fit of one GLM family. Returns ``(coef, weights, mu, deviance, working_residual)``.

    Fisher scoring written as iteratively reweighted least squares (Nelder &
    Wedderburn 1972; McCullagh & Nelder 1989, sec. 2.5, eqs. 2.12-2.13;
    Green 1984): with ``eta = X b + offset`` and ``mu = g^{-1}(eta)``, regress
    the working response ``z = eta - offset + (y - mu) / (dmu/deta)`` on
    ``X`` with weights ``W = prior (dmu/deta)^2 / V(mu)``. Rows with zero prior
    weight or a zero / non-finite ``dmu/deta`` sit out the step. A step whose
    deviance is not finite is halved back toward the previous coefficients.
    Iteration stops when ``|D - D_old| / (|D| + 0.1) < eps`` (the rule
    documented for R's ``glm.control``) or after ``maxit`` steps.

    ``weights`` are those of the least-squares step that produced the
    returned coefficients; the working residual and deviance are at the
    final mean.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    prior = np.asarray(prior, dtype=float)
    offset = np.asarray(offset, dtype=float)
    n, p = x.shape
    linkinv, mu_eta, variance, deviance_of, eta = _glm_start(y, family, prior, theta)
    if etastart is not None:
        eta = np.asarray(etastart, dtype=float).copy()
    mu = linkinv(eta)
    last_deviance = deviance_of(y, mu, prior)
    last_coef: np.ndarray | None = None
    coef = np.zeros(p)
    step_weights = np.zeros(n)
    deviance = last_deviance
    for _iteration in range(max(int(maxit), 1)):
        slope = mu_eta(eta)
        var = variance(mu)
        usable = (prior > 0.0) & np.isfinite(slope) & (slope != 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            working_weight = np.where(usable, prior * slope**2 / var, 0.0)
            working_response = np.where(usable, eta - offset + (y - mu) / slope, 0.0)
        usable &= np.isfinite(working_weight) & np.isfinite(working_response)
        if not np.any(usable):
            raise ValueError("no observations are usable in the IRLS step")
        working_weight = np.where(usable, working_weight, 0.0)
        coef = _weighted_ls_step(
            x[usable], working_response[usable], np.sqrt(working_weight[usable])
        )
        eta = x @ coef + offset
        mu = linkinv(eta)
        with np.errstate(all="ignore"):
            deviance = deviance_of(y, mu, prior)
        halvings = 0
        while not np.isfinite(deviance):
            if last_coef is None or halvings >= max(int(maxit), 1):
                raise ValueError("IRLS found no coefficients with a finite deviance")
            halvings += 1
            coef = 0.5 * (coef + last_coef)
            eta = x @ coef + offset
            mu = linkinv(eta)
            with np.errstate(all="ignore"):
                deviance = deviance_of(y, mu, prior)
        step_weights = working_weight
        converged = abs(deviance - last_deviance) / (abs(deviance) + 0.1) < eps
        last_deviance = deviance
        last_coef = coef
        if converged:
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        working_residual = (y - mu) / mu_eta(eta)
    return coef, step_weights, mu, float(deviance), working_residual


@njit(cache=True)
def _logit_deviance(y, mu, prior):
    """Binomial deviance 2 sum w [y log(y/mu) + (1-y) log((1-y)/(1-mu))]
    with mu clipped to [1e-12, 1 - 1e-12] (McCullagh & Nelder 1989, sec. 4.4)."""
    total = 0.0
    for i in range(y.shape[0]):
        m = min(max(mu[i], 1e-12), 1.0 - 1e-12)
        yi = y[i]
        term = 0.0
        if yi > 0.0:
            term += yi * np.log(yi / m)
        if yi < 1.0:
            term += (1.0 - yi) * np.log((1.0 - yi) / (1.0 - m))
        total += prior[i] * 2.0 * term
    return total


@njit(cache=True)
def _logistic(value):
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    e = np.exp(value)
    return e / (1.0 + e)


@njit(cache=True)
def _binomial_irls_numba(y, x, prior, offset, eps, maxit):
    """Logit-link binomial IRLS on contiguous float64 arrays.

    Returns ``(coef, weights, mu, deviance, working_residual, eta)``.

    The same scoring iteration as ``_glm_irls_python`` (McCullagh & Nelder
    1989, sec. 2.5) specialised to the canonical logit link, where
    ``dmu/deta = V(mu) = mu (1 - mu)`` and the working weight is
    ``prior mu (1 - mu)``. Start ``mu = (prior y + 0.5) / (prior + 1)``. The
    normal equations are solved by a Cholesky factorisation; a singular or
    non-finite step returns NaN coefficients so that the caller can redo the
    fit with the least-squares fallback.
    """
    n, p = x.shape
    eta = np.empty(n)
    mu = np.empty(n)
    for i in range(n):
        start = (prior[i] * y[i] + 0.5) / (prior[i] + 1.0)
        eta[i] = np.log(start / (1.0 - start))
        mu[i] = start
    last_deviance = _logit_deviance(y, mu, prior)
    deviance = last_deviance
    coef = np.zeros(p)
    last_coef = np.zeros(p)
    have_old = False
    weights = np.zeros(n)
    step_weights = np.zeros(n)
    gram = np.empty((p, p))
    rhs = np.empty(p)
    chol = np.empty((p, p))
    iterations = maxit if maxit > 0 else 1
    for _iteration in range(iterations):
        # Accumulate X'WX and X'Wz over the usable rows.
        gram[:, :] = 0.0
        rhs[:] = 0.0
        for i in range(n):
            slope = mu[i] * (1.0 - mu[i])
            if prior[i] > 0.0 and slope != 0.0 and np.isfinite(slope):
                w = prior[i] * slope
                z = eta[i] - offset[i] + (y[i] - mu[i]) / slope
                weights[i] = w
                for a in range(p):
                    xa = x[i, a] * w
                    rhs[a] += xa * z
                    for b in range(a + 1):
                        gram[a, b] += xa * x[i, b]
            else:
                weights[i] = 0.0
        # Cholesky factor of the Gram matrix with a relative pivot check.
        chol[:, :] = 0.0
        for a in range(p):
            for b in range(a + 1):
                s = gram[a, b]
                for k in range(b):
                    s -= chol[a, k] * chol[b, k]
                if a == b:
                    if not (s > 1e-12 * gram[a, a]) or not np.isfinite(s):
                        coef[:] = np.nan
                        return coef, step_weights, mu, deviance, mu, eta
                    chol[a, a] = np.sqrt(s)
                else:
                    chol[a, b] = s / chol[b, b]
        for a in range(p):
            s = rhs[a]
            for k in range(a):
                s -= chol[a, k] * coef[k]
            coef[a] = s / chol[a, a]
        for a in range(p - 1, -1, -1):
            s = coef[a]
            for k in range(a + 1, p):
                s -= chol[k, a] * coef[k]
            coef[a] = s / chol[a, a]
        for a in range(p):
            if not np.isfinite(coef[a]):
                coef[:] = np.nan
                return coef, step_weights, mu, deviance, mu, eta
        for i in range(n):
            s = offset[i]
            for a in range(p):
                s += x[i, a] * coef[a]
            eta[i] = s
            mu[i] = _logistic(s)
        deviance = _logit_deviance(y, mu, prior)
        halvings = 0
        while not np.isfinite(deviance):
            if not have_old or halvings >= iterations:
                coef[:] = np.nan
                return coef, step_weights, mu, deviance, mu, eta
            halvings += 1
            for a in range(p):
                coef[a] = 0.5 * (coef[a] + last_coef[a])
            for i in range(n):
                s = offset[i]
                for a in range(p):
                    s += x[i, a] * coef[a]
                eta[i] = s
                mu[i] = _logistic(s)
            deviance = _logit_deviance(y, mu, prior)
        step_weights[:] = weights
        converged = abs(deviance - last_deviance) / (abs(deviance) + 0.1) < eps
        last_deviance = deviance
        last_coef[:] = coef
        have_old = True
        if converged:
            break
    working = np.empty(n)
    for i in range(n):
        slope = mu[i] * (1.0 - mu[i])
        working[i] = (y[i] - mu[i]) / slope if slope != 0.0 else np.inf * (y[i] - mu[i])
    return coef, step_weights, mu, deviance, working, eta


def _take_aligned(series: pl.Series, row_index: np.ndarray) -> pl.Series:
    if series.null_count() == 0 and int(row_index.shape[0]) == series.len():
        return series
    return series.gather(np.asarray(row_index, dtype=np.int64))


def _optional_numeric(
    data: pl.DataFrame,
    ref: ColumnRef | None,
    row_index: np.ndarray,
) -> np.ndarray | None:
    if ref is None:
        return None
    series = column_series(data, ref)
    return np.asarray(_take_aligned(series, row_index).to_numpy(), dtype=float)


def _from_formula(
    data: pl.DataFrame,
    formula: str,
    weights: ColumnRef | None,
    offset: ColumnRef | None,
) -> tuple[Design, np.ndarray, np.ndarray | None, np.ndarray]:
    """Response, optional weights, and offset aligned to a Wilkinson design."""
    built = model_matrix(formula, data)
    if built.random_effects:
        raise ValueError("random effects belong in fit_mixed, for example (1 | group)")
    reject_survival_syntax(built)
    design = built.design
    y = np.asarray(built.y, dtype=float)
    off = np.zeros(design.n_obs) if built.offset is None else np.asarray(built.offset, dtype=float)
    keep = np.ones(design.n_obs, dtype=bool)
    w = None
    if weights is not None:
        values = column_series(data, weights).gather(design.row_index.tolist())
        keep &= values.is_not_null().to_numpy()
        w = np.asarray(values.to_numpy(), dtype=float)
    if offset is not None:
        values = column_series(data, offset).gather(design.row_index.tolist())
        keep &= values.is_not_null().to_numpy()
        off = off + np.asarray(values.to_numpy(), dtype=float)
    if not bool(keep.all()):
        design = _subset_design(design, keep)
        y = y[keep]
        off = off[keep]
        if w is not None:
            w = w[keep]
    return design, y, w, off


def _subset_design(design: Design, keep: np.ndarray) -> Design:
    return Design(
        x=design.x[keep],
        names=list(design.names),
        row_index=design.row_index[keep],
        predictors=list(design.predictors),
        intercept=design.intercept,
        recipes=design.recipes,
        factor_levels=design.factor_levels,
        computed=design.computed,
    )


def _ols_refit(
    y_values: np.ndarray,
    offset_values: np.ndarray,
    weight_values: np.ndarray | None,
    design: Design,
):
    def refit(index: np.ndarray, y: np.ndarray | None = None) -> Fit:
        x = design.x[index]
        if y is None:
            response = (y_values - offset_values)[index]
        else:
            response = np.asarray(y, dtype=float)
        ww = np.ones(index.shape[0]) if weight_values is None else weight_values[index]
        solved = _solve_ols(response, x, np.asarray(ww, dtype=float))
        n = x.shape[0]
        rss = float(solved["rss"])
        child = Fit(
            coefficients=solved["coefficients"],
            covariance=solved["covariance"],
            names=list(design.names),
            n_obs=n,
            log_likelihood=solved["log_likelihood"],
            residual_df=solved["residual_df"],
            family="ols",
            x=x,
            y=response,
            row_index=design.row_index[index],
            design=design,
            weights=ww,
            offset=np.zeros(n),
            working_residuals=solved["residuals"],
            working_weights=ww,
            hat_values=solved["hat"],
            dispersion=rss / n,
            deviance=rss,
            scale=solved["sigma2"],
        )
        return child

    return refit


def _glm_refit(
    y_values: np.ndarray,
    offset_values: np.ndarray,
    weight_values: np.ndarray | None,
    family: str,
    design: Design,
):
    def refit(index: np.ndarray, y: np.ndarray | None = None) -> Fit:
        x = design.x[index]
        response = y_values[index] if y is None else np.asarray(y, dtype=float)
        w = np.ones(index.shape[0]) if weight_values is None else weight_values[index]
        off = offset_values[index]
        return _fit_glm_arrays(
            response,
            x,
            family,
            w,
            off,
            list(design.names),
            design.row_index[index],
            design,
        )

    return refit
