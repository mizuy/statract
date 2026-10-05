"""Ordinary and generalized linear models on Polars frames.

``fit_ols`` follows ``lm`` (residual scale, t intervals). ``fit_glm`` follows
``glm`` for gaussian, binomial, poisson, and gamma, with the same default links
(identity, logit, log, inverse). ``tidy`` uses z intervals for GLMs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
from numba import njit
from scipy import special, stats

from .design import ColumnRef, Design, build_design, column_series, design_matrix
from .formula import is_formula, model_matrix, reject_survival_syntax

_FAMILIES = ("gaussian", "binomial", "poisson", "gamma")
_LINKS = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
}


def _family_object(family: str):
    import statsmodels.api as sm

    if family == "gaussian":
        return sm.families.Gaussian()
    if family == "binomial":
        return sm.families.Binomial()
    if family == "poisson":
        return sm.families.Poisson()
    if family == "gamma":
        return sm.families.Gamma()
    raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")


def _link_from_eta(family: str, eta: np.ndarray) -> np.ndarray:
    if family == "gaussian":
        return eta
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    if family == "poisson":
        return np.exp(eta)
    if family == "gamma":
        return 1.0 / eta
    raise ValueError(family)


@dataclass
class Fit:
    """Fitted linear or generalized linear model.

    ``covariance`` is the model-based covariance. Sandwich estimators live in
    ``covariance.py`` and read ``score_contributions`` and ``bread``.
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
    w = np.asarray(weights, dtype=float)
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
    """
    import statsmodels.api as sm

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
        result = sm.OLS(response, design.x, missing="raise").fit()
    else:
        result = sm.WLS(response, design.x, weights=w, missing="raise").fit()
    resid = np.asarray(result.resid, dtype=float)
    rss = float(np.sum(w * resid**2))
    df_resid = int(result.df_resid)
    sigma2 = rss / df_resid if df_resid else np.nan
    cov = np.asarray(result.normalized_cov_params, dtype=float) * sigma2
    working_weights = np.asarray(w, dtype=float)
    hat = _hat(design.x, working_weights)
    n = design.n_obs
    ll = float(result.llf)
    fit = Fit(
        coefficients=np.asarray(result.params, dtype=float),
        covariance=cov,
        names=list(design.names),
        n_obs=n,
        log_likelihood=ll,
        residual_df=df_resid,
        family="ols",
        x=design.x,
        y=y,
        row_index=design.row_index,
        design=design,
        weights=working_weights,
        offset=off,
        working_residuals=resid,
        working_weights=working_weights,
        hat_values=hat,
        dispersion=rss / n,
        deviance=rss,
        scale=sigma2,
        aic=float(result.aic),
        bic=float(result.bic),
        r_squared=float(result.rsquared),
        adj_r_squared=float(result.rsquared_adj),
        f_statistic=float(result.fvalue) if result.fvalue is not None else None,
        f_p_value=float(result.f_pvalue) if result.f_pvalue is not None else None,
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
    ``"y ~ x + stage"``.
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
    if family == "gaussian":
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The InversePower link function")
            result = sm.GLM(
                y, x, family=_family_object(family), offset=off, var_weights=w
            ).fit(maxiter=100, tol=1e-12, disp=0)
        working_resid = np.asarray(result.resid_working, dtype=float)
        df_resid = int(result.df_resid)
        working_w, dispersion, cov, hat = _glm_sandwich_pieces(
            y, x, family, w, off, working_resid, df_resid
        )
        coef = np.asarray(result.params, dtype=float)
        log_likelihood = float(result.llf)
        deviance = float(result.deviance)
        scale = float(result.scale)
        aic = float(result.aic)
        bic = float(getattr(result, "bic_llf", result.bic))
    else:
        # One IRLS supplies the coefficients and the weights R keeps from the
        # converging least-squares step. A second pass, or statsmodels' own
        # IRLS, repeats that work and stops at a different point under separation.
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
    """IRLS weights stored by ``glm.fit`` after the converging iteration."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Coefficients, weights, mean, deviance, and working residual.

    ``glm.fit`` recomputes the working residual at the final mean, and keeps
    the weights from the least-squares step that produced those coefficients.
    The step solves the normal equations. A singular step falls back to a
    least-squares factorization.
    """
    y = np.asarray(y, dtype=float)
    prior = np.asarray(prior, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if family == "binomial":
        return _binomial_irls(y, x, prior, offset, eps, maxit)
    linkinv, mu_eta, variance, deviance, eta = _glm_start(y, family, prior)
    mu = linkinv(eta)
    dev_old = deviance(y, mu, prior)
    dev_new = dev_old
    n = y.shape[0]
    good = np.ones(n, dtype=bool)
    w_good = np.ones(n)
    coef = np.zeros(x.shape[1])
    coef_old: np.ndarray | None = None
    for _ in range(maxit):
        mu_eta_val = mu_eta(eta)
        good = (prior > 0) & (mu_eta_val != 0) & np.isfinite(mu_eta_val)
        if not np.any(good):
            w_good = np.ones(0)
            break
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta_val[good]
        w_good = np.sqrt((prior[good] * mu_eta_val[good] ** 2) / variance(mu[good]))
        coef = _weighted_coef(x[good], z, w_good)
        eta = x @ coef + offset
        mu = linkinv(eta)
        dev_new = deviance(y, mu, prior)
        if coef_old is not None and not np.isfinite(dev_new):
            coef = 0.5 * (coef + coef_old)
            eta = x @ coef + offset
            mu = linkinv(eta)
            dev_new = deviance(y, mu, prior)
        if abs(dev_new - dev_old) / (0.1 + abs(dev_new)) < eps:
            break
        dev_old = dev_new
        coef_old = coef
    weights = np.zeros(n)
    weights[good] = w_good**2
    mu_eta_final = mu_eta(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        working = (y - mu) / mu_eta_final
    return coef, weights, mu, float(dev_new), working


def _weighted_coef(x_good: np.ndarray, z: np.ndarray, w_good: np.ndarray) -> np.ndarray:
    """Solve weighted least squares for one IRLS step."""
    sw = w_good**2
    xtw = x_good.T * sw
    try:
        coef = np.linalg.solve(xtw @ x_good, xtw @ z)
    except np.linalg.LinAlgError:
        coef = None
    if coef is None or not np.all(np.isfinite(coef)):
        coef, *_ = np.linalg.lstsq(x_good * w_good[:, None], z * w_good, rcond=None)
    return coef


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


def _glm_start(y: np.ndarray, family: str, prior: np.ndarray):
    """Initial linear predictor and family maps, matching ``glm``'s defaults."""
    if family == "binomial":
        mustart = (prior * y + 0.5) / (prior + 1.0)
        eta = np.log(mustart / (1.0 - mustart))

        def linkinv(eta: np.ndarray) -> np.ndarray:
            return _expit(eta)

        def mu_eta(eta: np.ndarray) -> np.ndarray:
            mu = linkinv(eta)
            return mu * (1.0 - mu)

        def variance(mu: np.ndarray) -> np.ndarray:
            return mu * (1.0 - mu)

        def deviance(y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
            mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
            out = np.empty_like(y)
            zero = y == 0
            one = y == 1
            mid = ~(zero | one)
            out[zero] = -2.0 * np.log(1.0 - mu[zero])
            out[one] = -2.0 * np.log(mu[one])
            if np.any(mid):
                yy = y[mid]
                mm = mu[mid]
                out[mid] = 2.0 * (yy * np.log(yy / mm) + (1.0 - yy) * np.log((1.0 - yy) / (1.0 - mm)))
            return float(np.sum(wt * out))

    elif family == "poisson":
        eta = np.log(y + 0.1)

        def linkinv(eta: np.ndarray) -> np.ndarray:
            return np.exp(eta)

        def mu_eta(eta: np.ndarray) -> np.ndarray:
            return np.exp(eta)

        def variance(mu: np.ndarray) -> np.ndarray:
            return mu

        def deviance(y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
            mu = np.maximum(mu, 1e-16)
            resid = mu * wt
            pos = y > 0
            resid = resid.copy()
            resid[pos] = (wt * (y * np.log(np.maximum(y, 1e-16) / mu) - (y - mu)))[pos]
            return float(np.sum(2.0 * resid))

    elif family == "gamma":
        eta = 1.0 / np.maximum(y, 1e-8)

        def linkinv(eta: np.ndarray) -> np.ndarray:
            return 1.0 / eta

        def mu_eta(eta: np.ndarray) -> np.ndarray:
            return -1.0 / np.square(eta)

        def variance(mu: np.ndarray) -> np.ndarray:
            return np.square(mu)

        def deviance(y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
            ratio = np.where(y == 0, 1.0, y / mu)
            return float(np.sum(-2.0 * wt * (np.log(ratio) - (y - mu) / mu)))

    elif family == "gaussian":
        eta = np.array(y, dtype=float, copy=True)

        def linkinv(eta: np.ndarray) -> np.ndarray:
            return eta

        def mu_eta(eta: np.ndarray) -> np.ndarray:
            return np.ones_like(eta)

        def variance(mu: np.ndarray) -> np.ndarray:
            return np.ones_like(mu)

        def deviance(y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
            return float(np.sum(wt * (y - mu) ** 2))

    else:
        raise ValueError(family)
    return linkinv, mu_eta, variance, deviance, eta


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


def _glm_irls_python(y, x, family, prior, offset, *, eps, maxit):
    linkinv, mu_eta, variance, deviance, eta = _glm_start(y, family, prior)
    mu = linkinv(eta)
    dev_old = deviance(y, mu, prior)
    dev_new = dev_old
    n = y.shape[0]
    good = np.ones(n, dtype=bool)
    w_good = np.ones(n)
    coef = np.zeros(x.shape[1])
    coef_old: np.ndarray | None = None
    for _ in range(maxit):
        mu_eta_val = mu_eta(eta)
        good = (prior > 0) & (mu_eta_val != 0) & np.isfinite(mu_eta_val)
        if not np.any(good):
            w_good = np.ones(0)
            break
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta_val[good]
        w_good = np.sqrt((prior[good] * mu_eta_val[good] ** 2) / variance(mu[good]))
        coef = _weighted_coef(x[good], z, w_good)
        eta = x @ coef + offset
        mu = linkinv(eta)
        dev_new = deviance(y, mu, prior)
        if coef_old is not None and not np.isfinite(dev_new):
            coef = 0.5 * (coef + coef_old)
            eta = x @ coef + offset
            mu = linkinv(eta)
            dev_new = deviance(y, mu, prior)
        if abs(dev_new - dev_old) / (0.1 + abs(dev_new)) < eps:
            break
        dev_old = dev_new
        coef_old = coef
    weights = np.zeros(n)
    weights[good] = w_good**2
    mu_eta_final = mu_eta(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        working = (y - mu) / mu_eta_final
    return coef, weights, mu, float(dev_new), working


@njit(cache=True)
def _expit_vec(eta):
    out = np.empty_like(eta)
    for i in range(eta.shape[0]):
        z = eta[i]
        if z >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            out[i] = ez / (1.0 + ez)
    return out


@njit(cache=True)
def _binom_deviance(y, mu, wt):
    total = 0.0
    for i in range(y.shape[0]):
        m = mu[i]
        if m < 1e-12:
            m = 1e-12
        elif m > 1.0 - 1e-12:
            m = 1.0 - 1e-12
        yi = y[i]
        if yi == 0.0:
            term = -2.0 * np.log(1.0 - m)
        elif yi == 1.0:
            term = -2.0 * np.log(m)
        else:
            term = 2.0 * (yi * np.log(yi / m) + (1.0 - yi) * np.log((1.0 - yi) / (1.0 - m)))
        total += wt[i] * term
    return total


@njit(cache=True)
def _binomial_irls_numba(y, x, prior, offset, eps, maxit):
    n = y.shape[0]
    p = x.shape[1]
    mustart = (prior * y + 0.5) / (prior + 1.0)
    eta = np.log(mustart / (1.0 - mustart))
    mu = _expit_vec(eta)
    dev_old = _binom_deviance(y, mu, prior)
    dev_new = dev_old
    coef = np.zeros(p)
    coef_old = np.zeros(p)
    have_old = False
    good = np.ones(n, dtype=np.uint8)
    weights = np.zeros(n)
    for _ in range(maxit):
        mu_eta = mu * (1.0 - mu)
        n_good = 0
        for i in range(n):
            ok = prior[i] > 0.0 and mu_eta[i] != 0.0 and np.isfinite(mu_eta[i])
            good[i] = 1 if ok else 0
            n_good += good[i]
        if n_good == 0:
            break
        x_good = np.empty((n_good, p))
        z = np.empty(n_good)
        w_good = np.empty(n_good)
        g = 0
        for i in range(n):
            if good[i] == 0:
                continue
            x_good[g] = x[i]
            z[g] = (eta[i] - offset[i]) + (y[i] - mu[i]) / mu_eta[i]
            w_good[g] = np.sqrt((prior[i] * mu_eta[i] * mu_eta[i]) / (mu[i] * (1.0 - mu[i])))
            g += 1
        sw = w_good * w_good
        xtw = np.empty((p, n_good))
        for j in range(p):
            for i in range(n_good):
                xtw[j, i] = x_good[i, j] * sw[i]
        xtwx = xtw @ x_good
        xty = xtw @ z
        step = np.linalg.solve(xtwx, xty)
        if not np.all(np.isfinite(step)):
            coef = np.full(p, np.nan)
            break
        coef = step
        eta = x @ coef + offset
        mu = _expit_vec(eta)
        dev_new = _binom_deviance(y, mu, prior)
        if have_old and not np.isfinite(dev_new):
            coef = 0.5 * (coef + coef_old)
            eta = x @ coef + offset
            mu = _expit_vec(eta)
            dev_new = _binom_deviance(y, mu, prior)
        for i in range(n):
            weights[i] = 0.0
        g = 0
        for i in range(n):
            if good[i] == 0:
                continue
            weights[i] = w_good[g] * w_good[g]
            g += 1
        if abs(dev_new - dev_old) / (0.1 + abs(dev_new)) < eps:
            break
        dev_old = dev_new
        coef_old = coef.copy()
        have_old = True
    mu_eta_final = mu * (1.0 - mu)
    working = np.empty(n)
    for i in range(n):
        if mu_eta_final[i] == 0.0:
            working[i] = np.nan
        else:
            working[i] = (y[i] - mu[i]) / mu_eta_final[i]
    return coef, weights, mu, dev_new, working, eta


def _expit(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta, dtype=float)
    positive = eta >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    out[~positive] = exp_eta / (1.0 + exp_eta)
    return out


def _take_aligned(series: pl.Series, row_index: np.ndarray) -> pl.Series:
    return series.gather(row_index.tolist())


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
        import statsmodels.api as sm

        x = design.x[index]
        if y is None:
            response = (y_values - offset_values)[index]
        else:
            response = np.asarray(y, dtype=float)
        ww = np.ones(index.shape[0]) if weight_values is None else weight_values[index]
        if np.allclose(ww, 1):
            result = sm.OLS(response, x, missing="raise").fit()
        else:
            result = sm.WLS(response, x, weights=ww, missing="raise").fit()
        resid = np.asarray(result.resid, dtype=float)
        rss = float(np.sum(ww * resid**2))
        df_resid = int(result.df_resid)
        sigma2 = rss / df_resid if df_resid else np.nan
        n = x.shape[0]
        child = Fit(
            coefficients=np.asarray(result.params, dtype=float),
            covariance=np.asarray(result.normalized_cov_params, dtype=float) * sigma2,
            names=list(design.names),
            n_obs=n,
            log_likelihood=float(result.llf),
            residual_df=df_resid,
            family="ols",
            x=x,
            y=response,
            row_index=design.row_index[index],
            design=design,
            weights=ww,
            offset=np.zeros(n),
            working_residuals=resid,
            working_weights=ww,
            hat_values=_hat(x, ww),
            dispersion=rss / n,
            deviance=rss,
            scale=sigma2,
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
