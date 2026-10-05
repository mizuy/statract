"""Coefficient tests and OLS diagnostics.

Wald and likelihood-ratio comparisons take two fitted models. The coefficient
test uses a t reference for ordinary least squares and a normal reference for
GLMs. Durbin–Watson p-values use the normal approximation that ``dwtest`` uses
when the sample has at least 100 rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from .design import Design
from .fit import Fit


@dataclass(frozen=True)
class TestResult:
    """A scalar hypothesis test."""

    statistic: float
    p_value: float
    df: tuple[float, ...]
    distribution: str
    method: str

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "statistic": [self.statistic],
                "p_value": [self.p_value],
                "df": [self.df[0] if len(self.df) == 1 else float("nan")],
                "df1": [self.df[0] if self.df else None],
                "df2": [self.df[1] if len(self.df) > 1 else None],
                "distribution": [self.distribution],
                "method": [self.method],
            }
        )


def coefficient_test(
    fit: Fit,
    covariance: np.ndarray | None = None,
    *,
    df: float | None = None,
    level: float = 0.95,
) -> pl.DataFrame:
    """Coefficient table under an optional covariance.

    ``covariance`` may be a matrix or a callable ``fit -> matrix``.
    """
    if callable(covariance):
        cov = np.asarray(covariance(fit), dtype=float)
    elif covariance is None:
        cov = fit.covariance
    else:
        cov = np.asarray(covariance, dtype=float)
    viewed = Fit(
        coefficients=fit.coefficients,
        covariance=cov,
        names=list(fit.names),
        n_obs=fit.n_obs,
        log_likelihood=fit.log_likelihood,
        residual_df=fit.residual_df,
        family=fit.family,
        x=fit.x,
        y=fit.y,
        row_index=fit.row_index,
        design=fit.design,
        weights=fit.weights,
        offset=fit.offset,
        working_residuals=fit.working_residuals,
        working_weights=fit.working_weights,
        hat_values=fit.hat_values,
        dispersion=fit.dispersion,
    )
    return viewed.tidy(level=level, df=df)


def coefficient_interval(fit: Fit, covariance: np.ndarray | None = None, *, level: float = 0.95) -> pl.DataFrame:
    """Confidence limits for each coefficient."""
    table = coefficient_test(fit, covariance, level=level)
    return table.select(["term", "estimate", "conf_low", "conf_high"])


def wald_test(
    full: Fit,
    reduced: Fit,
    covariance: np.ndarray | None = None,
    *,
    distribution: str = "f",
) -> TestResult:
    """Wald comparison of nested models.

    The statistic is the quadratic form in the coefficients that ``full`` has
    and ``reduced`` does not. ``distribution`` is ``"f"`` or ``"chi2"``.
    """
    if callable(covariance):
        cov = np.asarray(covariance(full), dtype=float)
    elif covariance is None:
        cov = full.covariance
    else:
        cov = np.asarray(covariance, dtype=float)
    missing = [i for i, name in enumerate(full.names) if name not in reduced.names]
    if not missing:
        raise ValueError("the models are not nested, or they have the same coefficients")
    beta = full.coefficients[missing]
    block = cov[np.ix_(missing, missing)]
    chi2 = float(beta @ np.linalg.solve(block, beta))
    df1 = float(len(missing))
    if distribution == "chi2":
        p_value = float(stats.chi2.sf(chi2, df1))
        return TestResult(chi2, p_value, (df1,), "chi2", "wald")
    if distribution != "f":
        raise ValueError("distribution must be 'f' or 'chi2'")
    if full.residual_df is None or full.residual_df <= 0:
        raise ValueError("an F Wald test needs a positive residual degrees of freedom")
    df2 = float(full.residual_df)
    stat = chi2 / df1
    p_value = float(stats.f.sf(stat, df1, df2))
    return TestResult(stat, p_value, (df1, df2), "f", "wald")


def likelihood_ratio_test(full: Fit, reduced: Fit) -> TestResult:
    """Likelihood-ratio comparison. The statistic is ``2 (ll_full - ll_reduced)``."""
    if full.log_likelihood is None or reduced.log_likelihood is None:
        raise ValueError("both fits need a log likelihood")
    stat = float(2 * (full.log_likelihood - reduced.log_likelihood))
    df = float(len(full.coefficients) - len(reduced.coefficients))
    if df <= 0:
        raise ValueError("the first model should be the larger one")
    p_value = float(stats.chi2.sf(stat, df))
    return TestResult(stat, p_value, (df,), "chi2", "likelihood_ratio")


def breusch_pagan_test(fit: Fit, *, studentize: bool = True) -> TestResult:
    """Breusch–Pagan test. The default is the studentized ``n R^2`` statistic."""
    if fit.family != "ols":
        raise ValueError("breusch_pagan_test expects an ordinary least-squares fit")
    resid = fit.working_residuals
    weights = fit.weights
    n = float(np.sum(weights > 0))
    sigma2 = float(np.sum(weights * resid**2) / n)
    response = resid**2 - sigma2 if studentize else resid**2 / sigma2 - 1
    aux = _weighted_lstsq(fit.x, response, weights)
    if studentize:
        stat = n * float(np.sum(weights * aux**2) / np.sum(response**2))
    else:
        stat = 0.5 * float(np.sum(weights * aux**2))
    df = float(fit.x.shape[1] - 1)
    return TestResult(stat, float(stats.chi2.sf(stat, df)), (df,), "chi2", "breusch_pagan")


def durbin_watson_test(fit: Fit, *, alternative: str = "greater") -> TestResult:
    """Durbin–Watson statistic, with the normal approximation used for n >= 100."""
    if fit.family != "ols":
        raise ValueError("durbin_watson_test expects an ordinary least-squares fit")
    resid = fit.working_residuals
    stat = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
    x = fit.x
    n, k = x.shape
    q1 = np.linalg.pinv(x.T @ x)
    ax = np.zeros_like(x)
    ax[0] = x[0] - x[1]
    ax[-1] = x[-1] - x[-2]
    if n > 2:
        ax[1:-1] = -x[:-2] + 2 * x[1:-1] - x[2:]
    xaxq = x.T @ ax @ q1
    p_moment = 2 * (n - 1) - np.trace(xaxq)
    q_moment = 2 * (3 * n - 4) - 2 * np.trace(ax.T @ ax @ q1) + np.trace(xaxq @ xaxq)
    dmean = p_moment / (n - k)
    dvar = 2 / ((n - k) * (n - k + 2)) * (q_moment - p_moment * dmean)
    sd = float(np.sqrt(dvar))
    if alternative == "greater":
        p_value = float(stats.norm.cdf(stat, loc=dmean, scale=sd))
    elif alternative == "less":
        p_value = float(stats.norm.sf(stat, loc=dmean, scale=sd))
    elif alternative == "two.sided":
        p_value = float(2 * stats.norm.sf(abs(stat - dmean), loc=0, scale=sd))
    else:
        raise ValueError("alternative must be greater, less, or two.sided")
    return TestResult(stat, p_value, (float(n - k),), "normal", "durbin_watson")


def ramsey_reset_test(fit: Fit, *, power: tuple[int, ...] = (2, 3)) -> TestResult:
    """Ramsey RESET test using powers of the fitted values."""
    if fit.family != "ols":
        raise ValueError("ramsey_reset_test expects an ordinary least-squares fit")
    y = fit.y.copy()
    has_intercept = "(Intercept)" in fit.names
    if has_intercept:
        y = y - y.mean()
        scale = float(y.std(ddof=1))
        if scale > 0:
            y = y / scale
    fitted = fit.x @ np.linalg.lstsq(fit.x, y, rcond=None)[0]
    powers = [fitted**p for p in power]
    extra = np.column_stack(powers)
    x_full = np.column_stack([fit.x, extra])
    rss1 = float(np.sum(np.linalg.lstsq(fit.x, y, rcond=None)[1]))
    rss2 = float(np.sum(np.linalg.lstsq(x_full, y, rcond=None)[1]))
    df1 = float(extra.shape[1])
    df2 = float(fit.n_obs - x_full.shape[1])
    stat = (df2 / df1) * ((rss1 - rss2) / rss2)
    p_value = float(stats.f.sf(stat, df1, df2))
    return TestResult(stat, p_value, (df1, df2), "f", "ramsey_reset")


def breusch_godfrey_test(fit: Fit, *, order: int = 1, distribution: str = "chi2") -> TestResult:
    """Breusch–Godfrey serial-correlation test."""
    if fit.family != "ols":
        raise ValueError("breusch_godfrey_test expects an ordinary least-squares fit")
    resid = fit.working_residuals
    n, k = fit.x.shape
    lags = []
    for lag in range(1, order + 1):
        column = np.zeros(n)
        column[lag:] = resid[:-lag]
        lags.append(column)
    z = np.column_stack(lags)
    x_full = np.column_stack([fit.x, z])
    beta = np.linalg.lstsq(x_full, resid, rcond=None)[0]
    fitted = x_full @ beta
    if distribution == "chi2":
        stat = float(n * np.sum(fitted**2) / np.sum(resid**2))
        df = float(order)
        p_value = float(stats.chi2.sf(stat, df))
        return TestResult(stat, p_value, (df,), "chi2", "breusch_godfrey")
    if distribution != "f":
        raise ValueError("distribution must be 'chi2' or 'f'")
    uresid = resid - fitted
    df1 = float(order)
    df2 = float(n - k - order)
    stat = ((np.sum(resid**2) - np.sum(uresid**2)) / df1) / (np.sum(uresid**2) / df2)
    p_value = float(stats.f.sf(stat, df1, df2))
    return TestResult(float(stat), p_value, (df1, df2), "f", "breusch_godfrey")


def check_collinearity(fit: Fit, *, ci: float = 0.95) -> pl.DataFrame:
    """Variance inflation factors, in the sense of ``performance::check_collinearity``.

    Each row is one model term. A factor contributes one generalized VIF
    (Fox and Monette), not one row per dummy. ``se_factor`` is
    ``VIF^(1/(2 * df))``, with ``df`` equal to the number of dummy columns of
    a main-effect factor and 1 otherwise. Intervals follow the same transform
    of the coefficient of determination that ``performance`` uses.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    groups = _collinearity_terms(fit.design)
    if len(groups) < 2:
        raise ValueError("collinearity needs at least two terms")
    start = 1 if fit.design.intercept and fit.names and fit.names[0] == "(Intercept)" else 0
    covariance = np.asarray(fit.covariance, dtype=float)
    if start:
        covariance = covariance[start:, start:]
    scale = np.sqrt(np.clip(np.diag(covariance), 0, None))
    corr = covariance / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    vifs = np.array([_generalized_vif(corr, [c - start for c in cols]) for _label, cols, _df in groups])
    se_factor = np.array([vif ** (1 / (2 * df)) for vif, (_label, _cols, df) in zip(vifs, groups, strict=True)])
    r_low, r_high = _vif_r_interval(vifs, n=fit.n_obs, p=len(fit.coefficients), ci=ci)
    with np.errstate(divide="ignore", invalid="ignore"):
        vif_low = 1 / (1 - r_low)
        vif_high = 1 / (1 - r_high)
        tolerance = 1 / vifs
    return pl.DataFrame(
        {
            "term": [label for label, _cols, _df in groups],
            "vif": vifs,
            "vif_ci_low": vif_low,
            "vif_ci_high": vif_high,
            "se_factor": se_factor,
            "tolerance": tolerance,
            "tolerance_ci_low": 1 - r_high,
            "tolerance_ci_high": 1 - r_low,
        }
    )


def _collinearity_terms(design: Design) -> list[tuple[str, list[int], int]]:
    """Term label, column indexes in the full design, and the VIF degrees of freedom."""
    start = 1 if design.intercept and design.names and design.names[0] == "(Intercept)" else 0
    if design.recipes is not None:
        grouped: list[tuple[tuple[str, ...], list[int]]] = []
        for index in range(start, len(design.names)):
            symbols = tuple(symbol for symbol, _level in design.recipes[index])
            if not symbols:
                continue
            if grouped and grouped[-1][0] == symbols:
                grouped[-1][1].append(index)
            else:
                grouped.append((symbols, [index]))
        terms: list[tuple[str, list[int], int]] = []
        levels = design.factor_levels or {}
        for symbols, cols in grouped:
            df = len(levels[symbols[0]]) - 1 if len(symbols) == 1 and symbols[0] in levels else 1
            terms.append((":".join(symbols), cols, max(df, 1)))
        return terms
    terms = []
    col = start
    for spec in design.predictors:
        if spec.kind == "factor":
            width = max(len(spec.levels) - 1, 1)
            df = width
        else:
            width = 1
            df = 1
        terms.append((spec.name, list(range(col, col + width)), df))
        col += width
    return terms


def _generalized_vif(corr: np.ndarray, columns: list[int]) -> float:
    """Fox–Monette GVIF from the correlation matrix of the coefficients."""
    index = np.asarray(columns, dtype=int)
    rest = np.setdiff1d(np.arange(corr.shape[0]), index, assume_unique=False)
    det_all = float(np.linalg.det(corr))
    det_term = float(np.linalg.det(corr[np.ix_(index, index)]))
    det_rest = 1.0 if rest.size == 0 else float(np.linalg.det(corr[np.ix_(rest, rest)]))
    if not np.isfinite(det_all) or abs(det_all) < 1e-15:
        return float("inf")
    return det_term * det_rest / det_all


def _vif_r_interval(vif: np.ndarray, *, n: int, p: int, ci: float) -> tuple[np.ndarray, np.ndarray]:
    """Map VIF to an interval on the ``1 - 1/VIF`` scale, then return that scale."""
    from scipy.special import expit, logit

    r = 1.0 - 1.0 / np.asarray(vif, dtype=float)
    z = float(stats.norm.ppf((1 + ci) / 2))
    denom = (n**2 - 1) * (n + 3)
    se = np.sqrt((1 - r**2) ** 2 * (n - p - 1) ** 2 / denom)
    se_log = se / (r * (1 - r))
    with np.errstate(divide="ignore", invalid="ignore"):
        center = logit(r)
        low = expit(center - z * se_log)
        high = expit(center + z * se_log)
    return low, high


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(weights)
    beta = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
    return x @ beta
