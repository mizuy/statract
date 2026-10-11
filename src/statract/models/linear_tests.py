"""Coefficient tests and OLS diagnostics.

Wald and likelihood-ratio comparisons take two fitted models. The coefficient
test uses a t reference for ordinary least squares and a normal reference for
GLMs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ._profile import profile_limit
from .design import Design
from .fit import Fit, _glm_start


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


def profile_interval(fit: Fit, *, level: float = 0.95) -> pl.DataFrame:
    """Profile-likelihood confidence limits for a ``fit_glm`` fit, as ``confint.glm``.

    For each coefficient the limit is where the signed root of
    ``(D(b) - D) / phi`` equals the normal quantile. ``D(b)`` is the deviance
    with the coefficient fixed at ``b`` and the others refitted, and ``phi``
    is 1 for binomial and poisson and ``summary.glm``'s dispersion for
    gaussian and gamma. The quantile is normal for every family, as in
    ``confint.profile.glm``. The limit is solved for directly (Brent's
    method) rather than read off ``MASS``'s spline through the profile
    points, so the two agree within 1e-3 of a standard error (at most
    2.5e-4 in the tests), not to machine precision. A side where the
    deviance levels off before the quantile (separation) gives an infinite
    limit; R gives ``NA`` there.
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if fit.family not in {"binomial", "poisson", "gamma", "gaussian"}:
        raise ValueError("profile_interval needs a fit_glm fit (binomial, poisson, gamma, or gaussian)")
    x = np.asarray(fit.x, dtype=float)
    y = np.asarray(fit.y, dtype=float)
    prior = np.asarray(fit.weights, dtype=float)
    offset = np.asarray(fit.offset, dtype=float)
    family = fit.family
    linkinv, mu_eta, variance, deviance, _eta = _glm_start(y, family, prior)
    pieces = (y, prior, linkinv, mu_eta, variance, deviance, family)
    beta, dev0 = _glm_refit_fixed(x, offset, x @ fit.coefficients + offset, pieces)
    dispersion = 1.0 if family in {"binomial", "poisson"} else float(fit.scale)
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    q = float(stats.norm.ppf(0.5 + level / 2))
    low = np.empty(len(beta))
    high = np.empty(len(beta))
    for j in range(len(beta)):
        keep = np.arange(len(beta)) != j
        xj = x[:, keep]
        state = {"eta": x @ beta + offset}

        def drop(b: float, j: int = j, xj: np.ndarray = xj, state: dict = state) -> float:
            fixed = offset + x[:, j] * b
            coef, dev = _glm_refit_fixed(xj, fixed, state["eta"], pieces)
            if np.all(np.isfinite(coef)):
                state["eta"] = xj @ coef + fixed
            return (dev - dev0) / dispersion

        low[j] = profile_limit(drop, beta[j], se[j], -q)
        high[j] = profile_limit(drop, beta[j], se[j], q)
    return pl.DataFrame({"term": list(fit.names), "estimate": fit.coefficients, "conf_low": low, "conf_high": high})


def _glm_refit_fixed(
    x: np.ndarray,
    offset: np.ndarray,
    eta: np.ndarray,
    pieces: tuple,
    *,
    maxit: int = 100,
) -> tuple[np.ndarray, float]:
    """IRLS from ``eta`` to a tight optimum: coefficients and deviance.

    All four default links are canonical, so IRLS is Newton's method. A step
    to an invalid mean or a larger deviance is halved toward the last one.

    Fisher scoring as in McCullagh & Nelder (1989, sec. 2.5): with
    ``mu = linkinv(eta)``, working weights ``w = prior * mu_eta^2 / V(mu)`` and
    working response ``z = eta - offset + (y - mu) / mu_eta``, the next
    coefficients solve the weighted least-squares problem of ``z`` on ``x``.
    The first iterate is one scoring step from the supplied ``eta`` (a warm
    start). Steps are halved along the coefficient path, which keeps the mean
    valid for the gamma inverse link (positive eta is a convex set). Iteration stops
    when the largest coefficient change is below 1e-11 relative and the
    deviance change below 1e-13 relative, or after ``maxit`` iterations; under
    separation the coefficients drift while the deviance levels off, which the
    caller detects.
    """
    y, prior, linkinv, mu_eta, variance, deviance_fn, family = pieces
    x = np.asarray(x, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.zeros(0), float(deviance_fn(y, linkinv(offset), prior))

    def valid(eta_values: np.ndarray) -> bool:
        if not np.all(np.isfinite(eta_values)):
            return False
        if family == "gamma":
            return bool(np.all(eta_values > 0))
        return True

    def deviance_at(eta_values: np.ndarray) -> float:
        if not valid(eta_values):
            return np.inf
        value = float(deviance_fn(y, linkinv(eta_values), prior))
        return value if np.isfinite(value) else np.inf

    def wls_step(eta_values: np.ndarray) -> np.ndarray:
        mu = linkinv(eta_values)
        derivative = mu_eta(eta_values)
        var = variance(mu)
        usable = (np.abs(derivative) > np.finfo(float).tiny) & (var > 0) & (prior > 0)
        safe_derivative = np.where(usable, derivative, 1.0)
        working = np.where(usable, eta_values - offset + (y - mu) / safe_derivative, 0.0)
        weight = np.where(usable, prior * derivative**2 / np.where(usable, var, 1.0), 0.0)
        root = np.sqrt(weight)
        solution, *_ = np.linalg.lstsq(x * root[:, None], working * root, rcond=None)
        return solution

    # The warm start ``eta`` usually lies off the affine space x b + offset
    # (the fixed coefficient moved), so its deviance is no reference for step
    # halving. Start instead from a point inside the space: one scoring step
    # from the warm eta, else its least-squares projection, else a constant
    # mean on the link scale.
    starts = []
    warm = np.asarray(eta, dtype=float)
    if valid(warm):
        starts.append(wls_step(warm))
    starts.append(np.linalg.lstsq(x, warm - offset, rcond=None)[0])
    current_beta = None
    current_dev = np.inf
    for candidate in starts:
        candidate_dev = deviance_at(x @ candidate + offset)
        if candidate_dev < current_dev:
            current_beta, current_dev = candidate, candidate_dev
    if current_beta is None:
        mean = float(np.average(y, weights=prior)) if np.sum(prior) > 0 else float(np.mean(y))
        if family == "binomial":
            mean = min(max(mean, 1e-3), 1 - 1e-3)
            start = np.log(mean / (1 - mean))
        elif family == "poisson":
            start = np.log(max(mean, 1e-3))
        elif family == "gamma":
            start = 1.0 / max(mean, 1e-8)
        else:
            start = mean
        current_beta = np.linalg.lstsq(x, np.full_like(offset, start) - offset, rcond=None)[0]
        current_dev = deviance_at(x @ current_beta + offset)
    current_eta = x @ current_beta + offset

    for _ in range(maxit):
        proposal = wls_step(current_eta)
        fraction = 1.0
        accepted = False
        for _halving in range(60):
            trial_beta = current_beta + fraction * (proposal - current_beta)
            trial_eta = x @ trial_beta + offset
            trial_dev = deviance_at(trial_eta)
            if np.isfinite(trial_dev) and trial_dev <= current_dev + 1e-12 * (abs(current_dev) + 1.0):
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            break
        beta_change = float(np.max(np.abs(trial_beta - current_beta)))
        dev_change = abs(trial_dev - current_dev)
        current_beta, current_eta, current_dev = trial_beta, trial_eta, trial_dev
        scale = 1.0 + float(np.max(np.abs(current_beta)))
        if beta_change <= 1e-11 * scale and dev_change <= 1e-13 * (abs(current_dev) + 1.0):
            break
    return np.asarray(current_beta, dtype=float), float(current_dev)

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
    """Breusch–Pagan test for heteroskedasticity of an OLS fit. Returns :class:`TestResult`.

    Breusch & Pagan (1979), *Econometrica* 47:1287: regress
    ``g = e^2 / sigma2`` (``sigma2 = sum w e^2 / n``) on the design and take
    half the explained sum of squares, chi-squared on ``k - 1`` df. With
    ``studentize=True`` (default) Koenker's (1981), *J Econometrics* 17:107
    robust version ``n R^2`` of the regression of ``e^2`` on the design is
    used instead. Case weights weight the auxiliary regression.
    """
    if fit.family != "ols":
        raise ValueError("breusch_pagan_test expects an ordinary least-squares fit")
    design = np.asarray(fit.x, dtype=float)
    resid = np.asarray(fit.working_residuals, dtype=float)
    weights = np.ones(resid.size) if fit.weights is None else np.asarray(fit.weights, dtype=float)
    keep = weights > 0
    design, resid, weights = design[keep], resid[keep], weights[keep]
    n_used = int(resid.size)
    sigma2 = float(np.sum(weights * resid**2) / n_used)
    response = resid**2 / sigma2
    root_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], response * root_w, rcond=None)
    fitted = design @ coef
    centre = float(np.sum(weights * response) / np.sum(weights))
    explained = float(np.sum(weights * (fitted - centre) ** 2))
    if studentize:
        total = float(np.sum(weights * (response - centre) ** 2))
        stat = n_used * explained / total
    else:
        stat = explained / 2
    df = float(design.shape[1] - 1)
    return TestResult(float(stat), float(stats.chi2.sf(stat, df)), (df,), "chi2", "breusch_pagan")


def _durbin_watson_moments(design: np.ndarray) -> tuple[float, float, int]:
    """Exact null mean and variance of the Durbin–Watson d given the design.

    Durbin & Watson (1950), *Biometrika* 37:409, sec. 3: with M the residual
    maker and A the first-difference matrix (d = e'Ae / e'e), and
    nu = n - k, E d = tr(MA) / nu and
    Var d = 2 [tr((MA)^2) - tr(MA) E d] / (nu (nu + 2)).
    With an orthonormal basis Q of the columns, tr(MA) = tr(A) - tr(Q'AQ) and
    tr((MA)^2) = tr(A^2) - 2 tr(Q'A^2 Q) + tr((Q'AQ)^2).
    """
    n = design.shape[0]
    basis, triangle = np.linalg.qr(design)
    rank = int(np.sum(np.abs(np.diag(triangle)) > 1e-10 * max(1.0, float(np.max(np.abs(np.diag(triangle)))))))
    if rank < design.shape[1]:
        u, s, _ = np.linalg.svd(design, full_matrices=False)
        basis = u[:, s > 1e-10 * s[0]]
        rank = basis.shape[1]

    def apply_a(v: np.ndarray) -> np.ndarray:
        # A v where A = D'D and D is the (n-1) x n first-difference matrix.
        step = np.diff(v, axis=0)
        out = np.zeros_like(v)
        out[:-1] -= step
        out[1:] += step
        return out

    a_basis = apply_a(basis)
    qaq = basis.T @ a_basis
    trace_a = 2.0 * (n - 1)  # diagonal 1, 2, ..., 2, 1
    trace_a2 = 6.0 * n - 8.0  # squared row norms: 2 end rows of 2, n - 2 interior rows of 6
    # tr(Q'A^2Q) = ||AQ||_F^2 since A is symmetric.
    trace_ma = trace_a - float(np.trace(qaq))
    trace_ma2 = trace_a2 - 2.0 * float(np.sum(a_basis**2)) + float(np.sum(qaq * qaq.T))
    nu = n - rank
    mean = trace_ma / nu
    variance = 2.0 * (trace_ma2 - trace_ma * mean) / (nu * (nu + 2))
    return mean, variance, rank


def durbin_watson_test(fit: Fit, *, alternative: str = "greater") -> TestResult:
    """Durbin–Watson test for first-order autocorrelation of OLS residuals. Returns :class:`TestResult`.

    d = sum (e_t - e_{t-1})^2 / sum e_t^2 (Durbin & Watson 1950, 1951). The
    p value uses the normal approximation with the exact null mean and
    variance of d given the design (Durbin & Watson 1950, sec. 3; 1971,
    *Biometrika* 58:1). ``"greater"`` tests positive autocorrelation (small
    d), ``"less"`` negative, ``"two.sided"`` either.
    """
    if alternative not in ("greater", "less", "two.sided"):
        raise ValueError("alternative must be greater, less, or two.sided")
    if fit.family != "ols":
        raise ValueError("durbin_watson_test expects an ordinary least-squares fit")
    resid = np.asarray(fit.working_residuals, dtype=float)
    design = np.asarray(fit.x, dtype=float)
    d = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
    mean, variance, rank = _durbin_watson_moments(design)
    z = (d - mean) / np.sqrt(variance)
    if alternative == "greater":
        p_value = float(stats.norm.cdf(z))
    elif alternative == "less":
        p_value = float(stats.norm.sf(z))
    else:
        p_value = float(2 * stats.norm.sf(abs(z)))
    return TestResult(d, p_value, (float(design.shape[0] - rank),), "normal", "durbin_watson")


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
    """Breusch–Godfrey test for serial correlation up to ``order``. Returns :class:`TestResult`.

    Breusch (1978), *Aust Econ Pap* 17:334; Godfrey (1978), *Econometrica*
    46:1293: regress the OLS residuals on the design and their first
    ``order`` lags (pre-sample lags set to 0). ``"chi2"``: ``n R^2`` of that
    regression on ``order`` df. ``"f"``: the F test that the lag coefficients
    are zero, on ``(order, n - k - order)`` df.
    """
    if distribution not in ("chi2", "f"):
        raise ValueError("distribution must be 'chi2' or 'f'")
    if fit.family != "ols":
        raise ValueError("breusch_godfrey_test expects an ordinary least-squares fit")
    order = int(order)
    if order < 1:
        raise ValueError("order must be a positive integer")
    resid = np.asarray(fit.working_residuals, dtype=float)
    design = np.asarray(fit.x, dtype=float)
    n = resid.size
    lags = np.zeros((n, order))
    for lag in range(1, order + 1):
        lags[lag:, lag - 1] = resid[: n - lag]
    augmented = np.column_stack([design, lags])
    coef, *_ = np.linalg.lstsq(augmented, resid, rcond=None)
    rss_aux = float(np.sum((resid - augmented @ coef) ** 2))
    rss_base = float(np.sum(resid**2))
    if distribution == "chi2":
        stat = n * (1.0 - rss_aux / rss_base)
        df = (float(order),)
        p_value = float(stats.chi2.sf(stat, order))
        return TestResult(float(stat), p_value, df, "chi2", "breusch_godfrey")
    rank = int(np.linalg.matrix_rank(design))
    df2 = float(n - rank - order)
    stat = ((rss_base - rss_aux) / order) / (rss_aux / df2)
    return TestResult(float(stat), float(stats.f.sf(stat, order, df2)), (float(order), df2), "f", "breusch_godfrey")


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
    """Lower and upper confidence limits on the ``1 - 1/VIF`` scale, one per term.

    R_j^2 = 1 - 1/VIF_j is the squared multiple correlation of term j on the
    others. Its large-sample standard error is
    SE = sqrt((1 - R^2)^2 (n - p - 1)^2 / ((n^2 - 1)(n + 3))) (Olkin & Finn
    1995, *Psychol Bull* 118:155; Cohen, Cohen, West & Aiken 2003, sec. 3.6).
    The interval is built on the logit scale with the delta method,
    logit(R^2) +/- z SE / (R^2 (1 - R^2)), z at (1 + ci)/2, and mapped back,
    which keeps it inside [0, 1] (cf. Marcoulides & Raykov 2019, *Educ
    Psychol Meas* 79:874). Infinite or undefined VIFs give numpy's inf/nan.
    """
    vif = np.asarray(vif, dtype=float)
    z = float(stats.norm.ppf((1 + ci) / 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r2 = 1.0 - 1.0 / vif
        se = np.sqrt((1.0 - r2) ** 2 * (n - p - 1) ** 2 / ((n**2 - 1.0) * (n + 3.0)))
        logit = np.log(r2 / (1.0 - r2))
        half = z * se / (r2 * (1.0 - r2))
        low = 1.0 / (1.0 + np.exp(-(logit - half)))
        high = 1.0 / (1.0 + np.exp(-(logit + half)))
    return low, high

