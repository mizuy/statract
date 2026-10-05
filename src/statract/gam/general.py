"""Penalized additive models beyond a single cubic-regression smooth.

Bases follow mgcv's constructors for cubic regression, P-splines, cyclic
cubics, random effects, and a 1-dimensional thin-plate regression spline.
Tensor products use a row-wise Kronecker product of cubic regression margins.
A tensor interaction centers each margin first. Prior weights and an offset
enter the penalized score. Smoothing parameters minimize the profiled REML, ML,
or GCV score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import special, stats
from scipy.linalg import solve
from scipy.optimize import minimize, minimize_scalar

from ..design import ColumnRef, column_series

_FAMILIES = {"gaussian", "binomial", "poisson", "gamma"}
_METHODS = {"reml", "ml", "gcv"}


@dataclass
class _Block:
    name: str
    columns: np.ndarray
    penalty: np.ndarray
    null_dim: int
    rebuild: Callable[[dict[str, np.ndarray]], np.ndarray]
    extra_penalties: tuple[np.ndarray, ...] = ()


@dataclass
class GeneralFit:
    """A fitted additive model with one smoothing parameter per penalty."""

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    smoothing_parameters: np.ndarray
    edf: float
    block_edf: np.ndarray
    sig2: float
    reml: float
    x: np.ndarray
    y: np.ndarray
    family: str
    method: str
    converged: bool
    blocks: list[_Block]
    kept: dict[str, np.ndarray]
    penalty_counts: list[int]
    parametric: list[tuple]
    offset: np.ndarray | None = None
    offset_name: str | None = None

    @property
    def smoothing_parameter(self) -> float:
        return float(self.smoothing_parameters[0]) if self.smoothing_parameters.size else float("nan")

    def tidy(self) -> pl.DataFrame:
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        stat = self.coefficients / se
        return pl.DataFrame(
            {
                "term": self.names,
                "estimate": self.coefficients,
                "std_error": se,
                "statistic": stat,
                "p_value": 2 * stats.norm.sf(np.abs(stat)),
            }
        )

    def smooth_table(self) -> pl.DataFrame:
        terms: list[str] = []
        edf: list[float] = []
        smoothing: list[float] = []
        cursor = 0
        for block, count, block_edf in zip(self.blocks, self.penalty_counts, self.block_edf, strict=True):
            for index in range(count):
                terms.append(block.name if count == 1 else f"{block.name}[{index + 1}]")
                edf.append(float(block_edf))
                smoothing.append(float(self.smoothing_parameters[cursor + index]))
            cursor += count
        return pl.DataFrame(
            {
                "term": terms,
                "edf": edf,
                "smoothing_parameter": smoothing,
                "reml": [self.reml] * len(terms),
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray | pl.DataFrame:
        if kind not in {"link", "response", "terms"}:
            raise ValueError("kind must be link, response, or terms")
        design = self.x if data is None else _design_from_blocks(
            _columns(data, self.kept.keys()), self.blocks, self.parametric
        )
        if kind == "terms":
            return _term_frame(design, self.coefficients, self.names, self.blocks)
        eta = design @ self.coefficients
        eta = eta + _offset_vector(self, data, eta.shape[0])
        if kind == "link" or self.family == "gaussian":
            return eta
        return _mean(self.family, eta)

    def partial_effect(self, term: str, *, n: int = 100) -> pl.DataFrame:
        """Smooth contribution and pointwise standard error on a grid or at the data."""
        block = next((item for item in self.blocks if item.name == term), None)
        if block is None:
            raise KeyError(f"smooth {term!r} is not in the fit")
        sl = block.columns
        beta = self.coefficients[sl]
        cov = self.covariance[np.ix_(sl, sl)]
        grid = _effect_grid(self.kept, block, n)
        z = block.rebuild(grid)
        eta = z @ beta
        var = np.sum((z @ cov) * z, axis=1)
        se = np.sqrt(np.clip(var, 0, None))
        frame = {name: grid[name] for name in grid}
        frame["estimate"] = eta
        frame["std_error"] = se
        return pl.DataFrame(frame)

    def check(self) -> pl.DataFrame:
        """Effective degrees of freedom against the basis size, plus a k-index.

        The k-index is the residual variance divided by the variance of residuals
        differenced along the smooth's covariate. Values well below 1 say ``k``
        is too small.
        """
        eta = self.x @ self.coefficients
        if self.offset is not None:
            eta = eta + self.offset
        resid = self.y - _mean(self.family, eta)
        rows = []
        for block, edf in zip(self.blocks, self.block_edf, strict=True):
            k = int(block.penalty.shape[0] + block.null_dim)
            rows.append(
                {
                    "term": block.name,
                    "k": k,
                    "edf": float(edf),
                    "k_index": _k_index(resid, self.kept.get(block.name.split("[", 1)[0])),
                }
            )
        return pl.DataFrame(rows)

    def concurvity(self) -> pl.DataFrame:
        """Observed concurvity: R² of each smooth on the sum of the other terms."""
        contributions = []
        for block in self.blocks:
            sl = block.columns
            contributions.append(self.x[:, sl] @ self.coefficients[sl])
        rows = []
        total = np.sum(contributions, axis=0) if contributions else np.zeros(self.n_obs)
        for block, part in zip(self.blocks, contributions, strict=True):
            other = total - part
            rows.append({"term": block.name, "observed": _rsquared(part, other)})
        return pl.DataFrame(rows)


def compare_gams(*fits: GeneralFit) -> pl.DataFrame:
    """AIC and the fitting criterion for nested or alternative smoothers."""
    rows = []
    for index, fit in enumerate(fits):
        aic = fit.n_obs * np.log(fit.sig2) + 2 * fit.edf if fit.family == "gaussian" else np.nan
        rows.append(
            {
                "model": index + 1,
                "n_obs": fit.n_obs,
                "edf": fit.edf,
                "criterion": fit.reml,
                "aic": aic,
            }
        )
    return pl.DataFrame(rows)


def fit_general(
    data: pl.DataFrame,
    outcome: ColumnRef,
    smooths: list,
    *,
    predictors: list[str] | None,
    family: str,
    method: str,
    select: bool,
    gamma: float,
    weights: str | None = None,
    offset: str | None = None,
) -> GeneralFit:
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if not smooths:
        raise ValueError("gam needs at least one smooth")
    used = _used_columns(smooths, predictors)
    y_name = column_series(data, outcome).name
    keep = np.ones(data.height, dtype=bool)
    for name in [y_name, *used, *([weights] if weights else []), *([offset] if offset else [])]:
        keep &= column_series(data, name).is_not_null().to_numpy()
    kept = {name: np.asarray(column_series(data, name).filter(pl.Series(keep)).to_numpy()) for name in used}
    y = np.asarray(column_series(data, outcome).filter(pl.Series(keep)).to_numpy(), dtype=float)
    weight_values = _positive_weights(data, weights, keep)
    offset_values = None if offset is None else _aligned_numeric(data, offset, keep)
    blocks = [_build_block(spec, kept) for spec in smooths]
    parametric = [np.ones(y.shape[0])]
    parametric_names = ["(Intercept)"]
    parametric_spec: list[tuple] = []
    for name in predictors or []:
        series = kept[name]
        if _is_factor(data, name):
            levels = _factor_levels(data, name, series)
            for level in levels[1:]:
                parametric_names.append(f"{name}{level}")
                parametric.append((np.asarray(series).astype(str) == str(level)).astype(float))
                parametric_spec.append(("factor", name, str(level)))
        else:
            parametric_names.append(name)
            parametric.append(np.asarray(series, dtype=float))
            parametric_spec.append(("numeric", name))
    penalty_counts = [1 + len(block.extra_penalties) for block in blocks]
    pieces = list(parametric)
    names = list(parametric_names)
    penalties = []
    cursor = len(pieces)
    for block in blocks:
        z = block.rebuild(kept)
        width = z.shape[1]
        sl = np.arange(cursor, cursor + width)
        block.columns = sl
        penalties.append((sl, block.penalty))
        for extra in block.extra_penalties:
            penalties.append((sl, extra))
        for col in range(width):
            names.append(f"{block.name}[{col}]")
        pieces.append(z)
        cursor += width
    if select:
        for sl, matrix in list(penalties):
            null_pen = _null_space_penalty(matrix)
            if null_pen is not None:
                penalties.append((np.asarray(sl), null_pen))
    design = np.column_stack(pieces)
    mp = 1 + len(parametric_spec) + sum(block.null_dim for block in blocks)
    lams, score, beta, sig2, edf, block_edf, converged = _optimize(
        design,
        y,
        penalties,
        [block.null_dim for block in blocks],
        family,
        method,
        mp,
        gamma,
        weight_values,
        offset_values,
    )
    cov = _covariance(design, y, beta, penalties, lams, family, sig2, weight_values, offset_values)
    return GeneralFit(
        coefficients=beta,
        covariance=cov,
        names=names,
        n_obs=int(y.shape[0]),
        smoothing_parameters=lams,
        edf=edf,
        block_edf=block_edf,
        sig2=sig2,
        reml=score,
        x=design,
        y=y,
        family=family,
        method=method,
        converged=converged,
        blocks=blocks,
        kept=kept,
        penalty_counts=penalty_counts,
        parametric=parametric_spec,
        offset=offset_values,
        offset_name=offset,
    )


def _optimize(design, y, penalties, null_dims, family, method, mp, gamma, weights=None, offset=None):
    n_pen = len(penalties)

    def objective(log_sp: np.ndarray) -> float:
        lam = np.exp(np.asarray(log_sp, dtype=float))
        score, *_rest = _fit_once(design, y, penalties, lam, family, method, mp, gamma, weights, offset)
        return score

    if n_pen == 1:
        opt = minimize_scalar(lambda value: objective(np.array([value])), bounds=(-12, 18), method="bounded", options={"xatol": 1e-8})
        log_sp = np.array([opt.x])
        converged = bool(opt.success)
    else:
        opt = minimize(objective, np.zeros(n_pen), method="L-BFGS-B", bounds=[(-12, 18)] * n_pen)
        log_sp = np.asarray(opt.x, dtype=float)
        converged = bool(opt.success)
    lam = np.exp(log_sp)
    score, beta, sig2, edf, block_edf = _fit_once(
        design, y, penalties, lam, family, method, mp, gamma, weights, offset
    )
    return lam, score, beta, sig2, edf, block_edf, converged


def _fit_once(design, y, penalties, lam, family, method, mp, gamma, weights=None, offset=None):
    penalty = _total_penalty(design.shape[1], penalties, lam)
    if family == "gaussian":
        if weights is None and offset is None:
            beta, rss, pen, edf, block_edf = _penalized_least_squares(design, y, penalty, penalties)
            phi_df = y.shape[0] if method == "ml" else y.shape[0] - mp
            phi = (rss + pen) / phi_df
            sig2 = rss / max(y.shape[0] - edf, 1.0)
            if method == "gcv":
                score = y.shape[0] * rss / (y.shape[0] - gamma * edf) ** 2
            else:
                score = _reml_score(design, penalty, phi, phi_df, method)
            return score, beta, float(sig2), float(edf), block_edf
        return _fit_gaussian_weighted(design, y, penalty, penalties, method, mp, gamma, weights, offset)
    if weights is None and offset is None:
        beta = _pirls(design, y, penalty, family)
        beta, rss, pen, edf, block_edf = _penalized_weighted(design, y, penalty, penalties, beta, family)
        dev = _deviance(family, y, _mean(family, design @ beta))
        sig2 = 1.0 if family in {"binomial", "poisson"} else dev / max(y.shape[0] - edf, 1.0)
        if method == "gcv":
            score = y.shape[0] * dev / (y.shape[0] - gamma * edf) ** 2
        elif family in {"binomial", "poisson"}:
            score = _known_scale_score(design, y, beta, penalty, family, method, mp)
        else:
            score, sig2 = _gamma_reml(design, y, beta, penalty, mp, method)
        return score, beta, float(sig2), float(edf), block_edf
    return _fit_glm_weighted(design, y, penalty, penalties, family, method, mp, gamma, weights, offset)


def _penalized_least_squares(design, y, penalty, penalties):
    xtx = design.T @ design
    sys = xtx + penalty
    beta = solve(sys, design.T @ y, assume_a="pos")
    fitted = design @ beta
    rss = float(np.sum((y - fitted) ** 2))
    pen = float(beta @ penalty @ beta)
    influence = solve(sys, xtx, assume_a="pos")
    edf = float(np.trace(influence))
    block_edf = _edf_by_block(influence, penalties)
    return beta, rss, pen, edf, block_edf


def _fit_gaussian_weighted(design, y, penalty, penalties, method, mp, gamma, weights, offset):
    """Gaussian REML with prior weights and an offset.

    The observation count stays ``n``, not the sum of the weights. The score
    subtracts ``sum(log w) / 2``, which is zero when every weight is 1.
    """
    prior = np.ones(y.shape[0]) if weights is None else weights
    target = y if offset is None else y - offset
    xtx = design.T @ (prior[:, None] * design)
    sys = xtx + penalty
    beta = solve(sys, design.T @ (prior * target), assume_a="pos")
    rss = float(np.sum(prior * (target - design @ beta) ** 2))
    pen = float(beta @ penalty @ beta)
    influence = solve(sys, xtx, assume_a="pos")
    edf = float(np.trace(influence))
    block_edf = _edf_by_block(influence, penalties)
    n = y.shape[0]
    phi_df = n if method == "ml" else n - mp
    phi = (rss + pen) / phi_df
    sig2 = rss / max(n - edf, 1.0)
    if method == "gcv":
        score = n * rss / (n - gamma * edf) ** 2
    else:
        score = _weighted_reml_score(sys, penalty, phi, phi_df, method, prior, n)
    return score, beta, float(sig2), float(edf), block_edf


def _weighted_reml_score(sys, penalty, phi, phi_df, method, weights, n) -> float:
    _sign, logdet_p = np.linalg.slogdet(sys)
    eig = np.linalg.eigvalsh(penalty)
    positive = eig[eig > eig.max() * 1e-10]
    logdet_s = float(np.sum(np.log(positive)))
    width = phi_df if method == "reml" else n
    return float(width / 2 * (1 + np.log(2 * np.pi * phi)) + 0.5 * (logdet_p - logdet_s) - 0.5 * np.sum(np.log(weights)))


def _fit_glm_weighted(design, y, penalty, penalties, family, method, mp, gamma, weights, offset):
    """PIRLS with prior weights and an offset, then the Laplace REML score."""
    prior = np.ones(y.shape[0]) if weights is None else weights
    off = np.zeros(y.shape[0]) if offset is None else offset
    beta = _pirls_offset(design, y, penalty, family, prior, off)
    eta = np.clip(design @ beta + off, -20, 20)
    _z, fisher = _working(family, y, eta)
    w = fisher * prior
    xtwx = design.T @ (w[:, None] * design)
    sys = xtwx + penalty
    beta = solve(sys, design.T @ (w * (_z - off)), assume_a="pos")
    influence = solve(sys, xtwx, assume_a="pos")
    edf = float(np.trace(influence))
    block_edf = _edf_by_block(influence, penalties)
    mu = _mean(family, np.clip(design @ beta + off, -20, 20))
    dev = _deviance_weighted(family, y, mu, prior)
    sig2 = 1.0 if family in {"binomial", "poisson"} else dev / max(y.shape[0] - edf, 1.0)
    if method == "gcv":
        score = y.shape[0] * dev / (y.shape[0] - gamma * edf) ** 2
    elif family in {"binomial", "poisson"}:
        score = _known_scale_score_weighted(design, y, beta, penalty, family, method, mp, prior, off)
    else:
        score, sig2 = _gamma_reml_weighted(design, y, beta, penalty, mp, method, prior, off)
    return score, beta, float(sig2), float(edf), block_edf


def _pirls_offset(design, y, penalty, family, prior, offset, iterations: int = 25) -> np.ndarray:
    beta = np.zeros(design.shape[1])
    beta[0] = _intercept_start(family, y)
    for _ in range(iterations):
        eta = np.clip(design @ beta + offset, -20, 20)
        z, fisher = _working(family, y, eta)
        w = fisher * prior
        updated = solve(design.T @ (w[:, None] * design) + penalty, design.T @ (w * (z - offset)), assume_a="pos")
        if np.max(np.abs(updated - beta)) < 1e-8:
            return updated
        beta = updated
    return beta


def _penalized_weighted(design, y, penalty, penalties, beta, family):
    eta = design @ beta
    _z, w = _working(family, y, eta)
    xtwx = design.T @ (w[:, None] * design)
    sys = xtwx + penalty
    beta = solve(sys, design.T @ (w * _z), assume_a="pos")
    pen = float(beta @ penalty @ beta)
    influence = solve(sys, xtwx, assume_a="pos")
    edf = float(np.trace(influence))
    block_edf = _edf_by_block(influence, penalties)
    rss = float(np.sum(w * (design @ beta - _z) ** 2))
    return beta, rss, pen, edf, block_edf


def _pirls(design, y, penalty, family, iterations: int = 25) -> np.ndarray:
    beta = np.zeros(design.shape[1])
    beta[0] = _intercept_start(family, y)
    for _ in range(iterations):
        eta = np.clip(design @ beta, -20, 20)
        z, w = _working(family, y, eta)
        sys = design.T @ (w[:, None] * design) + penalty
        updated = solve(sys, design.T @ (w * z), assume_a="pos")
        if np.max(np.abs(updated - beta)) < 1e-8:
            return updated
        beta = updated
    return beta


def _working(family: str, y: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = _mean(family, eta)
    if family == "binomial":
        deriv = np.clip(mu * (1 - mu), 1e-8, None)
        weight = deriv
    elif family == "poisson":
        deriv = np.clip(mu, 1e-8, None)
        weight = deriv
    elif family == "gamma":
        deriv = np.clip(-(mu**2), -1e8, -1e-8)
        weight = np.clip(mu**2, 1e-8, None)
    else:
        raise ValueError(family)
    return eta + (y - mu) / deriv, weight


def _mean(family: str, eta: np.ndarray) -> np.ndarray:
    if family == "gaussian":
        return eta
    if family == "binomial":
        return special.expit(eta)
    if family == "poisson":
        return np.exp(np.clip(eta, -20, 20))
    if family == "gamma":
        return 1 / np.clip(eta, 1e-6, None)
    raise ValueError(family)


def _deviance(family: str, y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-8, None)
    if family == "binomial":
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        term = np.zeros_like(y, dtype=float)
        pos = y > 0
        term[pos] += y[pos] * np.log(y[pos] / mu[pos])
        neg = y < 1
        term[neg] += (1 - y[neg]) * np.log((1 - y[neg]) / (1 - mu[neg]))
        return float(2 * np.sum(term))
    if family == "poisson":
        term = np.zeros_like(y, dtype=float)
        pos = y > 0
        term[pos] = y[pos] * np.log(y[pos] / mu[pos])
        return float(2 * np.sum(term - (y - mu)))
    if family == "gamma":
        return float(2 * np.sum((y - mu) / mu - np.log(y / mu)))
    return float(np.sum((y - mu) ** 2))


def _intercept_start(family: str, y: np.ndarray) -> float:
    mean = float(np.clip(np.mean(y), 1e-4, None))
    if family == "binomial":
        return float(special.logit(np.clip(mean, 1e-4, 1 - 1e-4)))
    if family == "poisson":
        return float(np.log(mean))
    if family == "gamma":
        return 1 / mean
    return mean


def _reml_score(design, penalty, phi, phi_df, method) -> float:
    sys = design.T @ design + penalty
    _sign, logdet_p = np.linalg.slogdet(sys)
    eig = np.linalg.eigvalsh(penalty)
    positive = eig[eig > max(eig.max(), 1.0) * 1e-10]
    logdet_s = float(np.sum(np.log(positive))) if positive.size else 0.0
    width = phi_df if method == "reml" else design.shape[0]
    return float(width / 2 * (1 + np.log(2 * np.pi * phi)) + 0.5 * (logdet_p - logdet_s))


def _logdet_penalty(penalty: np.ndarray) -> float:
    """Log determinant of a penalty, ignoring its null space."""
    sym = 0.5 * (penalty + penalty.T)
    eig = np.linalg.eigvalsh(sym)
    positive = eig[eig > max(float(np.max(np.abs(eig))), 1.0) * 1e-10]
    return float(np.sum(np.log(positive))) if positive.size else 0.0


def _penalized_hessian(design, w, penalty) -> tuple[np.ndarray, float]:
    hessian = design.T @ (w[:, None] * design) + penalty
    _sign, logdet_h = np.linalg.slogdet(hessian)
    return hessian, float(logdet_h)


def _known_scale_score(design, y, beta, penalty, family, method, mp: int) -> float:
    """Laplace REML for binomial and Poisson, where the scale is 1."""
    eta = np.clip(design @ beta, -20, 20)
    mu = _mean(family, eta)
    _z, w = _working(family, y, eta)
    _hessian, logdet_h = _penalized_hessian(design, w, penalty)
    logdet_s = _logdet_penalty(penalty)
    pen = float(beta @ penalty @ beta)
    loglik = _log_likelihood(family, y, mu)
    if method == "ml":
        return float(-loglik + 0.5 * pen)
    return float(-loglik + 0.5 * pen + 0.5 * (logdet_h - logdet_s) - 0.5 * mp * np.log(2 * np.pi))


def _gamma_loglik(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    nu = 1.0 / phi
    return float(np.sum(-special.gammaln(nu) - nu * np.log(phi * mu) + (nu - 1) * np.log(y) - y / (phi * mu)))


def _gamma_reml(design, y, beta, penalty, mp: int, method: str) -> tuple[float, float]:
    """Profile the gamma dispersion out of the Laplace REML score."""
    mu = np.clip(_mean("gamma", design @ beta), 1e-8, None)
    pen = float(beta @ penalty @ beta)
    logdet_s = _logdet_penalty(penalty)
    _z, w = _working("gamma", y, design @ beta)
    hessian = design.T @ (w[:, None] * design) + penalty
    _sign, logdet_h = np.linalg.slogdet(hessian)
    n, p = design.shape
    eig = np.linalg.eigvalsh(0.5 * (penalty + penalty.T))
    rank = float(np.sum(eig > max(float(np.max(np.abs(eig))), 1.0) * 1e-10))
    const = 0.0 if method == "ml" else 0.5 * mp * np.log(2 * np.pi)

    def objective(phi: float) -> float:
        ld_h = logdet_h - p * np.log(phi)
        ld_s = logdet_s - rank * np.log(phi)
        return -_gamma_loglik(y, mu, phi) + pen / (2 * phi) + 0.5 * (ld_h - ld_s) - const

    opt = minimize_scalar(objective, bounds=(1e-4, 1e3), method="bounded")
    return float(opt.fun), float(opt.x)


def _known_scale_score_weighted(design, y, beta, penalty, family, method, mp, prior, offset) -> float:
    eta = np.clip(design @ beta + offset, -20, 20)
    mu = _mean(family, eta)
    _z, fisher = _working(family, y, eta)
    _hessian, logdet_h = _penalized_hessian(design, fisher * prior, penalty)
    logdet_s = _logdet_penalty(penalty)
    pen = float(beta @ penalty @ beta)
    loglik = _log_likelihood_weighted(family, y, mu, prior)
    if method == "ml":
        return float(-loglik + 0.5 * pen)
    return float(-loglik + 0.5 * pen + 0.5 * (logdet_h - logdet_s) - 0.5 * mp * np.log(2 * np.pi))


def _gamma_reml_weighted(design, y, beta, penalty, mp, method, prior, offset) -> tuple[float, float]:
    mu = np.clip(_mean("gamma", design @ beta + offset), 1e-8, None)
    pen = float(beta @ penalty @ beta)
    logdet_s = _logdet_penalty(penalty)
    _z, fisher = _working("gamma", y, design @ beta + offset)
    hessian = design.T @ ((fisher * prior)[:, None] * design) + penalty
    _sign, logdet_h = np.linalg.slogdet(hessian)
    p = design.shape[1]
    eig = np.linalg.eigvalsh(0.5 * (penalty + penalty.T))
    rank = float(np.sum(eig > max(float(np.max(np.abs(eig))), 1.0) * 1e-10))
    const = 0.0 if method == "ml" else 0.5 * mp * np.log(2 * np.pi)

    def objective(phi: float) -> float:
        ld_h = logdet_h - p * np.log(phi)
        ld_s = logdet_s - rank * np.log(phi)
        return -_gamma_loglik_weighted(y, mu, phi, prior) + pen / (2 * phi) + 0.5 * (ld_h - ld_s) - const

    opt = minimize_scalar(objective, bounds=(1e-4, 1e3), method="bounded")
    return float(opt.fun), float(opt.x)


def _gamma_loglik_weighted(y: np.ndarray, mu: np.ndarray, phi: float, weights: np.ndarray) -> float:
    nu = 1.0 / phi
    terms = -special.gammaln(nu) - nu * np.log(phi * mu) + (nu - 1) * np.log(y) - y / (phi * mu)
    return float(np.sum(weights * terms))


def _log_likelihood(family: str, y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-8, None)
    if family == "binomial":
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    if family == "poisson":
        return float(np.sum(y * np.log(mu) - mu - special.gammaln(y + 1)))
    raise ValueError(family)


def _log_likelihood_weighted(family: str, y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> float:
    mu = np.clip(mu, 1e-8, None)
    if family == "binomial":
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        return float(np.sum(weights * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
    if family == "poisson":
        return float(np.sum(weights * (y * np.log(mu) - mu - special.gammaln(y + 1))))
    raise ValueError(family)


def _deviance_weighted(family: str, y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> float:
    mu = np.clip(mu, 1e-8, None)
    if family == "binomial":
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        term = np.zeros_like(y, dtype=float)
        pos = y > 0
        term[pos] += y[pos] * np.log(y[pos] / mu[pos])
        neg = y < 1
        term[neg] += (1 - y[neg]) * np.log((1 - y[neg]) / (1 - mu[neg]))
        return float(2 * np.sum(weights * term))
    if family == "poisson":
        term = np.zeros_like(y, dtype=float)
        pos = y > 0
        term[pos] = y[pos] * np.log(y[pos] / mu[pos])
        return float(2 * np.sum(weights * (term - (y - mu))))
    if family == "gamma":
        return float(2 * np.sum(weights * ((y - mu) / mu - np.log(y / mu))))
    return float(np.sum(weights * (y - mu) ** 2))


def _covariance(design, y, beta, penalties, lam, family, sig2, weights=None, offset=None) -> np.ndarray:
    penalty = _total_penalty(design.shape[1], penalties, lam)
    if family == "gaussian":
        xtx = design.T @ design if weights is None else design.T @ (weights[:, None] * design)
    else:
        eta = design @ beta if offset is None else design @ beta + offset
        _z, w = _working(family, y, eta)
        if weights is not None:
            w = w * weights
        xtx = design.T @ (w[:, None] * design)
    scale = sig2 if family in {"gaussian", "gamma"} else 1.0
    return scale * np.linalg.pinv(xtx + penalty)


def _edf_by_block(influence: np.ndarray, penalties) -> np.ndarray:
    """One effective-df per smooth. Penalties that share a column slice are one smooth."""
    seen: list[tuple[int, int, int]] = []
    out: list[float] = []
    for sl, _matrix in penalties:
        key = (int(sl[0]), int(sl[-1]), int(sl.size))
        if key in seen:
            continue
        seen.append(key)
        out.append(float(np.trace(influence[np.ix_(sl, sl)])))
    return np.asarray(out, dtype=float)


def _null_space_penalty(matrix: np.ndarray) -> np.ndarray | None:
    """Identity on the null space of a penalty, for ``select=True``."""
    sym = 0.5 * (matrix + matrix.T)
    eigval, eigvec = np.linalg.eigh(sym)
    tol = max(float(np.max(np.abs(eigval))), 1.0) * 1e-8
    vec = eigvec[:, eigval <= tol]
    if vec.size == 0:
        return None
    return vec @ vec.T


def _total_penalty(width: int, penalties, lam: np.ndarray) -> np.ndarray:
    penalty = np.zeros((width, width))
    for weight, (sl, matrix) in zip(lam, penalties, strict=True):
        penalty[np.ix_(sl, sl)] += weight * matrix
    return penalty


def _build_block(spec, kept: dict[str, np.ndarray]) -> _Block:
    if spec.basis == "te":
        return _tensor_block(spec, kept)
    if spec.basis == "ti":
        return _tensor_interaction_block(spec, kept)
    if spec.by is not None and _looks_like_factor(kept[spec.by]):
        return _by_factor_block(spec, kept)
    basis, penalty, null_dim, rebuild = _univariate(spec, kept)
    if spec.by is not None:
        by = np.asarray(kept[spec.by], dtype=float)
        basis = basis * by[:, None]

        def rebuild(columns, _rebuild=rebuild, by_name=spec.by):
            return _rebuild(columns) * np.asarray(columns[by_name], dtype=float)[:, None]

    basis, q, penalty = _absorb(basis, penalty, null_dim > 0)
    null_after = max(null_dim - 1, 0) if null_dim else 0

    def constrained(columns, _rebuild=rebuild, _q=q):
        return _rebuild(columns) @ _q

    return _Block(spec.column if spec.basis != "te" else spec.column, constrained(kept) * 0 + 0, penalty, null_after, constrained)


def _univariate(spec, kept):
    x = np.asarray(kept[spec.column])
    if spec.basis == "cr":
        from . import _cr_basis

        knots = np.quantile(np.unique(x.astype(float)), np.linspace(0, 1, spec.k))
        basis, penalty = _cr_basis(x.astype(float), knots)

        def rebuild(columns, _knots=knots):
            raw, _pen = _cr_basis(np.asarray(columns[spec.column], dtype=float), _knots)
            return raw

        return basis, penalty, 2, rebuild
    if spec.basis == "ps":
        return _ps(spec.column, x.astype(float), spec.k)
    if spec.basis == "cc":
        return _cc(spec.column, x.astype(float), spec.k)
    if spec.basis == "re":
        return _re(spec.column, x)
    if spec.basis == "tp":
        return _tp(spec.column, x.astype(float), spec.k)
    raise ValueError(f"basis {spec.basis!r} is not available")


def _absorb(basis: np.ndarray, penalty: np.ndarray, constrain: bool):
    if not constrain:
        scale = np.linalg.norm(penalty, ord=1) / max(np.linalg.norm(basis, ord=np.inf) ** 2, 1e-12)
        return basis, np.eye(basis.shape[1]), penalty / scale
    from . import _absorb_constraint

    return _absorb_constraint(basis, penalty)


def _ps(name: str, x: np.ndarray, k: int):
    from scipy.interpolate import BSpline

    m1, m2 = 2, 2
    nk = k - m1
    xl, xu = float(x.min()), float(x.max())
    span = xu - xl
    xl -= span * 0.001
    xu += span * 0.001
    dx = (xu - xl) / (nk - 1)
    knots = np.linspace(xl - dx * (m1 + 1), xu + dx * (m1 + 1), nk + 2 * m1 + 2)
    basis = BSpline.design_matrix(x, knots, m1 + 1).toarray()
    diff = np.diff(np.eye(k), n=m2, axis=0)
    penalty = diff.T @ diff

    def rebuild(columns, _knots=knots, _degree=m1 + 1):
        return BSpline.design_matrix(np.asarray(columns[name], dtype=float), _knots, _degree).toarray()

    return basis, penalty, m2, rebuild


def _cc(name: str, x: np.ndarray, k: int):
    knots = _place_knots(x, k)
    um_b, um_d = _cyclic_bd(knots)
    bd = solve(um_b, um_d, assume_a="pos")
    penalty = um_d.T @ bd
    penalty = 0.5 * (penalty + penalty.T)
    basis = _cyclic_design(x, knots, bd)

    def rebuild(columns, _knots=knots, _bd=bd):
        return _cyclic_design(np.asarray(columns[name], dtype=float), _knots, _bd)

    return basis, penalty, 1, rebuild


def _re(name: str, x: np.ndarray):
    levels = _level_labels(x)
    basis = np.column_stack([(np.asarray(x).astype(str) == level).astype(float) for level in levels])
    penalty = np.eye(basis.shape[1])

    def rebuild(columns, _levels=levels):
        values = np.asarray(columns[name]).astype(str)
        return np.column_stack([(values == level).astype(float) for level in _levels])

    return basis, penalty, 0, rebuild


def _tp(name: str, x: np.ndarray, k: int):
    """1D thin-plate regression spline, truncated like mgcv's eigen step.

    The penalty null space is the constant and the linear term. Columns are
    scaled to root-mean-square 1 before the sum-to-zero constraint.
    """
    x = np.asarray(x, dtype=float)
    knots, inverse = np.unique(x, return_inverse=True)
    n_unique = len(knots)
    k = min(k, n_unique)
    diff = knots[:, None] - knots[None, :]
    energy = np.abs(diff) ** 3 / 12
    null = np.column_stack([np.ones(n_unique), knots])
    eigval, eigvec = np.linalg.eigh(energy)
    order = np.argsort(np.abs(eigval))[::-1][:k]
    vec = eigvec[:, order]
    val = eigval[order]
    constraint = null.T @ vec
    q, _r = np.linalg.qr(constraint.T, mode="complete")
    null_dim = null.shape[1]
    z = q[:, null_dim:]
    wiggly = (vec * val) @ z
    basis_unique = np.column_stack([wiggly, null])
    penalty = np.zeros((k, k))
    penalty[: k - null_dim, : k - null_dim] = (z.T * val) @ z
    scales = np.sqrt(np.mean(basis_unique**2, axis=0))
    scales[scales == 0] = 1.0
    basis_unique = basis_unique / scales
    penalty = penalty / np.outer(scales, scales)
    basis = basis_unique[inverse]
    full = np.column_stack([energy, null])
    mapping, *_rest = np.linalg.lstsq(full, basis_unique, rcond=None)

    def rebuild(columns, _knots=knots, _map=mapping):
        xx = np.asarray(columns[name], dtype=float)
        gap = xx[:, None] - _knots[None, :]
        radial = np.abs(gap) ** 3 / 12
        poly = np.column_stack([np.ones(len(xx)), xx])
        return np.column_stack([radial, poly]) @ _map

    return basis, penalty, null_dim, rebuild


def _tensor_block(spec, kept) -> _Block:
    """Row-Kronecker of raw cubic margins, then one sum-to-zero constraint.

    ``te`` keeps a smoothing parameter per margin. Constraining each margin
    first would drop the main-effect null space that mgcv retains.
    """
    from . import _cr_basis

    names = (spec.column, *spec.extra)
    ks = (spec.k, *spec.extra_k)
    raw_margins = []
    for name, k in zip(names, ks, strict=True):
        x = np.asarray(kept[name], dtype=float)
        knots = np.quantile(np.unique(x), np.linspace(0, 1, k))
        basis, penalty = _cr_basis(x, knots)
        raw_margins.append((name, knots, basis, penalty))
    basis = raw_margins[0][2]
    pens = [raw_margins[0][3]]
    for _name, _knots, margin, penalty in raw_margins[1:]:
        n_left, n_right = basis.shape[1], margin.shape[1]
        pens = [np.kron(pen, np.eye(n_right)) for pen in pens]
        pens.append(np.kron(np.eye(n_left), penalty))
        basis = _row_kronecker(basis, margin)
    denom = max(float(np.linalg.norm(basis, ord=np.inf) ** 2), 1e-12)
    scaled = [pen * denom / max(float(np.linalg.norm(pen, ord=1)), 1e-12) for pen in pens]
    center = basis.mean(axis=0)
    q, _r = np.linalg.qr(center.reshape(-1, 1), mode="complete")
    q2 = q[:, 1:]
    pens_c = [q2.T @ pen @ q2 for pen in scaled]
    joint = np.sum(pens_c, axis=0)
    eig = np.linalg.eigvalsh(0.5 * (joint + joint.T))
    tol = max(float(np.max(np.abs(eig))), 1.0) * 1e-8
    null_after = int(np.sum(eig <= tol))
    stored = [(name, knots) for name, knots, _basis, _pen in raw_margins]

    def rebuild(columns, _stored=stored, _q=q2):
        acc = None
        for name, knots in _stored:
            raw, _pen = _cr_basis(np.asarray(columns[name], dtype=float), knots)
            acc = raw if acc is None else _row_kronecker(acc, raw)
        return acc @ _q

    label = "te(" + ",".join(names) + ")"
    return _Block(
        label,
        np.arange(q2.shape[1]),
        pens_c[0],
        null_after,
        rebuild,
        extra_penalties=tuple(pens_c[1:]),
    )


def _tensor_interaction_block(spec, kept) -> _Block:
    """Tensor product of centered margins, so the main effects are not in the basis.

    Each cubic margin is sum-to-zero constrained before the Kronecker product.
    Penalties are then scaled by the tensor design, matching ``mgcv``'s ``ti``.
    """
    from . import _cr_basis

    names = (spec.column, *spec.extra)
    ks = (spec.k, *spec.extra_k)
    margins = []
    for name, k in zip(names, ks, strict=True):
        x = np.asarray(kept[name], dtype=float)
        knots = np.quantile(np.unique(x), np.linspace(0, 1, k))
        basis, penalty = _cr_basis(x, knots)
        constrained, q2, pen = _absorb(basis, penalty, True)
        margins.append((name, knots, constrained, pen, q2))
    basis = margins[0][2]
    pens = [margins[0][3]]
    for _name, _knots, margin, penalty, _q2 in margins[1:]:
        n_left, n_right = basis.shape[1], margin.shape[1]
        pens = [np.kron(pen, np.eye(n_right)) for pen in pens]
        pens.append(np.kron(np.eye(n_left), penalty))
        basis = _row_kronecker(basis, margin)
    denom = max(float(np.linalg.norm(basis, ord=np.inf) ** 2), 1e-12)
    scaled = [pen * denom / max(float(np.linalg.norm(pen, ord=1)), 1e-12) for pen in pens]
    joint = np.sum(scaled, axis=0)
    eig = np.linalg.eigvalsh(0.5 * (joint + joint.T))
    tol = max(float(np.max(np.abs(eig))), 1.0) * 1e-8
    null_after = int(np.sum(eig <= tol))
    stored = [(name, knots, q2) for name, knots, _basis, _pen, q2 in margins]

    def rebuild(columns, _stored=stored):
        acc = None
        for name, knots, q2 in _stored:
            raw, _pen = _cr_basis(np.asarray(columns[name], dtype=float), knots)
            margin = raw @ q2
            acc = margin if acc is None else _row_kronecker(acc, margin)
        return acc

    label = "ti(" + ",".join(names) + ")"
    return _Block(
        label,
        np.arange(basis.shape[1]),
        scaled[0],
        null_after,
        rebuild,
        extra_penalties=tuple(scaled[1:]),
    )


def _tensor_product(left, left_pen, right, right_pen):
    basis = _row_kronecker(left, right)
    n_l, n_r = left.shape[1], right.shape[1]
    pen_l = np.kron(left_pen, np.eye(n_r))
    pen_r = np.kron(np.eye(n_l), right_pen)
    return basis, pen_l + pen_r


def _row_kronecker(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n, a = left.shape
    b = right.shape[1]
    return (left[:, :, None] * right[:, None, :]).reshape(n, a * b)


def _by_factor_block(spec, kept) -> _Block:
    """One smooth per factor level.

    Knots and the sum-to-zero constraint come from the whole covariate, as in
    ``mgcv``'s ``s(x, by=factor)``. Each level then keeps its own penalty.
    """
    levels = _level_labels(kept[spec.by])
    child = type(spec)(column=spec.column, k=spec.k, basis=spec.basis, by=None, extra=spec.extra, extra_k=spec.extra_k)
    raw, penalty, null_dim, rebuild = _univariate(child, kept)
    constrained, q, penalty = _absorb(raw, penalty, null_dim > 0)
    parts = []
    penalties = []
    nulls: list[int] = []
    kept_levels: list[str] = []
    for level in levels:
        mask = np.asarray(kept[spec.by]).astype(str) == level
        if int(mask.sum()) < 3:
            continue
        full = np.zeros_like(constrained)
        full[mask] = constrained[mask]
        parts.append(full)
        penalties.append(np.array(penalty, copy=True))
        nulls.append(max(null_dim - 1, 0) if null_dim else 0)
        kept_levels.append(level)
    if not parts:
        raise ValueError(f"by-factor smooth {spec.column!r} has no level with enough rows")
    basis = np.column_stack(parts)
    pens = []
    cursor = 0
    for level_penalty in penalties:
        full_pen = np.zeros((basis.shape[1], basis.shape[1]))
        width = level_penalty.shape[0]
        full_pen[cursor : cursor + width, cursor : cursor + width] = level_penalty
        pens.append(full_pen)
        cursor += width

    def rebuild_all(columns, _rebuild=rebuild, _q=q, _levels=tuple(kept_levels), _by=spec.by, _col=spec.column):
        values = np.asarray(columns[_col], dtype=float)
        group = np.asarray(columns[_by]).astype(str)
        shared = _rebuild({_col: values}) @ _q
        pieces = []
        for level in _levels:
            indicator = group == level
            pieces.append(shared * indicator[:, None])
        return np.column_stack(pieces)

    return _Block(
        f"{spec.column}|{spec.by}",
        np.arange(basis.shape[1]),
        pens[0],
        int(sum(nulls)),
        rebuild_all,
        extra_penalties=tuple(pens[1:]),
    )


def _with_null_penalties(blocks: list[_Block]) -> list[_Block]:
    """Extra ridge on the null space of each smooth (Marra and Wood)."""
    extra = []
    for block in blocks:
        if block.null_dim <= 0:
            continue
        null_pen = np.zeros_like(block.penalty)
        # The trailing null_dim columns are the least penalized after the eigen
        # ordering used by the cubic and thin-plate bases. A small identity there
        # lets `select=True` shrink a smooth toward zero.
        null_pen[-block.null_dim :, -block.null_dim :] = np.eye(block.null_dim)
        twin = _Block(block.name + ":null", block.columns, null_pen, 0, block.rebuild)
        extra.append(twin)
    return blocks + extra


def _block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    width = sum(matrix.shape[0] for matrix in matrices)
    out = np.zeros((width, width))
    cursor = 0
    for matrix in matrices:
        n = matrix.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = matrix
        cursor += n
    return out


def _place_knots(x: np.ndarray, nk: int) -> np.ndarray:
    values = np.sort(np.unique(x))
    n = len(values)
    if nk >= n:
        nk = max(n, 4)
    if nk == 2:
        return np.array([values[0], values[-1]])
    delta = (n - 1) / (nk - 1)
    knot = np.zeros(nk)
    knot[0] = values[0]
    knot[-1] = values[-1]
    for i, position in enumerate(np.arange(1, nk - 1) * delta):
        lower = int(np.floor(position))
        frac = position - lower
        knot[i + 1] = values[lower] * (1 - frac) + values[min(lower + 1, n - 1)] * frac
    return knot


def _cyclic_bd(knots: np.ndarray):
    h = np.diff(knots)
    # The last gap closes the circle.
    h = np.append(h, knots[0] + (knots[-1] - knots[0]) - knots[-1] + (knots[1] - knots[0]) * 0 + (knots[-1] - knots[0]))
    # mgcv's cyclic basis uses length(knots) points and n = length - 1 intervals
    # that include the wrap from the last knot back to the first.
    n = len(knots) - 1
    step = np.diff(knots)
    step = np.append(step[:-1] if False else np.diff(knots), (knots[-1] - knots[0]))
    # The constructor's getBD is called on k, and n = length(k) - 1 after h = diff.
    # h has length n, and the wrap uses h[n] which is h[length-1] = last diff.
    # Re-read: h <- diff(x); n <- n - 1 so h has the interior diffs only if x includes
    # both endpoints and the circle is coded by indexing h[n] as h[length(h)] in R
    # which is the last element. So h is diff(knots), length nk-1, and h[n] means h[nk-1]
    # the last difference. There is no extra wrap distance: the last interval is the
    # last diff, and the circle identifies knot 1 with knot n in the (n) x (n) system
    # where n = length(knots) - 1. So the basis dimension is nk - 1, not nk.
    h = np.diff(knots)
    n = len(h)
    b = np.zeros((n, n))
    d = np.zeros((n, n))
    b[0, 0] = (h[-1] + h[0]) / 3
    b[0, 1] = h[0] / 6
    b[0, -1] = h[-1] / 6
    d[0, 0] = -(1 / h[0] + 1 / h[-1])
    d[0, 1] = 1 / h[0]
    d[0, -1] = 1 / h[-1]
    for i in range(1, n - 1):
        b[i, i - 1] = h[i - 1] / 6
        b[i, i] = (h[i - 1] + h[i]) / 3
        b[i, i + 1] = h[i] / 6
        d[i, i - 1] = 1 / h[i - 1]
        d[i, i] = -(1 / h[i - 1] + 1 / h[i])
        d[i, i + 1] = 1 / h[i]
    b[-1, -2] = h[-2] / 6
    b[-1, -1] = (h[-2] + h[-1]) / 3
    b[-1, 0] = h[-1] / 6
    d[-1, -2] = 1 / h[-2]
    d[-1, -1] = -(1 / h[-2] + 1 / h[-1])
    d[-1, 0] = 1 / h[-1]
    return b, d


def _cyclic_design(x: np.ndarray, knots: np.ndarray, bd: np.ndarray) -> np.ndarray:
    wrapped = _cwrap(knots[0], knots[-1], np.asarray(x, dtype=float))
    h = np.diff(knots)
    n = len(knots)
    # j is the upper knot index in R's 1-based loop `for (i in n:2) j[x <= knots[i]] <- i`.
    upper = np.searchsorted(knots, wrapped, side="left")
    upper = np.clip(upper, 1, n - 1)
    lower = upper - 1
    # R then sets j[j == n] <- 1, so the upper index wraps. Our bd has n-1 rows.
    upper_wrapped = np.where(upper == n - 1, 0, upper)
    # The R matrix is (n-1) x (n-1). Indices j1 = j-1 are 0-based lower.
    span = h[lower]
    left = knots[upper] - wrapped
    right = wrapped - knots[lower]
    eye = np.eye(n - 1)
    out = (
        bd[lower] * (left**3)[:, None] / (6 * span)[:, None]
        + bd[upper_wrapped] * (right**3)[:, None] / (6 * span)[:, None]
        - bd[lower] * (span * left / 6)[:, None]
        - bd[upper_wrapped] * (span * right / 6)[:, None]
        + eye[lower] * (left / span)[:, None]
        + eye[upper_wrapped] * (right / span)[:, None]
    )
    return out


def _cwrap(lower: float, upper: float, x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=float, copy=True)
    span = upper - lower
    high = out > upper
    out[high] = lower + np.mod(out[high] - upper, span)
    low = out < lower
    out[low] = upper - np.mod(lower - out[low], span)
    return out


def _offset_vector(fit: GeneralFit, data: pl.DataFrame | None, n: int) -> np.ndarray:
    if fit.offset_name is None:
        return np.zeros(n)
    if data is None:
        return fit.offset
    return np.asarray(column_series(data, fit.offset_name).to_numpy(), dtype=float)


def _positive_weights(data: pl.DataFrame, name: str | None, keep: np.ndarray) -> np.ndarray | None:
    if name is None:
        return None
    values = _aligned_numeric(data, name, keep)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be positive and finite")
    return values


def _aligned_numeric(data: pl.DataFrame, name: str, keep: np.ndarray) -> np.ndarray:
    return np.asarray(column_series(data, name).filter(pl.Series(keep)).to_numpy(), dtype=float)


def _used_columns(smooths, predictors) -> list[str]:
    names: list[str] = []
    for spec in smooths:
        for name in (spec.column, *spec.extra, spec.by):
            if name and name not in names:
                names.append(name)
    for name in predictors or []:
        if name not in names:
            names.append(name)
    return names


def _columns(data: pl.DataFrame, names) -> dict[str, np.ndarray]:
    return {name: np.asarray(column_series(data, name).to_numpy()) for name in names}


def _design_from_blocks(columns: dict[str, np.ndarray], blocks: list[_Block], parametric: list[tuple]) -> np.ndarray:
    n = len(next(iter(columns.values())))
    pieces = [np.ones(n)]
    for spec in parametric:
        if spec[0] == "numeric":
            pieces.append(np.asarray(columns[spec[1]], dtype=float))
        else:
            pieces.append((np.asarray(columns[spec[1]]).astype(str) == spec[2]).astype(float))
    for block in blocks:
        pieces.append(block.rebuild(columns))
    return np.column_stack(pieces)


def _term_frame(design: np.ndarray, beta: np.ndarray, names: list[str], blocks: list[_Block]) -> pl.DataFrame:
    columns = {"(Intercept)": design[:, 0] * beta[0]}
    cursor = 1
    for name in names[1:]:
        if any(name.startswith(f"{block.name}[") for block in blocks):
            break
        columns[name] = design[:, cursor] * beta[cursor]
        cursor += 1
    for block in blocks:
        sl = block.columns
        columns[block.name] = design[:, sl] @ beta[sl]
    return pl.DataFrame(columns)


def _is_factor(data: pl.DataFrame, name: str) -> bool:
    dtype = data.schema[name]
    return isinstance(dtype, (pl.Enum, pl.Categorical, pl.Utf8, pl.String))


def _factor_levels(data: pl.DataFrame, name: str, values: np.ndarray) -> list[str]:
    dtype = data.schema[name]
    if isinstance(dtype, pl.Enum):
        return [str(level) for level in dtype.categories.to_list()]
    return sorted({str(value) for value in values})


def _looks_like_factor(values: np.ndarray) -> bool:
    return values.dtype.kind in {"U", "O", "S"} or np.issubdtype(values.dtype, np.integer)


def _level_labels(values: np.ndarray) -> list[str]:
    return sorted({str(value) for value in values})


def _effect_grid(kept: dict[str, np.ndarray], block: _Block, n: int) -> dict[str, np.ndarray]:
    name = block.name.split("|", 1)[0].split("(", 1)[-1].split(",", 1)[0].rstrip(")")
    if name not in kept:
        name = next(iter(kept))
    x = np.asarray(kept[name], dtype=float)
    grid = {key: np.repeat(np.asarray(value)[:1], n) for key, value in kept.items()}
    grid[name] = np.linspace(np.nanmin(x), np.nanmax(x), n)
    return grid


def _k_index(resid: np.ndarray, covariate: np.ndarray | None) -> float:
    if covariate is None or covariate.dtype.kind not in {"f", "i", "u"}:
        return float("nan")
    order = np.argsort(np.asarray(covariate, dtype=float), kind="mergesort")
    ordered = resid[order]
    if ordered.size < 3:
        return float("nan")
    differenced = np.diff(ordered)
    denom = float(np.var(differenced))
    if denom <= 0:
        return float("nan")
    return float(np.var(ordered) / denom)


def _rsquared(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 0.0
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(right)), right]), left, rcond=None)[0]
    fitted = beta[0] + beta[1] * right
    resid = left - fitted
    total = left - left.mean()
    return float(1 - np.sum(resid**2) / np.sum(total**2))
