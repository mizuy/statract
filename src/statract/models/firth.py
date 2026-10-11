"""Firth's penalized-likelihood logistic regression (the model of ``logistf::logistf``).

Written for statract from the papers:

* Firth, D. (1993). Bias reduction of maximum likelihood estimates.
  Biometrika 80(1), 27-38. The log-likelihood gets the Jeffreys-prior penalty
  ``0.5 log det I(beta)``, with ``I(beta) = X'WX`` and
  ``W = diag(w p (1 - p))``.
* Heinze, G., Schemper, M. (2002). A solution to the problem of separation in
  logistic regression. Statistics in Medicine 21, 2409-2419. The modified
  score ``U*(beta) = X'(w (y - p) + h (1/2 - p))``, where ``h`` is the
  diagonal of the weighted hat matrix ``W^(1/2) X I^(-1) X' W^(1/2)``; the
  Newton-Raphson iteration ``beta + I^(-1) U*``; profile penalized-likelihood
  confidence intervals; and penalized likelihood-ratio tests, in which a
  restricted model keeps the penalty of the full design.

The estimate stays finite under complete or quasi-complete separation. The
controls follow the help page of R's ``logistf.control()``, used only as a
description of the behavior: at most ``maxit`` (25) iterations, no
coefficient moves more than ``maxstep`` (5) in one step, the step is halved
at most ``maxhs`` (0) times while the penalized log-likelihood falls, and the
iteration stops once the largest step, the largest modified score, and the
change in the penalized log-likelihood are all below ``tol`` (1e-5). No
logistf code was used.

The covariance is the inverse of ``I(beta)`` at the estimate. ``tidy``
defaults to profile penalized-likelihood intervals and penalized
likelihood-ratio p-values. A restricted fit fixes one coefficient (or, for
the global test, every coefficient but the intercept) and maximizes the
penalized log-likelihood of the full design over the others; restricted fits
are solved to 1e-10. The global test compares with the model whose only free
coefficient is the intercept.

Profile limits are solved for exactly (``_profile.profile_limit``), with
each restricted fit warm-started from the last. Under complete separation
with several covariates the restricted penalized likelihood can have two
local maxima; the limit then follows the one reached continuously from the
estimate.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import linalg, stats

from ._profile import profile_limit
from .design import ColumnRef, Design, build_design, column_series, design_matrix
from .fit import _from_formula, _optional_numeric, _outcome_array, _take_aligned
from .formula import is_formula

# Restricted fits (tests and profile limits) are solved to this tolerance.
_TIGHT = 1e-10


def _expit(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _penalized_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, off: np.ndarray) -> float:
    """``sum w (y eta - log(1 + e^eta)) + 0.5 log det X'WX``."""
    eta = x @ beta + off
    loglik = float(w @ (y * eta - np.logaddexp(0.0, eta)))
    p = _expit(eta)
    info = (x * (w * p * (1.0 - p))[:, None]).T @ x
    try:
        chol = linalg.cholesky(info, lower=True)
    except linalg.LinAlgError:
        return -np.inf
    return loglik + float(np.sum(np.log(np.diag(chol))))


def _score_and_info(beta, x, y, w, off) -> tuple[np.ndarray, np.ndarray]:
    """Modified score ``U*`` and Fisher information ``I`` at ``beta``."""
    p = _expit(x @ beta + off)
    v = w * p * (1.0 - p)
    info = (x * v[:, None]).T @ x
    inv = linalg.inv(info)
    h = v * np.einsum("ij,jk,ik->i", x, inv, x)
    score = x.T @ (w * (y - p) + h * (0.5 - p))
    return score, info


@dataclass
class _Newton:
    beta: np.ndarray
    loglik: float
    iterations: int
    converged: bool


def _newton(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    off: np.ndarray,
    start: np.ndarray,
    free: np.ndarray,
    *,
    maxit: int,
    maxstep: float,
    maxhs: int,
    tol: float,
    slack: float = 0.0,
) -> _Newton:
    """Newton-Raphson on the modified score over the ``free`` coefficients.

    The others stay at their values in ``start``. The step on the free block
    is ``I_ff^(-1) U*_f``, scaled so the largest component is at most
    ``maxstep``, then halved up to ``maxhs`` times while the penalized
    log-likelihood is lower than before. Convergence needs the largest step,
    the largest free score (both before the step), and the change in the
    penalized log-likelihood all below ``tol``. ``slack`` lets the
    log-likelihood fall by rounding error without halving.
    """
    beta = np.array(start, dtype=float)
    loglik = _penalized_loglik(beta, x, y, w, off)
    if free.size == 0:
        return _Newton(beta, loglik, 0, True)
    for it in range(1, maxit + 1):
        score, info = _score_and_info(beta, x, y, w, off)
        u = score[free]
        delta = linalg.solve(info[np.ix_(free, free)], u, assume_a="pos")
        biggest = float(np.max(np.abs(delta)))
        if biggest > maxstep:
            delta = delta * (maxstep / biggest)
        trial = beta.copy()
        trial[free] += delta
        new = _penalized_loglik(trial, x, y, w, off)
        for _ in range(maxhs):
            if new >= loglik - slack:
                break
            delta = delta / 2.0
            trial = beta.copy()
            trial[free] += delta
            new = _penalized_loglik(trial, x, y, w, off)
        done = float(np.max(np.abs(delta))) <= tol and float(np.max(np.abs(u))) < tol and abs(new - loglik) < tol
        beta, loglik = trial, new
        if done:
            return _Newton(beta, loglik, it, True)
    return _Newton(beta, loglik, maxit, False)


def _restricted(x, y, w, off, start: np.ndarray, free: np.ndarray) -> _Newton:
    """Penalized likelihood maximized over ``free`` to ``_TIGHT``, with ascent guaranteed by halving."""
    slack = 1e-12 * max(1.0, float(np.sum(w)))
    return _newton(x, y, w, off, start, free, maxit=500, maxstep=5.0, maxhs=60, tol=_TIGHT, slack=slack)


@dataclass
class FirthFit:
    """Firth logistic regression.

    ``log_likelihood`` is the penalized log-likelihood at the estimate.
    ``log_likelihood_null`` is the penalized log-likelihood of the model whose
    only free coefficient is the intercept, used by the global test.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    log_likelihood: float
    log_likelihood_null: float
    n_obs: int
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    offset: np.ndarray
    row_index: np.ndarray
    design: Design
    iterations: int
    converged: bool
    _profiles: dict[float, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)
    _lr: np.ndarray | None = field(default=None, repr=False)
    _offset_column: ColumnRef | None = field(default=None, repr=False)
    _formula_offset: bool = field(default=False, repr=False)

    @property
    def std_errors(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def df(self) -> int:
        """Degrees of freedom of the global test: coefficients other than the intercept."""
        return len(self.names) - (1 if "(Intercept)" in self.names else 0)

    def _fixed_at(self, j: int, value: float, start: np.ndarray) -> _Newton:
        free = np.array([k for k in range(len(self.names)) if k != j], dtype=np.int64)
        begin = np.array(start, dtype=float)
        begin[j] = value
        return _restricted(self.x, self.y, self.weights, self.offset, begin, free)

    def likelihood_ratio(self) -> np.ndarray:
        """Penalized likelihood-ratio chi-square for each coefficient being 0 (1 df)."""
        if self._lr is None:
            out = np.empty(len(self.names))
            for j in range(len(self.names)):
                restricted = self._fixed_at(j, 0.0, self.coefficients)
                out[j] = max(2.0 * (self.log_likelihood - restricted.loglik), 0.0)
            self._lr = out
        return self._lr

    def profile_interval(self, *, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Profile penalized-likelihood limits (``logistf(pl = TRUE)``'s ``ci.lower`` / ``ci.upper``)."""
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        key = float(level)
        if key in self._profiles:
            return self._profiles[key]
        q = float(stats.norm.ppf(0.5 + level / 2.0))
        p = len(self.names)
        low, high = np.empty(p), np.empty(p)
        se = self.std_errors
        for j in range(p):
            for side in (-1.0, 1.0):
                warm = [self.coefficients.copy()]

                def drop(b: float, j: int = j, warm: list = warm) -> float:
                    fit = self._fixed_at(j, b, warm[0])
                    if np.isfinite(fit.loglik):
                        warm[0] = fit.beta
                    return 2.0 * (self.log_likelihood - fit.loglik)

                limit = profile_limit(drop, float(self.coefficients[j]), float(se[j]), side * q)
                if side < 0:
                    low[j] = limit
                else:
                    high[j] = limit
        self._profiles[key] = (low, high)
        return low, high

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False, method: str = "profile") -> pl.DataFrame:
        """Coefficient table.

        ``method="profile"`` (``logistf``'s default) gives profile penalized
        likelihood intervals, and ``statistic`` is the penalized
        likelihood-ratio chi-square with its p-value. ``method="wald"`` gives
        ``pl = FALSE``: z statistics and Wald intervals from ``covariance``.
        With ``exponentiate`` the estimate and limits are odds ratios;
        ``std_error`` stays on the log-odds scale.
        """
        beta = self.coefficients
        se = self.std_errors
        if method == "profile":
            statistic = self.likelihood_ratio()
            p_value = stats.chi2.sf(statistic, 1)
            low, high = self.profile_interval(level=level)
        elif method == "wald":
            statistic = beta / se
            p_value = 2.0 * stats.norm.sf(np.abs(statistic))
            z = float(stats.norm.ppf(0.5 + level / 2.0))
            low, high = beta - z * se, beta + z * se
        else:
            raise ValueError('method must be "profile" or "wald"')
        estimate = beta
        if exponentiate:
            estimate, low, high = np.exp(beta), np.exp(low), np.exp(high)
        return pl.DataFrame(
            {
                "term": list(self.names),
                "estimate": estimate,
                "std_error": se,
                "statistic": statistic,
                "p_value": p_value,
                "conf_low": low,
                "conf_high": high,
            }
        )

    def glance(self) -> pl.DataFrame:
        """Global penalized likelihood-ratio test and the fit summary."""
        statistic = max(2.0 * (self.log_likelihood - self.log_likelihood_null), 0.0)
        df = self.df
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "log_likelihood": [self.log_likelihood],
                "log_likelihood_null": [self.log_likelihood_null],
                "statistic": [statistic],
                "df": [df],
                "p_value": [float(stats.chi2.sf(statistic, df)) if df > 0 else float("nan")],
                "iterations": [self.iterations],
                "converged": [self.converged],
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        """Probability (``kind="response"``) or log odds (``"link"``).

        With ``data=None`` the fitted rows are used, with their offset. New
        rows read an ``offset=`` column from ``data``; a fit with an
        ``offset()`` term in the formula cannot predict new rows.
        """
        if kind not in ("response", "link"):
            raise ValueError('kind must be "response" or "link"')
        if data is None:
            eta = self.x @ self.coefficients + self.offset
        else:
            if self._formula_offset:
                raise ValueError("a fit with an offset() term cannot predict new rows; pass offset= as a column")
            extra = [self._offset_column] if self._offset_column is not None else None
            built = build_design(data, self.design, extra=extra)
            eta = built.x @ self.coefficients
            if self._offset_column is not None:
                eta = eta + np.asarray(column_series(data, self._offset_column).to_numpy(), dtype=float)[built.row_index]
        return eta if kind == "link" else _expit(eta)


def fit_firth(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    maxit: int = 25,
    maxstep: float = 5.0,
    maxhs: int = 0,
    tol: float = 1e-5,
) -> FirthFit:
    """Firth logistic regression for a 0/1 outcome (the model of ``logistf::logistf``).

    ``outcome`` is a column or a Wilkinson formula such as ``"y ~ arm + age"``.
    ``weights`` are frequency weights. ``maxit``, ``maxstep``, ``maxhs``, and
    ``tol`` play the roles of ``logistf.control()``'s ``maxit``, ``maxstep``,
    ``maxhs``, and ``lconv`` / ``gconv`` / ``xconv``.
    """
    formula_offset = False
    if is_formula(outcome):
        if predictors is not None:
            raise ValueError("pass a Wilkinson formula or predictors, not both")
        design, y, w, off = _from_formula(data, str(outcome), weights, offset)
        if off is not None and offset is None:
            formula_offset = bool(np.any(np.asarray(off) != 0))
        elif off is not None and offset is not None:
            column = _optional_numeric(data, offset, design.row_index)
            formula_offset = bool(np.any(np.asarray(off) - np.asarray(column) != 0))
    else:
        if predictors is None:
            raise ValueError("predictors are required when outcome is a column")
        extra: list[ColumnRef] = [outcome]
        if weights is not None:
            extra.append(weights)
        if offset is not None:
            extra.append(offset)
        design = design_matrix(data, predictors, extra=extra)
        y = _outcome_array(_take_aligned(column_series(data, outcome), design.row_index))
        w = _optional_numeric(data, weights, design.row_index)
        off = _optional_numeric(data, offset, design.row_index)
    y = np.asarray(y, dtype=float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("fit_firth needs a 0/1 outcome")
    n = design.n_obs
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(n) if off is None else np.asarray(off, dtype=float)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    x = np.asarray(design.x, dtype=float)
    p = x.shape[1]
    if p == 0:
        raise ValueError("the design has no columns")

    main = _newton(x, y, w, off, np.zeros(p), np.arange(p), maxit=maxit, maxstep=maxstep, maxhs=maxhs, tol=tol)
    if not main.converged:
        warnings.warn(
            f"fit_firth did not converge in {maxit} iterations; raise maxit",
            RuntimeWarning,
            stacklevel=2,
        )
    _score, info = _score_and_info(main.beta, x, y, w, off)
    covariance = linalg.inv(info)
    covariance = (covariance + covariance.T) / 2.0

    names = list(design.names)
    if "(Intercept)" in names:
        start = np.zeros(p)
        free = np.array([names.index("(Intercept)")], dtype=np.int64)
        null = _restricted(x, y, w, off, start, free)
        loglik_null = null.loglik
    else:
        loglik_null = _penalized_loglik(np.zeros(p), x, y, w, off)

    return FirthFit(
        coefficients=main.beta,
        covariance=covariance,
        names=names,
        log_likelihood=main.loglik,
        log_likelihood_null=float(loglik_null),
        n_obs=n,
        x=x,
        y=y,
        weights=w,
        offset=off,
        row_index=design.row_index,
        design=design,
        iterations=main.iterations,
        converged=main.converged,
        _offset_column=offset,
        _formula_offset=formula_offset,
    )
