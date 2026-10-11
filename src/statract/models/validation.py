"""Bootstrap validation and calibration of prediction models, after ``rms``.

``validate_logistic`` follows ``rms::validate.lrm``, ``calibrate_logistic``
follows ``rms::calibrate`` for an ``lrm`` fit, ``validate_cox`` follows
``rms::validate.cph`` and ``calibrate_cox`` follows ``rms::calibrate.cph`` with
``cmethod = "KM"``. All use the ordinary bootstrap of
``predab.resample``: each repetition fits the model on a resample and scores it
on that resample (training) and on the whole sample (test). The optimism is the
mean training index minus the mean test index.

The fits stop where rms stops them, which is early: ``lrm.fit`` stops once
``-2 log L`` changes by less than 0.025, and ``cph`` uses ``eps = 1e-4`` on the
relative change of the partial likelihood. Both fits are plain
Newton–Raphson iterations with these documented stopping rules. The
coefficients can therefore differ from the fully converged ``fit_glm`` and
``cox_ph`` (up to about 1e-4 relative in small samples), and the indexes
match rms instead. The calibration curves use the statsmodels-derived lowess
in ``statract._core.lowess``.

R's random numbers are not reproduced. Pass ``indices`` (``B`` rows of
0-based row numbers of the complete cases) to reuse given resamples; with
the same resamples the results equal rms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from numba import njit
from scipy import special
from scipy.stats import rankdata

from .._core import mpl as _mpl  # noqa: F401
from .._core.lowess import interpolate, lowess
from .design import ColumnRef
from .fit import fit_glm

import matplotlib.pyplot as plt

_LOGISTIC_INDEXES = ["Dxy", "R2", "Intercept", "Slope", "Emax", "D", "U", "Q", "B", "g", "gp"]
_MIN_CALIBRATION_EVENTS = 5
_COX_INDEXES = ["Dxy", "R2", "Slope", "D", "U", "Q", "g"]


# ---------------------------------------------------------------------------
# Small helpers (same values as the Hmisc functions)
# ---------------------------------------------------------------------------


def gini_mean_difference(x: np.ndarray) -> float:
    """Gini's mean difference: the mean of |x_i - x_j| over all pairs i != j.

    With order statistics x_(1) <= ... <= x_(n) this is
    2 / (n (n - 1)) * sum_i (2 i - n - 1) x_(i) (David 1968). The values are
    centered first, which leaves the result unchanged because the
    coefficients sum to zero. Same value as ``Hmisc::GiniMd``.
    """
    values = np.asarray(x, dtype=float)
    n = values.size
    if n < 2:
        return float("nan")
    ordered = np.sort(values - values.mean())
    rank_coef = 2.0 * np.arange(1, n + 1) - n - 1
    return float(2.0 * np.sum(rank_coef * ordered) / (n * (n - 1)))


def _somers2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = x.size
    n1 = float(np.sum(y == 1))
    if n1 == 0 or n1 == n:
        return float("nan"), float("nan")
    mean_rank = float(np.mean(rankdata(x)[y == 1]))
    c = (mean_rank - (n1 + 1) / 2.0) / (n - n1)
    return c, 2.0 * (c - 0.5)


@njit(cache=True)
def _surv_concordance(time, event, risk_rank, n_rank):
    """Concordant, discordant and risk-tied pair counts for censored data.

    Harrell et al. (1982, JAMA 247:2543) and Harrell, Lee & Mark (1996, Stat
    Med 15:361, sec. 5.5): a pair is usable when the shorter follow-up ends in
    an event. Two events at the same time are not usable; a censored time
    equal to an event time is treated as the longer one. Among usable pairs
    the subject with the event is concordant when its risk rank is higher.

    Subjects are swept from the longest time down while a Fenwick (binary
    indexed) tree counts the risk ranks of everyone with a longer time, so the
    cost is O(n log n). Within one time, censored subjects enter the tree
    before the events are scored and events enter after.
    """
    n = time.shape[0]
    order = np.argsort(-time, kind="mergesort")
    tree = np.zeros(n_rank + 1)
    concordant = 0.0
    discordant = 0.0
    tied = 0.0
    entered = 0.0
    start = 0
    while start < n:
        stop = start
        t0 = time[order[start]]
        while stop < n and time[order[stop]] == t0:
            stop += 1
        for k in range(start, stop):
            r = order[k]
            if event[r] == 0:
                pos = risk_rank[r] + 1
                while pos <= n_rank:
                    tree[pos] += 1.0
                    pos += pos & (-pos)
                entered += 1.0
        for k in range(start, stop):
            r = order[k]
            if event[r] != 0:
                # number of longer-time subjects with rank < own, and with rank <= own
                below = 0.0
                pos = risk_rank[r]
                while pos > 0:
                    below += tree[pos]
                    pos -= pos & (-pos)
                at_or_below = 0.0
                pos = risk_rank[r] + 1
                while pos > 0:
                    at_or_below += tree[pos]
                    pos -= pos & (-pos)
                concordant += below
                tied += at_or_below - below
                discordant += entered - at_or_below
        for k in range(start, stop):
            r = order[k]
            if event[r] != 0:
                pos = risk_rank[r] + 1
                while pos <= n_rank:
                    tree[pos] += 1.0
                    pos += pos & (-pos)
                entered += 1.0
        start = stop
    return concordant, discordant, tied


def _dxy_cens(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    uniq, rank = np.unique(np.asarray(risk, dtype=float), return_inverse=True)
    conc, disc, tied = _surv_concordance(
        np.ascontiguousarray(time, dtype=float),
        np.ascontiguousarray(event, dtype=np.int64),
        np.ascontiguousarray(rank, dtype=np.int64),
        int(uniq.size),
    )
    total = conc + disc + tied
    if total == 0:
        return float("nan")
    return float((conc - disc) / total)


def somers_dxy(
    prediction: np.ndarray,
    outcome: np.ndarray,
    event: np.ndarray | None = None,
) -> pl.DataFrame:
    """Apparent C index and Somers' Dxy of a prediction.

    With a binary ``outcome`` this is ``Hmisc::somers2``. With ``event`` the
    outcome is a follow-up time and the prediction is a risk score (higher means
    an earlier event), as ``rms`` scores a Cox linear predictor
    (``dxy.cens(..., type = "hazard")``). Pairs with tied event times are not used.
    """
    x = np.asarray(prediction, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if event is None:
        keep = ~(np.isnan(x) | np.isnan(y))
        x, y = x[keep], y[keep]
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("outcome must be 0/1 when event is not given")
        c, dxy = _somers2(x, y)
    else:
        e = np.asarray(event, dtype=float)
        keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(e))
        x, y, e = x[keep], y[keep], e[keep]
        dxy = _dxy_cens(x, y, e > 0)
        c = dxy / 2.0 + 0.5
    return pl.DataFrame({"c_index": [c], "dxy": [dxy], "n": [int(x.size)]})


# ---------------------------------------------------------------------------
# Binary logistic fit, stopped as lrm.fit documents
# ---------------------------------------------------------------------------


@dataclass
class _LrmFit:
    coefficients: np.ndarray
    deviance: tuple[float, float]  # -2 log L of the intercept-only model and of the fit
    fail: bool

    @property
    def model_lr(self) -> float:
        return self.deviance[0] - self.deviance[1]


def _binary_deviance(eta: np.ndarray, y: np.ndarray) -> float:
    """-2 log L of a Bernoulli model with logit ``eta``."""
    return float(-2.0 * np.sum(y * eta - np.logaddexp(0.0, eta)))


def _lrm_fit(x: np.ndarray, y: np.ndarray, *, maxit: int = 12, eps: float = 0.025, tol: float = 1e-7) -> _LrmFit:
    """Binary logistic regression by Newton–Raphson (McCullagh & Nelder 1989, sec. 4.4).

    The score is ``X'(y - mu)`` and the information ``X' W X`` with
    ``W = diag(mu (1 - mu))`` (Hosmer & Lemeshow 2000, sec. 1.2). The
    iteration starts at the intercept-only fit ``logit(mean y)``. Convergence
    is declared when ``-2 log L`` changes by less than ``eps`` (the rule the
    ``lrm.fit`` help page documents); the Newton step computed at that
    point is still applied, so the estimate is one quadratic step past the
    convergence check. This reproduces the printed rms estimates. A step that
    raises ``-2 log L`` is halved. ``tol`` is the smallest allowed eigenvalue
    of the scaled information (a singularity check).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float).reshape(y.size, -1)
    n, p = x.shape
    design = np.column_stack([np.ones(n), x])
    ybar = float(y.mean()) if n else float("nan")
    if not (0.0 < ybar < 1.0):
        return _LrmFit(np.full(p + 1, np.nan), (0.0, 0.0), True)
    beta = np.zeros(p + 1)
    beta[0] = np.log(ybar / (1.0 - ybar))
    null_dev = _binary_deviance(np.full(n, beta[0]), y)
    if p == 0:
        return _LrmFit(beta, (null_dev, null_dev), False)

    def newton_step(b):
        mu = special.expit(design @ b)
        info = design.T @ (design * (mu * (1.0 - mu))[:, None])
        score = design.T @ (y - mu)
        d = np.diag(info)
        if not (np.all(np.isfinite(info)) and np.all(d > 0.0)):
            return None
        scale = np.sqrt(d)
        scaled = info / np.outer(scale, scale)
        if np.min(np.linalg.eigvalsh(scaled)) < tol:
            return None
        return np.linalg.solve(scaled, score / scale) / scale

    dev = null_dev
    converged = False
    for _ in range(maxit):
        step = newton_step(beta)
        if step is None:
            return _LrmFit(np.full(p + 1, np.nan), (null_dev, np.nan), True)
        if converged:
            beta = beta + step
            dev = _binary_deviance(design @ beta, y)
            break
        new_beta = beta + step
        new_dev = _binary_deviance(design @ new_beta, y)
        halvings = 0
        while not (np.isfinite(new_dev) and new_dev <= dev + eps / 10.0) and halvings < 30:
            new_beta = (new_beta + beta) / 2.0
            new_dev = _binary_deviance(design @ new_beta, y)
            halvings += 1
        if not np.isfinite(new_dev):
            return _LrmFit(np.full(p + 1, np.nan), (null_dev, np.nan), True)
        converged = abs(dev - new_dev) < eps
        beta, dev = new_beta, new_dev
    else:
        if not converged:
            return _LrmFit(beta, (null_dev, dev), True)
    return _LrmFit(beta, (null_dev, dev), False)


def _logistic_index(xb: np.ndarray, y: np.ndarray, fit: _LrmFit | None) -> np.ndarray:
    """Dxy, R2, Intercept, Slope, D, U, Q, B, g, gp of a logistic linear predictor.

    Definitions from Harrell (2015, RMS 2nd ed., sec. 4.10, 5.3, 10.9),
    Harrell, Lee & Mark (1996, sec. 5) and Steyerberg (2019, ch. 15):
    Nagelkerke R2 ``(1 - exp(-LR/n)) / (1 - exp(-L0/n))``; calibration
    intercept and slope from the logistic regression of ``y`` on ``xb``;
    discrimination ``D = (LR - 1)/n``; unreliability ``U = (LR_cal - 2)/n``
    where ``LR_cal`` tests intercept 0 and slope 1; ``Q = D - U``; Brier score;
    Gini's mean difference ``g`` of the linear predictor and ``gp`` of the
    probabilities.

    With ``fit`` (apparent or training rows) the calibration line is the
    identity and ``LR`` is the model's. Without it (frozen predictions on the
    test rows) ``LR`` is that of the recalibration fit of ``y`` on ``xb``, and
    ``g``, ``gp`` use the recalibrated predictor.
    """
    xb = np.asarray(xb, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    _, dxy = _somers2(xb, y)
    brier = float(np.mean((special.expit(xb) - y) ** 2))
    if fit is not None:
        null_dev, model_dev = fit.deviance
        intercept, slope = 0.0, 1.0
        lr = null_dev - model_dev
        lr_cal = 0.0
    else:
        cal = _lrm_fit(xb[:, None], y)
        if cal.fail:
            out = np.full(10, np.nan)
            out[0], out[7] = dxy, brier
            return out
        null_dev, model_dev = cal.deviance
        intercept, slope = (float(v) for v in cal.coefficients)
        lr = null_dev - model_dev
        lr_cal = _binary_deviance(xb, y) - model_dev
    r2 = (1.0 - np.exp(-lr / n)) / (1.0 - np.exp(-null_dev / n))
    d = (lr - 1.0) / n
    u = (lr_cal - 2.0) / n
    lp = intercept + slope * xb
    return np.array(
        [
            dxy,
            r2,
            intercept,
            slope,
            d,
            u,
            d - u,
            brier,
            gini_mean_difference(lp),
            gini_mean_difference(special.expit(lp)),
        ]
    )


# ---------------------------------------------------------------------------
# Cox fit, stopped as cph documents
# ---------------------------------------------------------------------------


@dataclass
class _CoxFit:
    coefficients: np.ndarray
    loglik: tuple[float, float]  # at the initial value and at the estimate
    fail: bool


def _cox_layout(x, time, event):
    from ..surv.cox import _ordinary_layout

    n = time.size
    return _ordinary_layout(
        np.ascontiguousarray(x, dtype=float),
        np.asarray(time, dtype=float),
        np.asarray(event, dtype=bool),
        np.ones(n),
        np.zeros(n),
        np.zeros(n, dtype=np.int64),
    )


def _cox_newton_step(grad: np.ndarray, hess: np.ndarray, tol: float = 1e-9) -> np.ndarray | None:
    """``I^{-1} U`` with ``I = -hess``; ``None`` when ``I`` is not positive definite."""
    info = -hess
    scale = np.sqrt(np.diag(info)) if np.all(np.diag(info) > 0.0) else None
    if scale is None or not np.all(np.isfinite(info)):
        return None
    scaled = info / np.outer(scale, scale)
    if np.min(np.linalg.eigvalsh(scaled)) < tol:
        return None
    return np.linalg.solve(scaled, grad / scale) / scale


def _cox_fit(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    efron: bool = True,
    iter_max: int = 10,
    eps: float = 1e-4,
    init: np.ndarray | None = None,
) -> _CoxFit:
    """Cox model on right-censored data without strata, stopped where ``cph`` stops.

    Newton–Raphson on the Efron (or Breslow) log partial likelihood
    (Kalbfleisch and Prentice 2002, sec. 4.2), with the partial likelihood,
    score and information from ``statract.surv.cox``. The iteration starts
    at ``init`` (zeros by default) and stops at the first estimate whose log
    partial likelihood changes by less than ``eps`` relative to the previous
    one, the rule ``coxph.control`` documents. ``cph``'s defaults are
    ``eps = 1e-4`` and ``iter.max = 10``. A step that lowers the log partial
    likelihood is halved. ``iter_max = 0`` only scores ``init``.

    ``fail`` is set when the information matrix is singular, the likelihood
    is not finite, or ``iter_max`` steps do not converge.
    """
    from ..surv.cox import _score_layout

    p = x.shape[1]
    layout = _cox_layout(x, time, event)
    beta = np.zeros(p) if init is None else np.asarray(init, dtype=float).copy()
    ll, grad, hess = _score_layout(beta, layout, efron)
    ll0 = float(ll)
    if not np.isfinite(ll):
        return _CoxFit(np.full(p, np.nan), (ll0, np.nan), True)
    if iter_max <= 0:
        return _CoxFit(beta, (ll0, ll0), False)
    for _ in range(iter_max):
        step = _cox_newton_step(grad, hess)
        if step is None:
            return _CoxFit(np.full(p, np.nan), (ll0, np.nan), True)
        new_beta = beta + step
        new_ll, new_grad, new_hess = _score_layout(new_beta, layout, efron)
        halvings = 0
        while not (np.isfinite(new_ll) and new_ll >= ll) and halvings < 30:
            new_beta = (new_beta + beta) / 2.0
            new_ll, new_grad, new_hess = _score_layout(new_beta, layout, efron)
            halvings += 1
        if not np.isfinite(new_ll):
            return _CoxFit(np.full(p, np.nan), (ll0, np.nan), True)
        converged = abs(1.0 - ll / new_ll) < eps if new_ll != 0.0 else ll == new_ll
        beta, ll, grad, hess = new_beta, new_ll, new_grad, new_hess
        if converged:
            return _CoxFit(beta, (ll0, float(ll)), False)
    return _CoxFit(beta, (ll0, float(ll)), True)


def _cox_index(xb, time, event, fit: _CoxFit | None, efron: bool = True) -> np.ndarray:
    """Dxy, R2, Slope, D, U, Q, g of a Cox linear predictor.

    Harrell (2015, RMS 2nd ed., sec. 20.9) and Harrell, Lee & Mark (1996,
    sec. 5). With ``L0 = -2 log`` partial likelihood at ``beta = 0`` and the
    model likelihood ratio ``LR``:

    - Dxy: Somers' Dxy, higher ``xb`` meaning an earlier event;
    - R2: Nagelkerke ``(1 - exp(-LR/n)) / (1 - exp(-L0/n))``, ``n`` subjects;
    - Slope: 1, or the coefficient of the Cox model on ``xb`` alone (test rows);
    - D = (LR - 1) / L0 and U = (LR_cal - k) / L0, Q = D - U, where ``LR_cal``
      compares the refitted slope with slope 1. The null partial-likelihood
      deviance ``L0`` replaces ``n`` as the scale for censored data. The
      penalty ``k`` is 2 for the apparent and training rows (where
      ``LR_cal = 0``) and 0 for the test rows; this asymmetry is what the rms
      output shows (checked against the fixture to 1e-10);
    - g: Gini's mean difference of ``slope * xb``.

    A constant ``xb`` carries no information: the refit is skipped, the slope
    is NaN and R2 = 0, D = -1 / L0, U = 0, Q = D and g = 0 are reported, the slope is
    NaN and Dxy is 0 (every usable pair is a risk tie).
    """
    xb = np.asarray(xb, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    n = xb.size
    dxy = _dxy_cens(xb, time, event)
    if fit is not None:
        ll0, ll = fit.loglik
        slope = 1.0
        lr_cal = 0.0
        penalty = 2.0
    else:
        penalty = 0.0
        column = xb[:, None]
        at_one = _cox_fit(column, time, event, efron=efron, iter_max=0, init=np.ones(1))
        if np.ptp(xb) == 0.0:
            ll0 = at_one.loglik[0]
            null_dev = -2.0 * ll0
            out = np.full(7, np.nan)
            out[0] = dxy
            out[1] = 0.0
            out[3] = -1.0 / null_dev
            out[4] = 0.0
            out[5] = out[3]
            out[6] = 0.0
            return out
        refit = _cox_fit(column, time, event, efron=efron)
        if refit.fail or not np.all(np.isfinite(refit.coefficients)):
            raise RuntimeError("the Cox recalibration fit on the test sample failed")
        ll0, ll = refit.loglik
        slope = float(refit.coefficients[0])
        lr_cal = 2.0 * (ll - at_one.loglik[0])
    null_dev = -2.0 * ll0
    lr = 2.0 * (ll - ll0)
    r2 = (1.0 - np.exp(-lr / n)) / (1.0 - np.exp(-null_dev / n))
    d = (lr - 1.0) / null_dev
    u = (lr_cal - penalty) / null_dev
    return np.array([dxy, r2, slope, d, u, d - u, gini_mean_difference(slope * xb)])


# ---------------------------------------------------------------------------
# Resampling (predab.resample, method = "boot")
# ---------------------------------------------------------------------------


def _resamples(n: int, B: int, seed, indices) -> np.ndarray:
    if indices is not None:
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[1] != n:
            raise ValueError(f"indices must have shape (B, {n})")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("indices must be integers")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError("indices must be 0-based row numbers of the complete cases")
        return idx.astype(np.int64)
    if B < 1:
        raise ValueError("B must be at least 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def _accumulate(rows, index_orig: np.ndarray):
    """Means over successful repetitions with the per-index NA rule of predab.resample."""
    k = index_orig.size
    train_sum = np.zeros(k)
    test_sum = np.zeros(k)
    num = np.zeros(k)
    for train, test in rows:
        na = np.isnan(train + test)
        num += ~na
        train_sum += np.where(na, 0.0, train)
        test_sum += np.where(na, 0.0, test)
    with np.errstate(invalid="ignore", divide="ignore"):
        training = train_sum / num
        test = test_sum / num
    optimism = training - test
    return training, test, optimism, num


def _validation_table(names, index_orig, training, test, optimism, num) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "index": names,
            "index_orig": index_orig,
            "training": training,
            "test": test,
            "optimism": optimism,
            "index_corrected": index_orig - optimism,
            "n": num,
        }
    )


def _logistic_data(data, outcome, predictors):
    fit = fit_glm(data, outcome, predictors, family="binomial")
    y = np.asarray(fit.y, dtype=float)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("the outcome must be binary (0/1 or boolean)")
    names = list(fit.names)
    if "(Intercept)" not in names:
        raise ValueError("the model needs an intercept")
    keep = [j for j, nm in enumerate(names) if nm != "(Intercept)"]
    if not keep:
        raise ValueError("the model needs at least one predictor")
    x = np.asarray(fit.x, dtype=float)[:, keep]
    return x, y, [names[j] for j in keep]


def _emax(intercept: float, slope: float) -> float:
    """Largest |p - expit(a + b logit p)| for the bias-corrected calibration line.

    Emax of Harrell (2015, RMS 2nd ed., sec. 10.9; ``validate.lrm`` help):
    the largest absolute difference between a predicted probability ``p``
    and its calibrated value. The maximum is taken over the grid
    ``p = 0, 0.0005, ..., 1`` (2001 points), which is the resolution that
    reproduces the rms output; the exact maximum differs by up to about
    5e-7 relative. At ``p = 0`` and ``p = 1`` the calibrated value equals
    ``p`` when the slope is positive.
    """
    a, b = float(intercept), float(slope)
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    p = np.arange(2001) * 0.0005
    with np.errstate(divide="ignore", invalid="ignore"):
        cal = special.expit(a + b * special.logit(p))
    return float(np.nanmax(np.abs(p - cal)))


def validate_logistic(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    B: int = 40,
    seed: int | np.random.Generator | None = None,
    indices: np.ndarray | None = None,
) -> pl.DataFrame:
    """Bootstrap optimism of a binary logistic model, ``rms::validate.lrm``.

    ``outcome`` is a 0/1 column with ``predictors``, or a Wilkinson formula
    such as ``"y ~ age + sex"`` as in :func:`statract.fit_glm`. Rows with a
    missing value are dropped first.

    Returns one row per index (``Dxy``, ``R2``, ``Intercept``, ``Slope``,
    ``Emax``, ``D``, ``U``, ``Q``, ``B``, ``g``, ``gp``) with ``index_orig``,
    ``training``, ``test``, ``optimism``, ``index_corrected`` and ``n``
    (repetitions used). ``indices`` is a ``B`` x n matrix of 0-based rows of the
    complete cases; without it the resamples come from ``seed``.
    """
    x, y, _ = _logistic_data(data, outcome, predictors)
    n = y.size
    orig = _lrm_fit(x, y)
    if orig.fail:
        raise RuntimeError("the logistic model did not converge on the full sample")
    xb_orig = orig.coefficients[0] + x @ orig.coefficients[1:]
    index_orig = _logistic_index(xb_orig, y, orig)
    rows = []
    for train in _resamples(n, B, seed, indices):
        f = _lrm_fit(x[train], y[train])
        if f.fail:
            continue
        xb = f.coefficients[0] + x @ f.coefficients[1:]
        rows.append((_logistic_index(xb[train], y[train], f), _logistic_index(xb, y, None)))
    training, test, optimism, num = _accumulate(rows, index_orig)
    corrected = index_orig - optimism
    emax = _emax(float(corrected[2]), float(corrected[3]))
    # Emax follows Slope; it is 0 in the apparent and training columns.
    return _validation_table(
        _LOGISTIC_INDEXES,
        np.insert(index_orig, 4, 0.0),
        np.insert(training, 4, 0.0),
        np.insert(test, 4, emax),
        np.insert(optimism, 4, emax),
        np.insert(num, 4, num[0]),
    ).with_columns(
        pl.when(pl.col("index") == "Emax").then(emax).otherwise(pl.col("index_corrected")).alias("index_corrected")
    )


# ---------------------------------------------------------------------------
# calibrate (lrm)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCurve:
    """Bootstrap calibration curve (``rms::calibrate`` for ``lrm``).

    ``table`` has ``predy`` (grid of predicted probabilities),
    ``calibrated_orig`` (apparent lowess curve), ``calibrated_corrected``
    (bias-corrected curve), ``optimism`` and ``n`` (repetitions that covered
    the grid point). The error summaries are those of ``print.calibrate``:
    the bias-corrected curve interpolated at each subject's prediction.
    """

    table: pl.DataFrame
    predicted: np.ndarray
    mean_absolute_error: float
    mean_squared_error: float
    quantile_90: float
    B: int
    n: int


def _r_seq(start: float, stop: float, length: int) -> np.ndarray:
    if length == 1:
        return np.array([start])
    if start == stop:
        return np.full(length, start)
    n1 = length - 1
    out = start + np.arange(length) * ((stop - start) / n1)
    out[-1] = stop
    return out


def _cal_curve(xb: np.ndarray, y: np.ndarray, predy: np.ndarray) -> np.ndarray:
    sx, sy = lowess(special.expit(xb), y, iter=0)
    return interpolate(sx, sy, predy, ties="first")


def calibrate_logistic(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    B: int = 40,
    seed: int | np.random.Generator | None = None,
    indices: np.ndarray | None = None,
    predy: np.ndarray | None = None,
) -> CalibrationCurve:
    """Bootstrap overfitting-corrected calibration curve of a logistic model.

    Same as ``rms::calibrate`` on ``lrm`` with the default lowess smoother
    (``lowess(p, y, iter = 0)``). The default grid ``predy`` is 50 equally spaced
    points from the 5th smallest to the 5th largest predicted probability.
    ``indices`` and ``seed`` work as in :func:`validate_logistic`.
    """
    x, y, _ = _logistic_data(data, outcome, predictors)
    n = y.size
    orig = _lrm_fit(x, y)
    if orig.fail:
        raise RuntimeError("the logistic model did not converge on the full sample")
    xb_orig = orig.coefficients[0] + x @ orig.coefficients[1:]
    predicted = special.expit(xb_orig)
    if predy is None:
        if n < 11:
            raise ValueError("need n > 10 when predy is not given")
        p = np.sort(predicted)
        predy = _r_seq(p[4], p[n - 5], 50)
    predy = np.asarray(predy, dtype=float)
    orig_cal = _cal_curve(xb_orig, y, predy)
    rows = []
    resamples = _resamples(n, B, seed, indices)
    for train in resamples:
        f = _lrm_fit(x[train], y[train], tol=1e-13)
        if f.fail:
            continue
        xb = f.coefficients[0] + x @ f.coefficients[1:]
        rows.append((_cal_curve(xb[train], y[train], predy) - predy, _cal_curve(xb, y, predy) - predy))
    _, _, optimism, num = _accumulate(rows, orig_cal)
    corrected = orig_cal - optimism
    ok = ~np.isnan(predy + corrected)
    err = predicted - interpolate(predy[ok], corrected[ok], predicted, ties="mean")
    abs_err = np.abs(err[~np.isnan(err)])
    table = pl.DataFrame(
        {
            "predy": predy,
            "calibrated_orig": orig_cal,
            "calibrated_corrected": corrected,
            "optimism": optimism,
            "n": num,
        }
    )
    return CalibrationCurve(
        table=table,
        predicted=predicted,
        mean_absolute_error=float(np.mean(abs_err)) if abs_err.size else float("nan"),
        mean_squared_error=float(np.mean(abs_err**2)) if abs_err.size else float("nan"),
        quantile_90=float(np.quantile(abs_err, 0.9)) if abs_err.size else float("nan"),
        B=int(resamples.shape[0]),
        n=int(n),
    )


def plot_calibration_curve(
    result: CalibrationCurve,
    path: str | Path,
    *,
    title: str = "Calibration",
    rug: bool = True,
) -> Path:
    """Apparent, bias-corrected and ideal lines, as ``plot.calibrate``."""
    t = result.table
    p = t["predy"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    ax.plot(p, t["calibrated_orig"].to_numpy(), linestyle=":", color="black", label="Apparent")
    ax.plot(p, t["calibrated_corrected"].to_numpy(), color="C0", label="Bias-corrected")
    if rug and result.predicted.size:
        ax.plot(
            result.predicted,
            np.zeros_like(result.predicted),
            "|",
            color="gray",
            alpha=0.4,
            markersize=8,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title(title)
    ax.text(
        0.98,
        0.02,
        f"B = {result.B}, mean |error| = {result.mean_absolute_error:.3f}, n = {result.n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# validate.cph
# ---------------------------------------------------------------------------


def _cox_data(data, time, event, predictors, ties):
    from ..surv.cox import cox_ph

    if ties not in {"efron", "breslow"}:
        raise ValueError("ties must be 'efron' or 'breslow'")
    fit = cox_ph(data, time, event, predictors, ties=ties)
    if fit.x.shape[1] == 0:
        raise ValueError("the model needs at least one predictor")
    if np.unique(fit.strata).size > 1 or np.any(fit.entry != 0) or np.any(fit.offset != 0):
        raise ValueError("strata, entry times and offsets are not supported")
    x = np.asarray(fit.x, dtype=float)
    t = np.asarray(fit.time, dtype=float)
    e = np.asarray(fit.event, dtype=float) > 0
    return x, t, e, ties == "efron"


def validate_cox(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef | None = None,
    predictors: list[ColumnRef] | None = None,
    *,
    B: int = 40,
    seed: int | np.random.Generator | None = None,
    indices: np.ndarray | None = None,
    ties: str = "efron",
) -> pl.DataFrame:
    """Bootstrap optimism of a Cox model, ``rms::validate.cph``.

    ``time`` is a column with ``event`` and ``predictors``, or a formula such as
    ``"Surv(time, status) ~ age + sex"`` as in :func:`statract.cox_ph`. Strata,
    weights, offsets and entry times are not supported.

    Returns the rows ``Dxy``, ``R2``, ``Slope``, ``D``, ``U``, ``Q`` and ``g``
    with the columns of :func:`validate_logistic`. ``Dxy`` is positive when a
    higher linear predictor goes with earlier events.
    """
    x, t, e, efron = _cox_data(data, time, event, predictors, ties)
    n = t.size
    # The full-sample fit is kept even when it reaches the iteration limit.
    orig = _cox_fit(x, t, e, efron=efron)
    index_orig = _cox_index(x @ orig.coefficients, t, e, orig)
    rows = []
    for train in _resamples(n, B, seed, indices):
        f = _cox_fit(x[train], t[train], e[train], efron=efron)
        if f.fail or np.any(np.isnan(f.coefficients)):
            continue
        xb = x @ f.coefficients
        rows.append((_cox_index(xb[train], t[train], e[train], f), _cox_index(xb, t, e, None, efron)))
    training, test, optimism, num = _accumulate(rows, index_orig)
    return _validation_table(_COX_INDEXES, index_orig, training, test, optimism, num)



# ---------------------------------------------------------------------------
# calibrate.cph, cmethod = "KM"
# ---------------------------------------------------------------------------


def _baseline_survival_at(x, time, event, beta, u, efron):
    """Survival at ``u`` for a subject at the covariate means (``cph(surv = TRUE)``).

    Same as ``survfit.cph``: exp(-cumulative hazard), with the Efron hazard for
    ``ties = "efron"`` and the Aalen (Breslow) hazard otherwise.
    """
    means = x.mean(axis=0)
    risk = np.exp((x - means) @ beta)
    cumhaz = _cumulative_hazard_at(
        np.ascontiguousarray(time, dtype=float),
        np.ascontiguousarray(event, dtype=np.int64),
        np.ascontiguousarray(risk),
        float(u) + 1e-6,
        efron,
    )
    return float(np.exp(-cumhaz)), float(means @ beta)


@njit(cache=True)
def _cumulative_hazard_at(time, event, risk, upto, efron):
    order = np.argsort(time, kind="mergesort")
    n = time.shape[0]
    # Risk sums per distinct time, then the risk-set sums from the last time back.
    starts = [0]
    for k in range(1, n):
        if time[order[k]] != time[order[k - 1]]:
            starts.append(k)
    m = len(starts)
    total = np.zeros(m)
    dsum = np.zeros(m)
    nd = np.zeros(m, dtype=np.int64)
    for g in range(m):
        end = starts[g + 1] if g + 1 < m else n
        for k in range(starts[g], end):
            r = order[k]
            total[g] += risk[r]
            if event[r] == 1:
                dsum[g] += risk[r]
                nd[g] += 1
    nrisk = np.zeros(m)
    acc = 0.0
    for g in range(m - 1, -1, -1):
        acc += total[g]
        nrisk[g] = acc
    cumhaz = 0.0
    for g in range(m):
        if time[order[starts[g]]] > upto:
            break
        d = nd[g]
        if d == 0:
            continue
        if efron and d > 1:
            for k in range(d):
                cumhaz += 1.0 / (nrisk[g] - k / d * dsum[g])
        else:
            cumhaz += d / nrisk[g]
    return cumhaz


def _r_quantile7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Sample quantiles of type 7 (Hyndman and Fan 1996), R's default: numpy's ``"linear"``."""
    return np.quantile(np.asarray(x, dtype=float), probs, method="linear")


def _km_at(time: np.ndarray, event: np.ndarray, u: float) -> tuple[float, float]:
    """Kaplan–Meier S(u) and the Greenwood standard error of log S(u).

    Kaplan & Meier (1958): S(u) = prod over event times t <= u of
    (1 - d_t / r_t). Greenwood's formula gives
    Var(log S(u)) = sum d_t / (r_t (r_t - d_t)). When ``u`` is after the
    group's last follow-up time S(u) is not estimable and both are NaN.
    """
    if time.size == 0 or u > time.max():
        return float("nan"), float("nan")
    event_times = np.unique(time[event & (time <= u)])
    if event_times.size == 0:
        return 1.0, 0.0
    at_risk = np.searchsorted(np.sort(time), event_times, side="left")
    at_risk = time.size - at_risk
    deaths = np.array([np.sum(event & (time == t)) for t in event_times], dtype=float)
    surv = float(np.prod(1.0 - deaths / at_risk))
    with np.errstate(divide="ignore", invalid="ignore"):
        var_log = float(np.sum(deaths / (at_risk * (at_risk - deaths))))
    return surv, float(np.sqrt(var_log))


def _cut_groups(pred: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Interval number of each prediction for the cut points ``cuts``.

    Intervals are ``[c_k, c_{k+1})`` with the last one closed, and the outer
    cut points are moved out to the range of the data (``Hmisc::cut2`` help
    page).
    """
    edges = np.asarray(cuts, dtype=float).copy()
    edges[0] = min(edges[0], float(pred.min()))
    edges[-1] = max(edges[-1], float(pred.max()))
    group = np.searchsorted(edges, pred, side="right") - 1
    return np.clip(group, 0, edges.size - 2)


def _groupkm(pred: np.ndarray, time: np.ndarray, event: np.ndarray, u: float, cuts: np.ndarray) -> np.ndarray:
    """Kaplan–Meier survival at ``u`` within groups of the prediction (``rms::groupkm``).

    Columns: mean prediction, number of subjects, number of events,
    KM S(u) and the Greenwood standard error of log S(u). One row per
    interval of ``cuts``, so that bootstrap samples line up with the full
    sample; an interval without members has n = 0 and NaN elsewhere, which
    the bootstrap summary then leaves out for that group.
    """
    pred = np.asarray(pred, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    group = _cut_groups(pred, cuts)
    n_groups = max(np.asarray(cuts).size - 1, 1)
    out = np.full((n_groups, 5), np.nan)
    for k in range(n_groups):
        member = group == k
        out[k, 1] = float(member.sum())
        if not member.any():
            continue
        surv, se = _km_at(time[member], event[member], u)
        out[k] = [pred[member].mean(), float(member.sum()), float(event[member].sum()), surv, se]
    return out


def _calibration_gap(
    pred_surv: np.ndarray, time: np.ndarray, event: np.ndarray, u: float, cuts: np.ndarray, what: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Observed minus predicted survival at ``u`` per group (Harrell 2015, sec. 20.6).

    Returns ``(dist, table)`` with ``table`` from :func:`_groupkm` and
    ``dist`` the Kaplan–Meier estimate minus the mean prediction
    (``what = "observed-predicted"``) or the Kaplan–Meier estimate itself.
    ``None`` when the sample has fewer than 5 events, too few to estimate a
    survival curve in each group.
    """
    event = np.asarray(event, dtype=bool)
    if int(event.sum()) < _MIN_CALIBRATION_EVENTS:
        return None
    tab = _groupkm(pred_surv, time, event, u, cuts)
    dist = tab[:, 3] - tab[:, 0] if what == "observed-predicted" else tab[:, 3].copy()
    return dist, tab


@dataclass
class SurvivalCalibration:
    """Bootstrap calibration of predicted survival at ``u`` (``calibrate.cph``, KM).

    ``table`` has one row per group of predicted survival: ``mean_predicted``,
    ``KM`` (Kaplan-Meier at ``u``), ``KM_corrected`` (KM minus the optimism),
    ``std_err`` (of log KM), and the bootstrap columns ``index_orig``,
    ``training``, ``test``, ``mean_optimism``, ``mean_corrected`` and ``n``.
    ``predicted`` is each subject's predicted survival at ``u``.
    """

    table: pl.DataFrame
    predicted: np.ndarray
    cuts: np.ndarray
    u: float
    B: int
    n: int


def calibrate_cox(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef | None = None,
    predictors: list[ColumnRef] | None = None,
    *,
    u: float,
    m: int = 150,
    cuts: np.ndarray | None = None,
    B: int = 40,
    seed: int | np.random.Generator | None = None,
    indices: np.ndarray | None = None,
    ties: str = "efron",
    what: str = "observed-predicted",
) -> SurvivalCalibration:
    """Bootstrap calibration of a Cox model at time ``u``, ``rms::calibrate.cph``
    with ``cmethod = "KM"``.

    Subjects are grouped by predicted survival at ``u`` (about ``m`` per group,
    or at ``cuts``). In each group the Kaplan-Meier estimate at ``u`` is compared
    with the mean prediction; the bootstrap estimates the optimism of
    ``KM - predicted`` (``what = "observed-predicted"``) or of ``KM``
    (``what = "observed"``). Same as calling ``cph(..., surv = TRUE, time.inc = u)``
    before ``calibrate``. ``indices`` and ``seed`` work as in
    :func:`validate_logistic`.
    """
    if what not in {"observed-predicted", "observed"}:
        raise ValueError("what must be 'observed-predicted' or 'observed'")
    x, t, e, efron = _cox_data(data, time, event, predictors, ties)
    n = t.size
    if u > t.max():
        raise ValueError("u is after the last follow-up time")
    orig = _cox_fit(x, t, e, efron=efron)
    s0, center = _baseline_survival_at(x, t, e, orig.coefficients, u, efron)
    predicted = s0 ** np.exp(x @ orig.coefficients - center)
    if cuts is None:
        g = max(1, n // m)
        cuts = np.unique(_r_quantile7(np.concatenate([[0.0, 1.0], predicted]), _r_seq(0.0, 1.0, g + 1)))
    cuts = np.asarray(cuts, dtype=float)

    first = _calibration_gap(s0 ** np.exp(x @ orig.coefficients - center), t, e, u, cuts, what)
    if first is None:
        raise ValueError("too few events in the sample to calibrate")
    index_orig, pred_obs = first
    rows = []
    resamples = _resamples(n, B, seed, indices)
    for train in resamples:
        tt, ee = t[train], e[train]
        if u > tt.max():
            raise ValueError("a bootstrap sample ends before u")
        f = _cox_fit(x[train], tt, ee, efron=efron)
        if np.any(np.isnan(f.coefficients)):
            continue
        fs0, fcenter = _baseline_survival_at(x[train], tt, ee, f.coefficients, u, efron)
        xb = x @ f.coefficients
        tr = _calibration_gap(fs0 ** np.exp(xb[train] - fcenter), tt, ee, u, cuts, what)
        if tr is None:
            continue
        te = _calibration_gap(fs0 ** np.exp(xb - fcenter), t, e, u, cuts, what)
        rows.append((tr[0], np.full(index_orig.size, np.nan) if te is None else te[0]))
    training, test, optimism, num = _accumulate(rows, index_orig)
    table = pl.DataFrame(
        {
            "index_orig": index_orig,
            "training": training,
            "test": test,
            "mean_optimism": optimism,
            "mean_corrected": index_orig - optimism,
            "n": num,
            "mean_predicted": pred_obs[:, 0],
            "KM": pred_obs[:, 3],
            "KM_corrected": pred_obs[:, 3] - optimism,
            "std_err": pred_obs[:, 4],
        }
    )
    return SurvivalCalibration(table=table, predicted=predicted, cuts=cuts, u=float(u), B=int(resamples.shape[0]), n=int(n))
