"""Bootstrap validation and calibration of prediction models, after ``rms``.

``validate_logistic`` follows ``rms::validate.lrm``, ``calibrate_logistic``
follows ``rms::calibrate`` for an ``lrm`` fit, ``validate_cox`` follows
``rms::validate.cph`` and ``calibrate_cox`` follows ``rms::calibrate.cph`` with
``cmethod = "KM"``. All use the ordinary bootstrap of
``predab.resample``: each repetition fits the model on a resample and scores it
on that resample (training) and on the whole sample (test). The optimism is the
mean training index minus the mean test index.

The fits follow the R fitters step for step, because rms stops them early:
``lrm.fit`` stops once ``-2 log L`` changes by less than 0.025, and ``cph``
uses ``eps = 1e-4`` on the relative change of the partial likelihood. The
coefficients can therefore differ from the fully converged ``fit_glm`` and
``cox_ph`` (up to about 1e-4 relative in small samples), and the indexes
match rms instead.

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

from . import _mpl as _mpl  # noqa: F401
from ._lowess_r import approx_r, lowess_r
from .design import ColumnRef
from .fit import fit_glm

import matplotlib.pyplot as plt

_LOGISTIC_INDEXES = ["Dxy", "R2", "Intercept", "Slope", "Emax", "D", "U", "Q", "B", "g", "gp"]
_COX_INDEXES = ["Dxy", "R2", "Slope", "D", "U", "Q", "g"]


# ---------------------------------------------------------------------------
# Small helpers shared with Hmisc
# ---------------------------------------------------------------------------


def gini_mean_difference(x: np.ndarray) -> float:
    """Gini's mean difference, ``Hmisc::GiniMd``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    w = 4.0 * (np.arange(1, n + 1) - (n - 1) / 2.0) / n / (n - 1)
    return float(np.sum(w * np.sort(x - x.mean())))


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
    # survConcordance.fit counts for right-censored data. A pair is usable when
    # the shorter time is an event; a censored time tied with an event time
    # counts as longer, and two events at one time are a time tie (not used).
    # Times are walked from the last; a Fenwick tree over risk ranks holds the
    # subjects still at risk.
    n = time.shape[0]
    order = np.argsort(-time, kind="mergesort")
    tree = np.zeros(n_rank + 1)
    count = np.zeros(n_rank)
    conc = 0.0
    disc = 0.0
    tied = 0.0
    inserted = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and time[order[j]] == time[order[i]]:
            j += 1
        for k in range(i, j):
            r = order[k]
            if event[r] == 0:
                pos = risk_rank[r] + 1
                while pos <= n_rank:
                    tree[pos] += 1.0
                    pos += pos & (-pos)
                count[risk_rank[r]] += 1.0
                inserted += 1.0
        for k in range(i, j):
            r = order[k]
            if event[r] == 1:
                below = 0.0
                pos = risk_rank[r]
                while pos > 0:
                    below += tree[pos]
                    pos -= pos & (-pos)
                same = count[risk_rank[r]]
                conc += below
                tied += same
                disc += inserted - below - same
        for k in range(i, j):
            r = order[k]
            if event[r] == 1:
                pos = risk_rank[r] + 1
                while pos <= n_rank:
                    tree[pos] += 1.0
                    pos += pos & (-pos)
                count[risk_rank[r]] += 1.0
                inserted += 1.0
        i = j
    return conc, disc, tied


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
# lrm.fit (binary), step for step
# ---------------------------------------------------------------------------


@dataclass
class _LrmFit:
    coefficients: np.ndarray
    deviance: tuple[float, float]  # -2 log L of the intercept-only model and of the fit
    fail: bool

    @property
    def model_lr(self) -> float:
        return self.deviance[0] - self.deviance[1]


def _lrm_llogit(beta: np.ndarray, xd: np.ndarray, y: np.ndarray):
    bx = xd @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(bx, -30.0, 30.0)))
    if np.any((y == 0) & (p >= 1.0)) or np.any((y == 1) & (p <= 0.0)):
        return np.nan, None, None, True
    ll = float(np.sum(np.where(y == 1, np.log(p), np.log1p(-p))))
    u = xd.T @ (y - p)
    v = (xd * (p * (1.0 - p))[:, None]).T @ xd
    return -2.0 * ll, u, v, False


def _lrm_fit(x: np.ndarray, y: np.ndarray, *, maxit: int = 12, eps: float = 0.025, tol: float = 1e-7) -> _LrmFit:
    """``rms::lrm.fit`` for a 0/1 outcome (Fortran ``lrmfit``)."""
    n = y.size
    n1 = float(np.sum(y))
    counts = np.array([n - n1, n1])
    null_dev = float(-2.0 * np.sum(counts * np.log(counts / n))) if np.all(counts > 0) else 0.0
    if n1 == 0 or n1 == n:
        return _LrmFit(np.full(1 + x.shape[1], np.nan), (null_dev, null_dev), True)
    xd = np.column_stack([np.ones(n), x])
    nvi = xd.shape[1]
    beta = np.zeros(nvi)
    beta[0] = np.log(n1 / n / (1.0 - n1 / n))
    if nvi == 1:
        return _LrmFit(beta, (null_dev, null_dev), False)
    oldl = 1e30
    deltab = np.zeros(nvi)
    curstp = 1.0
    converged = False
    for it in range(1, maxit):
        ll, u, v, dvrg = _lrm_llogit(beta, xd, y)
        if not dvrg:
            dmax = float(np.max(np.abs(u)))
            if dmax < 1e-9 and abs(ll - oldl) < 0.1 * eps:
                converged = True
                break
        if dvrg or ll > oldl:
            if it == 1:
                return _LrmFit(beta, (null_dev, np.nan), True)
            curstp /= 2.0
            beta = beta - curstp * deltab
            continue
        curstp = 1.0
        if np.linalg.matrix_rank(v, tol=tol * np.max(np.abs(v))) < nvi:
            return _LrmFit(beta, (null_dev, np.nan), True)
        deltab = np.linalg.solve(v, u)
        beta = beta + deltab
        if abs(oldl - ll) <= eps:
            converged = True
            break
        oldl = ll
    if not converged and maxit > 2:
        return _LrmFit(beta, (null_dev, np.nan), True)
    ll, _, _, dvrg = _lrm_llogit(beta, xd, y)
    return _LrmFit(beta, (null_dev, ll), bool(dvrg))


def _lrm_stats(fit: _LrmFit, lp: np.ndarray) -> dict[str, float]:
    n = lp.size
    lr = fit.model_lr
    r2 = (1.0 - np.exp(-lr / n)) / (1.0 - np.exp(-fit.deviance[0] / n))
    return {
        "lr": lr,
        "R2": float(r2),
        "g": gini_mean_difference(lp),
        "gp": gini_mean_difference(special.expit(lp)),
    }


def _logistic_index(xb: np.ndarray, y: np.ndarray, fit: _LrmFit | None) -> np.ndarray:
    """``discrim`` of validate.lrm. ``fit`` is the fit on these rows (training
    or original sample); ``None`` scores frozen predictions (test sample)."""
    n = xb.size
    _, dxy = _somers2(xb, y)
    if fit is not None:
        st = _lrm_stats(fit, xb)
        intercept, shrink = 0.0, 1.0
        d = (st["lr"] - 1.0) / n
        u = -2.0 / n
        r2, g, gp = st["R2"], st["g"], st["gp"]
    else:
        refit = _lrm_fit(xb[:, None], y, tol=1e-13)
        intercept, shrink = float(refit.coefficients[0]), float(refit.coefficients[1])
        st = _lrm_stats(refit, intercept + shrink * xb)
        d = (st["lr"] - 1.0) / n
        l01 = -2.0 * np.sum(y * xb - np.log1p(np.exp(xb)))
        u = (l01 - refit.deviance[1] - 2.0) / n
        r2 = st["R2"]
        g = gini_mean_difference(shrink * xb)
        gp = gini_mean_difference(special.expit(intercept + shrink * xb))
    brier = float(np.sum((y - special.expit(xb)) ** 2) / n)
    return np.array([dxy, r2, intercept, shrink, d, u, d - u, brier, g, gp])


# ---------------------------------------------------------------------------
# coxph.fit (coxfit6), step for step
# ---------------------------------------------------------------------------


@dataclass
class _CoxFit:
    coefficients: np.ndarray
    loglik: tuple[float, float]  # at the initial value and at the estimate
    fail: bool


def _cox_layout(x, time, event):
    from .surv.cox import _ordinary_layout

    n = time.size
    return _ordinary_layout(
        np.ascontiguousarray(x, dtype=float),
        np.asarray(time, dtype=float),
        np.asarray(event, dtype=bool),
        np.ones(n),
        np.zeros(n),
        np.zeros(n, dtype=np.int64),
    )


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
    """``rms:::coxphFit`` on right-censored data without strata."""
    from .surv.cox import _score_layout

    layout = _cox_layout(x, time, event)
    p = x.shape[1]
    beta = np.zeros(p) if init is None else np.asarray(init, dtype=float).copy()

    def score(b):
        ll, grad, hess = _score_layout(b, layout, efron)
        return float(ll), grad, -hess

    ll0, u, imat = score(beta)
    if iter_max == 0:
        return _CoxFit(beta, (ll0, ll0), False)
    try:
        step = np.linalg.solve(imat, u)
    except np.linalg.LinAlgError:
        return _CoxFit(beta, (ll0, np.nan), True)
    newbeta = beta + step
    best = ll0
    halving = False
    iters = 0
    newlk = ll0
    converged = False
    for it in range(1, iter_max + 1):
        iters = it
        newlk, u, imat = score(newbeta)
        if np.isfinite(newlk) and abs(1.0 - best / newlk) <= eps and not halving:
            converged = True
            break
        if it == iter_max:
            break
        if not np.isfinite(newlk) or newlk < best:
            halving = True
            newbeta = (newbeta + beta) / 2.0
        else:
            halving = False
            best = newlk
            try:
                step = np.linalg.solve(imat, u)
            except np.linalg.LinAlgError:
                return _CoxFit(newbeta, (ll0, np.nan), True)
            beta = newbeta
            newbeta = newbeta + step
    fail = (not converged) or (iter_max > 1 and iters >= iter_max)
    return _CoxFit(newbeta, (ll0, newlk), bool(fail))


def _cox_index(xb, time, event, fit: _CoxFit | None, efron: bool = True) -> np.ndarray:
    """``discrim`` of validate.cph without strata."""
    n = xb.size
    if np.unique(xb).size == 1:
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    if fit is not None:
        lr = -2.0 * (fit.loglik[0] - fit.loglik[1])
        ll0 = -2.0 * fit.loglik[0]
        slope = 1.0
        d = (lr - 1.0) / ll0
        u = -2.0 / ll0
        r2 = (1.0 - np.exp(-lr / n)) / (1.0 - np.exp(-ll0 / n))
        g = gini_mean_difference(xb)
    else:
        col = xb[:, None]
        f = _cox_fit(col, time, event, efron=efron)
        if f.fail:
            raise RuntimeError("Cox refit of the test sample failed (rms: fit failure in discrim)")
        lr = -2.0 * (f.loglik[0] - f.loglik[1])
        ll0 = -2.0 * f.loglik[0]
        slope = float(f.coefficients[0])
        d = (lr - 1.0) / ll0
        r2 = (1.0 - np.exp(-lr / n)) / (1.0 - np.exp(-ll0 / n))
        frozen = _cox_fit(col, time, event, efron=efron, iter_max=0, init=np.array([1.0]))
        u = -2.0 * (frozen.loglik[1] - f.loglik[1]) / ll0
        g = gini_mean_difference(slope * xb)
    dxy = _dxy_cens(xb, time, event)
    return np.array([dxy, r2, slope, d, u, d - u, g])


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
    p = np.arange(2001) * 0.0005
    with np.errstate(divide="ignore", invalid="ignore"):
        logit = np.log(p / (1.0 - p))
        emax_p = special.expit(corrected[2] + corrected[3] * logit)
    emax = float(np.nanmax(np.abs(p - emax_p)))
    # validate.lrm adds Emax after Slope: 0 for the apparent and training columns.
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
    sx, sy = lowess_r(special.expit(xb), y, iter=0)
    return approx_r(sx, sy, predy, ties="first")


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
    err = predicted - approx_r(predy[ok], corrected[ok], predicted, ties="mean")
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
    from .surv.cox import cox_ph

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
    # cph keeps a fit that ran out of iterations; so does this.
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
    # Risk sums per distinct time, then sums from the last time back (rcumsum).
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
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    index = 1.0 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = x[lo - 1]
    h = index - lo
    use = (index > lo) & (x[hi - 1] != qs)
    out = qs.copy()
    out[use] = (1.0 - h[use]) * qs[use] + h[use] * x[hi[use] - 1]
    return out


def _cut2_codes(x: np.ndarray, cuts: np.ndarray) -> tuple[np.ndarray, int]:
    """Group codes (0-based, -1 for none) of ``Hmisc::cut2(x, cuts)``."""
    xu = np.unique(np.concatenate([x[~np.isnan(x)], cuts]))
    min_dif = np.min(np.diff(xu)) / 2.0 if xu.size > 1 else 0.0
    cuts = np.asarray(cuts, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if lo < cuts[0]:
        cuts = np.concatenate([[lo], cuts])
    if hi > cuts.max():
        cuts = np.concatenate([cuts, [hi]])
    k2 = cuts - min_dif
    k2[-1] = cuts[-1]
    # cut(x, k2): right-closed intervals (k2[i], k2[i + 1]].
    codes = np.searchsorted(k2, x, side="left") - 1
    codes[(x <= k2[0]) | (x > k2[-1]) | np.isnan(x)] = -1
    return codes, k2.size - 1


def _km_at(time: np.ndarray, event: np.ndarray, u: float) -> tuple[float, float]:
    """Kaplan-Meier survival and Greenwood SE of log survival at ``u`` (survfitKM)."""
    times, inverse = np.unique(time, return_inverse=True)
    at = np.bincount(inverse, minlength=times.size).astype(float)
    dead = np.bincount(inverse, weights=event.astype(float), minlength=times.size)
    nrisk = np.cumsum(at[::-1])[::-1]
    keep = (times <= u + 1e-6) & (dead > 0)
    d, r = dead[keep], nrisk[keep]
    s = float(np.prod(1.0 - d / r))
    with np.errstate(divide="ignore"):
        var = float(np.sum(np.where(r > d, d / (r * (r - d)), np.inf)))
    if u > times[-1] + 1e-6 and s > 0:
        return float("nan"), float("nan")
    return s, float(np.sqrt(var))


def _groupkm(pred: np.ndarray, time: np.ndarray, event: np.ndarray, u: float, cuts: np.ndarray) -> np.ndarray:
    """``rms::groupkm`` columns x, n, events, KM, std.err for the given cuts."""
    pred = np.where(np.abs(pred) < 1e-10, 0.0, pred)
    codes, g = _cut2_codes(pred, cuts)
    out = np.full((g, 5), np.nan)
    for i in range(g):
        s = codes == i
        nobs = int(s.sum())
        if nobs < 2:
            out[i] = [pred[s].mean() if nobs == 1 else np.nan, 0, 0, np.nan, np.nan]
            continue
        km, se = _km_at(time[s], event[s], u)
        out[i] = [pred[s].mean(), nobs, float(event[s].sum()), km, se]
    return out


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

    def distance(xb, tt, ee, fit_s0, fit_center):
        # distance() of calibrate.cph: grouped KM against the mean prediction.
        if ee.sum() < 5:
            return None
        cox = fit_s0 ** np.exp(xb - fit_center)
        tab = _groupkm(cox, tt, ee, u, cuts)
        dist = tab[:, 3] if what == "observed" else tab[:, 3] - tab[:, 0]
        return dist, tab

    first = distance(x @ orig.coefficients, t, e, s0, center)
    if first is None:
        raise ValueError("fewer than 5 events")
    index_orig, pred_obs = first
    rows = []
    resamples = _resamples(n, B, seed, indices)
    for train in resamples:
        tt, ee = t[train], e[train]
        if ee.sum() < 5:
            continue
        if u > tt.max():
            raise ValueError("a bootstrap sample ends before u (rms: subscript out of bounds)")
        f = _cox_fit(x[train], tt, ee, efron=efron)
        if np.any(np.isnan(f.coefficients)):
            continue
        fs0, fcenter = _baseline_survival_at(x[train], tt, ee, f.coefficients, u, efron)
        xb = x @ f.coefficients
        tr = distance(xb[train], tt, ee, fs0, fcenter)
        te = distance(xb, t, e, fs0, fcenter)
        nan = np.full(index_orig.size, np.nan)
        rows.append((nan if tr is None else tr[0], nan if te is None else te[0]))
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
