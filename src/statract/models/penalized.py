"""Penalized regression: ridge, lasso, and elastic net paths, and their cross-validation.

The objective is the one documented for R's ``glmnet``: the negative log
likelihood with the weights scaled to sum to one, plus
``lambda * sum(v_j * ((1 - alpha) / 2 * b_j**2 + alpha * |b_j|))``, where the
penalty factors ``v_j`` are rescaled to sum to the number of columns. Columns
are standardized to unit weighted variance by default and the penalty applies
on that scale; for the gaussian family the response is also scaled to unit
variance, as glmnet's documentation describes. Coefficients are returned on
the original scale and the intercept is not penalized.

The solver was written for statract from the published algorithms:

* Friedman, J., Hastie, T., Tibshirani, R. (2010). Regularization paths for
  generalized linear models via coordinate descent. Journal of Statistical
  Software 33(1). Cyclic coordinate descent with soft thresholding, warm
  starts along a decreasing lambda sequence, active-set sweeps, and a
  penalized weighted least-squares (IRLS) inner problem for the binomial and
  Poisson families.
* Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011). Regularization
  paths for Cox's proportional hazards model via coordinate descent. Journal
  of Statistical Software 39(5). The Breslow partial likelihood is handled
  the same way, with the diagonal of its Hessian in the linear predictor as
  the working weights.
* Tibshirani, R., Bien, J., Friedman, J., Hastie, T., Simon, N., Taylor, J.,
  Tibshirani, R. J. (2012). Strong rules for discarding predictors in
  lasso-type problems. Journal of the Royal Statistical Society B 74(2). The
  sequential strong rule screens columns, and every screened-out column is
  checked against the Karush-Kuhn-Tucker conditions before a solution is
  accepted.

The user-facing conventions (the automatic lambda sequence from
``lambda_max`` down to ``lambda_min_ratio * lambda_max``, the early end of
the path, ``penalty_factor``, ``thresh``, linear interpolation in ``coef``
and ``predict``, and ``cv.glmnet``'s ``cvm``, ``cvsd``, ``lambda.min`` and
``lambda.1se``) follow glmnet's help pages. No glmnet code was used; glmnet
4.1-8 is used only to check the output (``tests/r_oracle``). Each subproblem
is solved more tightly than glmnet's default ``thresh``, so results agree
with glmnet to its own convergence error (about 1e-4 relative on the
coefficients), not bit for bit.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import _penalized_cd as cd
from .design import ColumnRef, Design, build_design, column_series
from .formula import model_matrix

FAMILIES = ("gaussian", "binomial", "poisson", "cox")
_FAMILY_CODE = {"gaussian": cd.GAUSSIAN, "binomial": cd.BINOMIAL, "poisson": cd.POISSON, "cox": cd.COX}
MEASURES = ("deviance", "mse", "mae", "class", "auc")

# Path controls documented in glmnet.control().
_FDEV = 1e-5  # minimum change in the deviance ratio before the path stops
_DEVMAX = 0.999  # the path stops once this fraction of the deviance is explained
_MNLAM = 5  # minimum number of lambdas before the path may stop
_ALPHA_FLOOR = 1e-3  # lambda_max uses max(alpha, 1e-3), so ridge has a finite start
_PROB_CV = 1e-5  # probabilities are clipped here for the binomial CV deviance
# Each coordinate-descent problem stops when the largest ``col_scale * delta**2`` is
# below ``thresh * _TIGHT`` times the null deviance per unit weight, a much
# tighter stop than glmnet's ``thresh`` alone.
_TIGHT = 1e-4


# ---------------------------------------------------------------------------
# Data


@dataclass
class _Data:
    """Model inputs after the formula, aligned to ``design.row_index``."""

    x: np.ndarray  # n x p, no intercept column
    y: np.ndarray  # response, or the event indicator for Cox
    w: np.ndarray  # prior weights as given
    offset: np.ndarray | None
    time: np.ndarray | None
    pen_factor: np.ndarray  # penalty factors as given (inf excludes)
    intercept: bool

    def subset(self, rows: np.ndarray) -> _Data:
        return _Data(
            x=self.x[rows],
            y=self.y[rows],
            w=self.w[rows],
            offset=None if self.offset is None else self.offset[rows],
            time=None if self.time is None else self.time[rows],
            pen_factor=self.pen_factor,
            intercept=self.intercept,
        )


def _prepare(
    data: pl.DataFrame,
    formula: str,
    family: str,
    weights: ColumnRef | None,
    offset: ColumnRef | None,
    penalty_factor: Mapping[str, float] | Sequence[float] | None,
) -> tuple[_Data, Design, list[str], bool, bool]:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    built = model_matrix(formula, data)
    if built.random_effects:
        raise ValueError("random effects are not supported in penalized_regression")
    if built.strata or built.cluster is not None:
        raise ValueError("strata() and cluster() are not supported in penalized_regression")
    design = built.design
    rows = design.row_index
    if family == "cox":
        if built.surv is None:
            raise ValueError('family="cox" needs a Surv(time, status) response')
        if built.surv.entry is not None:
            raise ValueError("Surv(start, stop, status) is not supported; use Surv(time, status)")
    elif built.surv is not None:
        raise ValueError(f'a Surv() response needs family="cox", not {family!r}')

    keep = np.ones(rows.size, dtype=bool)
    w = np.ones(rows.size)
    if weights is not None:
        series = column_series(data, weights)
        keep &= series.is_not_null().to_numpy()[rows]
        w = np.asarray(series.fill_null(0).to_numpy(), dtype=float)[rows]
    off_col = None
    if offset is not None:
        series = column_series(data, offset)
        keep &= series.is_not_null().to_numpy()[rows]
        off_col = np.asarray(series.fill_null(0).to_numpy(), dtype=float)[rows]
    time = None
    if family == "cox":
        series = column_series(data, built.surv.time)
        time = np.asarray(series.to_numpy(), dtype=float)[rows]

    has_formula_offset = built.offset is not None
    total_offset = None
    if has_formula_offset or off_col is not None:
        total_offset = np.zeros(rows.size)
        if has_formula_offset:
            total_offset = total_offset + built.offset
        if off_col is not None:
            total_offset = total_offset + off_col

    x_full = design.x
    names = list(design.names)
    has_int_col = bool(names) and names[0] == "(Intercept)"
    if has_int_col:
        x_full = x_full[:, 1:]
        names = names[1:]
    intercept = has_int_col and family != "cox"
    y = np.asarray(built.y, dtype=float)

    if not np.all(keep):
        x_full, y, w = x_full[keep], y[keep], w[keep]
        time = None if time is None else time[keep]
        total_offset = None if total_offset is None else total_offset[keep]
        design = Design(
            x=design.x[keep],
            names=design.names,
            row_index=rows[keep],
            predictors=design.predictors,
            intercept=design.intercept,
            recipes=design.recipes,
            factor_levels=design.factor_levels,
            computed=design.computed,
        )
    if not names:
        raise ValueError("penalized_regression needs at least one column besides the intercept")
    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and not all zero")
    if family == "binomial" and not np.all((y == 0) | (y == 1)):
        raise ValueError('family="binomial" needs a 0/1 response')
    if family == "poisson" and np.any(y < 0):
        raise ValueError('family="poisson" needs a non-negative response')
    if family == "cox":
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("the Surv() status must be 0/1")
        if np.any(time <= 0):
            raise ValueError("Surv() times must be positive")

    pen_factor = _penalty_vector(penalty_factor, names)
    data_ = _Data(x=np.ascontiguousarray(x_full), y=y, w=w, offset=total_offset, time=time, pen_factor=pen_factor, intercept=intercept)
    return data_, design, names, has_formula_offset, off_col is not None


def _penalty_vector(penalty_factor, names: list[str]) -> np.ndarray:
    p = len(names)
    if penalty_factor is None:
        return np.ones(p)
    if isinstance(penalty_factor, Mapping):
        pen_factor = np.ones(p)
        for key, value in penalty_factor.items():
            if key not in names:
                raise ValueError(f"penalty_factor column {key!r} is not in the design {names}")
            pen_factor[names.index(key)] = float(value)
    else:
        pen_factor = np.asarray(list(penalty_factor), dtype=float)
        if pen_factor.shape != (p,):
            raise ValueError(f"penalty_factor needs {p} values, one per design column {names}")
    if np.any(np.isnan(pen_factor)) or np.any(pen_factor < 0):
        raise ValueError("penalty_factor values must be non-negative")
    return pen_factor


# ---------------------------------------------------------------------------
# Solver


class _Problem:
    """One family's standardized problem; ``solve`` walks a lambda sequence."""

    def __init__(self, d: _Data, family: str, alpha: float, standardize: bool, thresh: float, max_iter: int):
        self.family = family
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        x, n = d.x, d.x.shape[0]
        self.w_total = float(d.w.sum())
        w = d.w / self.w_total
        self.w = w
        self.y = d.y
        self.off = np.zeros(n) if d.offset is None else np.asarray(d.offset, dtype=float)
        self.fit_int = d.intercept
        p = x.shape[1]
        self.p = p

        mean = w @ x
        # Cox is invariant to a shift of the linear predictor, so its columns
        # are centered too; without an intercept they are not.
        centered = self.fit_int or family == "cox"
        m = mean if centered else np.zeros(p)
        var = w @ (x - mean) ** 2
        if standardize:
            s = np.sqrt(var) if centered else np.sqrt(np.maximum(w @ x**2 - mean**2, 0.0))
        else:
            s = np.ones(p)
        if centered:
            usable = var > 1e-14 * np.maximum(1.0, w @ x**2)
        else:
            usable = w @ x**2 > 0
        usable &= np.isfinite(d.pen_factor) & (s > 0)
        self.cols = np.flatnonzero(usable)
        self.m = m[self.cols]
        self.s = s[self.cols]
        pen_factor = d.pen_factor[self.cols].astype(float)
        if pen_factor.size and pen_factor.sum() > 0:
            pen_factor = pen_factor * pen_factor.size / pen_factor.sum()
        self.pen_factor = pen_factor
        self.k = self.cols.size
        self.z = np.ascontiguousarray((x[:, self.cols] - self.m) / self.s) if self.k else np.zeros((n, 0))

        if family == "gaussian":
            yt = d.y - self.off
            if self.fit_int:
                y_mean = w @ yt
                y_scale = math.sqrt(max(w @ (yt - y_mean) ** 2, 0.0))
            else:
                y_scale = math.sqrt(max(w @ yt**2, 0.0))
            if y_scale == 0.0:
                y_scale = 1.0
            self.y_scale = y_scale
            self.target = yt / y_scale
        else:
            self.y_scale = 1.0
        if family == "cox":
            order = np.argsort(d.time, kind="stable")
            t = d.time[order]
            breaks = np.flatnonzero(np.diff(t) != 0) + 1
            self.order = order.astype(np.int64)
            self.starts = np.concatenate([[0], breaks]).astype(np.int64)
            self.ends = np.concatenate([breaks, [n]]).astype(np.int64)
            self.status = d.y.astype(float)
            # The weights enter the deviance unnormalized, as glmnet reports it.
            self.w_raw = d.w.astype(float)
            ev = np.zeros(self.starts.size)
            for g in range(self.starts.size):
                idx = order[self.starts[g] : self.ends[g]]
                ev[g] = float(np.sum(d.w[idx] * self.status[idx]))
            ev = ev[ev > 0]
            self.cox_saturated = float(-np.sum(ev * np.log(ev)))

        self.null_dev_std = self._null_deviance_std()
        self.null_deviance = self._to_raw_deviance(self.null_dev_std)
        self.tol = thresh * _TIGHT * max(self.null_dev_std, 1e-300)
        self.gam = np.zeros(self.k)
        self.b0 = np.zeros(1)
        self.grad = np.zeros(self.k)
        self.passes = 0
        self._empty_f = np.zeros(0)
        self._empty_i = np.zeros(0, dtype=np.int64)

    # -- likelihood pieces in the standardized problem (weights sum to one)

    def _eta(self, b0: float, gam: np.ndarray) -> np.ndarray:
        nz = np.flatnonzero(gam)
        eta = self.off + b0 if self.family != "gaussian" else np.full(self.z.shape[0], b0)
        if nz.size:
            eta = eta + self.z[:, nz] @ gam[nz]
        return eta

    def _std_deviance(self, eta: np.ndarray) -> float:
        """Deviance per unit weight (gaussian: on the scaled response)."""
        w, y = self.w, self.y
        if self.family == "gaussian":
            r = self.target - eta
            return float(w @ (r * r))
        if self.family == "binomial":
            return 2.0 * float(w @ (np.logaddexp(0.0, eta) - y * eta))
        if self.family == "poisson":
            mu = np.exp(eta)
            ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            return 2.0 * float(w @ (ylog - (y - mu)))
        ll = cd.cox_loglik(eta, self.w_raw, self.status, self.order, self.starts, self.ends)
        return 2.0 * (self.cox_saturated - ll) / self.w_total

    def _to_raw_deviance(self, dev_std: float) -> float:
        return dev_std * self.w_total * (self.y_scale**2 if self.family == "gaussian" else 1.0)

    def _null_deviance_std(self) -> float:
        w, y, off = self.w, self.y, self.off
        if self.family == "gaussian":
            t = self.target
            return float(w @ (t - (w @ t)) ** 2) if self.fit_int else float(w @ t**2)
        if self.family == "cox":
            return self._std_deviance(off)
        if not self.fit_int:
            return self._std_deviance(off)
        # Intercept-only model with the offset, by Newton's method.
        if self.family == "binomial":
            ybar = float(w @ y)
            if ybar <= 0.0 or ybar >= 1.0:
                return 0.0
            c = math.log(ybar / (1 - ybar)) - float(w @ off)
            for _ in range(100):
                pr = 1.0 / (1.0 + np.exp(-(off + c)))
                step = float(w @ (y - pr)) / max(float(w @ (pr * (1 - pr))), 1e-300)
                c += step
                if abs(step) < 1e-13:
                    break
        else:
            c = math.log(float(w @ y) / float(w @ np.exp(off)))
        return self._std_deviance(off + c)

    # -- one lambda

    def _kernel_args(self):
        if self.family == "gaussian":
            return self.target, np.zeros_like(self.target)
        return self.y, self.off

    def solve(self, lam: float, strong: np.ndarray, *, kkt: bool = True) -> np.ndarray:
        """Solve at ``lam`` on ``strong`` plus any KKT violators; returns the final set.

        ``kkt=False`` solves on ``strong`` only (the null fit). ``self.grad``
        receives the score of every column at the solution.
        """
        in_set = np.zeros(self.k, dtype=bool)
        in_set[strong] = True
        l1 = lam * self.alpha * self.pen_factor
        l2 = lam * (1.0 - self.alpha) * self.pen_factor
        y, off = self._kernel_args()
        if self.family == "cox":
            cox = (self.status, self.order, self.starts, self.ends)
        else:
            cox = (self._empty_f, self._empty_i, self._empty_i, self._empty_i)
        self.passes += cd.solve_lambda(
            _FAMILY_CODE[self.family], self.z, y, self.w, off, self.gam, self.b0, self.fit_int, in_set,
            l1, l2, self.tol, max(self.max_iter - self.passes, 1), kkt, *cox, self.grad,
        )
        return np.flatnonzero(in_set)


@dataclass
class _PathResult:
    lambdas: np.ndarray
    intercepts: np.ndarray
    coefs: np.ndarray
    df: np.ndarray
    dev_ratio: np.ndarray
    null_deviance: float
    n_passes: int


def _fit_path(
    d: _Data,
    family: str,
    alpha: float,
    lambdas: Sequence[float] | None,
    n_lambda: int,
    lambda_min_ratio: float | None,
    standardize: bool,
    thresh: float,
    max_iter: int,
) -> _PathResult:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    n, p = d.x.shape
    pr = _Problem(d, family, alpha, standardize, thresh, max_iter)
    unpen = np.flatnonzero(pr.pen_factor == 0).astype(np.int64)
    # Null model: intercept and unpenalized columns only.
    pr.solve(0.0, unpen, kkt=False)
    grad = pr.grad
    pen = pr.pen_factor > 0
    if pen.any():
        lam_max = float(np.max(np.abs(grad[pen]) / pr.pen_factor[pen])) / max(alpha, _ALPHA_FLOOR)
    else:
        lam_max = 0.0
    auto = lambdas is None
    if auto:
        if n_lambda < 1:
            raise ValueError("n_lambda must be at least 1")
        ratio = lambda_min_ratio if lambda_min_ratio is not None else (0.01 if n < p else 1e-4)
        if not 0 < ratio < 1:
            raise ValueError("lambda_min_ratio must be in (0, 1)")
        if n_lambda == 1:
            seq = np.array([lam_max])
        else:
            seq = lam_max * ratio ** (np.arange(n_lambda) / (n_lambda - 1))
    else:
        user = np.asarray(list(lambdas), dtype=float)
        if user.size == 0 or np.any(user < 0) or not np.all(np.isfinite(user)):
            raise ValueError("lambdas must be non-negative and finite")
        seq = np.sort(user)[::-1] / pr.y_scale

    intercept_list, beta_list, dr_list, df_list, lam_list = [], [], [], [], []
    ever = np.zeros(pr.k, dtype=bool)
    ever[unpen] = True
    prev = lam_max
    for step, lam in enumerate(seq):
        strong_cut = alpha * pr.pen_factor * (2.0 * lam - prev)
        strong = ever | (np.abs(pr.grad) >= strong_cut)
        if not (auto and step == 0):
            # The automatic path starts at the null model, the exact solution
            # at lambda_max (and glmnet's report there for alpha < 1e-3 too).
            idx = pr.solve(float(lam), np.flatnonzero(strong).astype(np.int64))
            ever[idx[pr.gam[idx] != 0]] = True
        prev = lam
        beta = np.zeros(p)
        beta[pr.cols] = pr.gam * pr.y_scale / pr.s
        if pr.fit_int:
            intercept_value = pr.y_scale * pr.b0[0] - float(beta[pr.cols] @ pr.m)
        else:
            intercept_value = 0.0
        eta = pr._eta(pr.b0[0], pr.gam)
        dev_std = pr._std_deviance(eta) if family != "gaussian" else float(pr.w @ (pr.target - eta) ** 2)
        dr = 1.0 - dev_std / pr.null_dev_std if pr.null_dev_std > 0 else 0.0
        intercept_list.append(intercept_value)
        beta_list.append(beta)
        dr_list.append(dr)
        df_list.append(int(np.count_nonzero(beta)))
        lam_list.append(lam * pr.y_scale)
        if pr.passes >= max_iter:
            warnings.warn(
                f"coordinate descent reached max_iter={max_iter} passes; the path stops at lambda {lam * pr.y_scale:.4g}",
                RuntimeWarning,
                stacklevel=3,
            )
            break
        stalled = step + 1 >= _MNLAM and dr_list[-1] - dr_list[-2] < _FDEV * dr_list[-1]
        if auto and (stalled or dr_list[-1] > _DEVMAX):
            break
    return _PathResult(
        lambdas=np.asarray(lam_list),
        intercepts=np.asarray(intercept_list) if d.intercept else np.zeros(len(intercept_list)),
        coefs=np.column_stack(beta_list) if beta_list else np.zeros((p, 0)),
        df=np.asarray(df_list, dtype=np.int64),
        dev_ratio=np.asarray(dr_list),
        null_deviance=pr.null_deviance,
        n_passes=pr.passes,
    )



# ---------------------------------------------------------------------------
# Interpolation along the path


def _interp_weights(path_lambdas: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left index, right index, and left weight for linear interpolation in lambda.

    ``path_lambdas`` decrease. Values outside the path are clamped to its ends.
    """
    lam = np.asarray(path_lambdas, dtype=float)
    s = np.asarray(s, dtype=float)
    if lam.size == 1:
        zero = np.zeros(s.size, dtype=np.int64)
        return zero, zero, np.ones(s.size)
    sc = np.clip(s, lam[-1], lam[0])
    rev = lam[::-1]  # increasing
    pos = np.clip(np.searchsorted(rev, sc, side="left"), 1, lam.size - 1)
    right = lam.size - pos  # the smaller lambda, in decreasing order
    left = right - 1
    span = lam[left] - lam[right]
    safe = np.where(span > 0, span, 1.0)
    frac = np.where(span > 0, (sc - lam[right]) / safe, 1.0)
    return left.astype(np.int64), right.astype(np.int64), frac


def _interp(matrix: np.ndarray, path_lambdas: np.ndarray, s: np.ndarray) -> np.ndarray:
    left, right, frac = _interp_weights(path_lambdas, s)
    return matrix[:, left] * frac + matrix[:, right] * (1.0 - frac)


def _lambda_arg(lambda_) -> tuple[np.ndarray | None, bool]:
    if lambda_ is None:
        return None, False
    if np.ndim(lambda_) == 0:
        value = float(lambda_)
        if not np.isfinite(value) or value < 0:
            raise ValueError("lambda_ must be non-negative")
        return np.array([value]), True
    arr = np.asarray(list(lambda_), dtype=float)
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("lambda_ must be non-negative")
    return arr, False


def _link_to(family: str, eta: np.ndarray, kind: str) -> np.ndarray:
    if kind == "link":
        return eta
    if kind == "class":
        return (eta > 0).astype(np.int64)
    if family == "gaussian":
        return eta
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    return np.exp(eta)


# ---------------------------------------------------------------------------
# Results


@dataclass
class PenalizedPath:
    """A ridge, lasso, or elastic-net lambda path.

    ``lambdas`` decrease. ``coefs`` is ``p`` by ``len(lambdas)`` on the
    original scale with ``names`` as rows. ``intercepts`` are zero for Cox and
    for formulas without an intercept. ``df`` counts the nonzero coefficients
    and ``dev_ratio`` is the fraction of the null deviance explained.
    """

    family: str
    alpha: float
    names: list[str]
    lambdas: np.ndarray
    intercepts: np.ndarray
    coefs: np.ndarray
    df: np.ndarray
    dev_ratio: np.ndarray
    null_deviance: float
    n_passes: int
    intercept: bool
    standardize: bool
    n_obs: int
    design: Design
    has_offset: bool = False
    offset_column: ColumnRef | None = None
    _data: _Data | None = field(default=None, repr=False)
    _formula_offset: bool = field(default=False, repr=False)

    @property
    def coef_names(self) -> list[str]:
        return (["(Intercept)"] if self.intercept else []) + list(self.names)

    def _full(self) -> np.ndarray:
        if self.intercept:
            return np.vstack([self.intercepts[None, :], self.coefs])
        return self.coefs

    def coef(self, lambda_: float | Sequence[float] | None = None) -> np.ndarray:
        """Coefficients at ``lambda_`` (``coef.glmnet`` with ``exact=FALSE``).

        Values between path lambdas are interpolated linearly in lambda, and
        values outside the path are clamped to its ends. A scalar gives a
        vector, a sequence gives one column per lambda, and ``None`` the whole
        path. The intercept comes first unless the family is Cox or the formula
        has no intercept; ``coef_names`` lists the rows.
        """
        s, scalar = _lambda_arg(lambda_)
        full = self._full()
        if s is None:
            return full.copy()
        out = _interp(full, self.lambdas, s)
        return out[:, 0] if scalar else out

    def _new_design(self, data: pl.DataFrame, offset) -> tuple[np.ndarray, np.ndarray | None]:
        extra = [self.offset_column] if (offset is None and isinstance(self.offset_column, str)) else None
        built = build_design(data, self.design, extra=extra)
        x = built.x
        if self.design.names and self.design.names[0] == "(Intercept)":
            x = x[:, 1:]
        off = None
        if offset is not None:
            if isinstance(offset, (str, pl.Expr)):
                series = column_series(data, offset)
                off = np.asarray(series.to_numpy(), dtype=float)[built.row_index]
            else:
                off = np.asarray(list(offset), dtype=float)
                if off.size == data.height:
                    off = off[built.row_index]
                elif off.size != x.shape[0]:
                    raise ValueError("offset needs one value per row of data")
        elif self._formula_offset:
            raise ValueError("this fit has an offset() term; pass offset= (the total offset of the new rows)")
        elif self.offset_column is not None:
            series = column_series(data, self.offset_column)
            off = np.asarray(series.to_numpy(), dtype=float)[built.row_index]
        return x, off

    def predict(
        self,
        data: pl.DataFrame | None = None,
        lambda_: float | Sequence[float] | None = None,
        *,
        kind: str = "link",
        offset: ColumnRef | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Predictions at ``lambda_`` (``predict.glmnet``, ``exact=FALSE``).

        ``kind`` is ``"link"``, ``"response"`` (mean, probability, or relative
        risk ``exp(eta)`` for Cox), or ``"class"`` (binomial: 1 when the link
        is positive). ``data=None`` uses the fitted rows. With new data, rows
        with a null in a used column are dropped, as in ``build_design``. A fit
        with ``offset=`` reads the same column from ``data``; an ``offset()``
        term in the formula needs ``offset=`` here. A scalar ``lambda_`` gives
        a vector, otherwise one column per lambda.
        """
        if kind not in ("link", "response", "class"):
            raise ValueError('kind must be "link", "response", or "class"')
        if kind == "class" and self.family != "binomial":
            raise ValueError('kind="class" is only for family="binomial"')
        if data is None:
            if self._data is None:
                raise ValueError("this fit keeps no data; pass data=")
            x = self._data.x
            off = self._data.offset if offset is None else np.asarray(list(offset), dtype=float)
        else:
            x, off = self._new_design(data, offset)
        s, scalar = _lambda_arg(lambda_)
        if s is None:
            beta, intercept_value = self.coefs, self.intercepts
        else:
            beta = _interp(self.coefs, self.lambdas, s)
            intercept_value = _interp(self.intercepts[None, :], self.lambdas, s)[0]
        eta = x @ beta + (intercept_value if self.intercept else 0.0)
        if off is not None:
            eta = eta + np.asarray(off, dtype=float)[:, None]
        out = _link_to(self.family, eta, kind)
        return out[:, 0] if scalar else out

    def path(self) -> pl.DataFrame:
        """``print.glmnet``: one row per lambda with ``df`` and ``dev_ratio``."""
        return pl.DataFrame(
            {
                "step": np.arange(1, self.lambdas.size + 1),
                "lambda": self.lambdas,
                "df": self.df,
                "dev_ratio": self.dev_ratio,
            }
        )

    def tidy(self, lambda_: float | Sequence[float] | None = None) -> pl.DataFrame:
        """Long table of ``lambda``, ``term``, ``estimate`` (all terms, zeros included)."""
        s, _scalar = _lambda_arg(lambda_)
        lam = self.lambdas if s is None else s
        coef = self.coef(None if s is None else lam)
        names = self.coef_names
        return pl.DataFrame(
            {
                "lambda": np.repeat(lam, len(names)),
                "term": names * lam.size,
                "estimate": coef.T.reshape(-1),
            }
        )

    def plot(self, *, xvar: str = "log_lambda", ax=None):
        """Coefficient paths against ``log(lambda)`` (``plot.glmnet``). Returns the axes.

        ``xvar`` is ``"log_lambda"``, ``"norm"`` (L1 norm of the coefficients),
        or ``"dev"`` (fraction of deviance explained).
        """
        import matplotlib.pyplot as plt

        from .._core import mpl as _mpl  # noqa: F401

        if xvar == "log_lambda":
            x, label = np.log(self.lambdas), "log(lambda)"
        elif xvar == "norm":
            x, label = np.abs(self.coefs).sum(axis=0), "L1 norm"
        elif xvar == "dev":
            x, label = self.dev_ratio, "fraction of deviance explained"
        else:
            raise ValueError('xvar must be "log_lambda", "norm", or "dev"')
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        for j, name in enumerate(self.names):
            ax.plot(x, self.coefs[j], label=name, linewidth=1)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("coefficient")
        return ax


def penalized_regression(
    data: pl.DataFrame,
    formula: str,
    *,
    family: str = "gaussian",
    alpha: float = 1.0,
    lambdas: Sequence[float] | None = None,
    n_lambda: int = 100,
    lambda_min_ratio: float | None = None,
    standardize: bool = True,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    penalty_factor: Mapping[str, float] | Sequence[float] | None = None,
    thresh: float = 1e-7,
    max_iter: int = 100_000,
) -> PenalizedPath:
    """Fit a ridge, lasso, or elastic-net path with glmnet's objective.

    ``formula`` is ``"y ~ x1 + x2"`` (``family`` ``"gaussian"``,
    ``"binomial"`` with a 0/1 outcome, or ``"poisson"``) or
    ``"Surv(time, status) ~ x1 + x2"`` (``"cox"``, Breslow ties). Factors
    become dummy columns as in ``model.matrix``, and the intercept column is
    not penalized; ``y ~ x - 1`` fits without an intercept (glmnet's
    ``intercept=FALSE``). ``alpha=1`` is the lasso and ``alpha=0`` ridge.

    ``lambdas=None`` builds the automatic path: ``n_lambda`` values from
    ``lambda_max`` down to ``lambda_min_ratio * lambda_max`` (0.01 when there
    are fewer rows than columns, else 1e-4). The path stops early once the
    fraction of deviance explained grows by less than 1e-5 of itself between
    two lambdas (after the first five) or passes 0.999. ``penalty_factor``
    maps design column names (or lists one value per column) to relative
    penalties; ``0`` leaves a column unpenalized and ``inf`` drops it.
    ``weights`` and ``offset`` are columns.

    ``thresh`` is the convergence threshold: each coordinate-descent problem
    runs until no coefficient update changes the objective by more than
    ``thresh * 1e-4`` times the null deviance per unit weight. ``max_iter``
    caps the total number of coordinate sweeps.
    """
    if thresh <= 0:
        raise ValueError("thresh must be positive")
    d, design, names, formula_offset, column_offset = _prepare(data, formula, family, weights, offset, penalty_factor)
    res = _fit_path(d, family, alpha, lambdas, n_lambda, lambda_min_ratio, standardize, thresh, max_iter)
    return PenalizedPath(
        family=family,
        alpha=float(alpha),
        names=names,
        lambdas=res.lambdas,
        intercepts=res.intercepts,
        coefs=res.coefs,
        df=res.df,
        dev_ratio=res.dev_ratio,
        null_deviance=res.null_deviance,
        n_passes=res.n_passes,
        intercept=d.intercept,
        standardize=bool(standardize),
        n_obs=int(d.x.shape[0]),
        design=design,
        has_offset=d.offset is not None,
        offset_column=offset if column_offset else None,
        _data=d,
        _formula_offset=formula_offset,
    )


# ---------------------------------------------------------------------------
# Cross-validation


@dataclass
class PenalizedCV:
    """``cv.glmnet`` result.

    ``cvm`` is the mean cross-validated error at each of ``lambdas`` (the
    full-data path), ``cvsd`` its standard error, and ``nzero`` the number of
    nonzero coefficients. ``lambda_min`` minimizes ``cvm`` (AUC is maximized);
    ``lambda_1se`` is the largest lambda within one standard error of it.
    """

    fit: PenalizedPath
    measure: str
    lambdas: np.ndarray
    cvm: np.ndarray
    cvsd: np.ndarray
    nzero: np.ndarray
    lambda_min: float
    lambda_1se: float
    index_min: int
    index_1se: int
    foldid: np.ndarray

    @property
    def cvup(self) -> np.ndarray:
        return self.cvm + self.cvsd

    @property
    def cvlo(self) -> np.ndarray:
        return self.cvm - self.cvsd

    def _lambda(self, lambda_: float | str) -> float:
        if isinstance(lambda_, str):
            if lambda_ == "min":
                return self.lambda_min
            if lambda_ == "1se":
                return self.lambda_1se
            raise ValueError('lambda_ must be a number, "min", or "1se"')
        return float(lambda_)

    def coef(self, lambda_: float | str = "1se") -> np.ndarray:
        """Coefficients of the full-data path at ``lambda_`` (``"1se"`` by default, as ``coef.cv.glmnet``)."""
        return self.fit.coef(self._lambda(lambda_))

    def predict(
        self,
        data: pl.DataFrame | None = None,
        lambda_: float | str = "1se",
        *,
        kind: str = "link",
        offset: ColumnRef | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Predictions of the full-data path at ``lambda_`` (``predict.cv.glmnet``)."""
        return self.fit.predict(data, self._lambda(lambda_), kind=kind, offset=offset)

    def table(self) -> pl.DataFrame:
        """One row per lambda: ``lambda``, ``cvm``, ``cvsd``, ``cvup``, ``cvlo``, ``nzero``."""
        return pl.DataFrame(
            {
                "lambda": self.lambdas,
                "cvm": self.cvm,
                "cvsd": self.cvsd,
                "cvup": self.cvup,
                "cvlo": self.cvlo,
                "nzero": self.nzero,
            }
        )

    def tidy(self, lambda_: float | str = "1se") -> pl.DataFrame:
        """``lambda``, ``term``, ``estimate`` of the full-data path at ``lambda_``."""
        return self.fit.tidy(self._lambda(lambda_))

    def plot(self, *, ax=None):
        """``plot.cv.glmnet``: ``cvm`` with ``cvsd`` bars against ``log(lambda)``. Returns the axes."""
        import matplotlib.pyplot as plt

        from .._core import mpl as _mpl  # noqa: F401

        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        x = np.log(self.lambdas)
        ax.errorbar(x, self.cvm, yerr=self.cvsd, fmt="o", markersize=3, color="tab:red", ecolor="gray", elinewidth=0.8)
        ax.axvline(math.log(self.lambda_min), linestyle="--", color="gray", linewidth=0.8)
        ax.axvline(math.log(self.lambda_1se), linestyle="--", color="gray", linewidth=0.8)
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel(self.measure)
        return ax


def _fold_vector(data: pl.DataFrame, rows: np.ndarray, foldid, n_folds: int, seed: int | None) -> np.ndarray:
    n = rows.size
    if foldid is None:
        if n_folds < 3:
            raise ValueError("n_folds must be at least 3")
        rng = np.random.default_rng(seed)
        return rng.permutation(np.arange(n) % n_folds) + 1
    if isinstance(foldid, (str, pl.Expr)):
        values = np.asarray(column_series(data, foldid).to_numpy())[rows]
    else:
        values = np.asarray(list(foldid))
        if values.size == data.height:
            values = values[rows]
        elif values.size != n:
            raise ValueError("foldid needs one value per row of data")
    return np.asarray(values, dtype=np.int64)


def _auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted area under the ROC curve; ties count one half."""
    cases, controls = y == 1, y == 0
    wc, wn = w[cases], w[controls]
    if wc.sum() <= 0 or wn.sum() <= 0:
        return float("nan")
    pc, pn = p[cases], p[controls]
    order = np.argsort(pn, kind="stable")
    pn, wn_sorted = pn[order], wn[order]
    cum = np.concatenate([[0.0], np.cumsum(wn_sorted)])
    below = cum[np.searchsorted(pn, pc, side="left")]
    at_or_below = cum[np.searchsorted(pn, pc, side="right")]
    score = below + 0.5 * (at_or_below - below)
    return float(np.sum(wc * score) / (wc.sum() * wn.sum()))


def _cox_deviance(eta: np.ndarray, d: _Data) -> float:
    """Breslow deviance ``2 * (saturated - fitted)`` log partial likelihood."""
    order = np.argsort(d.time, kind="stable")
    t = d.time[order]
    breaks = np.flatnonzero(np.diff(t) != 0) + 1
    starts = np.concatenate([[0], breaks]).astype(np.int64)
    ends = np.concatenate([breaks, [t.size]]).astype(np.int64)
    status = d.y.astype(float)
    # cv.glmnet's Cox deviance scales the weights to mean one within each sample.
    w = d.w.astype(float) * (d.w.size / float(d.w.sum()))
    ll = cd.cox_loglik(np.ascontiguousarray(eta, dtype=float), w, status, order.astype(np.int64), starts, ends)
    ev = np.array([np.sum(w[order[a:b]] * status[order[a:b]]) for a, b in zip(starts, ends, strict=True)])
    ev = ev[ev > 0]
    return 2.0 * (float(-np.sum(ev * np.log(ev))) - ll)


def _cv_errors(family: str, measure: str, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Per-row error at each lambda (columns), as ``cv.glmnet`` defines each measure."""
    y = y[:, None]
    mu = _link_to(family, eta, "response")
    # glmnet codes a binomial response as two indicator columns, so the
    # squared and absolute errors count both classes.
    scale = 2.0 if family == "binomial" else 1.0
    if measure == "mse":
        return scale * (y - mu) ** 2
    if measure == "mae":
        return scale * np.abs(y - mu)
    if measure == "class":
        return (y != (eta > 0)).astype(float)
    if family == "binomial":
        pc = np.clip(mu, _PROB_CV, 1 - _PROB_CV)
        return -2.0 * (y * np.log(pc) + (1 - y) * np.log(1 - pc))
    ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (ylog - (y - mu))


def cv_penalized(
    data: pl.DataFrame,
    formula: str,
    *,
    family: str = "gaussian",
    alpha: float = 1.0,
    measure: str = "default",
    foldid: ColumnRef | Sequence[int] | None = None,
    n_folds: int = 10,
    seed: int | None = None,
    grouped: bool = True,
    lambdas: Sequence[float] | None = None,
    n_lambda: int = 100,
    lambda_min_ratio: float | None = None,
    standardize: bool = True,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    penalty_factor: Mapping[str, float] | Sequence[float] | None = None,
    thresh: float = 1e-7,
    max_iter: int = 100_000,
) -> PenalizedCV:
    """Cross-validate the lambda path, as ``cv.glmnet`` defines it.

    ``foldid`` (a column or one fold number per row, 1 to K) fixes the
    folds, as in R; without it the folds are a seeded permutation, which
    differs from R's ``sample``. Each fold refits the path on the other folds
    (with its own automatic lambda sequence unless ``lambdas`` is given) and
    is predicted at the full-data lambdas by linear interpolation.

    ``measure="default"`` is ``"mse"`` for gaussian (the same as
    ``"deviance"`` there) and ``"deviance"`` otherwise. ``"mse"`` and
    ``"mae"`` work for gaussian, binomial, and poisson; ``"class"`` and
    ``"auc"`` for binomial. Cox uses the partial-likelihood deviance with R's
    ``grouped`` rule (full-data minus training deviance per fold). With
    ``grouped=True`` the error is averaged within folds first, so ``cvsd``
    is the standard error of the fold means. The other arguments are those
    of ``penalized_regression``.
    """
    if measure == "default":
        measure = "mse" if family == "gaussian" else "deviance"
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES} or 'default'")
    if family == "gaussian" and measure == "deviance":
        measure = "mse"
    allowed = {
        "gaussian": ("mse", "mae"),
        "binomial": ("deviance", "mse", "mae", "class", "auc"),
        "poisson": ("deviance", "mse", "mae"),
        "cox": ("deviance",),
    }
    if family in allowed and measure not in allowed[family]:
        raise ValueError(f"measure {measure!r} is not available for family {family!r}")
    fit = penalized_regression(
        data,
        formula,
        family=family,
        alpha=alpha,
        lambdas=lambdas,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        standardize=standardize,
        weights=weights,
        offset=offset,
        penalty_factor=penalty_factor,
        thresh=thresh,
        max_iter=max_iter,
    )
    d = fit._data
    assert d is not None
    folds = _fold_vector(data, fit.design.row_index, foldid, n_folds, seed)
    labels = np.unique(folds)
    if labels.size < 3:
        raise ValueError("cross-validation needs at least 3 folds")
    lam = fit.lambdas
    n_lam = lam.size
    w = d.w
    by_fold = grouped or family == "cox" or measure == "auc"
    raw = np.zeros((d.x.shape[0], n_lam))
    fold_err = np.zeros((labels.size, n_lam))
    fold_w = np.zeros(labels.size)
    for k, label in enumerate(labels):
        test = folds == label
        train = d.subset(np.flatnonzero(~test))
        res = _fit_path(train, family, alpha, lambdas, n_lambda, lambda_min_ratio, standardize, thresh, max_iter)
        beta = _interp(res.coefs, res.lambdas, lam)
        intercept_value = _interp(res.intercepts[None, :], res.lambdas, lam)[0]
        eta = d.x @ beta + (intercept_value if d.intercept else 0.0)
        if d.offset is not None:
            eta = eta + d.offset[:, None]
        if family == "cox":
            # Verweij and van Houwelingen's cross-validated partial likelihood:
            # the full-data deviance minus the training-fold deviance.
            full = np.array([_cox_deviance(eta[:, j], d) for j in range(n_lam)])
            part = np.array([_cox_deviance(eta[~test, j], train) for j in range(n_lam)])
            fold_w[k] = float(np.sum(w[test] * d.y[test]))
            fold_err[k] = (full - part) / fold_w[k]
            continue
        fold_w[k] = float(w[test].sum())
        if measure == "auc":
            prob = 1.0 / (1.0 + np.exp(-eta[test]))
            fold_err[k] = [_auc(d.y[test], prob[:, j], w[test]) for j in range(n_lam)]
            continue
        err = _cv_errors(family, measure, d.y[test], eta[test])
        raw[test] = err
        fold_err[k] = (w[test] @ err) / fold_w[k]
    if by_fold:
        cvm = (fold_w @ fold_err) / fold_w.sum()
        cvsd = np.sqrt((fold_w @ (fold_err - cvm) ** 2) / fold_w.sum() / (labels.size - 1))
    else:
        cvm = (w @ raw) / w.sum()
        cvsd = np.sqrt((w @ (raw - cvm) ** 2) / w.sum() / (w.size - 1))
    score = -cvm if measure == "auc" else cvm
    best = float(np.nanmin(score))
    index_min = int(np.flatnonzero(score <= best)[0])
    index_1se = int(np.flatnonzero(score <= best + cvsd[index_min])[0])
    return PenalizedCV(
        fit=fit,
        measure=measure,
        lambdas=lam,
        cvm=cvm,
        cvsd=cvsd,
        nzero=fit.df.copy(),
        lambda_min=float(lam[index_min]),
        lambda_1se=float(lam[index_1se]),
        index_min=index_min,
        index_1se=index_1se,
        foldid=folds,
    )
