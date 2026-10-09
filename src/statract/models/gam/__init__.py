"""Penalized additive models with the cubic-regression basis used by mgcv.

``method="reml"`` profiles the Gaussian REML score. The reported smoothing
parameter multiplies the scaled penalty, and coefficients are in the
sum-to-zero constrained space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit
from scipy import stats
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

from ..design import ColumnRef, column_series


_BASES = {"cr", "tp", "cc", "ps", "re", "te", "ti"}


@dataclass(frozen=True)
class Smooth:
    column: str
    k: int = 10
    basis: str = "cr"
    by: str | None = None
    extra: tuple[str, ...] = ()
    extra_k: tuple[int, ...] = ()


def smooth(column: str, *, k: int = 10, basis: str = "cr", by: str | None = None) -> Smooth:
    """A univariate smooth.

    ``basis`` is ``cr`` (cubic regression), ``tp`` (thin plate), ``cc``
    (cyclic cubic), ``ps`` (P-spline), or ``re`` (random effect). ``by`` is a
    factor, which fits one smooth per level, or a numeric column, which
    multiplies the basis.
    """
    if basis not in {"cr", "tp", "cc", "ps", "re"}:
        raise ValueError("basis must be cr, tp, cc, ps, or re")
    if basis != "re" and k < 3:
        raise ValueError("k must be at least 3")
    return Smooth(column=column, k=k, basis=basis, by=by)


def tensor_smooth(*columns: str, k: tuple[int, ...] = (8, 8)) -> Smooth:
    """A tensor-product smooth. Each margin is a cubic regression spline."""
    return _tensor(columns, k, "te")


def tensor_interaction(*columns: str, k: tuple[int, ...] = (8, 8)) -> Smooth:
    """A tensor interaction. Main effects of each margin are removed, as in ``ti``."""
    return _tensor(columns, k, "ti")


def _tensor(columns: tuple[str, ...], k: tuple[int, ...], basis: str) -> Smooth:
    if len(columns) < 2:
        raise ValueError("a tensor smooth needs at least two columns")
    if len(k) != len(columns):
        raise ValueError("k must have one value per column")
    if any(value < 3 for value in k):
        raise ValueError("k must be at least 3")
    return Smooth(column=columns[0], k=k[0], basis=basis, extra=tuple(columns[1:]), extra_k=tuple(k[1:]))


@dataclass
class GamFit:
    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    smoothing_parameter: float
    edf: float
    sig2: float
    reml: float
    x: np.ndarray
    y: np.ndarray
    penalty: np.ndarray
    family: str
    converged: bool

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
        return pl.DataFrame(
            {
                "term": ["s"],
                "edf": [self.edf],
                "smoothing_parameter": [self.smoothing_parameter],
                "reml": [self.reml],
            }
        )

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        if kind not in {"link", "response"}:
            raise ValueError("kind must be link or response")
        if data is None:
            eta = self.x @ self.coefficients
        else:
            built = _model_matrix(data, self._smooth, self._knots, self._q2, self._column)
            eta = built @ self.coefficients
        return eta


def compare_gams(*fits):
    """AIC and the fitting criterion for two or more additive models."""
    from .general import compare_gams as _compare_gams

    return _compare_gams(*fits)


def gam(
    data: pl.DataFrame,
    outcome: ColumnRef,
    smooths: list[Smooth],
    *,
    predictors: list[str] | None = None,
    family: str = "gaussian",
    method: str = "reml",
    select: bool = False,
    gamma: float = 1.0,
    weights: str | None = None,
    offset: str | None = None,
) -> GamFit:
    """Fit a penalized additive model.

    The single cubic-regression Gaussian model uses the REML path checked
    against ``mgcv``. Other bases (``tp``, ``cc``, ``ps``, ``re``), tensor
    products and interactions, ``by``, binomial, Poisson, gamma, weights,
    offsets, and ``ml`` / ``gcv`` use the penalized score. A tensor or a
    factor ``by`` has one smoothing parameter per margin or level.
    ``weights`` and ``offset`` are column names.
    """
    simple = (
        family == "gaussian"
        and method == "reml"
        and not select
        and gamma == 1.0
        and not predictors
        and weights is None
        and offset is None
        and len(smooths) == 1
        and smooths[0].basis == "cr"
        and smooths[0].by is None
        and not smooths[0].extra
    )
    if not simple:
        from .general import fit_general

        return fit_general(
            data,
            outcome,
            smooths,
            predictors=predictors,
            family=family,
            method=method,
            select=select,
            gamma=gamma,
            weights=weights,
            offset=offset,
        )
    spec = smooths[0]
    y = np.asarray(column_series(data, outcome).to_numpy(), dtype=float)
    x_raw = np.asarray(column_series(data, spec.column).to_numpy(), dtype=float)
    knots = np.quantile(np.unique(x_raw), np.linspace(0, 1, spec.k))
    basis, penalty = _cr_basis(x_raw, knots)
    constrained, q2, penalty_c = _absorb_constraint(basis, penalty)
    design = np.column_stack([np.ones(len(y)), constrained])
    # The penalty pads a zero for the intercept.
    pen = np.zeros((design.shape[1], design.shape[1]))
    pen[1:, 1:] = penalty_c
    mp = 2  # intercept plus the linear null space of one cr smooth

    def reml(log_sp: float) -> float:
        return _reml(np.exp(log_sp), design, y, pen, mp)[0]

    opt = minimize_scalar(reml, bounds=(-8, 12), method="bounded", options={"xatol": 1e-8})
    sp = float(np.exp(opt.x))
    score, beta, sig2, edf = _reml(sp, design, y, pen, mp)
    xtx = design.T @ design
    cov = sig2 * np.linalg.pinv(xtx + sp * pen)
    names = ["(Intercept)", *[f"{spec.column}[{i}]" for i in range(constrained.shape[1])]]
    fit = GamFit(
        coefficients=beta,
        covariance=cov,
        names=names,
        n_obs=len(y),
        smoothing_parameter=sp,
        edf=edf,
        sig2=sig2,
        reml=score,
        x=design,
        y=y,
        penalty=pen,
        family=family,
        converged=bool(opt.success),
    )
    fit._smooth = spec
    fit._knots = knots
    fit._q2 = q2
    fit._column = spec.column
    return fit


def _reml(sp: float, design: np.ndarray, y: np.ndarray, penalty: np.ndarray, mp: int):
    n, p = design.shape
    xtx = design.T @ design
    sys = xtx + sp * penalty
    beta = solve(sys, design.T @ y, assume_a="pos")
    fitted = design @ beta
    dev = float(np.sum((y - fitted) ** 2))
    pen = float(beta @ (sp * penalty) @ beta)
    phi = (dev + pen) / (n - mp)
    sign_p, logdet_p = np.linalg.slogdet(sys)
    eig = np.linalg.eigvalsh(sp * penalty)
    positive = eig[eig > eig.max() * 1e-10]
    logdet_s = float(np.sum(np.log(positive)))
    score = (n - mp) / 2 * (1 + np.log(2 * np.pi * phi)) + 0.5 * (logdet_p - logdet_s)
    edf = float(np.trace(solve(sys, xtx, assume_a="pos")))
    sig2 = dev / (n - edf)
    return float(score), beta, float(sig2), edf


def _model_matrix(data, spec: Smooth, knots: np.ndarray, q2: np.ndarray, column: str) -> np.ndarray:
    x_raw = np.asarray(column_series(data, column).to_numpy(), dtype=float)
    basis, _ = _cr_basis(x_raw, knots, penalty=False)
    return np.column_stack([np.ones(len(x_raw)), basis @ q2])


def _absorb_constraint(basis: np.ndarray, penalty: np.ndarray):
    center = basis.mean(axis=0)
    scale = np.linalg.norm(penalty, ord=1) / np.linalg.norm(basis, ord=np.inf) ** 2
    penalty = penalty / scale
    # QR of the centering constraint. The null space of the row vector is Q[:, 1:].
    q, _ = np.linalg.qr(center.reshape(-1, 1), mode="complete")
    q2 = q[:, 1:]
    return basis @ q2, q2, q2.T @ penalty @ q2


def _cr_basis(x: np.ndarray, knots: np.ndarray, *, penalty: bool = True):
    f_map, pen = _cr_penalty(knots)
    n = len(x)
    k = len(knots)
    out = np.zeros((n, k))
    _cr_basis_rows(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(knots, dtype=np.float64),
        np.ascontiguousarray(f_map, dtype=np.float64),
        out,
    )
    return out, pen


@njit(cache=True)
def _cr_basis_rows(x, knots, f_map, out):
    """Cubic-regression basis at each x. ``out`` is zeros on entry."""
    n = x.shape[0]
    k = knots.shape[0]
    kmin = knots[0]
    kmax = knots[-1]
    for i in range(n):
        xi = x[i]
        if xi < kmin or xi > kmax:
            if xi < kmin:
                h = knots[1] - kmin
                xik = xi - kmin
                cjm = -xik * h / 3.0
                cjp = -xik * h / 6.0
                for col in range(k):
                    out[i, col] = cjm * f_map[col, 0] + cjp * f_map[col, 1]
                out[i, 0] += 1.0 - xik / h
                out[i, 1] += xik / h
            else:
                j = k - 1
                h = kmax - knots[j - 1]
                xik = xi - kmax
                cjm = xik * h / 6.0
                cjp = xik * h / 3.0
                for col in range(k):
                    out[i, col] = cjm * f_map[col, j - 1] + cjp * f_map[col, j]
                out[i, k - 2] += -xik / h
                out[i, k - 1] += 1.0 + xik / h
            continue
        lo = 0
        hi = k
        while lo < hi:
            mid = (lo + hi) // 2
            if knots[mid] <= xi:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j < 0:
            j = 0
        if j > k - 2:
            j = k - 2
        h = knots[j + 1] - knots[j]
        left = knots[j + 1] - xi
        right = xi - knots[j]
        ajm = left / h
        ajp = right / h
        cjm = (left * ((left * left) / h - h)) / 6.0
        cjp = (right * ((right * right) / h - h)) / 6.0
        for col in range(k):
            out[i, col] = cjm * f_map[col, j] + cjp * f_map[col, j + 1]
        out[i, j] += ajm
        out[i, j + 1] += ajp


def _cr_penalty(knots: np.ndarray):
    """Map knot values to second derivatives and the integrated-square penalty."""
    h = np.diff(knots)
    n = len(knots)
    n2 = n - 2
    d = np.zeros((n2, n))
    for i in range(n2):
        d[i, i] = 1 / h[i]
        d[i, i + 1] = -1 / h[i] - 1 / h[i + 1]
        d[i, i + 2] = 1 / h[i + 1]
    b = np.zeros((n2, n2))
    for i in range(n2):
        b[i, i] = (h[i] + h[i + 1]) / 3
        if i + 1 < n2:
            b[i, i + 1] = b[i + 1, i] = h[i + 1] / 6
    g = solve(b, d, assume_a="pos")
    f_map = np.zeros((n, n))
    f_map[:, 1:-1] = g.T
    penalty = d.T @ g
    penalty = 0.5 * (penalty + penalty.T)
    return f_map, penalty
