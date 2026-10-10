"""Regression spline bases for formulas: ``ns``, ``bs``, and ``rcs``.

``ns`` and ``bs`` follow R's ``splines`` package and ``rcs`` follows
``rms::rcs`` (``Hmisc::rcspline.eval`` with ``inclx=TRUE`` and ``norm=2``).
Knots come from the non-missing values of the argument when the formula is
first evaluated and are kept for prediction, as R's ``makepredictcall`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

BASES = ("ns", "bs", "rcs")


@dataclass(frozen=True)
class SplineSpec:
    """A resolved basis: fixed knots, so new rows reuse the fitted columns."""

    kind: str
    knots: tuple[float, ...]
    boundary: tuple[float, float] = (0.0, 0.0)
    degree: int = 3
    intercept: bool = False
    suffixes: tuple[str, ...] = ()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "ns":
            return _ns_basis(x, np.asarray(self.knots), self.boundary, self.intercept)
        if self.kind == "bs":
            return _bs_basis(x, np.asarray(self.knots), self.boundary, self.degree, self.intercept)
        return _rcs_basis(x, np.asarray(self.knots))


def resolve(kind: str, x: np.ndarray, args: dict[str, object], name: str) -> SplineSpec:
    """Fix the knots of ``kind`` from the non-missing values ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError(f"{kind}() has no non-missing values")
    if kind == "rcs":
        return _resolve_rcs(x, args, name)
    return _resolve_bspline(kind, x, args)


def _quantile7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R's ``quantile(type = 7)``."""
    xs = np.sort(x)
    n = xs.size
    h = (n - 1) * np.asarray(probs, dtype=float)
    lo = np.floor(h + 4 * np.finfo(float).eps).astype(int)
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = h - lo
    frac[np.abs(frac) < 4 * np.finfo(float).eps] = 0.0
    return xs[lo] + frac * (xs[hi] - xs[lo])


def _flag(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be TRUE or FALSE")


def _vector(value: object, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a numeric vector")
    return arr


def _resolve_bspline(kind: str, x: np.ndarray, args: dict[str, object]) -> SplineSpec:
    intercept = _flag(args.get("intercept", False), "intercept")
    degree = 3
    if kind == "bs":
        degree = int(_vector(args.get("degree", 3), "degree")[0])
        if degree < 1:
            raise ValueError("degree must be a positive integer")
    elif "degree" in args:
        raise ValueError("ns() has no degree argument")
    if "Boundary.knots" in args:
        boundary = np.sort(_vector(args["Boundary.knots"], "Boundary.knots"))
        if boundary.size != 2:
            raise ValueError("Boundary.knots takes two values")
    else:
        boundary = np.array([x.min(), x.max()])
    knots = args.get("knots")
    df = args.get("df")
    if knots is not None:
        interior = np.sort(_vector(knots, "knots"))
    elif df is not None:
        df_value = int(_vector(df, "df")[0])
        if kind == "ns":
            n_interior = df_value - 1 - int(intercept)
        else:
            n_interior = df_value - (degree + 1) + (1 - int(intercept))
        if n_interior < 0:
            n_interior = 0
        if n_interior > 0:
            inside = x[(x >= boundary[0]) & (x <= boundary[1])]
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            interior = _quantile7(inside, probs)
        else:
            interior = np.zeros(0)
    else:
        interior = np.zeros(0)
    width = (interior.size + (1 if kind == "ns" else degree) + int(intercept))
    return SplineSpec(
        kind=kind,
        knots=tuple(float(k) for k in interior),
        boundary=(float(boundary[0]), float(boundary[1])),
        degree=degree,
        intercept=intercept,
        suffixes=tuple(str(i) for i in range(1, width + 1)),
    )


def _all_knots(interior: np.ndarray, boundary: tuple[float, float], order: int) -> np.ndarray:
    return np.sort(np.concatenate([np.repeat(boundary, order), interior]))


def _spline_design(knots: np.ndarray, x: np.ndarray, order: int, derivs: int = 0) -> np.ndarray:
    """``splines::splineDesign`` for points inside the boundary knots."""
    n_basis = knots.size - order
    spline = BSpline(knots, np.eye(n_basis), order - 1, extrapolate=True)
    if derivs:
        spline = spline.derivative(derivs)
    return np.atleast_2d(spline(np.asarray(x, dtype=float)))


def _bs_basis(x, interior, boundary, degree, intercept) -> np.ndarray:
    order = degree + 1
    knots = _all_knots(interior, boundary, order)
    n_basis = knots.size - order
    basis = np.zeros((x.size, n_basis))
    left = x < boundary[0]
    right = x > boundary[1]
    inside = ~(left | right)
    if np.any(inside):
        basis[inside] = _spline_design(knots, x[inside], order)
    scale = np.array([float(np.prod(np.arange(1, j + 1))) for j in range(order)])
    for mask, pivot in ((left, boundary[0]), (right, boundary[1])):
        if not np.any(mask):
            continue
        taylor = np.column_stack([(x[mask] - pivot) ** j for j in range(order)])
        derivs = np.vstack([_spline_design(knots, np.array([pivot]), order, j) for j in range(order)])
        basis[mask] = taylor @ (derivs / scale[:, None])
    return basis if intercept else basis[:, 1:]


def _householder_qty(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``qr.qty(qr(a), b)`` for a full-column-rank ``a`` (LINPACK sign rule)."""
    a = a.astype(float).copy()
    b = b.astype(float).copy()
    n, p = a.shape
    for j in range(p):
        col = a[j:, j]
        norm = np.linalg.norm(col)
        if norm == 0.0:
            continue
        alpha = norm if col[0] >= 0 else -norm
        v = col.copy()
        v[0] += alpha
        vv = float(v @ v)
        a[j:, j:] -= np.outer(v, (2.0 / vv) * (v @ a[j:, j:]))
        b[j:] -= np.outer(v, (2.0 / vv) * (v @ b[j:]))
    return b


def _ns_basis(x, interior, boundary, intercept) -> np.ndarray:
    order = 4
    knots = _all_knots(interior, boundary, order)
    n_basis = knots.size - order
    basis = np.zeros((x.size, n_basis))
    left = x < boundary[0]
    right = x > boundary[1]
    inside = ~(left | right)
    if np.any(inside):
        basis[inside] = _spline_design(knots, x[inside], order)
    for mask, pivot in ((left, boundary[0]), (right, boundary[1])):
        if not np.any(mask):
            continue
        value = _spline_design(knots, np.array([pivot]), order, 0)
        slope = _spline_design(knots, np.array([pivot]), order, 1)
        basis[mask] = value + np.outer(x[mask] - pivot, slope[0])
    const = _spline_design(knots, np.asarray(boundary), order, 2)
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]
    projected = _householder_qty(const.T, basis.T).T
    return projected[:, 2:]


def _resolve_rcs(x: np.ndarray, args: dict[str, object], name: str) -> SplineSpec:
    parms = args.get("parms", 5)
    values = _vector(parms, "rcs() knots")
    if values.size == 1:
        nk = int(values[0])
        knots = _rcs_knots(x, nk)
    else:
        knots = values
    knots = np.unique(knots)
    if knots.size < 3:
        raise ValueError("rcs() needs at least 3 unique knots")
    primes = [name + "'" * j for j in range(knots.size - 1)]
    return SplineSpec(kind="rcs", knots=tuple(float(k) for k in knots), suffixes=tuple(primes))


def _rcs_knots(xx: np.ndarray, nk: int, fractied: float = 0.05) -> np.ndarray:
    """``Hmisc::rcspline.eval(knots.only = TRUE)`` default knot placement."""
    n = xx.size
    if n < 6:
        raise ValueError("rcs() needs at least 6 non-missing values to place knots")
    if nk < 3:
        raise ValueError("rcs() needs at least 3 knots")
    xu = np.unique(xx)
    nxu = xu.size
    if nxu - 2 <= nk:
        return xu[1:-1]
    outer = 0.05 if nk > 3 else 0.1
    if nk > 6:
        outer = 0.025
    nke = nk
    first: list[float] = []
    last: list[float] = []
    override_first = override_last = False
    if 0 < fractied < 1:
        uniq, counts = np.unique(xx, return_counts=True)
        freq = counts / n
        if freq[1:-1].max() < fractied:
            if freq[0] >= fractied:
                first = [float(xx[xx > xx.min()].min())]
                xx = xx[xx > first[0]]
                nke -= 1
                override_first = True
            if freq[-1] >= fractied:
                last = [float(xx[xx < xx.max()].max())]
                xx = xx[xx < last[0]]
                nke -= 1
                override_last = True
    if nke == 1:
        knots = np.array([float(np.median(xx))])
    else:
        if nxu <= nke:
            knots = xu.astype(float)
        else:
            probs = np.linspace(0.5, 1 - outer, nke) if nke == 2 else np.linspace(outer, 1 - outer, nke)
            knots = _quantile7(xx, probs)
            if np.unique(knots).size < min(nke, 3):
                raise ValueError("rcs() could not place distinct knots; pass the knots, for example rcs(x, c(1, 5, 9))")
        if xx.size < 100:
            xs = np.sort(xx)
            if not override_first:
                knots[0] = xs[4]
            if not override_last:
                knots[nke - 1] = xs[xs.size - 5]
    return np.concatenate([first, knots, last])


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    nk = knots.size
    kd = (knots[-1] - knots[0]) ** (2.0 / 3.0)
    k_last, k_prev = knots[-1], knots[-2]
    cols = [x]
    for j in range(nk - 2):
        term = np.maximum((x - knots[j]) / kd, 0.0) ** 3 + (
            (k_prev - knots[j]) * np.maximum((x - k_last) / kd, 0.0) ** 3
            - (k_last - knots[j]) * np.maximum((x - k_prev) / kd, 0.0) ** 3
        ) / (k_last - k_prev)
        cols.append(term)
    return np.column_stack(cols)
