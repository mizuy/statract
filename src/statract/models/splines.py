"""Regression spline bases for formulas: ``ns``, ``bs``, and ``rcs``.

``ns`` and ``bs`` follow R's ``splines`` package and ``rcs`` follows
``rms::rcs`` (restricted cubic splines, Harrell 2015, sec. 2.4.5).
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
    """B-spline basis matrix (n x columns) for fixed interior and boundary knots.

    The basis is the order ``degree + 1`` B-spline basis with each boundary knot
    repeated ``degree + 1`` times (de Boor 2001, ch. IX; Hastie, Tibshirani &
    Friedman 2009, ch. 5 appendix).  Beyond a boundary knot every basis
    function is continued by its degree-``degree`` Taylor polynomial about that
    knot, built from the derivatives there (de Boor 2001, ch. X).  The first
    column is dropped when ``intercept`` is false.
    """
    x = np.asarray(x, dtype=float).ravel()
    lower, upper = float(boundary[0]), float(boundary[1])
    order = degree + 1
    knots = _all_knots(np.asarray(interior, dtype=float), (lower, upper), order)
    n_basis = knots.size - order
    out = np.zeros((x.size, n_basis))
    inside = (x >= lower) & (x <= upper)
    if inside.any():
        out[inside] = _spline_design(knots, x[inside], order)
    for edge, mask in ((lower, x < lower), (upper, x > upper)):
        if not mask.any():
            continue
        step = x[mask] - edge
        taylor = np.zeros((step.size, n_basis))
        factorial = 1.0
        for d in range(order):
            if d:
                factorial *= d
            derivative = _spline_design(knots, np.array([edge]), order, derivs=d)[0]
            taylor += np.outer(step**d / factorial, derivative)
        out[mask] = taylor
    if not intercept:
        out = out[:, 1:]
    return out


def _ns_basis(x, interior, boundary, intercept) -> np.ndarray:
    """Natural cubic spline basis matrix (n x columns) for fixed knots.

    Cubic B-splines (boundary knots repeated four times) restricted to the
    subspace whose second derivative vanishes at both boundary knots
    (Hastie, Tibshirani & Friedman 2009, sec. 5.2.1; Green & Silverman 1994,
    ch. 2).  The restriction is a change of basis: with C the (m x 2) matrix
    of second derivatives of the m B-splines at the two boundary knots and
    C = Q R its complete Householder QR factorization, the natural-spline
    basis is B Q[:, 2:], the coordinates of B in the orthogonal complement of
    the columns of C.  Outside the boundary knots the B-splines are extended
    linearly (value plus first derivative at the nearer knot), so the result
    is linear there.
    """
    x = np.asarray(x, dtype=float).ravel()
    lower, upper = float(boundary[0]), float(boundary[1])
    order = 4
    knots = _all_knots(np.asarray(interior, dtype=float), (lower, upper), order)
    n_basis = knots.size - order
    basis = np.zeros((x.size, n_basis))
    inside = (x >= lower) & (x <= upper)
    if inside.any():
        basis[inside] = _spline_design(knots, x[inside], order)
    for edge, mask in ((lower, x < lower), (upper, x > upper)):
        if mask.any():
            value = _spline_design(knots, np.array([edge]), order)[0]
            slope = _spline_design(knots, np.array([edge]), order, derivs=1)[0]
            basis[mask] = value + np.outer(x[mask] - edge, slope)
    constraint = _spline_design(knots, np.array([lower, upper]), order, derivs=2)
    if not intercept:
        basis = basis[:, 1:]
        constraint = constraint[:, 1:]
    q, _ = np.linalg.qr(constraint.T, mode="complete")
    return basis @ q[:, 2:]


def _resolve_rcs(x: np.ndarray, args: dict[str, object], name: str) -> SplineSpec:
    parms = args.get("parms", 5)
    values = _vector(parms, "rcs() knots")
    if values.size == 1:
        knots = _rcs_knots(x, int(values[0]))
    else:
        knots = values
    knots = np.unique(knots)
    if knots.size < 3:
        raise ValueError("rcs() needs at least 3 unique knots")
    primes = [name + "'" * j for j in range(knots.size - 1)]
    return SplineSpec(kind="rcs", knots=tuple(float(k) for k in knots), suffixes=tuple(primes))


def _rcs_outer_fraction(n_knots: int) -> float:
    """Tail probability of the outermost default knot (Harrell 2015, Table 2.3)."""
    if n_knots <= 3:
        return 0.10
    if n_knots <= 6:
        return 0.05
    return 0.025


def _rcs_knots(values: np.ndarray, n_knots: int, tie_share: float = 0.05) -> np.ndarray:
    """Default knot locations (sorted float array) for ``rcs(x, nk)``.

    Knots sit at equally spaced sample quantiles between an outer tail
    fraction and one minus it (Harrell 2015, *Regression Modeling Strategies*,
    2nd ed., sec. 2.4.6, Table 2.3; Stone & Koo 1985).  With fewer than 100
    observations the outer knots move to the fifth smallest and fifth largest
    values.  When the minimum (or maximum) alone holds a share ``tie_share`` of
    the data while no interior value does, that end knot becomes the nearest
    distinct value inward and the other knots are placed on the observations
    strictly inside it.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    if n_knots < 3:
        raise ValueError("rcs() needs at least 3 knots")
    if n < 6:
        raise ValueError("rcs() needs at least 6 non-missing values to place knots")
    distinct, counts = np.unique(ordered, return_counts=True)
    if distinct.size - 2 <= n_knots:
        return distinct[1:-1].astype(float)

    frequency = counts / n
    inner_value_heavy = frequency[1:-1].max() >= tie_share
    pile_at_min = (not inner_value_heavy) and frequency[0] >= tie_share
    pile_at_max = (not inner_value_heavy) and frequency[-1] >= tie_share

    low_end = [distinct[1]] if pile_at_min else []
    high_end = [distinct[-2]] if pile_at_max else []
    keep = np.ones(n, dtype=bool)
    if pile_at_min:
        keep &= ordered > distinct[1]
    if pile_at_max:
        keep &= ordered < distinct[-2]
    inner = ordered[keep]
    n_inner = n_knots - len(low_end) - len(high_end)
    if n_inner == 1:
        spread = np.array([np.median(inner)])
    elif n_inner >= 2:
        tail = _rcs_outer_fraction(n_knots)
        spread = _quantile7(inner, np.linspace(tail, 1.0 - tail, n_inner))
        if n < 100:
            if not pile_at_min:
                spread[0] = ordered[4]
            if not pile_at_max:
                spread[-1] = ordered[-5]
    else:
        spread = np.zeros(0)
    knots = np.sort(np.concatenate([low_end, spread, high_end]).astype(float))
    if np.unique(knots).size < min(n_knots, 3):
        raise ValueError(
            "rcs() could not place distinct knots; pass the knots, for example rcs(x, c(1, 5, 9))"
        )
    return knots


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    n_knots = knots.size
    span = (knots[-1] - knots[0]) ** (2.0 / 3.0)
    last_knot, next_to_last = knots[-1], knots[-2]
    cols = [x]
    for j in range(n_knots - 2):
        term = np.maximum((x - knots[j]) / span, 0.0) ** 3 + (
            (next_to_last - knots[j]) * np.maximum((x - last_knot) / span, 0.0) ** 3
            - (last_knot - knots[j]) * np.maximum((x - next_to_last) / span, 0.0) ** 3
        ) / (last_knot - next_to_last)
        cols.append(term)
    return np.column_stack(cols)
