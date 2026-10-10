"""R's ``lowess`` (``clowess`` in ``stats/src/lowess.c``).

This is Cleveland's original algorithm as R ships it, with R's defaults
``f = 2/3``, ``iter = 3`` and ``delta = 0.01 * diff(range(x))``. It differs from
the statsmodels port in ``surv.diagnostics`` in the tie handling and in the
``delta`` skip, so ``rms::calibrate`` curves need this one.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _lowest(x, y, n, xs, nleft, nright, w, userw, rw):
    # 0-based translation of lowest(); nleft and nright are 0-based too.
    rng = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    j = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                q = r / h
                q = 1.0 - q * q * q
                w[j] = q * q * q
            if userw:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
    nrt = j - 1
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nrt + 1):
        w[j] /= a
    if h > 0.0:
        a = 0.0
        for j in range(nleft, nrt + 1):
            a += w[j] * x[j]
        b = xs - a
        c = 0.0
        for j in range(nleft, nrt + 1):
            c += w[j] * (x[j] - a) * (x[j] - a)
        if np.sqrt(c) > 0.001 * rng:
            b /= c
            for j in range(nleft, nrt + 1):
                w[j] *= b * (x[j] - a) + 1.0
    ys = 0.0
    for j in range(nleft, nrt + 1):
        ys += w[j] * y[j]
    return ys, True


@njit(cache=True)
def _partial_sort_value(values, k):
    s = np.sort(values)
    return s[k]


@njit(cache=True)
def _clowess(x, y, f, nsteps, delta):
    n = x.shape[0]
    ys = np.zeros(n)
    if n < 2:
        ys[0] = y[0]
        return ys
    rw = np.zeros(n)
    res = np.zeros(n)
    ns = max(2, min(n, int(f * n + 1e-7)))
    it = 1
    while it <= nsteps + 1:
        nleft = 0
        nright = ns - 1
        last = -1
        i = 0
        while True:
            if nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            val, ok = _lowest(x, y, n, x[i], nleft, nright, res, it > 1, rw)
            ys[i] = val if ok else y[i]
            if last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        for k in range(n):
            res[k] = y[k] - ys[k]
        sc = 0.0
        for k in range(n):
            sc += abs(res[k])
        sc /= n
        if it > nsteps:
            break
        for k in range(n):
            rw[k] = abs(res[k])
        m1 = n // 2
        if n % 2 == 0:
            m2 = n - m1 - 1
            cmad = 3.0 * (_partial_sort_value(rw, m1) + _partial_sort_value(rw, m2))
        else:
            cmad = 6.0 * _partial_sort_value(rw, m1)
        if cmad < 1e-7 * sc:
            break
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for k in range(n):
            r = abs(res[k])
            if r <= c1:
                rw[k] = 1.0
            elif r <= c9:
                q = r / cmad
                q = 1.0 - q * q
                rw[k] = q * q
            else:
                rw[k] = 0.0
        it += 1
    return ys


def lowess_r(
    x: np.ndarray,
    y: np.ndarray,
    *,
    f: float = 2.0 / 3.0,
    iter: int = 3,
    delta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``stats::lowess``: the sorted ``x`` and the smoothed ``y`` at each of them."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs = np.ascontiguousarray(x[order])
    yv = np.ascontiguousarray(y[order])
    if delta is None:
        delta = 0.01 * (xs[-1] - xs[0]) if xs.size else 0.0
    return xs, _clowess(xs, yv, float(f), int(iter), float(delta))


def approx_r(x: np.ndarray, y: np.ndarray, xout: np.ndarray, *, ties: str = "first") -> np.ndarray:
    """``stats::approx`` with ``rule = 1``: linear interpolation, NaN outside the range.

    Duplicated ``x`` keep the first ``y`` (``ties = function(x) x[1]``) or their
    mean (``ties = mean``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xout = np.asarray(xout, dtype=float)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    ux, start, counts = np.unique(x, return_index=True, return_counts=True)
    if ux.size < x.size:
        if ties == "first":
            y = y[start]
        elif ties == "mean":
            y = np.add.reduceat(y, start) / counts
        else:
            raise ValueError("ties must be 'first' or 'mean'")
        x = ux
    out = np.full(xout.shape, np.nan)
    if x.size == 0:
        return out
    if x.size == 1:
        out[xout == x[0]] = y[0]
        return out
    inside = (xout >= x[0]) & (xout <= x[-1])
    xo = xout[inside]
    j = np.clip(np.searchsorted(x, xo, side="right") - 1, 0, x.size - 2)
    x0 = x[j]
    x1 = x[j + 1]
    y0 = y[j]
    y1 = y[j + 1]
    # R's approx1: exact hits on the right end of an interval return that y.
    val = y0 + (y1 - y0) * ((xo - x0) / (x1 - x0))
    val = np.where(xo == x1, y1, val)
    val = np.where(xo == x0, y0, val)
    out[inside] = val
    return out
