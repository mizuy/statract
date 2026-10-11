"""Lowess smoothing and linear interpolation.

``lowess`` is Cleveland's (1979) locally weighted linear regression with
bisquare robustness passes. The code follows the ``lowess`` of
``statsmodels`` (``statsmodels/nonparametric/_smoothers_lowess.pyx``, BSD-3):
a sliding window of the ``frac * n`` nearest points, tricube weights, the
``delta`` skip with linear interpolation between fitted points, and tied ``x``
sharing one fit. The defaults are those documented for R's ``stats::lowess``
(``f = 2/3``, ``iter = 3``, ``delta = 0.01 * diff(range(x))``).

Cleveland, W. S. (1979). Robust locally weighted regression and smoothing
scatterplots. *Journal of the American Statistical Association* 74, 829-836.

The lowess code is derived from the statsmodels code and carries its notice:

    Copyright (C) 2006, Jonathan E. Taylor
    All rights reserved.

    Copyright (c) 2006-2008 Scipy Developers.
    All rights reserved.

    Copyright (c) 2009-2018 statsmodels Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of statsmodels nor the names of its contributors
         may be used to endorse or promote products derived from this software
         without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _local_fit(x, y, i, left, right, robust, weights):
    """Weighted straight line through ``x[left:right]`` evaluated at ``x[i]``."""
    xval = x[i]
    radius = max(xval - x[left], x[right - 1] - xval)
    if radius == 0.0:
        # Every neighbour ties with xval, so no line can be fitted.
        return y[i]
    total = 0.0
    nonzero = 0
    for j in range(left, right):
        d = abs(x[j] - xval) / radius
        t = 1.0 - d * d * d
        weights[j] = t * t * t * robust[j]
        total += weights[j]
        if weights[j] > 1e-12:
            nonzero += 1
    if nonzero < 2:
        return y[i]
    mean_x = 0.0
    for j in range(left, right):
        weights[j] /= total
        mean_x += weights[j] * x[j]
    spread = 0.0
    for j in range(left, right):
        spread += weights[j] * (x[j] - mean_x) ** 2
    spread = max(spread, 1e-12)
    value = 0.0
    for j in range(left, right):
        value += weights[j] * (1.0 + (xval - mean_x) * (x[j] - mean_x) / spread) * y[j]
    return value


@njit(cache=True)
def lowess_sorted(x, y, frac, iterations, delta):
    """Lowess fit at each point of the sorted ``x``.

    Each fit uses the ``frac * n`` nearest points with tricube weights. After a
    fit at ``x[i]``, points within ``delta`` of it are skipped and filled by
    linear interpolation, except the last such point, which is fitted. The
    robustness passes reweight by the bisquare of residuals over six times
    their median.
    """
    n = x.shape[0]
    k = int(frac * n + 1e-10)
    k = min(max(k, 2), n)
    fitted = np.empty(n)
    weights = np.empty(n)
    robust = np.ones(n)
    for _ in range(iterations + 1):
        left = 0
        right = k
        last = -1
        i = 0
        while True:
            while right < n and x[i] > (x[left] + x[right]) / 2.0:
                left += 1
                right += 1
            fitted[i] = _local_fit(x, y, i, left, right, robust, weights)
            if last >= 0 and i - last > 1:
                span = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / span
                    fitted[j] = alpha * fitted[i] + (1.0 - alpha) * fitted[last]
            last = i
            cut = x[last] + delta
            j = last + 1
            while j < n:
                if x[j] > cut:
                    break
                if x[j] == x[last]:
                    # Tied x share the fit.
                    fitted[j] = fitted[last]
                    last = j
                j += 1
            if last >= n - 1:
                break
            i = max(last + 1, j - 1)
        resid = np.abs(y - fitted)
        median = np.median(resid)
        for j in range(n):
            if median == 0.0:
                r = 1.0 if resid[j] > 0.0 else 0.0
            else:
                r = min(resid[j] / (6.0 * median), 1.0)
            t = 1.0 - r * r
            robust[j] = t * t
    return fitted


def lowess(
    x: np.ndarray,
    y: np.ndarray,
    *,
    f: float = 2.0 / 3.0,
    iter: int = 3,
    delta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The sorted ``x`` and the lowess fit at each of them.

    ``delta=None`` is ``0.01 * (max(x) - min(x))``. ``iter`` counts the
    robustness passes after the first fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs = np.ascontiguousarray(x[order])
    ys = np.ascontiguousarray(y[order])
    if xs.size == 0:
        return xs, ys
    if delta is None:
        delta = 0.01 * (xs[-1] - xs[0])
    return xs, lowess_sorted(xs, ys, float(f), int(iter), float(delta))


def interpolate(x: np.ndarray, y: np.ndarray, xout: np.ndarray, *, ties: str = "first") -> np.ndarray:
    """Linear interpolation of ``(x, y)`` at ``xout``; NaN outside the range of ``x``.

    Duplicated ``x`` keep their first ``y`` (``ties="first"``) or their mean
    (``ties="mean"``), as ``stats::approx`` does with ``rule = 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xout = np.asarray(xout, dtype=float)
    order = np.argsort(x, kind="stable")
    ux, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    ys = y[order]
    if ties == "first":
        uy = ys[start]
    elif ties == "mean":
        uy = np.add.reduceat(ys, start) / counts if ys.size else ys
    else:
        raise ValueError("ties must be 'first' or 'mean'")
    if ux.size == 0:
        return np.full(xout.shape, np.nan)
    return np.interp(xout, ux, uy, left=np.nan, right=np.nan)
