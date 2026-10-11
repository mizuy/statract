"""Numba kernels for ``penalized.py``: coordinate descent and Cox derivatives.

Written for statract from the published algorithms, not from any program's
source:

* Friedman, Hastie, Tibshirani (2010). Regularization paths for generalized
  linear models via coordinate descent. J Stat Softw 33(1). The soft-threshold
  update, the active-set cycling, and the weighted least-squares (IRLS) inner
  problem.
* Simon, Friedman, Hastie, Tibshirani (2011). Regularization paths for Cox's
  proportional hazards model via coordinate descent. J Stat Softw 39(5). The
  gradient and the diagonal of the Hessian of the Breslow log partial
  likelihood in the linear predictor.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _soft(z: float, t: float) -> float:
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


@njit(cache=True)
def _update(z, v, r, gam, j, l1, l2, col_scale):
    """One coordinate: returns ``col_scale * delta**2`` and updates ``r`` and ``gam``."""
    n = z.shape[0]
    g = 0.0
    for i in range(n):
        g += v[i] * r[i] * z[i, j]
    old = gam[j]
    g += col_scale[j] * old
    new = _soft(g, l1[j]) / (col_scale[j] + l2[j])
    d = new - old
    if d == 0.0:
        return 0.0
    gam[j] = new
    for i in range(n):
        r[i] -= d * z[i, j]
    return col_scale[j] * d * d


@njit(cache=True)
def _update_intercept(v, r, weight_sum, b0):
    n = v.shape[0]
    s = 0.0
    for i in range(n):
        s += v[i] * r[i]
    d = s / weight_sum
    if d == 0.0:
        return 0.0
    b0[0] += d
    for i in range(n):
        r[i] -= d
    return weight_sum * d * d


@njit(cache=True)
def wls_cd(z, v, r, gam, b0, fit_intercept, idx, l1, l2, col_scale, tol, max_pass):
    """Elastic-net weighted least squares by cyclic coordinate descent.

    Minimizes ``sum(v * r**2) / 2 + sum(l1 * |gam| + l2 * gam**2 / 2)`` over the
    coordinates ``idx`` (and the intercept ``b0[0]`` when ``fit_intercept``),
    where ``r`` is the current residual and is kept up to date. A full sweep
    over ``idx`` is followed by sweeps over the nonzero coordinates only until
    they settle; the loop ends when a full sweep changes nothing by more than
    ``tol`` (the largest ``col_scale * delta**2``). Returns the number of sweeps.
    """
    weight_sum = 0.0
    for i in range(v.shape[0]):
        weight_sum += v[i]
    passes = 0
    act = np.empty(idx.shape[0], dtype=np.int64)
    while True:
        max_change = 0.0
        if fit_intercept and weight_sum > 0.0:
            max_change = max(max_change, _update_intercept(v, r, weight_sum, b0))
        for k in range(idx.shape[0]):
            max_change = max(max_change, _update(z, v, r, gam, idx[k], l1, l2, col_scale))
        passes += 1
        if max_change < tol or passes >= max_pass:
            return passes
        m = 0
        for k in range(idx.shape[0]):
            if gam[idx[k]] != 0.0:
                act[m] = idx[k]
                m += 1
        while True:
            max_change = 0.0
            if fit_intercept and weight_sum > 0.0:
                max_change = max(max_change, _update_intercept(v, r, weight_sum, b0))
            for k in range(m):
                max_change = max(max_change, _update(z, v, r, gam, act[k], l1, l2, col_scale))
            passes += 1
            if max_change < tol or passes >= max_pass:
                break


@njit(cache=True)
def weighted_sq(z, v, idx, col_scale):
    n = z.shape[0]
    for k in range(idx.shape[0]):
        j = idx[k]
        s = 0.0
        for i in range(n):
            s += v[i] * z[i, j] * z[i, j]
        col_scale[j] = s


@njit(cache=True)
def cox_derivs(eta, w, status, order, starts, ends):
    """Breslow log partial likelihood, its gradient, and the Hessian diagonal in ``eta``.

    ``order`` sorts the rows by time; ``starts``/``ends`` delimit runs of equal
    times in that order. The risk set at time ``t`` is every row with time at
    least ``t``. Weights enter the risk sums and the event counts. Returns
    ``(loglik, grad, hdiag)`` where ``-hdiag`` is the diagonal of the Hessian.
    """
    n = eta.shape[0]
    shift = -np.inf
    for i in range(n):
        if w[i] > 0.0 and eta[i] > shift:
            shift = eta[i]
    if not np.isfinite(shift):
        shift = 0.0
    ew = np.empty(n)
    for i in range(n):
        ew[i] = w[i] * math.exp(eta[i] - shift)
    n_runs = starts.shape[0]
    risk = np.empty(n_runs)
    dsum = np.empty(n_runs)
    acc = 0.0
    for g in range(n_runs - 1, -1, -1):
        d = 0.0
        for k in range(starts[g], ends[g]):
            i = order[k]
            acc += ew[i]
            d += w[i] * status[i]
        risk[g] = acc
        dsum[g] = d
    loglik = 0.0
    grad = np.empty(n)
    hdiag = np.empty(n)
    c1 = 0.0
    c2 = 0.0
    for g in range(n_runs):
        if dsum[g] > 0.0:
            c1 += dsum[g] / risk[g]
            c2 += dsum[g] / (risk[g] * risk[g])
            loglik -= dsum[g] * (math.log(risk[g]) + shift)
        for k in range(starts[g], ends[g]):
            i = order[k]
            if status[i] != 0.0:
                loglik += w[i] * status[i] * eta[i]
            grad[i] = w[i] * status[i] - ew[i] * c1
            hdiag[i] = ew[i] * c1 - ew[i] * ew[i] * c2
    return loglik, grad, hdiag


@njit(cache=True)
def cox_loglik(eta, w, status, order, starts, ends):
    n = eta.shape[0]
    shift = -np.inf
    for i in range(n):
        if w[i] > 0.0 and eta[i] > shift:
            shift = eta[i]
    if not np.isfinite(shift):
        shift = 0.0
    n_runs = starts.shape[0]
    acc = 0.0
    loglik = 0.0
    for g in range(n_runs - 1, -1, -1):
        d = 0.0
        for k in range(starts[g], ends[g]):
            i = order[k]
            acc += w[i] * math.exp(eta[i] - shift)
            d += w[i] * status[i]
            if status[i] != 0.0:
                loglik += w[i] * status[i] * eta[i]
        if d > 0.0:
            loglik -= d * (math.log(acc) + shift)
    return loglik


# Family codes for ``solve_lambda``.
GAUSSIAN, BINOMIAL, POISSON, COX = 0, 1, 2, 3
_PMIN = 1e-9  # smallest fitted probability in the binomial working weights


@njit(cache=True)
def linear_predictor(z, gam, b0, off, out):
    n, k = z.shape
    for i in range(n):
        out[i] = off[i] + b0
    for j in range(k):
        g = gam[j]
        if g != 0.0:
            for i in range(n):
                out[i] += g * z[i, j]


@njit(cache=True)
def loss(fam, eta, y, w, status, order, starts, ends):
    """Negative log-likelihood per unit weight (gaussian: half the weighted RSS)."""
    n = eta.shape[0]
    s = 0.0
    if fam == GAUSSIAN:
        for i in range(n):
            d = y[i] - eta[i]
            s += 0.5 * w[i] * d * d
    elif fam == BINOMIAL:
        for i in range(n):
            e = eta[i]
            lae = e + math.log1p(math.exp(-e)) if e > 0 else math.log1p(math.exp(e))
            s += w[i] * (lae - y[i] * e)
    elif fam == POISSON:
        for i in range(n):
            s += w[i] * (math.exp(eta[i]) - y[i] * eta[i])
    else:
        s = -cox_loglik(eta, w, status, order, starts, ends)
    return s


@njit(cache=True)
def score_weights(fam, eta, y, w, status, order, starts, ends, u, v):
    """``u``: score in the linear predictor; ``v``: working weights."""
    n = eta.shape[0]
    if fam == GAUSSIAN:
        for i in range(n):
            u[i] = w[i] * (y[i] - eta[i])
            v[i] = w[i]
    elif fam == BINOMIAL:
        for i in range(n):
            p = 1.0 / (1.0 + math.exp(-eta[i]))
            u[i] = w[i] * (y[i] - p)
            pc = min(max(p, _PMIN), 1.0 - _PMIN)
            v[i] = w[i] * pc * (1.0 - pc)
    elif fam == POISSON:
        for i in range(n):
            mu = math.exp(eta[i])
            u[i] = w[i] * (y[i] - mu)
            v[i] = w[i] * mu
    else:
        _ll, g, h = cox_derivs(eta, w, status, order, starts, ends)
        for i in range(n):
            u[i] = g[i]
            v[i] = h[i] if h[i] > 0.0 else 0.0


@njit(cache=True)
def _penalty(gam, l1, l2):
    s = 0.0
    for j in range(gam.shape[0]):
        s += l1[j] * abs(gam[j]) + 0.5 * l2[j] * gam[j] * gam[j]
    return s


@njit(cache=True)
def solve_lambda(
    fam, z, y, w, off, gam, b0, fit_intercept, in_set, l1, l2, tol, max_pass, kkt,
    status, order, starts, ends, grad,
):
    """Solve one lambda: proximal Newton (IRLS) outer loop, coordinate descent inside.

    ``in_set`` marks the columns allowed to move. Each outer step solves the
    penalized weighted least-squares problem of the current quadratic
    approximation, then halves the step while the penalized objective rises.
    The outer loop ends when a step moves no coordinate by more than ``tol``
    (as ``col_scale * delta**2``). With ``kkt``, every column outside the set is then
    checked: a column whose score exceeds its L1 penalty joins the set and the
    problem is solved again. ``grad`` receives the score of every column.
    Returns the number of coordinate sweeps.
    """
    n, k = z.shape
    eta = np.empty(n)
    u = np.empty(n)
    v = np.empty(n)
    r = np.empty(n)
    col_scale = np.zeros(k)
    old = np.empty(k)
    passes = 0
    while True:
        m = 0
        for j in range(k):
            if in_set[j]:
                m += 1
        idx = np.empty(m, dtype=np.int64)
        m = 0
        for j in range(k):
            if in_set[j]:
                idx[m] = j
                m += 1
        linear_predictor(z, gam, b0[0], off, eta)
        obj = loss(fam, eta, y, w, status, order, starts, ends) + _penalty(gam, l1, l2)
        for _outer in range(500):
            score_weights(fam, eta, y, w, status, order, starts, ends, u, v)
            weight_sum = 0.0
            for i in range(n):
                r[i] = u[i] / v[i] if v[i] > 0.0 else 0.0
                weight_sum += v[i]
            weighted_sq(z, v, idx, col_scale)
            for j in range(k):
                old[j] = gam[j]
            old_b0 = b0[0]
            budget = max_pass - passes
            if budget < 1:
                budget = 1
            passes += wls_cd(z, v, r, gam, b0, fit_intercept, idx, l1, l2, col_scale, tol, budget)
            if fam == GAUSSIAN:
                break
            linear_predictor(z, gam, b0[0], off, eta)
            new = loss(fam, eta, y, w, status, order, starts, ends) + _penalty(gam, l1, l2)
            halvings = 0
            while not (new <= obj + 1e-13 * abs(obj)) and halvings < 30:
                for j in range(k):
                    gam[j] = 0.5 * (old[j] + gam[j])
                b0[0] = 0.5 * (old_b0 + b0[0])
                linear_predictor(z, gam, b0[0], off, eta)
                new = loss(fam, eta, y, w, status, order, starts, ends) + _penalty(gam, l1, l2)
                halvings += 1
            change = 0.0
            for t in range(m):
                j = idx[t]
                d = gam[j] - old[j]
                if col_scale[j] * d * d > change:
                    change = col_scale[j] * d * d
            if fit_intercept:
                d = b0[0] - old_b0
                if weight_sum * d * d > change:
                    change = weight_sum * d * d
            obj = new
            if change < tol or passes >= max_pass:
                break
        linear_predictor(z, gam, b0[0], off, eta)
        score_weights(fam, eta, y, w, status, order, starts, ends, u, v)
        for j in range(k):
            s = 0.0
            for i in range(n):
                s += z[i, j] * u[i]
            grad[j] = s
        if not kkt or passes >= max_pass:
            return passes
        added = False
        for j in range(k):
            if not in_set[j] and abs(grad[j]) > l1[j]:
                in_set[j] = True
                added = True
        if not added:
            return passes
