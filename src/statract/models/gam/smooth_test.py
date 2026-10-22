"""Tests that a smooth term is zero, from Wood (2013, *Biometrika* 100:221-228).

The statistic is a Wald statistic on the smooth's fitted values with a
rank-``r`` pseudo-inverse of their Bayesian covariance. ``r`` is the
smooth's reference degrees of freedom, the block trace of ``2F - FF`` with
``F = V_beta X'WX / phi``. For non-integer ``r`` the last two eigen
components enter through a 2 x 2 block, so that the null distribution is
``chi2_{k-2} + nu1 chi2_1 + nu2 chi2_1`` (sec. 2.2-2.3). The QR form of
sec. 2.4 keeps the work in the smooth's coefficient dimension.

Two choices the paper leaves open follow ``?summary.gam`` and agree with its
``s.table``: the rows of the basis are weighted by the square roots of the
working weights ("the components of the test statistic are weighted by the
iterative fitting weights"), and the signs of the last two eigenvectors are
taken so that the cross term of the 2 x 2 block is not negative. The
paper's statistic changes with those signs, which an eigen solver leaves
arbitrary.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def reference_df(influence: np.ndarray, columns: np.ndarray) -> float:
    """Block trace of ``2F - FF`` for one smooth (``F`` is the influence ``V_beta X'WX / phi``)."""
    sl = np.asarray(columns)
    ff = influence[sl, :] @ influence[:, sl]
    return float(2 * np.trace(influence[np.ix_(sl, sl)]) - np.trace(ff))


def wood_statistic(beta: np.ndarray, covariance: np.ndarray, basis: np.ndarray, rank: float):
    """``T_r`` and the weights of its null distribution.

    ``beta`` and ``covariance`` are the smooth's coefficients and their
    Bayesian covariance; ``basis`` is its model matrix at the data, with rows
    scaled by the square roots of the working weights. Returns
    ``(statistic, weights, dofs)``: under the null hypothesis the statistic
    is ``sum(weights[i] * chi2(dofs[i]))``.
    """
    rank = max(float(rank), 1.0)
    _q, r_factor = np.linalg.qr(basis)
    middle = r_factor @ covariance @ r_factor.T
    middle = 0.5 * (middle + middle.T)
    eigval, eigvec = np.linalg.eigh(middle)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    k = int(np.floor(rank)) + 1
    k = min(k, int(np.sum(eigval > eigval[0] * np.finfo(float).eps * len(eigval))))
    nu = min(rank - k + 1, 1.0)
    d = eigvec.T @ (r_factor @ beta)
    lead = float(np.sum(d[: k - 2] ** 2 / eigval[: k - 2])) if k > 2 else 0.0
    rho = np.sqrt(max(nu * (1 - nu), 0.0) / 2)
    scale = eigval[k - 2 : k] ** -0.5
    block = np.outer(scale, scale) * np.array([[1.0, rho], [rho, nu]])
    tail = np.abs(d[k - 2 : k])
    statistic = lead + float(tail @ block @ tail)
    nu1 = (nu + 1 + np.sqrt(max(1 - nu * nu, 0.0))) / 2
    nu2 = nu + 1 - nu1
    weights = [1.0, nu1, nu2]
    dofs = [k - 2, 1, 1]
    return statistic, weights, dofs


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(96)


def null_sf(x: np.ndarray | float, weights, dofs) -> np.ndarray:
    """``P(chi2(a) + nu1 Z1^2 + nu2 Z2^2 > x)`` for the null distribution of ``T_r``.

    ``weights`` and ``dofs`` are as returned by :func:`wood_statistic`:
    ``(1, nu1, nu2)`` and ``(a, 1, 1)``. Each normal is integrated out with
    Gauss-Legendre quadrature up to the point where the remaining sum is
    certain to exceed ``x``; past it the probability is the normal tail.
    Vectorized over ``x``.
    """
    _one, nu1, nu2 = (float(w) for w in weights)
    a = int(dofs[0])
    x = np.atleast_1d(np.asarray(x, dtype=float))

    def chisq_sf(y):
        if a == 0:
            return (y < 0).astype(float)
        return np.where(y <= 0, 1.0, stats.chi2.sf(np.maximum(y, 0.0), a))

    def add_normal(sf, y, nu):
        # P(S + nu Z^2 > y) from P(S > .) = sf(.), with Z standard normal.
        y = np.asarray(y, dtype=float)
        if nu <= 1e-12:
            return sf(y)
        edge = np.sqrt(np.maximum(y, 0.0) / nu)
        z = 0.5 * edge[..., None] * (_NODES + 1)
        inner = sf(y[..., None] - nu * z * z)
        dens = 2 * stats.norm.pdf(z)
        part = 0.5 * edge * np.sum(_WEIGHTS * dens * inner, axis=-1)
        return np.where(y <= 0, 1.0, part + 2 * stats.norm.sf(edge))

    def with_second(y):
        if a == 0 and nu2 > 1e-12:
            return np.where(y <= 0, 1.0, 2 * stats.norm.sf(np.sqrt(np.maximum(y, 0.0) / nu2)))
        return add_normal(chisq_sf, y, nu2)

    return np.clip(add_normal(with_second, x, nu1), 0.0, 1.0)


def smooth_p_value(statistic: float, weights, dofs, residual_df: float | None) -> float:
    """p-value for ``T_r``: known scale, or estimated scale with ``residual_df`` (sec. 2.3).

    With an estimated scale the p-value is ``P(Q > t W / kappa)`` with
    ``W ~ chi2(kappa)``, integrated over the quantiles of ``W``.
    """
    if residual_df is None:
        return float(null_sf(statistic, weights, dofs)[0])
    u = 0.5 * (_NODES + 1)
    w = stats.chi2.ppf(u, residual_df)
    return float(np.clip(0.5 * np.sum(_WEIGHTS * null_sf(statistic * w / residual_df, weights, dofs)), 0.0, 1.0))
