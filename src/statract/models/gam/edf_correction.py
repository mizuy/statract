"""Effective degrees of freedom corrected for smoothing parameter uncertainty.

Wood, Pya & Säfken (2016, *JASA* 111:1548-1563), sec. 4-5 and appendix D.
With ``rho = log(lambda)`` and ``V_rho`` the inverse Hessian of the fitting
criterion (REML or ML) in ``rho``, the posterior covariance of the
coefficients becomes ``V' = V + J V_rho J' + V''`` (eq. 7), where
``J = d beta / d rho`` and ``V''_{jm} = sum_{i,k,l} dR_ij/drho_k V_rho,kl
dR_im/drho_l`` for a factor ``R'R = V``. The corrected edf is
``tau2 = tr(I V')`` with ``I`` the unpenalized information. It is capped
at ``tau1 = tr(2F - FF)``, ``F = V I``, which ``?gamObject`` calls an upper
bound for it. The AIC of eq. 10 is ``-2 l + 2 tau2``.

``R`` is ``L^-1`` for the Cholesky factor ``L L' = V^-1`` (appendix D).
``V''`` depends on that choice of factor, so ``tau2`` matches mgcv only
to about 1% when the cap does not apply. Derivatives in ``rho`` are
central differences of full refits, so ``J`` and ``dR`` include the change
of the working weights.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_STEP_CRITERION = 1e-3
_STEP_FIT = 1e-4
_FLAT = 1e-3


def corrected_edf(
    rho: np.ndarray,
    refit: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
    information: np.ndarray,
) -> tuple[float, float, float]:
    """``(tau0, tau1, tau2)`` for a fit at log smoothing parameters ``rho``.

    ``refit(rho)`` returns ``(criterion, beta, V)``: the minimized fitting
    criterion, the coefficients, and their Bayesian covariance at fixed
    ``rho``. ``information`` is ``X'WX / phi`` at the fit.
    """
    rho = np.asarray(rho, dtype=float)
    m = rho.size
    _crit, beta, cov = refit(rho)
    influence = cov @ information
    tau0 = float(np.trace(influence))
    tau1 = float(np.trace(2 * influence - influence @ influence))
    if m == 0:
        return tau0, tau1, tau0

    def criterion(values: np.ndarray) -> float:
        return refit(values)[0]

    h = _STEP_CRITERION
    hessian = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            ei = np.eye(m)[i] * h
            ej = np.eye(m)[j] * h
            value = (
                criterion(rho + ei + ej)
                - criterion(rho + ei - ej)
                - criterion(rho - ei + ej)
                + criterion(rho - ei - ej)
            ) / (4 * h * h)
            hessian[i, j] = hessian[j, i] = value
    # A smoothing parameter that is effectively infinite (or zero) leaves the
    # criterion flat in its direction. Section 4 gives it no variance through
    # a pseudo-inverse; a curvature below _FLAT is treated as flat, since its
    # inverse would put the log smoothing parameter's SD above 30.
    keep = np.diag(hessian) > _FLAT
    v_rho = np.zeros((m, m))
    if keep.any():
        v_rho[np.ix_(keep, keep)] = np.linalg.pinv(hessian[np.ix_(keep, keep)], rcond=1e-8)

    def factor(v: np.ndarray) -> np.ndarray:
        lower = np.linalg.cholesky(np.linalg.inv(v))
        return np.linalg.inv(lower)

    jac = np.zeros((beta.size, m))
    d_factor = []
    for k in range(m):
        step = np.eye(m)[k] * _STEP_FIT
        _cp, beta_up, cov_up = refit(rho + step)
        _cm, beta_down, cov_down = refit(rho - step)
        jac[:, k] = (beta_up - beta_down) / (2 * _STEP_FIT)
        d_factor.append((factor(cov_up) - factor(cov_down)) / (2 * _STEP_FIT))
    second = np.zeros_like(cov)
    for k in range(m):
        for l in range(m):
            second += v_rho[k, l] * d_factor[k].T @ d_factor[l]
    corrected = cov + jac @ v_rho @ jac.T + second
    tau2 = float(np.trace(corrected @ information))
    return tau0, tau1, min(tau2, tau1)
