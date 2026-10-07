"""Laplace and adaptive Gauss–Hermite GLMMs for count outcomes.

One grouping factor with correlated random effects ``b_j = L u_j``,
``u_j ~ N(0, I)``. The conditional modes come from a Newton solve per group.
The marginal log-likelihood is the Laplace approximation (``n_agq=1``) or
adaptive Gauss–Hermite quadrature for a single random intercept. Both match
the objective of ``lme4::glmer`` (Poisson) and ``glmmTMB`` (``nbinom2``).

The outer optimisation runs over ``(beta, log diag L, offdiag L, log theta)``.
The fixed-effect covariance is the beta block of the inverse Hessian of the
negative marginal log-likelihood over all parameters, as in glmer
(``use.hessian=TRUE``) and glmmTMB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

COUNT_FAMILIES = ("poisson", "negative_binomial")
_LOG_SD_FLOOR = -20.0


@dataclass
class LaplaceResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    group_covariance: np.ndarray
    theta: float | None
    theta_std_error: float | None
    log_likelihood: float
    modes: np.ndarray
    converged: bool
    n_iter: int
    function_evals: int


class _Problem:
    def __init__(self, y, x, z, group, offset, family, n_agq):
        self.y = np.asarray(y, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.group = np.asarray(group, dtype=np.int64)
        self.offset = np.zeros(len(self.y)) if offset is None else np.asarray(offset, dtype=float)
        self.family = family
        self.n_agq = int(n_agq)
        self.n, self.p = self.x.shape
        self.q = self.z.shape[1]
        self.m = int(self.group.max()) + 1
        self.n_l = self.q * (self.q + 1) // 2
        self.n_par = self.p + self.n_l + (1 if family == "negative_binomial" else 0)
        self.lgamma_y1 = special.gammaln(self.y + 1.0)
        self.u = np.zeros((self.m, self.q))
        self.evals = 0
        self._const_key = None
        self._const = None
        if self.n_agq > 1:
            nodes, weights = np.polynomial.hermite.hermgauss(self.n_agq)
            self.gh_nodes = nodes
            self.gh_logw = np.log(weights) + nodes**2

    # parameters -------------------------------------------------------
    def unpack(self, phi):
        beta = phi[: self.p]
        lpar = phi[self.p : self.p + self.n_l]
        L = np.zeros((self.q, self.q))
        L[np.diag_indices(self.q)] = np.exp(lpar[: self.q])
        if self.q > 1:
            L[np.tril_indices(self.q, -1)] = lpar[self.q :]
        log_theta = phi[-1] if self.family == "negative_binomial" else None
        return beta, L, log_theta

    # family -----------------------------------------------------------
    def _terms(self, eta, log_theta):
        y = self.y
        if self.family == "poisson":
            mu = np.exp(eta)
            ll = y * eta - mu - self.lgamma_y1
            return ll, y - mu, mu
        theta = np.exp(log_theta)
        if self._const_key != log_theta:
            # The gamma-function terms do not depend on eta; compute them once per theta.
            self._const = special.gammaln(y + theta) - special.gammaln(theta) - self.lgamma_y1 + theta * log_theta
            self._const_key = log_theta
        log_tm = np.logaddexp(log_theta, eta)
        ratio = np.exp(eta - log_tm)  # mu / (theta + mu)
        ll = self._const + y * eta - (y + theta) * log_tm
        d1 = theta * (y - np.exp(eta)) * np.exp(-log_tm)
        w = (y + theta) * ratio * (1.0 - ratio)
        return ll, d1, w

    def _by_group(self, values):
        flat = values.reshape(len(values), -1)
        out = np.empty((self.m, flat.shape[1]))
        for k in range(flat.shape[1]):
            out[:, k] = np.bincount(self.group, weights=flat[:, k], minlength=self.m)
        return out.reshape((self.m,) + values.shape[1:])

    def _group_ll(self, eta, log_theta):
        ll, _, _ = self._terms(eta, log_theta)
        return np.bincount(self.group, weights=ll, minlength=self.m)

    # inner Newton -----------------------------------------------------
    def modes(self, beta, L, log_theta, u0):
        eta0 = self.x @ beta + self.offset
        zl = self.z @ L
        u = u0.copy()
        eye = np.eye(self.q)
        for _ in range(100):
            eta = eta0 + np.einsum("ij,ij->i", zl, u[self.group])
            ll, d1, w = self._terms(eta, log_theta)
            f_old = np.bincount(self.group, weights=ll, minlength=self.m) - 0.5 * np.sum(u * u, axis=1)
            grad = self._by_group(zl * d1[:, None]) - u
            hess = self._by_group(w[:, None, None] * zl[:, :, None] * zl[:, None, :]) + eye
            step = np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
            scale = np.ones(self.m)
            for _half in range(40):
                trial = u + scale[:, None] * step
                eta_t = eta0 + np.einsum("ij,ij->i", zl, trial[self.group])
                f_new = self._group_ll(eta_t, log_theta) - 0.5 * np.sum(trial * trial, axis=1)
                bad = ~(f_new >= f_old - 1e-12 * (1.0 + np.abs(f_old)))
                if not bad.any():
                    break
                scale[bad] *= 0.5
            u = trial
            if np.max(np.abs(scale[:, None] * step)) < 1e-11:
                break
        eta = eta0 + np.einsum("ij,ij->i", zl, u[self.group])
        ll, _, w = self._terms(eta, log_theta)
        hess = self._by_group(w[:, None, None] * zl[:, :, None] * zl[:, None, :]) + eye
        f = np.bincount(self.group, weights=ll, minlength=self.m) - 0.5 * np.sum(u * u, axis=1)
        return u, f, hess, eta0, zl

    def loglik(self, phi):
        self.evals += 1
        beta, L, log_theta = self.unpack(phi)
        u, f, hess, eta0, zl = self.modes(beta, L, log_theta, self.u)
        self.u = u
        if self.n_agq <= 1 or self.q != 1:
            _, logdet = np.linalg.slogdet(hess)
            return float(np.sum(f - 0.5 * logdet))
        # Adaptive Gauss–Hermite for a scalar random effect.
        h = hess[:, 0, 0]
        s = np.sqrt(2.0 / h)
        z1 = zl[:, 0]
        total = np.zeros((self.m, self.n_agq))
        for k, node in enumerate(self.gh_nodes):
            uk = u[:, 0] + s * node
            eta = eta0 + z1 * uk[self.group]
            total[:, k] = self._group_ll(eta, log_theta) - 0.5 * uk**2 + self.gh_logw[k]
        per_group = special.logsumexp(total, axis=1) + np.log(s) - 0.5 * np.log(2.0 * np.pi)
        return float(np.sum(per_group))


def _gradient(fun, phi, h):
    g = np.zeros_like(phi)
    for i in range(len(phi)):
        e = np.zeros_like(phi)
        e[i] = h[i]
        g[i] = (fun(phi + e) - fun(phi - e)) / (2 * h[i])
    return g


def _hessian(fun, phi, h):
    k = len(phi)
    f0 = fun(phi)
    out = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        out[i, i] = (fun(phi + ei) - 2 * f0 + fun(phi - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(k)
            ej[j] = h[j]
            val = (
                fun(phi + ei + ej) - fun(phi + ei - ej) - fun(phi - ei + ej) + fun(phi - ei - ej)
            ) / (4 * h[i] * h[j])
            out[i, j] = out[j, i] = val
    return out


def _glm_start(y, x, offset):
    """Poisson IRLS without random effects, for starting values."""
    beta = np.zeros(x.shape[1])
    mean = max(float(np.mean(y)), 1e-3)
    if x.shape[1] and np.allclose(x[:, 0], 1.0):
        beta[0] = np.log(mean) - float(np.mean(offset))
    for _ in range(50):
        eta = x @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta - offset + (y - mu) / mu
        xw = x * mu[:, None]
        new = np.linalg.lstsq(xw.T @ x, xw.T @ z, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    return beta


def fit_count_glmm(y, x, z, group, offset, *, family: str, n_agq: int = 1) -> LaplaceResult:
    """Fit a Poisson or NB2 GLMM by Laplace / adaptive Gauss–Hermite ML."""
    if family not in COUNT_FAMILIES:
        raise ValueError(f"family must be one of {COUNT_FAMILIES}")
    y = np.asarray(y, dtype=float)
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("count outcomes must be finite and non-negative")
    if n_agq > 1 and np.asarray(z).shape[1] != 1:
        raise ValueError("n_agq > 1 needs a single random effect, for example (1 | g)")
    prob = _Problem(y, x, z, group, offset, family, n_agq)
    beta0 = _glm_start(prob.y, prob.x, prob.offset)
    phi0 = np.concatenate([beta0, np.zeros(prob.n_l)])
    if family == "negative_binomial":
        mu = np.exp(prob.x @ beta0 + prob.offset)
        extra = float(np.mean((prob.y - mu) ** 2 - mu))
        theta0 = float(np.clip(np.mean(mu**2) / extra, 0.1, 100.0)) if extra > 0 else 10.0
        phi0 = np.append(phi0, np.log(theta0))

    def objective(phi):
        return -prob.loglik(phi)

    bounds = [(None, None)] * prob.n_par
    for i in range(prob.q):
        bounds[prob.p + i] = (_LOG_SD_FLOOR, None)
    first = optimize.minimize(objective, phi0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-9})
    phi = first.x
    n_iter = int(first.nit)

    # A random-effect SD that L-BFGS drives toward zero is a singular fit, as
    # glmer reports it. Pin it at the floor when that does not lower the
    # likelihood, and polish the rest with Newton steps.
    free = np.ones(prob.n_par, dtype=bool)
    for i in range(prob.q):
        at = prob.p + i
        if phi[at] < -8.0:
            pinned = phi.copy()
            pinned[at] = _LOG_SD_FLOOR
            if objective(pinned) <= objective(phi) + 1e-8:
                phi = pinned
        if phi[at] <= _LOG_SD_FLOOR + 1e-6:
            free[at] = False
    converged = bool(first.success)

    def sub(fun_full):
        def fun(sub_phi):
            full = phi.copy()
            full[free] = sub_phi
            return fun_full(full)

        return fun

    for _ in range(30):
        f_sub = sub(objective)
        h_grad = 1e-5 * np.maximum(1.0, np.abs(phi[free]))
        grad = _gradient(f_sub, phi[free], h_grad)
        hess = _hessian(f_sub, phi[free], 1e-3 * np.maximum(1.0, np.abs(phi[free])))
        # At the optimum the line search can fail on numerical noise alone, so
        # a small gradient counts as converged.
        converged = converged or bool(np.max(np.abs(grad)) < 1e-3)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        f_now = objective(phi)
        scale = 1.0
        while scale > 1e-4:
            trial = phi.copy()
            trial[free] = phi[free] - scale * step
            if objective(trial) <= f_now + 1e-10:
                break
            scale *= 0.5
        else:
            break
        phi = trial
        n_iter += 1
        if np.max(np.abs(scale * step)) < 1e-9:
            converged = True
            break

    f_sub = sub(objective)
    hess = _hessian(f_sub, phi[free], 1e-3 * np.maximum(1.0, np.abs(phi[free])))
    cov_free = np.linalg.pinv(hess)
    cov_full = np.full((prob.n_par, prob.n_par), np.nan)
    idx = np.flatnonzero(free)
    cov_full[np.ix_(idx, idx)] = cov_free
    beta, L, log_theta = prob.unpack(phi)
    ll = prob.loglik(phi)
    theta = theta_se = None
    if log_theta is not None:
        theta = float(np.exp(log_theta))
        theta_se = float(theta * np.sqrt(max(cov_full[-1, -1], 0.0)))
    return LaplaceResult(
        coefficients=np.asarray(beta, dtype=float),
        covariance=cov_full[: prob.p, : prob.p],
        group_covariance=L @ L.T,
        theta=theta,
        theta_std_error=theta_se,
        log_likelihood=ll,
        modes=prob.u @ L.T,
        converged=converged,
        n_iter=n_iter,
        function_evals=prob.evals,
    )
