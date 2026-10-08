"""Laplace and adaptive Gauss–Hermite GLMMs for binary and count outcomes.

One grouping factor with correlated random effects ``b_j = L u_j``,
``u_j ~ N(0, I)``. The conditional modes come from a Newton solve per group.
The marginal log-likelihood is the Laplace approximation (``n_agq=1``) or
adaptive Gauss–Hermite quadrature for a single random intercept. Both match
the objective of ``lme4::glmer`` (binomial, Poisson) and ``glmmTMB``
(``binomial``, ``poisson``, ``nbinom2``, zero-inflated and hurdle models).

Count models may add a zero part with fixed effects only:

- ``zero="inflated"``: a structural zero with probability ``pi``, else the
  count distribution (glmmTMB ``ziformula`` with ``poisson`` / ``nbinom2``).
- ``zero="hurdle"``: zero with probability ``pi``, else the zero-truncated
  count distribution (glmmTMB ``ziformula`` with ``truncated_poisson`` /
  ``truncated_nbinom2``).

The outer optimisation runs over ``(beta, log diag L, offdiag L, log theta,
gamma)``. The covariance is the inverse Hessian of the negative marginal
log-likelihood over all parameters, as in glmer (``use.hessian=TRUE``) and
glmmTMB. The Laplace term uses the exact Hessian in the random effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse, special
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg

COUNT_FAMILIES = ("poisson", "negative_binomial")
LAPLACE_FAMILIES = ("binomial", *COUNT_FAMILIES)
ZERO_PARTS = ("inflated", "hurdle")
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
    zero_coefficients: np.ndarray | None = None
    zero_covariance: np.ndarray | None = None


class _Problem:
    def __init__(self, y, x, z, group, offset, family, n_agq, zero, zx):
        self.y = np.asarray(y, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.group = np.asarray(group, dtype=np.int64)
        self.offset = np.zeros(len(self.y)) if offset is None else np.asarray(offset, dtype=float)
        self.family = family
        self.n_agq = int(n_agq)
        self.zero = zero
        self.zx = None if zx is None else np.asarray(zx, dtype=float)
        self.n, self.p = self.x.shape
        self.q = self.z.shape[1]
        self.m = int(self.group.max()) + 1
        self.n_l = self.q * (self.q + 1) // 2
        self.n_theta = 1 if family == "negative_binomial" else 0
        self.r = 0 if self.zx is None else self.zx.shape[1]
        self.n_par = self.p + self.n_l + self.n_theta + self.r
        self.lgamma_y1 = special.gammaln(self.y + 1.0)
        self.is_zero = self.y == 0
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
        at = self.p + self.n_l
        log_theta = phi[at] if self.n_theta else None
        gamma = phi[at + self.n_theta :] if self.r else None
        return beta, L, log_theta, gamma

    # family -----------------------------------------------------------
    def _count(self, eta, log_theta):
        """Count log-density, its eta derivatives, and log f(0) with derivatives."""
        y = self.y
        if self.family == "poisson":
            mu = np.exp(eta)
            ll = y * eta - mu - self.lgamma_y1
            return ll, y - mu, -mu, -mu, -mu, -mu
        theta = np.exp(log_theta)
        if self._const_key != log_theta:
            # The gamma-function terms do not depend on eta; compute them once per theta.
            self._const = special.gammaln(y + theta) - special.gammaln(theta) - self.lgamma_y1 + theta * log_theta
            self._const_key = log_theta
        log_tm = np.logaddexp(log_theta, eta)
        ratio = np.exp(eta - log_tm)  # mu / (theta + mu)
        ll = self._const + y * eta - (y + theta) * log_tm
        d1 = y - (y + theta) * ratio
        d2 = -(y + theta) * ratio * (1.0 - ratio)
        c = theta * (log_theta - log_tm)  # log f(0)
        c1 = -theta * ratio
        c2 = -theta * ratio * (1.0 - ratio)
        return ll, d1, d2, c, c1, c2

    def _terms(self, eta, log_theta, zeta):
        """Per-row log-likelihood, first derivative, and minus the second (eta)."""
        if self.family == "binomial":
            p = special.expit(eta)
            ll = self.y * eta - np.logaddexp(0.0, eta)
            return ll, self.y - p, p * (1.0 - p)
        ll, d1, d2, c, c1, c2 = self._count(eta, log_theta)
        if self.zero is None:
            return ll, d1, -d2
        log_pi = -np.logaddexp(0.0, -zeta)
        log_1mpi = -np.logaddexp(0.0, zeta)
        zero = self.is_zero
        if self.zero == "inflated":
            # y = 0: log(pi + (1 - pi) f(0)); y > 0: log(1 - pi) + log f(y).
            mix = np.logaddexp(log_pi, log_1mpi + c)
            share = np.exp(log_1mpi + c - mix)  # P(count zero | y = 0)
            out_ll = np.where(zero, mix, log_1mpi + ll)
            out_d1 = np.where(zero, share * c1, d1)
            out_d2 = np.where(zero, share * c2 + share * (1.0 - share) * c1**2, d2)
            return out_ll, out_d1, -out_d2
        # Hurdle: y = 0: log pi; y > 0: log(1 - pi) + log f(y) - log(1 - f(0)).
        log_pos = np.log(-np.expm1(np.minimum(c, -1e-300)))
        s = np.exp(c - log_pos)  # f(0) / (1 - f(0))
        out_ll = np.where(zero, log_pi, log_1mpi + ll - log_pos)
        out_d1 = np.where(zero, 0.0, d1 + s * c1)
        out_d2 = np.where(zero, 0.0, d2 + s * c2 + s * (1.0 + s) * c1**2)
        return out_ll, out_d1, -out_d2

    def _by_group(self, values):
        flat = values.reshape(len(values), -1)
        out = np.empty((self.m, flat.shape[1]))
        for k in range(flat.shape[1]):
            out[:, k] = np.bincount(self.group, weights=flat[:, k], minlength=self.m)
        return out.reshape((self.m,) + values.shape[1:])

    def _group_ll(self, eta, log_theta, zeta):
        ll, _, _ = self._terms(eta, log_theta, zeta)
        return np.bincount(self.group, weights=ll, minlength=self.m)

    # inner Newton -----------------------------------------------------
    def modes(self, beta, L, log_theta, zeta, u0):
        eta0 = self.x @ beta + self.offset
        zl = self.z @ L
        u = u0.copy()
        eye = np.eye(self.q)
        for _ in range(100):
            eta = eta0 + np.einsum("ij,ij->i", zl, u[self.group])
            ll, d1, w = self._terms(eta, log_theta, zeta)
            f_old = np.bincount(self.group, weights=ll, minlength=self.m) - 0.5 * np.sum(u * u, axis=1)
            grad = self._by_group(zl * d1[:, None]) - u
            # A zero-inflated row can have negative curvature; Newton uses its
            # positive part, which still ascends. The Laplace term below keeps
            # the exact Hessian.
            w_step = np.maximum(w, 0.0)
            hess = self._by_group(w_step[:, None, None] * zl[:, :, None] * zl[:, None, :]) + eye
            step = np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
            scale = np.ones(self.m)
            for _half in range(40):
                trial = u + scale[:, None] * step
                eta_t = eta0 + np.einsum("ij,ij->i", zl, trial[self.group])
                f_new = self._group_ll(eta_t, log_theta, zeta) - 0.5 * np.sum(trial * trial, axis=1)
                bad = ~(f_new >= f_old - 1e-12 * (1.0 + np.abs(f_old)))
                if not bad.any():
                    break
                scale[bad] *= 0.5
            u = trial
            if np.max(np.abs(scale[:, None] * step)) < 1e-11:
                break
        eta = eta0 + np.einsum("ij,ij->i", zl, u[self.group])
        ll, _, w = self._terms(eta, log_theta, zeta)
        hess = self._by_group(w[:, None, None] * zl[:, :, None] * zl[:, None, :]) + eye
        f = np.bincount(self.group, weights=ll, minlength=self.m) - 0.5 * np.sum(u * u, axis=1)
        return u, f, hess, eta0, zl

    def loglik(self, phi):
        self.evals += 1
        beta, L, log_theta, gamma = self.unpack(phi)
        zeta = None if gamma is None else self.zx @ gamma
        u, f, hess, eta0, zl = self.modes(beta, L, log_theta, zeta, self.u)
        self.u = u
        if self.n_agq <= 1 or self.q != 1:
            sign, logdet = np.linalg.slogdet(hess)
            if np.any(sign <= 0):
                return -np.inf
            return float(np.sum(f - 0.5 * logdet))
        # Adaptive Gauss–Hermite for a scalar random effect.
        h = hess[:, 0, 0]
        if np.any(h <= 0):
            return -np.inf
        s = np.sqrt(2.0 / h)
        z1 = zl[:, 0]
        total = np.zeros((self.m, self.n_agq))
        for k, node in enumerate(self.gh_nodes):
            uk = u[:, 0] + s * node
            eta = eta0 + z1 * uk[self.group]
            total[:, k] = self._group_ll(eta, log_theta, zeta) - 0.5 * uk**2 + self.gh_logw[k]
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


def _maximize(prob, phi0, sd_index):
    """Maximise ``prob.loglik``; return the optimum, its inverse Hessian, and status.

    ``sd_index`` lists the log-SD parameters that may hit the singular floor.
    """
    def objective(phi):
        value = -prob.loglik(phi)
        return value if np.isfinite(value) else 1e300

    bounds = [(None, None)] * prob.n_par
    for at in sd_index:
        bounds[at] = (_LOG_SD_FLOOR, None)
    first = optimize.minimize(objective, phi0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-9})
    phi = first.x
    n_iter = int(first.nit)

    # A random-effect SD that L-BFGS drives toward zero is a singular fit, as
    # glmer reports it. Pin it at the floor when that does not lower the
    # likelihood, and polish the rest with Newton steps.
    free = np.ones(prob.n_par, dtype=bool)
    for at in sd_index:
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
    return phi, cov_full, converged, n_iter


def _glm_start(y, x, offset, family):
    """IRLS without random effects (logit or log link), for starting values."""
    beta = np.zeros(x.shape[1])
    if x.shape[1] and np.allclose(x[:, 0], 1.0):
        if family == "binomial":
            mean = float(np.clip(np.mean(y), 0.01, 0.99))
            beta[0] = np.log(mean / (1 - mean)) - float(np.mean(offset))
        else:
            beta[0] = np.log(max(float(np.mean(y)), 1e-3)) - float(np.mean(offset))
    for _ in range(50):
        eta = np.clip(x @ beta + offset, -30, 30)
        if family == "binomial":
            mu = special.expit(eta)
            w = np.clip(mu * (1 - mu), 1e-10, None)
        else:
            mu = np.exp(eta)
            w = np.clip(mu, 1e-10, None)
        z = eta - offset + (y - mu) / w
        xw = x * w[:, None]
        new = np.linalg.lstsq(xw.T @ x, xw.T @ z, rcond=None)[0]
        if not np.all(np.isfinite(new)):
            break
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    return beta


def _check(y, family, zero, zero_x):
    if family not in LAPLACE_FAMILIES:
        raise ValueError(f"family must be one of {LAPLACE_FAMILIES}")
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("the outcome must be finite")
    if family == "binomial":
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("a binomial GLMM needs a 0/1 outcome")
    elif np.any(y < 0) or np.any(y != np.round(y)):
        raise ValueError("count outcomes must be non-negative whole numbers")
    if zero is not None:
        if zero not in ZERO_PARTS:
            raise ValueError(f"zero must be one of {ZERO_PARTS}")
        if family not in COUNT_FAMILIES:
            raise ValueError("zero-inflated and hurdle models need the Poisson or negative binomial family")
        if zero_x is None:
            zero_x = np.ones((len(y), 1))
        if zero == "hurdle" and not np.any(y > 0):
            raise ValueError("a hurdle model needs some positive counts")
    elif zero_x is not None:
        raise ValueError("zero_x needs zero='inflated' or 'hurdle'")
    return y, zero_x


def _start(prob):
    """Starting values: a GLM for beta, unit SDs, a moment theta, the zero share."""
    zero, family = prob.zero, prob.family
    fit_rows = prob.y > 0 if zero == "hurdle" else np.ones(prob.n, dtype=bool)
    beta0 = _glm_start(prob.y[fit_rows], prob.x[fit_rows], prob.offset[fit_rows], family)
    phi0 = np.concatenate([beta0, np.zeros(prob.n_l)])
    if family == "negative_binomial":
        mu = np.exp(prob.x @ beta0 + prob.offset)
        extra = float(np.mean((prob.y - mu) ** 2 - mu))
        theta0 = float(np.clip(np.mean(mu**2) / extra, 0.1, 100.0)) if extra > 0 else 10.0
        phi0 = np.append(phi0, np.log(theta0))
    if prob.r:
        gamma0 = np.zeros(prob.r)
        zero_share = float(np.clip(np.mean(prob.is_zero), 0.02, 0.98))
        start = zero_share if zero == "hurdle" else min(zero_share, 0.2)
        if np.allclose(prob.zx[:, 0], 1.0):
            gamma0[0] = np.log(start / (1 - start))
        phi0 = np.append(phi0, gamma0)
    return phi0


def fit_laplace_glmm(
    y,
    x,
    z,
    group,
    offset,
    *,
    family: str,
    n_agq: int = 1,
    zero: str | None = None,
    zero_x=None,
) -> LaplaceResult:
    """Fit a binomial, Poisson, or NB2 GLMM by Laplace / adaptive Gauss–Hermite ML."""
    y, zero_x = _check(y, family, zero, zero_x)
    if n_agq > 1 and np.asarray(z).shape[1] != 1:
        raise ValueError("n_agq > 1 needs a single random effect, for example (1 | g)")
    prob = _Problem(y, x, z, group, offset, family, n_agq, zero, zero_x)
    phi0 = _start(prob)

    phi, cov_full, converged, n_iter = _maximize(prob, phi0, [prob.p + i for i in range(prob.q)])
    beta, L, log_theta, gamma = prob.unpack(phi)
    ll = prob.loglik(phi)
    theta = theta_se = None
    at = prob.p + prob.n_l
    if log_theta is not None:
        theta = float(np.exp(log_theta))
        theta_se = float(theta * np.sqrt(max(cov_full[at, at], 0.0)))
    zero_cov = None
    if gamma is not None:
        g0 = at + prob.n_theta
        zero_cov = cov_full[g0:, g0:]
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
        zero_coefficients=None if gamma is None else np.asarray(gamma, dtype=float),
        zero_covariance=zero_cov,
    )


def fit_count_glmm(y, x, z, group, offset, *, family: str, n_agq: int = 1) -> LaplaceResult:
    """Fit a Poisson or NB2 GLMM. Kept for callers of the first version."""
    if family not in COUNT_FAMILIES:
        raise ValueError(f"family must be one of {COUNT_FAMILIES}")
    return fit_laplace_glmm(y, x, z, group, offset, family=family, n_agq=n_agq)


# Several random-effect terms -------------------------------------------------
#
# Crossed or nested grouping factors, and structured covariances, share one
# vector of spherical effects u ~ N(0, I). The linear predictor is
# eta = X beta + offset + A(L) u, with A sparse. The conditional modes come
# from a joint Newton solve on the sparse system A' W A + I, as in lme4's
# PIRLS and TMB's sparse Laplace.

STRUCTURES = ("us", "ar1")
# Largest Schur complement (in random effects outside the biggest term) that
# is factored densely; beyond it the whole system goes to sparse LU.
_SCHUR_MAX = 3000


@dataclass
class RandomTerm:
    """One random-effect term: ``z`` columns within the levels of ``group``.

    ``structure`` is ``"us"`` (unstructured, a Cholesky factor) or ``"ar1"``
    (``sigma^2 rho^|i - j|`` over the ``z`` columns, which are the levels of
    a time factor in order, as glmmTMB's ``ar1(time + 0 | group)``).
    """

    z: np.ndarray
    group: np.ndarray
    n_levels: int
    structure: str = "us"

    @property
    def q(self) -> int:
        return self.z.shape[1]

    @property
    def n_par(self) -> int:
        return 2 if self.structure == "ar1" else self.q * (self.q + 1) // 2

    @property
    def sd_slots(self) -> list[int]:
        """Positions of the log-SD parameters within this term's parameters."""
        return [0] if self.structure == "ar1" else list(range(self.q))

    def factor(self, par: np.ndarray) -> np.ndarray:
        q = self.q
        if self.structure == "ar1":
            sd = np.exp(par[0])
            rho = np.tanh(par[1])
            idx = np.arange(q)
            lag = idx[:, None] - idx[None, :]
            L = np.where(lag >= 0, rho ** np.clip(lag, 0, None), 0.0)
            L[1:, 1:] *= np.sqrt(1.0 - rho * rho)
            return sd * L
        L = np.zeros((q, q))
        L[np.diag_indices(q)] = np.exp(par[:q])
        if q > 1:
            L[np.tril_indices(q, -1)] = par[q:]
        return L

    def correlation(self, par: np.ndarray) -> float | None:
        return float(np.tanh(par[1])) if self.structure == "ar1" else None


class _SparseProblem(_Problem):
    def __init__(self, y, x, terms, offset, family, zero, zx):
        self.terms = list(terms)
        super().__init__(y, x, np.ones((len(y), 1)), np.zeros(len(y), dtype=np.int64), offset, family, 1, zero, zx)
        self.n_l = sum(t.n_par for t in self.terms)
        self.n_par = self.p + self.n_l + self.n_theta + self.r
        # Row i of A has nonzeros in the columns cols[i]; their values are
        # (z_k L_k)[i] for each term k, stacked in the same order.
        cols, self.u_base = [], []
        base = 0
        for t in self.terms:
            self.u_base.append(base)
            cols.append(base + t.group[:, None] * t.q + np.arange(t.q)[None, :])
            base += t.n_levels * t.q
        self.n_u = base
        self.cols = np.hstack(cols)
        width = self.cols.shape[1]
        # Fixed CSC pattern of A' W A + I; each row adds width^2 products.
        left = np.repeat(self.cols, width, axis=1)
        right = np.tile(self.cols, (1, width))
        diag = np.arange(self.n_u)
        keys = np.concatenate([(right * self.n_u + left).ravel(), diag * self.n_u + diag])
        unique, inverse = np.unique(keys, return_inverse=True)
        self.pair_pos = inverse[: self.n * width * width]
        self.diag_pos = inverse[self.n * width * width :]
        self.h_rows = (unique % self.n_u).astype(np.int32)
        self.h_cols = unique // self.n_u
        self.h_indptr = np.searchsorted(self.h_cols, np.arange(self.n_u + 1)).astype(np.int32)
        self.h_nnz = len(unique)
        self.u = np.zeros(self.n_u)
        self._setup_blocks()

    def _setup_blocks(self):
        """Split u into the largest term (block diagonal in H) and the rest.

        Each row touches one level of every term, so the largest term's
        block of H is block diagonal. Eliminating it leaves a dense Schur
        complement over the other terms, which stays small when the other
        grouping factors have few levels (examiners, sites, years).
        """
        sizes = [t.n_levels * t.q for t in self.terms]
        big = int(np.argmax(sizes))
        self.big = big
        self.n_rest = self.n_u - sizes[big]
        self.use_blocks = len(self.terms) > 1 and self.n_rest <= _SCHUR_MAX
        if not self.use_blocks:
            return
        widths = [t.q for t in self.terms]
        starts = np.concatenate([[0], np.cumsum(widths)])
        self.big_slice = slice(starts[big], starts[big + 1])
        rest = [k for k in range(len(self.terms)) if k != big]
        self.rest_slices = [slice(starts[k], starts[k + 1]) for k in rest]
        big_base = self.u_base[big]
        # Positions of u in [big block | rest block] order.
        in_big = np.zeros(self.n_u, dtype=bool)
        in_big[big_base : big_base + sizes[big]] = True
        self.big_idx = np.flatnonzero(in_big)
        self.rest_idx = np.flatnonzero(~in_big)
        rest_pos = np.full(self.n_u, -1)
        rest_pos[self.rest_idx] = np.arange(self.n_rest)
        self.rest_cols = np.hstack([rest_pos[self.cols[:, sl]] for sl in self.rest_slices])
        w_r = self.rest_cols.shape[1]
        self.rr_key = (self.rest_cols[:, :, None] * self.n_rest + self.rest_cols[:, None, :]).ravel()
        tb = self.terms[big]
        self.big_q = tb.q
        self.big_levels = tb.n_levels
        self.big_group = tb.group
        rows_b = (tb.group[:, None] * tb.q + np.arange(tb.q)[None, :])
        self.br_rows = np.repeat(rows_b, w_r, axis=1).ravel()
        self.br_cols = np.tile(self.rest_cols, (1, tb.q)).ravel()

    def unpack(self, phi):
        beta = phi[: self.p]
        factors = []
        at = self.p
        for t in self.terms:
            factors.append(t.factor(phi[at : at + t.n_par]))
            at += t.n_par
        log_theta = phi[at] if self.n_theta else None
        gamma = phi[at + self.n_theta :] if self.r else None
        return beta, factors, log_theta, gamma

    def term_params(self, phi):
        at = self.p
        out = []
        for t in self.terms:
            out.append(phi[at : at + t.n_par])
            at += t.n_par
        return out

    def sd_index(self) -> list[int]:
        at = self.p
        out = []
        for t in self.terms:
            out.extend(at + k for k in t.sd_slots)
            at += t.n_par
        return out

    def _factorize(self, vals, w):
        if self.use_blocks:
            return _BlockSystem(self, vals, w)
        products = (w[:, None, None] * vals[:, :, None] * vals[:, None, :]).ravel()
        data = np.bincount(self.pair_pos, weights=products, minlength=self.h_nnz)
        data[self.diag_pos] += 1.0
        hess = sparse.csc_matrix((data, self.h_rows, self.h_indptr), shape=(self.n_u, self.n_u))
        try:
            return linalg.splu(
                hess, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True}
            )
        except RuntimeError:
            return None

    def modes(self, beta, factors, log_theta, zeta, u0):
        """Joint Newton solve; returns the modes, the log-joint, and log det(H)."""
        eta0 = self.x @ beta + self.offset
        vals = np.hstack([t.z @ L for t, L in zip(self.terms, factors, strict=True)])
        cols = self.cols

        def value(v):
            ll, _, _ = self._terms(eta0 + np.einsum("ij,ij->i", vals, v[cols]), log_theta, zeta)
            return float(np.sum(ll) - 0.5 * v @ v)

        u = u0.copy()
        f_old = value(u)
        for _ in range(100):
            _, d1, w = self._terms(eta0 + np.einsum("ij,ij->i", vals, u[cols]), log_theta, zeta)
            grad = np.bincount(cols.ravel(), weights=(vals * d1[:, None]).ravel(), minlength=self.n_u) - u
            # Zero-inflated rows can curve the wrong way; Newton uses the
            # positive part, and the Laplace term below the exact curvature.
            lu = self._factorize(vals, np.maximum(w, 0.0))
            if lu is None or getattr(lu, "ok", True) is False:
                return u, f_old, None
            step = lu.solve(grad)
            scale = 1.0
            for _half in range(40):
                trial = u + scale * step
                f_new = value(trial)
                if f_new >= f_old - 1e-12 * (1.0 + abs(f_old)):
                    break
                scale *= 0.5
            u, f_old = trial, f_new
            if np.max(np.abs(scale * step)) < 1e-11:
                break
        _, _, w = self._terms(eta0 + np.einsum("ij,ij->i", vals, u[cols]), log_theta, zeta)
        return u, f_old, _logdet_lu(self._factorize(vals, w))

    def loglik(self, phi):
        self.evals += 1
        beta, factors, log_theta, gamma = self.unpack(phi)
        zeta = None if gamma is None else self.zx @ gamma
        u, f, logdet = self.modes(beta, factors, log_theta, zeta, self.u)
        if logdet is None:
            return -np.inf
        self.u = u
        return float(f - 0.5 * logdet)


class _BlockSystem:
    """Factor of H = A' W A + I by eliminating the largest term's blocks."""

    def __init__(self, prob: _SparseProblem, vals: np.ndarray, w: np.ndarray):
        self.prob = prob
        self.ok = False
        q, m = prob.big_q, prob.big_levels
        vb = vals[:, prob.big_slice]
        vr = np.hstack([vals[:, sl] for sl in prob.rest_slices])
        d = np.zeros((m, q, q))
        np.add.at(d, prob.big_group, w[:, None, None] * vb[:, :, None] * vb[:, None, :])
        d += np.eye(q)
        sign, logdet_d = np.linalg.slogdet(d)
        if np.any(sign <= 0):
            return
        self.d_inv = np.linalg.inv(d)
        n_r = prob.n_rest
        h_rr = np.bincount(
            prob.rr_key, weights=(w[:, None, None] * vr[:, :, None] * vr[:, None, :]).ravel(), minlength=n_r * n_r
        ).reshape(n_r, n_r)
        h_rr += np.eye(n_r)
        br_vals = (w[:, None, None] * vb[:, :, None] * vr[:, None, :]).ravel()
        self.h_br = sparse.csr_matrix((br_vals, (prob.br_rows, prob.br_cols)), shape=(m * q, n_r))
        d_inv_sparse = sparse.block_diag(list(self.d_inv), format="csr") if q > 1 else sparse.diags(self.d_inv[:, 0, 0])
        self.d_inv_br = d_inv_sparse @ self.h_br
        schur = h_rr - (self.h_br.T @ self.d_inv_br).toarray()
        try:
            self.chol = scipy_linalg.cho_factor(schur, lower=True)
        except np.linalg.LinAlgError:
            return
        self.logdet = float(np.sum(logdet_d) + 2.0 * np.sum(np.log(np.diag(self.chol[0]))))
        self.ok = True

    def _apply_d_inv(self, g):
        q = self.prob.big_q
        return np.einsum("mij,mj->mi", self.d_inv, g.reshape(-1, q)).ravel()

    def solve(self, grad):
        prob = self.prob
        g_b = grad[prob.big_idx]
        g_r = grad[prob.rest_idx]
        x_r = scipy_linalg.cho_solve(self.chol, g_r - self.h_br.T @ self._apply_d_inv(g_b))
        x_b = self._apply_d_inv(g_b - self.h_br @ x_r)
        out = np.empty_like(grad)
        out[prob.big_idx] = x_b
        out[prob.rest_idx] = x_r
        return out


def _logdet_lu(lu) -> float | None:
    """log det of a symmetric positive definite matrix from its factor."""
    if lu is None:
        return None
    if isinstance(lu, _BlockSystem):
        return lu.logdet if lu.ok else None
    diag = lu.U.diagonal()
    if np.any(diag <= 0):
        return None
    return float(np.sum(np.log(diag)))


@dataclass
class TermsResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    group_covariances: list[np.ndarray]
    correlations: list[float | None]
    theta: float | None
    theta_std_error: float | None
    log_likelihood: float
    modes: list[np.ndarray]
    converged: bool
    n_iter: int
    function_evals: int
    zero_coefficients: np.ndarray | None = None
    zero_covariance: np.ndarray | None = None


def fit_laplace_terms(
    y,
    x,
    terms: list[RandomTerm],
    offset,
    *,
    family: str,
    zero: str | None = None,
    zero_x=None,
) -> TermsResult:
    """Laplace ML for several random-effect terms (crossed, nested, ar1).

    The modes come from a joint sparse Newton solve. Matches glmmTMB's
    Laplace fit, including ``ar1(time + 0 | group)``.
    """
    y, zero_x = _check(y, family, zero, zero_x)
    for t in terms:
        if t.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}")
        if t.structure == "ar1" and t.q < 2:
            raise ValueError("ar1 needs a time factor with at least two levels")
    prob = _SparseProblem(y, x, terms, offset, family, zero, zero_x)
    phi0 = _start(prob)
    phi, cov_full, converged, n_iter = _maximize(prob, phi0, prob.sd_index())
    beta, factors, log_theta, gamma = prob.unpack(phi)
    ll = prob.loglik(phi)
    theta = theta_se = None
    at = prob.p + prob.n_l
    if log_theta is not None:
        theta = float(np.exp(log_theta))
        theta_se = float(theta * np.sqrt(max(cov_full[at, at], 0.0)))
    modes = []
    for t, base, L in zip(prob.terms, prob.u_base, factors, strict=True):
        u = prob.u[base : base + t.n_levels * t.q].reshape(t.n_levels, t.q)
        modes.append(u @ L.T)
    return TermsResult(
        coefficients=np.asarray(beta, dtype=float),
        covariance=cov_full[: prob.p, : prob.p],
        group_covariances=[L @ L.T for L in factors],
        correlations=[t.correlation(par) for t, par in zip(prob.terms, prob.term_params(phi), strict=True)],
        theta=theta,
        theta_std_error=theta_se,
        log_likelihood=ll,
        modes=modes,
        converged=converged,
        n_iter=n_iter,
        function_evals=prob.evals,
        zero_coefficients=None if gamma is None else np.asarray(gamma, dtype=float),
        zero_covariance=None if gamma is None else cov_full[at + prob.n_theta :, at + prob.n_theta :],
    )
