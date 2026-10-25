"""Cox models with Gaussian random effects (frailty), compared with ``coxme``.

For fixed variances the fixed effects and the random effects maximize the
penalized partial likelihood together. The variances maximize its Laplace
approximation, the integrated partial likelihood (Ripatti & Palmgren 2000).
Random effects are spherical, ``b = Lambda u`` with ``u ~ N(0, I)``, so a
variance of zero needs no special case. Design note:
``dev-notes/design/cox-mixed.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize, sparse, stats

from ..models.design import ColumnRef, Design, build_design, column_series
from ..models.fit import Fit
from ..models.formula import group_labels
from ..models.mixed import _effect_columns, _terms_variance_table
from .spec import parse_survival_formula

_MAX_NEWTON = 60
# Newton decrement (g' H^-1 g) at which the inner problem counts as solved.
_NEWTON_TOL = 1e-14
_START_SD = 0.3
_FD_STEP = 1e-5


@dataclass
class CoxMixedFit:
    """Mixed-effects Cox model.

    ``coefficients`` and ``covariance`` are the fixed effects; the covariance
    is the fixed-effect block of the inverse penalized information, as
    ``vcov(coxme)``. ``log_likelihood`` is the integrated (Laplace) partial
    log-likelihood. ``log_likelihood_penalized`` subtracts the random-effect
    penalty from ``log_likelihood_fitted``, the partial log-likelihood at the
    estimates. ``df`` is the effective number of parameters of the penalized
    fit.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    n_events: int
    log_likelihood: float
    log_likelihood_null: float
    log_likelihood_penalized: float
    log_likelihood_fitted: float
    df: float
    n_variance_parameters: int
    random_terms: list[dict[str, Any]]
    theta: np.ndarray
    converged: bool
    n_iter: int
    function_evals: int
    ties: str
    design: Design
    x: np.ndarray
    row_index: np.ndarray
    time: np.ndarray
    event: np.ndarray
    strata: np.ndarray
    entry: np.ndarray
    weights: np.ndarray
    offset: np.ndarray
    formula: str | None = None
    family: str = "cox"
    residual_df: int | None = None
    _z: Any = field(default=None, repr=False)
    _blups: np.ndarray | None = field(default=None, repr=False)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Fixed-effect table with Wald z tests, as the ``coxme`` print.

        With ``exponentiate=True``, ``estimate``, ``conf_low``, and
        ``conf_high`` are hazard ratios.
        """
        return Fit.tidy(self, level=level, exponentiate=exponentiate, df=np.inf)

    def variance_table(self) -> pl.DataFrame:
        """Random-effect variances, one row per term and column, as ``VarCorr``.

        ``correlation`` is the correlation with the intercept of the same group.
        """
        return _terms_variance_table(self.random_terms)

    def random_effects(self) -> pl.DataFrame:
        """Conditional modes (BLUPs): ``group``, ``level``, ``term``, ``blup``, as ``ranef``."""
        frames = []
        for term in self.random_terms:
            for k, name in enumerate(term["names"]):
                frames.append(
                    pl.DataFrame(
                        {"group": term["group"], "level": list(term["levels"]), "term": name, "blup": term["blups"][:, k]},
                        schema={"group": pl.String, "level": pl.String, "term": pl.String, "blup": pl.Float64},
                    )
                )
        return pl.concat(frames)

    def likelihood_table(self) -> pl.DataFrame:
        """Likelihood-ratio tests against the null model, as the ``coxme`` print.

        ``integrated`` uses the Laplace log-likelihood with one degree of
        freedom per fixed effect and variance parameter. ``penalized`` uses
        the partial log-likelihood at the estimates
        (``log_likelihood_fitted``) and ``df``, as ``coxme`` does. ``aic`` and ``bic`` follow
        the print: ``chisq - 2 df`` and ``chisq - log(events) df``, so larger
        is better. :meth:`glance` has the usual ``-2 log L`` forms.
        """
        rows = []
        p = len(self.coefficients)
        for label, ll, df in (
            ("integrated", self.log_likelihood, float(p + self.n_variance_parameters)),
            ("penalized", self.log_likelihood_fitted, self.df),
        ):
            chisq = 2 * (ll - self.log_likelihood_null)
            rows.append(
                {
                    "likelihood": label,
                    "chisq": chisq,
                    "df": df,
                    "p_value": float(stats.chi2.sf(chisq, df)),
                    "aic": chisq - 2 * df,
                    "bic": chisq - np.log(self.n_events) * df,
                }
            )
        return pl.DataFrame(rows)

    def glance(self) -> pl.DataFrame:
        """One-row summary. AIC and BIC use the integrated log-likelihood and events for BIC."""
        k = len(self.coefficients) + self.n_variance_parameters
        ll = self.log_likelihood
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "n_events": [self.n_events],
                "n_parameters": [k],
                "log_likelihood": [ll],
                "log_likelihood_null": [self.log_likelihood_null],
                "log_likelihood_penalized": [self.log_likelihood_penalized],
                "df_penalized": [self.df],
                "aic": [-2 * ll + 2 * k],
                "bic": [-2 * ll + np.log(self.n_events) * k],
            }
        )

    def predict(
        self, data: pl.DataFrame | None = None, *, kind: str = "linear_predictor", random: bool = True
    ) -> np.ndarray:
        """Linear predictor or relative risk.

        ``random=True`` adds each row's BLUPs; levels not in the fit get zero.
        ``random=False`` uses the fixed effects only. Offsets are included.
        """
        if kind not in {"linear_predictor", "risk"}:
            raise ValueError("kind must be linear_predictor or risk")
        if data is None:
            lp = self.x @ self.coefficients + self.offset
            if random:
                lp = lp + self._z @ self._blups
        else:
            design = build_design(data, self.design)
            lp = design.x @ self.coefficients
            if random:
                for term in self.random_terms:
                    lp = lp + _term_contribution(data, design, term)
        return lp if kind == "linear_predictor" else np.exp(lp)


def cox_mixed(
    data: pl.DataFrame,
    formula: str,
    *,
    weights: ColumnRef | None = None,
    ties: str = "efron",
    variances: dict[str, Any] | Sequence[Any] | None = None,
) -> CoxMixedFit:
    """Fit a Cox model with Gaussian random effects, as ``coxme::coxme``.

    ``formula`` is a ``Surv`` formula with one or more random-effect terms:
    ``"Surv(time, status) ~ age + (1 | hospital)"``. Terms may be crossed
    (``(1 | hospital) + (1 | surgeon)``), nested (``(1 | hospital/ward)``),
    or have correlated slopes (``(1 + age | hospital)``). As in lme4,
    ``(age | hospital)`` includes the intercept; write ``(0 + age | hospital)``
    for a slope alone (``coxme`` reads ``(age | hospital)`` that way).
    ``strata()``, ``offset()``, and ``Surv(start, stop, status)`` work as in
    :func:`cox_ph`.

    Variances are maximum likelihood estimates on the Laplace-approximated
    integrated partial likelihood. The information matrix is computed in
    full; ``coxme`` drops part of it for terms with more than 50 levels by
    default (its ``sparse`` control), which moves its estimates slightly.

    ``variances`` fixes the random-effect covariances instead of estimating
    them, as ``coxme(vfixed = ...)``: a dict keyed by group (``"hospital"``,
    ``"hospital:ward"``) or a list in term order, each a variance or a
    covariance matrix for a term with slopes.
    """
    if ties not in {"efron", "breslow"}:
        raise ValueError("ties must be 'efron' or 'breslow'")
    parsed = parse_survival_formula(
        data,
        formula,
        weights=weights,
        drop_intercept=True,
        allow_counting=True,
        allow_strata=True,
        allow_cluster=False,
        allow_random=True,
    )
    if not parsed.random_effects:
        raise ValueError("cox_mixed needs a random effect, for example (1 | hospital); use cox_ph without one")
    design = parsed.design
    terms = []
    for effect in parsed.random_effects:
        if not effect.correlated:
            raise ValueError("uncorrelated random slopes (||) are not supported; write separate terms")
        if effect.structure != "us":
            raise ValueError(f"{effect.structure}() random effects are not supported in cox_mixed")
        terms.append(_effect_columns(data, design, effect))
    idx = design.row_index
    weight_a = (
        np.ones(design.n_obs)
        if weights is None
        else np.asarray(column_series(data, weights).gather(idx.tolist()).to_numpy(), dtype=float)
    )
    event_a = np.asarray(parsed.event, dtype=float) > 0
    counting = parsed.entry_name is not None
    entry_a = parsed.entry if counting else np.full(design.n_obs, -np.inf)
    risk = _RiskSets(parsed.time, entry_a, event_a, weight_a, parsed.strata, ties == "efron")
    problem = _Problem(design.x, terms, parsed.offset, risk)
    fixed = None if variances is None else problem.theta_from(_variance_list(variances, terms))
    result = problem.fit(fixed)

    random_terms = []
    for t, (cov, blups) in zip(terms, problem.term_results(result.theta, result.u), strict=True):
        random_terms.append(
            {
                "group": t.group,
                "structure": "us",
                "names": list(t.names),
                "covariance": cov,
                "correlation": None,
                "levels": list(t.levels),
                "blups": blups,
            }
        )
    fit = CoxMixedFit(
        coefficients=result.beta,
        covariance=result.covariance,
        names=list(design.names),
        n_obs=design.n_obs,
        n_events=int(event_a.sum()),
        log_likelihood=result.integrated,
        log_likelihood_null=result.null,
        log_likelihood_penalized=result.penalized,
        log_likelihood_fitted=result.fitted,
        df=result.df,
        n_variance_parameters=problem.n_theta if fixed is None else 0,
        random_terms=random_terms,
        theta=result.theta,
        converged=result.converged,
        n_iter=result.n_iter,
        function_evals=result.function_evals,
        ties=ties,
        design=design,
        x=design.x,
        row_index=idx,
        time=parsed.time,
        event=event_a.astype(float),
        strata=parsed.strata,
        entry=parsed.entry,
        weights=weight_a,
        offset=parsed.offset,
        formula=formula,
        _z=problem.z,
        _blups=problem.lam(result.theta) @ result.u,
    )
    fit._strata_names = parsed.strata_names
    return fit


def _term_contribution(data: pl.DataFrame, design: Design, term: dict[str, Any]) -> np.ndarray:
    """``z_i' b`` of one term on new rows; unknown levels contribute zero."""
    from ..models.formula import RandomEffect

    names = term["names"]
    slopes = tuple(name for name in names if name != "(Intercept)")
    effect = RandomEffect(intercept="(Intercept)" in names, slopes=slopes, group=term["group"], correlated=True)
    cols = _effect_columns(data, design, effect)
    labels = group_labels(data, term["group"], design.row_index)
    position = {level: k for k, level in enumerate(term["levels"])}
    at = np.asarray([position.get(label, -1) for label in labels])
    known = at >= 0
    out = np.zeros(len(labels))
    out[known] = np.sum(cols.z[known] * term["blups"][at[known]], axis=1)
    return out


class _RiskSets:
    """Cox partial likelihood derivatives for a sparse design ``W``.

    Risk-set sums at each death time come from per-row buckets and a reverse
    cumulative sum. The second-moment part of the information is
    ``W' diag(r c) W`` with ``r c`` each row's expected count, so it stays
    sparse; only the ``a a'`` part (one row per death) is dense.
    """

    def __init__(self, time, entry, event, weights, strata, efron):
        n = len(time)
        _, sid = np.unique(np.asarray(strata).astype(str), return_inverse=True)
        idx_t = np.full(n, -1)
        idx_e = np.full(n, -1)
        base = np.full(n, -1)
        slices = []
        offset = 0
        for s in range(sid.max() + 1 if n else 0):
            rows = np.flatnonzero(sid == s)
            tau = np.unique(time[rows][event[rows]])
            kt = np.searchsorted(tau, time[rows], side="right") - 1
            ke = np.searchsorted(tau, entry[rows], side="right") - 1
            # A row is at risk at tau_k when entry < tau_k <= time.
            idx_t[rows] = np.where(kt >= 0, kt + offset, -1)
            idx_e[rows] = np.where((ke >= 0) & (ke < kt), ke + offset, -1)
            base[rows] = np.where(ke >= 0, ke + offset, offset - 1)
            slices.append(slice(offset, offset + len(tau)))
            offset += len(tau)
        k_total = offset
        live = (idx_t >= 0) & (idx_t > idx_e)
        rows_t = np.flatnonzero(live)
        rows_e = np.flatnonzero(live & (idx_e >= 0))
        self.bucket = sparse.csr_matrix(
            (
                np.r_[np.ones(len(rows_t)), -np.ones(len(rows_e))],
                (np.r_[idx_t[rows_t], idx_e[rows_e]], np.r_[rows_t, rows_e]),
            ),
            shape=(k_total, n),
        )
        dead = np.flatnonzero(event & live)
        self.deaths = sparse.csr_matrix((np.ones(len(dead)), (idx_t[dead], dead)), shape=(k_total, n))
        m = np.bincount(idx_t[dead], minlength=k_total)
        wd = np.bincount(idx_t[dead], weights=weights[dead], minlength=k_total)
        if efron:
            self.kk = np.repeat(np.arange(k_total), m)
            self.f = np.concatenate([np.arange(c) / c for c in m]) if k_total else np.zeros(0)
            self.omega = (wd / np.maximum(m, 1))[self.kk]
        else:
            self.kk = np.flatnonzero(m > 0)
            self.f = np.zeros(len(self.kk))
            self.omega = wd[self.kk]
        self.slices = slices
        self.k_total = k_total
        self.idx_t = idx_t
        self.base = base
        self.live = live
        self.event = event.astype(float)
        self.weights = weights
        self.dw = self.event * weights

    def _reverse_cumsum(self, values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values)
        for sl in self.slices:
            out[sl] = np.cumsum(values[sl][::-1], axis=0)[::-1]
        return out

    def loglik(self, eta: np.ndarray) -> float:
        shift = float(eta.max()) if eta.size else 0.0
        r = self.weights * np.exp(eta - shift)
        s0 = self._reverse_cumsum(self.bucket @ r)
        s0e = s0[self.kk] - self.f * (self.deaths @ r)[self.kk]
        return float(self.dw @ eta - self.omega @ (np.log(s0e) + shift))

    def evaluate(self, eta: np.ndarray, w: sparse.csr_matrix):
        shift = float(eta.max()) if eta.size else 0.0
        r = self.weights * np.exp(eta - shift)
        rw = sparse.csr_matrix(w.multiply(r[:, None]))
        s0 = self._reverse_cumsum(self.bucket @ r)
        s1 = self._reverse_cumsum((self.bucket @ rw).toarray())
        kk, f, omega = self.kk, self.f, self.omega
        s0e = s0[kk] - f * (self.deaths @ r)[kk]
        s1e = s1[kk] - f[:, None] * (self.deaths @ rw).toarray()[kk]
        a = s1e / s0e[:, None]
        ll = float(self.dw @ eta - omega @ (np.log(s0e) + shift))
        grad = np.asarray(w.T @ self.dw).ravel() - a.T @ omega
        h = np.bincount(kk, weights=omega / s0e, minlength=self.k_total)
        g = np.bincount(kk, weights=omega * f / s0e, minlength=self.k_total)
        cum = np.r_[0.0, np.cumsum(h)]
        c = np.zeros(len(eta))
        live = self.live
        c[live] = cum[self.idx_t[live] + 1] - cum[self.base[live] + 1] - self.event[live] * g[self.idx_t[live]]
        expected = sparse.csr_matrix(w.multiply((r * c)[:, None]))
        info = np.asarray((w.T @ expected).todense()) - (a * omega[:, None]).T @ a
        return ll, grad, info


@dataclass
class _Result:
    theta: np.ndarray
    u: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    integrated: float
    penalized: float
    fitted: float
    null: float
    df: float
    converged: bool
    n_iter: int
    function_evals: int


class _Problem:
    def __init__(self, x, terms, offset, risk: _RiskSets):
        self.x = np.asarray(x, dtype=float)
        self.terms = terms
        self.offset = np.asarray(offset, dtype=float)
        self.risk = risk
        n = self.x.shape[0]
        rows, cols, vals = [], [], []
        self.col_offsets = []
        self.par_offsets = []
        q = 0
        n_par = 0
        for t in terms:
            k = t.z.shape[1]
            rows.append(np.repeat(np.arange(n), k))
            cols.append(q + (t.codes[:, None] * k + np.arange(k)[None, :]).ravel())
            vals.append(t.z.ravel())
            self.col_offsets.append(q)
            self.par_offsets.append(n_par)
            q += len(t.levels) * k
            n_par += k * (k + 1) // 2
        self.q = q
        self.n_theta = n_par
        self.z = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, q))
        self.x_sparse = sparse.csr_matrix(self.x)
        self.p = self.x.shape[1]
        self.evals = 0
        self._warm = np.zeros(q + self.p)

    def factors(self, theta: np.ndarray) -> list[np.ndarray]:
        out = []
        for t, at in zip(self.terms, self.par_offsets, strict=True):
            k = t.z.shape[1]
            factor = np.zeros((k, k))
            factor[np.diag_indices(k)] = theta[at : at + k]
            if k > 1:
                factor[np.tril_indices(k, -1)] = theta[at + k : at + k * (k + 1) // 2]
            out.append(factor)
        return out

    def lam(self, theta: np.ndarray) -> sparse.csr_matrix:
        blocks = [sparse.kron(sparse.identity(len(t.levels)), f) for t, f in zip(self.terms, self.factors(theta), strict=True)]
        return sparse.csr_matrix(sparse.block_diag(blocks))

    def design(self, theta: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(sparse.hstack([self.z @ self.lam(theta), self.x_sparse]))

    def solve(self, theta: np.ndarray, start: np.ndarray):
        """Maximize the penalized partial likelihood over ``(u, beta)``."""
        w = self.design(theta)
        q = self.q
        pen = np.r_[np.ones(q), np.zeros(self.p)]

        def evaluate(c):
            ll, grad, info = self.risk.evaluate(w @ c + self.offset, w)
            return ll, ll - 0.5 * c[:q] @ c[:q], grad - pen * c, info + np.diag(pen)

        c = start.copy()
        ll, ppl, grad, hess = evaluate(c)
        converged = False
        for _ in range(_MAX_NEWTON):
            step = _solve(hess, grad)
            decrement = float(grad @ step)
            if decrement < _NEWTON_TOL:
                converged = True
                break
            lam = 1.0
            while lam > 1e-10:
                trial = c + lam * step
                eta = w @ trial + self.offset
                ppl_t = self.risk.loglik(eta) - 0.5 * trial[:q] @ trial[:q]
                if np.isfinite(ppl_t) and ppl_t >= ppl - 1e-12 * (1 + abs(ppl)):
                    break
                lam *= 0.5
            else:
                break
            c = trial
            ll, ppl, grad, hess = evaluate(c)
        self.evals += 1
        return c, ll, ppl, hess, converged

    def integrated(self, theta: np.ndarray, start: np.ndarray | None = None):
        c, ll, ppl, hess, converged = self.solve(theta, self._warm if start is None else start)
        if start is None and converged:
            # The next variance search step starts from this solution.
            self._warm = c
        chol = np.linalg.cholesky(hess[: self.q, : self.q])
        value = ppl - float(np.sum(np.log(np.diag(chol))))
        return value, c, ll, ppl, hess, converged

    def theta_from(self, covariances: list[np.ndarray]) -> np.ndarray:
        theta = np.zeros(self.n_theta)
        for t, at, cov in zip(self.terms, self.par_offsets, covariances, strict=True):
            k = t.z.shape[1]
            cov = np.atleast_2d(np.asarray(cov, dtype=float))
            if cov.shape != (k, k):
                raise ValueError(f"the variance of ({' + '.join(t.names)} | {t.group}) must be {k} x {k}")
            if not np.any(cov):
                continue
            try:
                factor = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f"the covariance of ({' + '.join(t.names)} | {t.group}) is not positive definite") from None
            theta[at : at + k] = np.diag(factor)
            if k > 1:
                theta[at + k : at + k * (k + 1) // 2] = factor[np.tril_indices(k, -1)]
        return theta

    def fit(self, fixed: np.ndarray | None = None) -> _Result:
        start = np.zeros(self.n_theta)
        for t, at in zip(self.terms, self.par_offsets, strict=True):
            start[at : at + t.z.shape[1]] = _START_SD
        # The fixed-effects-only Cox fit starts the inner problem.
        _, c0, *_ = self.integrated(np.zeros(self.n_theta))
        self._warm = c0
        bounds = []
        for t in self.terms:
            k = t.z.shape[1]
            bounds += [(0.0, None)] * k + [(None, None)] * (k * (k - 1) // 2)

        def objective(theta):
            value, c, *_ = self.integrated(np.asarray(theta, dtype=float))
            return -value, c

        n_iter = 0
        if fixed is not None:
            theta = fixed
            converged = True
        elif self.n_theta == 1:
            res = optimize.minimize_scalar(
                lambda s: objective(np.array([s]))[0], bounds=(0.0, 10.0), method="bounded", options={"xatol": 1e-9}
            )
            theta = np.array([res.x])
            converged = bool(res.success)
            n_iter = int(res.nfev)
        else:
            def fun(theta):
                theta = np.asarray(theta, dtype=float)
                value, c = objective(theta)
                center = c
                grad = np.zeros_like(theta)
                for j in range(len(theta)):
                    step = np.zeros_like(theta)
                    step[j] = _FD_STEP
                    up = self.integrated(theta + step, center)[0]
                    down = self.integrated(theta - step, center)[0]
                    grad[j] = -(up - down) / (2 * _FD_STEP)
                self._warm = center
                return value, grad

            res = optimize.minimize(
                fun, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 500}
            )
            theta = np.asarray(res.x, dtype=float)
            converged = bool(res.success)
            n_iter = int(res.nit)
        value, c, ll, ppl, hess, inner_ok = self.integrated(theta)
        inv = np.linalg.inv(hess)
        q = self.q
        null = self.risk.loglik(self.offset.copy())
        return _Result(
            theta=theta,
            u=c[:q],
            beta=c[q:],
            covariance=0.5 * (inv[q:, q:] + inv[q:, q:].T),
            integrated=value,
            penalized=ppl,
            fitted=ll,
            null=null,
            df=float(q + self.p - np.trace(inv[:q, :q])),
            converged=converged and inner_ok,
            n_iter=n_iter,
            function_evals=self.evals,
        )

    def term_results(self, theta: np.ndarray, u: np.ndarray):
        b = self.lam(theta) @ u
        for t, factor, at in zip(self.terms, self.factors(theta), self.col_offsets, strict=True):
            k = t.z.shape[1]
            levels = len(t.levels)
            yield factor @ factor.T, b[at : at + levels * k].reshape(levels, k)


def _variance_list(variances: dict[str, Any] | Sequence[Any], terms) -> list[np.ndarray]:
    if isinstance(variances, dict):
        groups = [t.group for t in terms]
        if len(set(groups)) < len(groups):
            raise ValueError("two terms share a group; pass variances as a list in term order")
        unknown = set(variances) - set(groups)
        if unknown or len(variances) != len(groups):
            raise ValueError(f"variances needs one entry for each group: {groups}")
        return [np.asarray(variances[g], dtype=float) for g in groups]
    out = [np.asarray(v, dtype=float) for v in variances]
    if len(out) != len(terms):
        raise ValueError(f"variances needs {len(terms)} entries, one per random-effect term")
    return out


def _solve(hess: np.ndarray, grad: np.ndarray) -> np.ndarray:
    try:
        factor = np.linalg.cholesky(hess)
        return np.linalg.solve(factor.T, np.linalg.solve(factor, grad))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(hess, grad, rcond=None)[0]
