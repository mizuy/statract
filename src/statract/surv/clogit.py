"""Conditional logistic regression for matched sets.

``survival::clogit`` rewrites the outcome as ``Surv(rep(1, n), y)`` and calls
``coxph``. The default ``method="exact"`` is the discrete partial likelihood:
within each stratum, the probability of the observed events given how many
occurred. ``"efron"`` and ``"breslow"`` (also ``"approximate"``) use that same
constant-time Cox model. Exact does not accept case weights or cluster-robust
variance, matching ``clogit``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl
from numba import njit

from ..models.design import ColumnRef, Design, column_series, design_matrix
from ..models.formula import is_formula, model_matrix
from .cox import CoxFit, _EPS, _MAX_ITER, _newton
from .spec import _drop_intercept, _subset_design, combine_strata

_METHODS = ("exact", "approximate", "efron", "breslow")


def conditional_logit(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    strata: ColumnRef | None = None,
    cluster: ColumnRef | None = None,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    method: str = "exact",
) -> CoxFit:
    """Fit a conditional logit, the matched-set form of ``survival::clogit``.

    ``outcome`` may be ``"bleed ~ eolm + strata(set)"``. The intercept column is
    removed and factors keep their treatment contrasts. ``method="exact"``
    enumerates the combinations inside each stratum. ``"approximate"`` is
    Breslow. One event per stratum makes exact, Efron, and Breslow the same
    likelihood.
    """
    ties = _ties(method)
    design, y, weight, off, strata_a, cluster_a, strata_names, strata_label = _design(
        data,
        outcome,
        predictors,
        strata=strata,
        cluster=cluster,
        weights=weights,
        offset=offset,
        ties=ties,
    )
    event = y > 0
    if ties == "exact":
        beta, info, converged, ll = _exact_newton(design.x, event, off, strata_a)
    else:
        beta, info, converged, ll = _newton(
            design.x,
            np.ones(design.n_obs),
            event,
            weight,
            off,
            np.zeros(design.n_obs),
            strata_a,
            ties,
        )
    fit = _as_cox(
        design,
        beta,
        info,
        converged,
        ll,
        event,
        weight,
        off,
        strata_a,
        ties,
        strata_names,
        strata_label,
    )
    if ties != "exact":
        fit._baseline_pending = (
            design.x,
            beta,
            np.ones(design.n_obs),
            event,
            weight,
            off,
            np.zeros(design.n_obs),
            strata_a,
            ties,
        )
    if ties != "exact" and cluster_a is not None:
        _apply_cluster(fit, cluster_a, weight)
    elif ties != "exact" and weights is not None:
        _apply_cluster(fit, np.arange(design.n_obs), weight)
    return fit


def _ties(method: str) -> str:
    if method not in _METHODS:
        raise ValueError("method must be 'exact', 'approximate', 'efron', or 'breslow'")
    if method == "approximate":
        return "breslow"
    return method


def _design(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None,
    *,
    strata: ColumnRef | None,
    cluster: ColumnRef | None,
    weights: ColumnRef | None,
    offset: ColumnRef | None,
    ties: str,
) -> tuple[Design, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...], str | None]:
    if is_formula(outcome):
        if predictors is not None:
            raise ValueError("pass a Wilkinson formula or predictors, not both")
        built = model_matrix(str(outcome), data)
        if built.surv is not None:
            raise ValueError("conditional_logit uses a binary outcome, for example bleed ~ eolm + strata(set)")
        if built.random_effects:
            raise ValueError("random effects belong in fit_mixed, for example (1 | group)")
        if built.strata and strata is not None:
            raise ValueError("pass strata() in the formula or strata=, not both")
        if built.cluster is not None and cluster is not None:
            raise ValueError("pass cluster() in the formula or cluster=, not both")
        if ties == "exact" and (built.cluster is not None or cluster is not None):
            raise ValueError("cluster is not available for method='exact'")
        design = _drop_intercept(built.design)
        y = _as_binary(built.y)
        off = np.zeros(design.n_obs) if built.offset is None else np.asarray(built.offset, dtype=float)
        strata_names = tuple(built.strata)
        cluster_name = built.cluster
    else:
        if predictors is None:
            raise ValueError("predictors are required when outcome is a column")
        if ties == "exact" and cluster is not None:
            raise ValueError("cluster is not available for method='exact'")
        strata_names = ()
        cluster_name = None
    if ties == "exact" and weights is not None:
        warnings.warn("weights ignored: not possible for the exact method", UserWarning, stacklevel=3)
        weights = None
    if not is_formula(outcome):
        extra: list[ColumnRef] = [outcome]
        for ref in (strata, cluster, weights, offset):
            if ref is not None:
                extra.append(ref)
        design = design_matrix(data, predictors, extra=extra, intercept=False)
        y = _as_binary(column_series(data, outcome).gather(design.row_index.tolist()).to_numpy())
        off = np.zeros(design.n_obs)
    if design.n_params == 0:
        raise ValueError("conditional_logit needs a covariate")

    keep = np.ones(design.n_obs, dtype=bool)
    extra_values: dict[str, np.ndarray] = {}
    for key, ref in (("weights", weights), ("offset", offset), ("strata", strata), ("cluster", cluster)):
        if ref is None:
            continue
        values = column_series(data, ref).gather(design.row_index.tolist())
        keep &= values.is_not_null().to_numpy()
        extra_values[key] = values.to_numpy()
    if cluster_name is not None:
        values = column_series(data, cluster_name).gather(design.row_index.tolist())
        keep &= values.is_not_null().to_numpy()
        extra_values["cluster"] = values.to_numpy()
    if not bool(keep.all()):
        design = _subset_design(design, keep)
        y = y[keep]
        off = off[keep]
        extra_values = {key: values[keep] for key, values in extra_values.items()}
    if design.n_obs == 0:
        raise ValueError("no complete rows")
    if offset is not None:
        off = off + np.asarray(extra_values["offset"], dtype=float)
    if strata is not None:
        strata_a = np.asarray(extra_values["strata"])
        strata_label = strata if isinstance(strata, str) else column_series(data, strata).name
        strata_names = (strata_label,)
    elif strata_names:
        strata_a = combine_strata(data, strata_names, design.row_index)
        strata_label = strata_names[0] if len(strata_names) == 1 else None
    else:
        strata_a = np.array(["_"] * design.n_obs)
        strata_label = None
    cluster_a = None
    if "cluster" in extra_values:
        cluster_a = np.asarray(extra_values["cluster"])
    weight = np.ones(design.n_obs) if weights is None else np.asarray(extra_values["weights"], dtype=float)
    if np.any(weight <= 0) or not np.isfinite(weight).all():
        raise ValueError("weights must be positive and finite")
    return design, y, weight, off, strata_a, cluster_a, strata_names, strata_label


def _as_binary(y: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be 0 or 1") from exc
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("outcome must be 0 or 1")
    return values


def _as_cox(
    design: Design,
    beta: np.ndarray,
    info: np.ndarray,
    converged: bool,
    ll: float,
    event: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    strata: np.ndarray,
    ties: str,
    strata_names: tuple[str, ...],
    strata_label: str | None,
) -> CoxFit:
    fit = CoxFit(
        coefficients=beta,
        covariance=np.linalg.pinv(info),
        names=list(design.names),
        n_obs=design.n_obs,
        log_likelihood=ll,
        residual_df=None,
        family="conditional_logit",
        x=design.x,
        y=event.astype(float),
        row_index=design.row_index,
        design=design,
        weights=weight,
        offset=offset,
        working_residuals=np.zeros(design.n_obs),
        working_weights=weight,
        hat_values=np.zeros(design.n_obs),
        dispersion=1.0,
        deviance=float(-2 * ll),
        time=np.ones(design.n_obs),
        event=event.astype(float),
        strata=strata,
        entry=np.zeros(design.n_obs),
        baseline_time=np.zeros(0),
        baseline_hazard_values=np.zeros(0),
        baseline_strata=np.empty(0, dtype=object),
        information=info,
        ties=ties,
        converged=converged,
    )
    fit._time_name = None
    fit._strata_name = strata_label
    fit._strata_names = strata_names
    fit._entry_name = None
    return fit


def _apply_cluster(fit: CoxFit, labels: np.ndarray, weight: np.ndarray) -> None:
    scores = fit.score_contributions() * weight[:, None]
    _, inverse = np.unique(labels, return_inverse=True)
    summed = np.zeros((int(inverse.max()) + 1, scores.shape[1]))
    np.add.at(summed, inverse, scores)
    meat = summed.T @ summed
    naive = np.linalg.pinv(fit.information)
    fit.covariance = naive @ meat @ naive


def _exact_newton(x: np.ndarray, event: np.ndarray, offset: np.ndarray, strata: np.ndarray):
    order, bounds = _stratum_bounds(strata)
    if not _any_informative(event[order], bounds):
        raise ValueError("no stratum has both an event and a non-event")
    x_o = np.ascontiguousarray(x[order], dtype=np.float64)
    event_o = np.ascontiguousarray(event[order], dtype=np.float64)
    offset_o = np.ascontiguousarray(offset[order], dtype=np.float64)
    bounds = np.ascontiguousarray(bounds, dtype=np.int64)

    def score(beta: np.ndarray):
        eta = np.ascontiguousarray(x_o @ beta + offset_o, dtype=np.float64)
        return _exact_blocks(x_o, event_o, eta, bounds)

    p = x.shape[1]
    beta = np.zeros(p)
    ll, grad, hess = score(beta)
    if not np.isfinite(ll):
        raise ValueError("conditional likelihood is not finite at the starting value")
    converged = False
    for _ in range(_MAX_ITER):
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        lam = 1.0
        accepted = False
        while lam > 1e-8:
            trial = beta + lam * step
            ll_t, grad_t, hess_t = score(trial)
            if np.isfinite(ll_t) and ll_t >= ll - 1e-8:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        if abs(ll_t - ll) <= _EPS * (abs(ll) + _EPS):
            beta, ll, grad, hess = trial, ll_t, grad_t, hess_t
            converged = True
            break
        beta, ll, grad, hess = trial, ll_t, grad_t, hess_t
    return beta, -hess, converged, float(ll)


def _stratum_bounds(strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _codes, inverse = np.unique(strata, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    ordered = inverse[order]
    if ordered.size == 0:
        return order, np.zeros(1, dtype=np.int64)
    change = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    bounds = np.empty(change.size + 2, dtype=np.int64)
    bounds[0] = 0
    bounds[1:-1] = change
    bounds[-1] = ordered.size
    return order, bounds


def _any_informative(event: np.ndarray, bounds: np.ndarray) -> bool:
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        k = float(np.sum(event[start:stop]))
        n = stop - start
        if 0 < k < n:
            return True
    return False


@njit(cache=True)
def _exact_blocks(x, event, eta, bounds):
    """Sum of the discrete conditional likelihood over stratum blocks."""
    p = x.shape[1]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    n_st = bounds.shape[0] - 1
    for s in range(n_st):
        start = bounds[s]
        stop = bounds[s + 1]
        add_ll, add_grad, add_hess, used = _exact_one(x[start:stop], event[start:stop], eta[start:stop])
        if used:
            if not np.isfinite(add_ll):
                return np.nan, grad, hess
            ll += add_ll
            grad += add_grad
            hess += add_hess
    return ll, grad, hess


@njit(cache=True)
def _exact_one(x, event, eta):
    """One stratum: combinations of size k, with risks scaled by the max linear predictor."""
    n, p = x.shape
    k = 0
    for i in range(n):
        k += event[i] > 0.5
    if k == 0 or k == n:
        return 0.0, np.zeros(p), np.zeros((p, p)), False
    eta_max = eta[0]
    for i in range(1, n):
        if eta[i] > eta_max:
            eta_max = eta[i]
    phi = np.zeros(k + 1)
    u = np.zeros((k + 1, p))
    v = np.zeros((k + 1, p, p))
    phi[0] = 1.0
    log_scale = 0.0
    for i in range(n):
        r = np.exp(eta[i] - eta_max)
        xi = x[i]
        upper = k if k < i + 1 else i + 1
        for j in range(upper, 0, -1):
            prev_phi = phi[j - 1]
            prev_u = u[j - 1].copy()
            prev_v = v[j - 1].copy()
            for a in range(p):
                for b in range(p):
                    v[j, a, b] += r * (
                        prev_v[a, b] + prev_u[a] * xi[b] + xi[a] * prev_u[b] + prev_phi * xi[a] * xi[b]
                    )
                u[j, a] += r * (prev_u[a] + prev_phi * xi[a])
            phi[j] += r * prev_phi
        scale = phi[0]
        for j in range(1, k + 1):
            if phi[j] > scale:
                scale = phi[j]
        if scale > 1e8 or (scale > 0.0 and scale < 1e-8):
            for j in range(k + 1):
                phi[j] /= scale
                for a in range(p):
                    u[j, a] /= scale
                    for b in range(p):
                        v[j, a, b] /= scale
            log_scale += np.log(scale)
    if phi[k] <= 0.0 or not np.isfinite(phi[k]):
        return np.nan, np.zeros(p), np.zeros((p, p)), True
    inv = 1.0 / phi[k]
    mean = u[k] * inv
    ll = 0.0
    grad = np.zeros(p)
    for i in range(n):
        if event[i] > 0.5:
            ll += eta[i]
            for a in range(p):
                grad[a] += x[i, a]
    ll -= k * eta_max + log_scale + np.log(phi[k])
    for a in range(p):
        grad[a] -= mean[a]
    hess = np.zeros((p, p))
    for a in range(p):
        for b in range(p):
            hess[a, b] = -(v[k, a, b] * inv - mean[a] * mean[b])
    return ll, grad, hess, True
