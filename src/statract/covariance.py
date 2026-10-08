"""Sandwich covariance estimators.

The estimators follow Zeileis, Köll and Graham (sandwich). ``hc_covariance``
covers HC0–HC5 and the constant estimator. ``cluster_covariance`` covers one-way
and multi-way clustering with the inclusion-exclusion meat. The Newey–West
estimator uses a Bartlett kernel and, by default, the lag
``floor(4 (n/100)^(2/9))`` without prewhitening.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .design import ColumnRef, column_series
from .fit import Fit


def meat(fit: Fit, cluster: np.ndarray | None = None) -> np.ndarray:
    """Empirical variance of the scores, divided by ``n``."""
    scores = fit.score_contributions()
    if cluster is not None:
        scores = _sum_by_cluster(scores, cluster)
    return scores.T @ scores / fit.n_obs


def hc_covariance(fit: Fit, kind: str = "HC3") -> np.ndarray:
    """Heteroskedasticity-consistent covariance, matching ``vcovHC``."""
    omega = _hc_omega(fit, kind)
    return _sandwich_from_omega(fit, omega)


def cluster_covariance(
    fit: Fit,
    cluster: ColumnRef | Sequence[ColumnRef] | np.ndarray,
    *,
    data=None,
    kind: str | None = None,
    adjust: bool = True,
) -> np.ndarray:
    """Cluster-robust covariance, matching ``vcovCL``.

    ``kind`` defaults to ``"HC1"`` for ordinary least squares and ``"HC0"``
    otherwise, which is the ``sandwich`` default. ``cluster`` is one column,
    a list of columns (pass ``data``), or an array aligned with the fitted rows.
    Several columns use the Cameron–Gelbach–Miller inclusion-exclusion meat,
    and ``adjust`` applies ``G / (G - 1)`` to each term (``cadjust``).

    A Cox fit (``cox_ph``) works too, with ``kind`` HC0 or HC1. The scores are
    the weighted score residuals and the bread is the model-based variance,
    as ``vcovCL`` on a ``coxph``. One cluster column then differs from
    ``cox_ph(cluster=)`` only by the ``G / (G - 1)`` factor (``adjust=False``
    gives the ``coxph`` value). Unlike ``vcovCL``, a single coefficient works.
    The multi-way result is not forced to be positive semidefinite.
    """
    cox = fit.family == "cox"
    if kind is None:
        kind = "HC1" if fit.family == "ols" else "HC0"
    kind = "HC0" if kind == "HC" else kind
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("cluster kind must be HC0, HC1, HC2, or HC3")
    if cox and kind not in {"HC0", "HC1"}:
        raise ValueError("a Cox fit takes cluster kind HC0 or HC1")
    groups = _as_groups(fit, cluster, data)
    if any(_has_null(groups[:, j]) for j in range(groups.shape[1])):
        raise ValueError("cluster has missing values; drop those rows before fitting")
    scores = fit.score_contributions()
    if cox:
        scores = scores * np.asarray(fit.weights, dtype=float)[:, None]
    n, k = scores.shape
    meat_total = np.zeros((k, k))
    codes = _factorize_groups(groups)
    from itertools import combinations

    for size in range(1, codes.shape[1] + 1):
        for columns in combinations(range(codes.shape[1]), size):
            inverse, n_groups = _combine_codes(codes, columns)
            if n_groups == 0:
                continue
            summed = np.zeros((n_groups, k))
            np.add.at(summed, inverse, scores)
            adj = n_groups / (n_groups - 1) if adjust and n_groups > 1 else 1.0
            sign = 1 if size % 2 == 1 else -1
            meat_total += sign * adj * (summed.T @ summed) / n
    if kind == "HC1":
        meat_total *= (n - 1) / (n - k)
    bread = fit.bread()
    return (bread @ meat_total @ bread) / n


def bootstrap_covariance(
    fit: Fit,
    *,
    cluster: np.ndarray | None = None,
    n: int = 250,
    kind: str = "xy",
    wild: str = "rademacher",
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap covariance of the coefficients.

    ``kind="xy"`` resamples rows (or clusters). ``kind="wild"`` multiplies
    residuals by Rademacher, Webb, or Mammen weights. The random generator is
    NumPy's, so standard errors are not bit-identical to ``vcovBS``.
    """
    rng = np.random.default_rng(seed)
    coefs = np.zeros((n, len(fit.coefficients)))
    if kind == "xy":
        for draw in range(n):
            if cluster is None:
                index = rng.integers(0, fit.n_obs, size=fit.n_obs)
            else:
                labels = np.asarray(cluster)
                unique = np.unique(labels)
                chosen = rng.choice(unique, size=len(unique), replace=True)
                index = np.concatenate([np.flatnonzero(labels == lab) for lab in chosen])
            coefs[draw] = fit.refit_rows(index).coefficients
    elif kind == "wild":
        fitted = fit.x @ fit.coefficients
        resid = fit.working_residuals
        for draw in range(n):
            shocks = _wild_weights(rng, fit.n_obs, wild)
            y_star = fitted + resid * shocks
            coefs[draw] = fit.refit_rows(np.arange(fit.n_obs), y=y_star).coefficients
    else:
        raise ValueError("kind must be 'xy' or 'wild'")
    centered = coefs - coefs.mean(axis=0)
    return (centered.T @ centered) / (n - 1)


def newey_west_covariance(fit: Fit, lags: int | None = None) -> np.ndarray:
    """Bartlett HAC covariance without prewhitening."""
    scores = fit.score_contributions()
    n, k = scores.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    if lags < 0:
        raise ValueError("lags must be non-negative")
    gamma0 = scores.T @ scores
    meat_total = gamma0
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat_total = meat_total + weight * (gamma + gamma.T)
    meat_total = meat_total / n
    bread = fit.bread()
    return (bread @ meat_total @ bread) / n


def _hc_omega(fit: Fit, kind: str) -> np.ndarray:
    kind = "HC0" if kind == "HC" else kind
    scores = fit.score_contributions()
    x = fit.x
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(scores, x, out=np.full_like(scores, np.nan), where=x != 0)
    residual = np.nanmean(ratio, axis=1)
    all_zero = np.all(np.abs(scores) < np.finfo(float).eps, axis=1)
    residual[all_zero] = 0.0
    residual = np.where(np.isfinite(residual), residual, 0.0)
    n = fit.n_obs
    df = n - x.shape[1]
    hat = np.asarray(fit.hat_values, dtype=float)
    if kind == "const":
        scale = float(np.sum(residual**2) / df)
        return np.full(n, scale)
    if kind == "HC0":
        return residual**2
    if kind == "HC1":
        return residual**2 * (n / df)
    if kind == "HC2":
        return residual**2 / (1 - hat)
    if kind == "HC3":
        return residual**2 / (1 - hat) ** 2
    if kind == "HC4":
        p = int(round(float(np.sum(hat))))
        delta = np.minimum(4, n * hat / p)
        return residual**2 / (1 - hat) ** delta
    if kind == "HC4m":
        p = int(round(float(np.sum(hat))))
        ratio_h = n * hat / p
        delta = np.minimum(1.0, ratio_h) + np.minimum(1.5, ratio_h)
        return residual**2 / (1 - hat) ** delta
    if kind == "HC5":
        p = int(round(float(np.sum(hat))))
        delta = np.minimum(n * hat / p, np.maximum(4, n * 0.7 * np.max(hat) / p))
        return residual**2 / np.sqrt((1 - hat) ** delta)
    raise ValueError(f"unknown HC kind {kind!r}")


def _sandwich_from_omega(fit: Fit, omega: np.ndarray) -> np.ndarray:
    x = fit.x
    bread = fit.bread()
    scaled = np.sqrt(omega)[:, None] * x
    meat_matrix = (scaled.T @ scaled) / fit.n_obs
    return (bread @ meat_matrix @ bread) / fit.n_obs


def _as_groups(fit: Fit, cluster: ColumnRef | Sequence[ColumnRef] | np.ndarray, data) -> np.ndarray:
    if isinstance(cluster, np.ndarray):
        groups = np.asarray(cluster)
        if groups.ndim == 1:
            groups = groups.reshape(-1, 1)
        if groups.shape[0] == fit.n_obs:
            return groups
        if groups.shape[0] > fit.n_obs and fit.row_index.max() < groups.shape[0]:
            return groups[fit.row_index]
        raise ValueError("cluster array must align with the fitted rows or the original frame")
    if data is None:
        raise ValueError("pass data when cluster is a column name")
    return cluster_from_frame(fit, data, cluster)


def _has_null(values: np.ndarray) -> bool:
    if values.dtype.kind == "f":
        return bool(np.any(np.isnan(values)))
    if values.dtype.kind == "O":
        return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
    return False


def _is_expr(value: object) -> bool:
    import polars as pl

    return isinstance(value, pl.Expr)


def cluster_from_frame(fit: Fit, data, cluster: ColumnRef | Sequence[ColumnRef]) -> np.ndarray:
    """Cluster labels for the rows that entered the fit."""
    if isinstance(cluster, str) or _is_expr(cluster):
        refs = [cluster]
    else:
        refs = list(cluster)
    columns = []
    for ref in refs:
        series = column_series(data, ref)
        columns.append(np.asarray(series.gather(fit.row_index.tolist()).to_list()))
    return np.column_stack(columns)


def _factorize_groups(groups: np.ndarray) -> np.ndarray:
    """Integer codes for each cluster column. Equal labels share a code."""
    groups = np.asarray(groups)
    if groups.ndim == 1:
        groups = groups.reshape(-1, 1)
    n, width = groups.shape
    codes = np.empty((n, width), dtype=np.int64)
    for j in range(width):
        _, inverse = np.unique(groups[:, j], return_inverse=True)
        codes[:, j] = inverse
    return codes


def _combine_codes(codes: np.ndarray, columns: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """One code per observed combination of ``columns``. The meat does not depend on order."""
    if not codes.size:
        return np.zeros(0, dtype=np.int64), 0
    if len(columns) == 1:
        labels = codes[:, columns[0]]
        return labels, int(labels.max()) + 1
    chosen = np.ascontiguousarray(codes[:, list(columns)])
    _, inverse = np.unique(chosen, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False), int(inverse.max()) + 1


def _sum_by_cluster(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(labels, return_inverse=True)
    summed = np.zeros((len(unique), scores.shape[1]))
    np.add.at(summed, inverse, scores)
    return summed


def _wild_weights(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=n)
    if kind == "webb":
        support = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        return rng.choice(support, size=n)
    if kind == "mammen":
        # Mammen (1993): two-point distribution with mean 0 and variance 1.
        low = -(np.sqrt(5) - 1) / 2
        high = (np.sqrt(5) + 1) / 2
        p_low = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        return np.where(rng.random(n) < p_low, low, high)
    raise ValueError("wild must be rademacher, webb, or mammen")
