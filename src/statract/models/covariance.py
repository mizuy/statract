"""Sandwich covariance estimators.

The estimators follow Zeileis, Köll and Graham (sandwich). ``hc_covariance``
covers HC0–HC5 and the constant estimator. ``cluster_covariance`` covers one-way
and multi-way clustering with the inclusion-exclusion meat. The Newey–West
estimator uses a Bartlett kernel and, by default, the lag
``floor(4 (n/100)^(2/9))`` without prewhitening.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from .design import ColumnRef, column_series
from .fit import Fit, _xtwx_inv


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
    """Cluster-robust covariance of the coefficients (k x k), matching ``vcovCL``.

    ``cluster`` is one column, a list of columns (pass ``data``), or an array
    aligned with the fitted rows. ``kind`` is ``"HC0"``–``"HC3"`` (default
    ``"HC1"`` for OLS, ``"HC0"`` otherwise); ``adjust`` toggles the
    small-sample cluster factor. Also accepts a Cox fit.

    Several cluster dimensions are combined by inclusion-exclusion over the
    non-empty subsets ``S`` of the dimensions (Cameron, Gelbach & Miller 2011,
    eq. 2.13): the meat is ``sum_S (-1)^(|S|+1) c_S sum_g U_g U_g' / n``,
    where ``g`` runs over the observed intersections of the clusters in
    ``S``, ``U_g`` is the sum of the score rows in ``g`` (Liang & Zeger 1986;
    Arellano 1987) and ``c_S = G_S / (G_S - 1)`` when ``adjust``. HC1 also
    multiplies by ``(n - 1) / (n - k)`` (Zeileis, Köll & Graham 2020, sec. 3).
    No positive-semidefinite correction is applied.

    HC2 and HC3 (one cluster dimension, not for Cox fits) replace each
    cluster's residual vector ``e_g`` by ``(I - H_gg)^(-1/2) e_g`` or
    ``(I - H_gg)^(-1) e_g``, with ``H_gg`` the cluster's block of the weighted
    hat matrix (Bell & McCaffrey 2002; Zeileis, Köll & Graham 2020, sec. 3.2).
    With one row per cluster and ``adjust=False`` they reduce to the HC2 and
    HC3 of :func:`hc_covariance`.
    """
    is_cox = fit.family == "cox"
    if kind is None:
        kind = "HC1" if fit.family == "ols" else "HC0"
    if kind == "HC":
        kind = "HC0"
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("cluster kind must be HC0, HC1, HC2, or HC3")
    if is_cox and kind not in {"HC0", "HC1"}:
        raise ValueError("a Cox fit takes cluster kind HC0 or HC1")
    groups = _as_groups(fit, cluster, data)
    if any(_has_null(groups[:, j]) for j in range(groups.shape[1])):
        raise ValueError("cluster has missing values; drop those rows before fitting")
    leverage_power = {"HC2": 0.5, "HC3": 1.0}.get(kind)
    if leverage_power is not None and groups.shape[1] > 1:
        raise ValueError("cluster kinds HC2 and HC3 take one cluster dimension")

    scores = np.asarray(fit.score_contributions(), dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if is_cox:
        scores = scores * np.asarray(fit.weights, dtype=float)[:, None]
    n, k = scores.shape
    codes = [_label_codes(groups[:, j]) for j in range(groups.shape[1])]
    meat_total = np.zeros((k, k))
    for size in range(1, len(codes) + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in combinations(range(len(codes)), size):
            if size == 1:
                cells = codes[subset[0]]
            else:
                stacked = np.column_stack([codes[j] for j in subset])
                cells = np.unique(stacked, axis=0, return_inverse=True)[1].reshape(-1)
            n_cells = int(cells.max()) + 1 if cells.size else 0
            if leverage_power is None:
                summed = np.zeros((n_cells, k))
                np.add.at(summed, cells, scores)
            else:
                summed = _leverage_adjusted_sums(fit, scores, cells, n_cells, leverage_power)
            factor = n_cells / (n_cells - 1.0) if adjust and n_cells > 1 else 1.0
            meat_total += sign * factor * (summed.T @ summed)
    meat_matrix = meat_total / n
    if kind == "HC1":
        meat_matrix = meat_matrix * (n - 1) / (n - k)
    bread = fit.bread()
    return (bread @ meat_matrix @ bread) / n


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
    """Per-observation weights (length n) of the HC meat for ``kind``.

    With ``r_i`` the scalar such that the score row is ``r_i x_i`` and
    ``h_i`` the leverage (``fit.hat_values``), the meat is
    ``X' diag(omega) X`` with (Zeileis 2004, sec. 2, eq. 6 and table 1):

    - HC0 ``r^2`` (White 1980);
    - HC1 ``r^2 n / (n - k)`` (MacKinnon & White 1985);
    - HC2 ``r^2 / (1 - h)`` and HC3 ``r^2 / (1 - h)^2`` (MacKinnon & White
      1985; Long & Ervin 2000);
    - HC4 ``r^2 / (1 - h)^d``, ``d = min(4, n h / p)`` (Cribari-Neto 2004);
    - HC4m ``d = min(1, n h / p) + min(1.5, n h / p)`` (Cribari-Neto & da
      Silva 2011);
    - HC5 ``r^2 / sqrt((1 - h)^d)``, ``d = min(n h / p, max(4, 0.7 n h_max / p))``
      (Cribari-Neto, Souza & Vasconcellos 2007);
    - ``"const"``: the constant ``sum r^2 / (n - k)``.

    ``p`` is the trace of the hat matrix rounded to an integer and ``k`` the
    number of columns of the design.
    """
    if kind == "HC":
        kind = "HC0"
    if kind not in {"HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5", "const"}:
        raise ValueError(f"unknown HC kind {kind!r}")
    resid = _score_scalars(fit.score_contributions(), fit.x)
    squared = resid**2
    n, k = fit.x.shape
    if kind == "HC0":
        return squared
    if kind == "HC1":
        return squared * n / (n - k)
    if kind == "const":
        return np.full(n, float(np.sum(squared)) / (n - k))
    leverage = np.asarray(fit.hat_values, dtype=float)
    complement = 1.0 - leverage
    if kind == "HC2":
        return squared / complement
    if kind == "HC3":
        return squared / complement**2
    trace = float(np.round(np.sum(leverage)))
    relative = n * leverage / trace
    if kind == "HC4":
        power = np.minimum(4.0, relative)
        return squared / complement**power
    if kind == "HC4m":
        power = np.minimum(1.0, relative) + np.minimum(1.5, relative)
        return squared / complement**power
    power = np.minimum(relative, max(4.0, 0.7 * n * float(np.max(leverage)) / trace))
    return squared / np.sqrt(complement**power)


def _score_scalars(scores: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The scalar ``r_i`` with ``score_i = r_i x_i``, row by row.

    Recovered as the projection ``<score_i, x_i> / <x_i, x_i>``. Rows with an
    all-zero score, or a non-finite result, give 0.
    """
    scores = np.asarray(scores, dtype=float)
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.sum(scores * x, axis=1) / np.sum(x * x, axis=1)
    resid[~np.any(scores != 0.0, axis=1)] = 0.0
    resid[~np.isfinite(resid)] = 0.0
    return resid


def _label_codes(labels: np.ndarray) -> np.ndarray:
    """Integer codes 0..G-1 for one column of cluster labels."""
    labels = np.asarray(labels)
    try:
        return np.unique(labels, return_inverse=True)[1].reshape(-1)
    except TypeError:
        # Mixed label types that do not sort: code by first appearance.
        seen: dict[object, int] = {}
        return np.array([seen.setdefault(label, len(seen)) for label in labels.tolist()], dtype=np.intp)


def _leverage_adjusted_sums(
    fit: Fit, scores: np.ndarray, cells: np.ndarray, n_cells: int, power: float
) -> np.ndarray:
    """Cluster score sums with the HC2 (``power=1/2``) or HC3 (``power=1``) adjustment.

    Writing the score row as ``r_i x_i`` and the weighted design row as
    ``a_i = sqrt(w_i) x_i``, the score of cluster ``g`` is ``A_g' e_g`` with
    ``e_i = r_i / sqrt(w_i)``. The adjusted score is
    ``A_g' (I - H_gg)^(-power) e_g`` with ``H_gg = A_g (A'A)^{-1} A_g'``
    (Bell & McCaffrey 2002, eq. 3; Cameron & Miller 2015, sec. VI.B). A
    zero eigenvalue of ``I - H_gg`` is dropped (generalised inverse).
    """
    x = np.asarray(fit.x, dtype=float)
    weights = np.asarray(fit.weights if fit.family == "ols" else fit.working_weights, dtype=float)
    root_w = np.sqrt(np.clip(weights, 0.0, None))
    weighted_x = x * root_w[:, None]
    resid = _score_scalars(scores, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        whitened = np.where(root_w > 0.0, resid / root_w, 0.0)
    gram_inv = _xtwx_inv(x, weights)
    summed = np.zeros((n_cells, x.shape[1]))
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(n_cells + 1))
    for g in range(n_cells):
        rows = order[bounds[g] : bounds[g + 1]]
        a_g = weighted_x[rows]
        block = np.eye(len(rows)) - a_g @ gram_inv @ a_g.T
        values, vectors = np.linalg.eigh((block + block.T) / 2.0)
        tolerance = np.finfo(float).eps * max(len(rows), 1) * 100.0
        scaled = np.zeros_like(values)
        keep = values > tolerance
        scaled[keep] = values[keep] ** (-power)
        adjusted = vectors @ (scaled * (vectors.T @ whitened[rows]))
        summed[g] = a_g.T @ adjusted
    return summed


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
