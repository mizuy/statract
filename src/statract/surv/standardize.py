"""Regression standardization after a Cox model, matching ``stdReg2::standardize_coxph``.

``measure="survival"`` averages the Breslow Cox survival over the sample with the
exposure set to each value. The variance is the sandwich of Sjölander (2016),
which includes the spread of the covariates. ``measure="rmean"`` is the
restricted mean survival time of Chen and Tsiatis (2001), with a separate Efron
Cox model in each exposure group.

Three places differ on purpose from R for ``rmean``; see ``docs/models/vs-r.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

from ..models.design import Design, build_design
from ..models.formula import _columns_in
from .cox import CoxFit, _baseline, _newton, cox_ph
from .spec import parse_survival_formula

_MEASURES = {"survival", "rmean"}
_CONTRASTS = {"difference", "ratio"}
_TRANSFORMS = {"log", "logit", "odds"}


@dataclass
class StandardizedSurvival:
    """Standardized survival or restricted mean, one estimate per time and exposure value.

    ``estimates`` has one row per time and one column per value. ``covariances``
    holds the covariance across values at each time.
    """

    measure: str
    exposure: str
    values: list[object]
    times: np.ndarray
    estimates: np.ndarray
    covariances: list[np.ndarray]
    n_obs: int
    fits: list[CoxFit] = field(default_factory=list, repr=False)

    def covariance(self, time: float) -> np.ndarray:
        """Covariance of the estimates across exposure values at ``time``."""
        return self.covariances[self._time_index(time)].copy()

    def tidy(
        self,
        *,
        contrast: str | None = None,
        reference: object = None,
        transform: str | None = None,
        ci: str = "plain",
        level: float = 0.95,
    ) -> pl.DataFrame:
        """Table as ``tidy()`` of a ``std_surv``. The transform is applied before the contrast."""
        if contrast is not None and contrast not in _CONTRASTS:
            raise ValueError("contrast must be None, 'difference', or 'ratio'")
        if transform is not None and transform not in _TRANSFORMS:
            raise ValueError("transform must be None, 'log', 'logit', or 'odds'")
        if ci not in {"plain", "log"}:
            raise ValueError("ci must be 'plain' or 'log'")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        if self.measure == "rmean" and transform in {"logit", "odds"}:
            raise ValueError(f"transform {transform!r} is not available for the restricted mean")
        if contrast is not None and reference is None:
            raise ValueError("a contrast needs a reference value")
        if contrast is None and reference is not None:
            raise ValueError("reference is used only with a contrast")
        if contrast == "difference" and ci == "log":
            raise ValueError("ci='log' does not fit a difference: the reference row is 0")
        ref = None
        if contrast is not None:
            matches = [i for i, value in enumerate(self.values) if value == reference]
            if not matches:
                raise ValueError("reference must be one of the values")
            ref = matches[0]
        z = abs(stats.norm.ppf((1 - level) / 2))
        n_values = len(self.values)
        rows: dict[str, list] = {"time": [], self.exposure: [], "estimate": [], "std_error": [], "conf_low": [], "conf_high": []}
        with np.errstate(divide="ignore", invalid="ignore"):
            for j, time in enumerate(self.times):
                est = self.estimates[j].astype(float).copy()
                cov = self.covariances[j].astype(float).copy()
                if transform is not None:
                    if transform == "log":
                        grad = 1 / est
                        est = np.log(est)
                    elif transform == "logit":
                        grad = 1 / (est * (1 - est))
                        est = np.log(est) - np.log(1 - est)
                    else:
                        grad = 1 / (1 - est) ** 2
                        est = est / (1 - est)
                    jac = np.diag(grad)
                    cov = jac.T @ cov @ jac
                if ref is not None:
                    if contrast == "difference":
                        jac = np.eye(n_values)
                        jac[ref, :] = -1
                        jac[ref, ref] = 0
                        est = est - est[ref]
                    else:
                        jac = np.diag(np.full(n_values, 1 / est[ref]))
                        jac[ref, :] = -est / est[ref] ** 2
                        jac[ref, ref] = 1
                        est = est / est[ref]
                    cov = jac.T @ cov @ jac
                    cov[ref, :] = 0
                    cov[:, ref] = 0
                se = np.sqrt(np.diag(cov))
                if ci == "plain":
                    low, high = est - z * se, est + z * se
                else:
                    low, high = est * np.exp(-z * se / est), est * np.exp(z * se / est)
                rows["time"].extend([float(time)] * n_values)
                rows[self.exposure].extend(self.values)
                rows["estimate"].extend(est.tolist())
                rows["std_error"].extend(se.tolist())
                rows["conf_low"].extend(low.tolist())
                rows["conf_high"].extend(high.tolist())
        return pl.DataFrame(rows, strict=False)

    def _time_index(self, time: float) -> int:
        gaps = np.abs(self.times - float(time))
        k = int(np.argmin(gaps))
        if gaps[k] > np.sqrt(np.finfo(float).eps):
            raise ValueError(f"no estimate at time {time}")
        return k


def standardize_cox(
    data: pl.DataFrame,
    formula: str,
    *,
    values: dict[str, Sequence[object]],
    times: float | Sequence[float],
    measure: str = "survival",
    cluster: str | None = None,
) -> StandardizedSurvival:
    """Standardize a Cox model over the sample, as ``stdReg2::standardize_coxph``.

    ``values`` names one exposure column and the values to set it to.
    ``measure="survival"`` returns the survival at each of ``times``.
    ``measure="rmean"`` returns the restricted mean survival up to the single
    time in ``times``; the exposure must be numeric 0/1. ``cluster`` is a column
    whose rows are summed before the sandwich (survival only).
    """
    if measure not in _MEASURES:
        raise ValueError("measure must be 'survival' or 'rmean'")
    if not isinstance(values, dict) or len(values) != 1:
        raise ValueError("values must name exactly one exposure, for example {'ope': [0, 1]}")
    exposure, levels = next(iter(values.items()))
    if exposure not in data.columns:
        raise KeyError(f"column {exposure!r} is not in the frame")
    levels = list(levels)
    if not levels:
        raise ValueError("values needs at least one value")
    time_grid = np.atleast_1d(np.asarray(times, dtype=float))
    if time_grid.size == 0:
        raise ValueError("times needs at least one time")
    parsed = parse_survival_formula(
        data,
        formula,
        drop_intercept=True,
        allow_counting=True,
        allow_strata=True,
        allow_cluster=True,
    )
    if parsed.strata_names:
        raise ValueError("strata() is not allowed in the formula")
    if parsed.cluster_name is not None:
        raise ValueError("cluster() is not allowed in the formula; pass cluster= (survival only)")
    if parsed.entry_name is not None:
        raise ValueError("Surv(start, stop, status) is not supported")
    if measure == "survival":
        return _survival(data, formula, parsed, exposure, levels, time_grid, cluster)
    if cluster is not None:
        raise ValueError("cluster is not available for the restricted mean")
    if time_grid.size != 1:
        raise ValueError("the restricted mean takes a single time")
    return _rmean(data, parsed, exposure, levels, float(time_grid[0]))


# ---------------------------------------------------------------- survival


def _survival(data, formula, parsed, exposure, levels, times, cluster) -> StandardizedSurvival:
    """Standardized survival at each of ``times`` for each exposure value, with
    the covariance across values at each time. Returns ``StandardizedSurvival``
    with ``measure="survival"``.

    theta_a(t) = mean_i exp(-Lambda_0(t) exp(x_i(a) beta)) with a Breslow Cox
    model (Sjölander 2016, sec. 3). The covariance is the M-estimation sandwich
    (Stefanski & Boos 2002) of the stacked estimating functions for theta,
    beta (Cox score residuals) and Lambda_0(t) (Breslow, written as
    n int dM_i / S0). Covariates are centered as R's coxph centers them: with
    R's assignment of the jump by time value the Breslow rows do not average
    to zero, so the bread depends on the parametrization.
    """
    used = data[parsed.design.row_index.tolist()]
    fit = cox_ph(used, formula, ties="breslow")
    n = used.height
    x = np.asarray(fit.x, dtype=float)
    p = x.shape[1]
    beta = np.asarray(fit.coefficients, dtype=float)
    center = _coxph_centers(x)
    xc = x - center
    risk = np.exp(xc @ beta)
    time = np.asarray(fit.time, dtype=float)
    dead = np.asarray(fit.event, dtype=float) > 0
    event_times = np.unique(time[dead])
    if event_times.size == 0 or event_times[0] > float(np.min(times)):
        raise ValueError("the earliest requested time is before the first event time")
    # Risk-set sums at each event time (right-censored data: time >= s).
    order = np.argsort(time, kind="mergesort")
    tail0 = np.concatenate([np.cumsum(risk[order][::-1])[::-1], [0.0]])
    tail1 = np.concatenate([np.cumsum((risk[:, None] * xc)[order][::-1], axis=0)[::-1], np.zeros((1, p))])
    at = np.searchsorted(time[order], event_times, side="left")
    s0 = tail0[at]
    s1 = tail1[at]
    n_dead = np.bincount(np.searchsorted(event_times, time[dead]), minlength=event_times.size).astype(float)
    jump = n_dead / s0
    # Row -> event time with the same value. R assigns 1/S0 by the time value,
    # so a row censored at an event time also carries it (docs/models/vs-r.md).
    pos = np.minimum(np.searchsorted(event_times, time), event_times.size - 1)
    on_event_time = event_times[pos] == time
    score_rows = fit.score_contributions()
    info = np.asarray(fit.information, dtype=float)
    designs = []
    for level in levels:
        frame = used.with_columns(_constant(used, exposure, level))
        designs.append(build_design(frame, fit.design).x - center)
    clusters = None
    if cluster is not None:
        labels = np.asarray([str(v) for v in used[cluster].to_list()], dtype=object)
        _, clusters = np.unique(labels, return_inverse=True)
    k = len(levels)
    size = k + p + 1
    estimates = np.ones((times.size, k))
    covariances: list[np.ndarray] = []
    for row, t in enumerate(times):
        if t <= 0:
            covariances.append(np.zeros((k, k)))
            continue
        upto = event_times <= t
        cum_hazard = float(jump[upto].sum())
        psi = np.zeros((n, size))
        bread = np.zeros((size, size))
        for j, design in enumerate(designs):
            level_risk = np.exp(design @ beta)
            surv = np.exp(-cum_hazard * level_risk)
            estimates[row, j] = surv.mean()
            psi[:, j] = surv - surv.mean()
            bread[j, j] = -1.0
            bread[j, k : k + p] = -(surv * cum_hazard * level_risk) @ design / n
            bread[j, -1] = -float(np.mean(surv * level_risk))
        psi[:, k : k + p] = score_rows
        bread[k : k + p, k : k + p] = -info / n
        # Breslow at t, n * int_0^t dM_i / S0 (its mean is Lambda_0(t) - Lambda_0(t)):
        #   n * [1{T_i <= t on an event time} / S0(T_i) - r_i sum_{s <= min(T_i, t)} dN(s) / S0(s)^2]
        # Only its spread enters the meat; d/dLambda_0 of the stacked mean is -1.
        own = on_event_time & (time <= t)
        steps = np.concatenate([[0.0], np.cumsum(np.where(upto, n_dead / s0**2, 0.0))])
        reach = np.searchsorted(event_times, np.minimum(time, t), side="right")
        psi[:, -1] = n * (np.where(own, 1.0 / s0[pos], 0.0) - risk * steps[reach])
        bread[-1, -1] = -1.0
        bread[-1, k : k + p] = -np.sum(np.where(own, 1.0 / s0[pos] ** 2, 0.0)[:, None] * s1[pos], axis=0)
        # Meat: sample covariance (m - 1) of the rows, or of their cluster sums.
        if clusters is None:
            summed = psi
            scale = 1.0 / n
        else:
            summed = np.zeros((int(clusters.max()) + 1, size))
            np.add.at(summed, clusters, psi)
            scale = summed.shape[0] / n**2
        summed = summed - summed.mean(axis=0)
        meat = summed.T @ summed / (summed.shape[0] - 1)
        inverse = np.linalg.inv(bread)
        covariances.append((inverse @ meat @ inverse.T * scale)[:k, :k])
    return StandardizedSurvival(
        measure="survival",
        exposure=exposure,
        values=list(levels),
        times=np.asarray(times, dtype=float),
        estimates=estimates,
        covariances=covariances,
        n_obs=n,
        fits=[fit],
    )


def _coxph_centers(x: np.ndarray) -> np.ndarray:
    """Column means, except 0 for columns whose values all lie in {-1, 0, 1} (as coxph)."""
    centers = x.mean(axis=0) if x.shape[0] else np.zeros(x.shape[1])
    coded = np.all(np.isin(x, [-1.0, 0.0, 1.0]), axis=0)
    return np.where(coded, 0.0, centers)


def _constant(frame: pl.DataFrame, name: str, value: object) -> pl.Series:
    dtype = frame.schema[name]
    series = pl.Series(name, [value] * frame.height)
    try:
        return series.cast(dtype, strict=True)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        if dtype.is_numeric():
            return series.cast(pl.Float64)
        raise ValueError(f"value {value!r} does not fit column {name!r} ({dtype})") from error


# ---------------------------------------------------------------- rmean


def _rmean(data, parsed, exposure, levels, tstar) -> StandardizedSurvival:
    """Standardized restricted mean survival up to ``tstar`` for a 0/1 exposure,
    with its 2 x 2 covariance. Returns ``StandardizedSurvival`` with
    ``measure="rmean"``.

    Chen & Tsiatis (2001): a Cox model (Efron ties) is fitted in each exposure
    group, and mu_a = n^-1 sum_i int_0^L S_a(t | Z_i) dt over all n subjects
    (eq. 4). The variance follows the Appendix estimator of var(phi_{0,i})
    (terms A.1-A.3), divided by n: the subject term (A.3), shared by the two
    groups; the martingale term int h(t)^2 / S0(t) dLambda(t) (A.2); and the
    coefficient term g' Sigma g (A.1), where Sigma integrates only over [0, L].
    The variance uses Breslow increments dN / S0.
    """
    if len(levels) != 2 or set(map(_as_number, levels)) != {0.0, 1.0}:
        raise ValueError("the restricted mean needs values [0, 1] for a 0/1 exposure")
    design = parsed.design
    column = data[exposure]
    if not column.dtype.is_numeric() and column.dtype != pl.Boolean:
        raise ValueError(f"the exposure {exposure!r} must be numeric 0/1")
    exposure_values = np.asarray(
        column.cast(pl.Float64).gather(design.row_index.tolist()).to_numpy(), dtype=float
    )
    present = ~np.isnan(exposure_values)
    if not np.all(np.isin(exposure_values[present], [0.0, 1.0])):
        raise ValueError(f"the exposure {exposure!r} must be coded 0/1")
    keep = _columns_without(design, exposure)
    if not keep:
        raise ValueError("the restricted mean needs a covariate besides the exposure")
    covariates = np.asarray(design.x[:, keep], dtype=float)[present]
    time = np.asarray(parsed.time, dtype=float)[present]
    event = np.asarray(parsed.event, dtype=float)[present] > 0
    group = exposure_values[present]
    n = time.size
    covariates = covariates - covariates.mean(axis=0)
    by_level: dict[float, tuple[float, np.ndarray, float]] = {}
    for level in (0.0, 1.0):
        by_level[level] = _group_rmean(covariates, time, event, group == level, float(tstar))
    order = [_as_number(level) for level in levels]
    means = np.array([by_level[level][0] for level in order])
    subject = np.column_stack([by_level[level][1] for level in order])
    centered = subject - subject.mean(axis=0)
    covariance = centered.T @ centered / n**2
    for k, level in enumerate(order):
        covariance[k, k] += by_level[level][2]
    return StandardizedSurvival(
        measure="rmean",
        exposure=exposure,
        values=list(levels),
        times=np.array([float(tstar)]),
        estimates=means[None, :],
        covariances=[covariance],
        n_obs=n,
        fits=[],
    )


def _as_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _group_rmean(z: np.ndarray, time: np.ndarray, event: np.ndarray, in_group: np.ndarray, tstar: float):
    """Restricted mean of one group's Cox model averaged over all subjects.

    Returns the mean, the per-subject restricted means and the group's
    model-based variance terms (Breslow martingale and coefficient terms).
    """
    zg = z[in_group]
    tg = time[in_group]
    eg = event[in_group]
    m = int(in_group.sum())
    ones = np.ones(m)
    zeros = np.zeros(m)
    one_stratum = np.full(m, "_", dtype=object)
    beta, _info, _converged, _ll = _newton(zg, tg, eg, ones, zeros, zeros, one_stratum, "efron")
    jump_times, jumps, _labels = _baseline(zg, beta, tg, eg, ones, zeros, zeros, one_stratum, "efron")
    jump_times = np.asarray(jump_times, dtype=float)
    jumps = np.asarray(jumps, dtype=float)
    within = jump_times <= tstar
    event_times = jump_times[within]
    d_hazard = jumps[within]
    cumulative = np.concatenate([[0.0], np.cumsum(d_hazard)])
    # S_i is constant on [grid_k, grid_k+1) with cumulative hazard cumulative[k].
    grid = np.concatenate([[0.0], event_times, [tstar]])
    widths = np.diff(grid)
    risk_all = np.exp(z @ beta)
    surv = np.exp(-np.outer(risk_all, cumulative))
    pieces = surv * widths
    per_subject = pieces.sum(axis=1)
    mean = float(per_subject.mean())
    if event_times.size == 0:
        return mean, per_subject, 0.0
    # tail[:, k] = int_{grid_k}^{tstar} S_i ; g at event k uses grid index k + 1
    tail = np.cumsum(pieces[:, ::-1], axis=1)[:, ::-1]
    g = (risk_all[:, None] * tail[:, 1 : event_times.size + 1]).mean(axis=0)
    # Chen & Tsiatis (2001), Appendix: the variance estimator uses the
    # Breslow increments dN / S0 (the survival itself uses the Efron fit), and
    # Sigma_0 integrates only over [0, L].
    risk_group = np.exp(zg @ beta)
    xz = risk_group[:, None] * zg
    order = np.argsort(tg, kind="mergesort")
    tail0 = np.concatenate([np.cumsum(risk_group[order][::-1])[::-1], [0.0]])
    tail1 = np.concatenate([np.cumsum(xz[order][::-1], axis=0)[::-1], np.zeros((1, zg.shape[1]))])
    tail2 = np.concatenate(
        [np.cumsum((xz[:, :, None] * zg[:, None, :])[order][::-1], axis=0)[::-1], np.zeros((1, zg.shape[1], zg.shape[1]))]
    )
    at = np.searchsorted(tg[order], event_times, side="left")
    s0 = tail0[at]
    zbar = tail1[at] / s0[:, None]
    deaths = np.bincount(
        np.searchsorted(event_times, tg[eg & (tg <= tstar)]), minlength=event_times.size
    ).astype(float)
    breslow = deaths / s0
    # Sigma_0 (times n_0): sum over deaths up to L of the risk-set covariance.
    covariance_z = tail2[at] / s0[:, None, None] - zbar[:, :, None] * zbar[:, None, :]
    information = np.tensordot(deaths, covariance_z, axes=1)
    # (A.2): int h(t)^2 / S0(t) dLambda(t), h(t) = g above.
    martingale_term = float(np.sum(g**2 * breslow / s0))
    # (A.1): g_0 = n^-1 sum_i int_0^L int_0^u S(u|Z_i) r_i {Zbar(t) - Z_i} dLambda(t) du
    weighted_tail = (tail[:, 1 : event_times.size + 1] @ breslow) * risk_all
    h = (g * breslow) @ zbar - (weighted_tail @ z) / z.shape[0]
    coefficient_term = float(h @ np.linalg.solve(information, h))
    return mean, per_subject, martingale_term + coefficient_term


def _columns_without(design: Design, exposure: str) -> list[int]:
    """Design columns whose term does not use ``exposure``."""
    computed = design.computed or {}
    if design.recipes is None:
        raise ValueError("a Surv formula is required")
    keep = []
    for j, recipe in enumerate(design.recipes):
        sources: set[str] = set()
        for symbol, _level in recipe:
            sources.update(_columns_in(computed[symbol]) if symbol in computed else [symbol])
        if exposure not in sources:
            keep.append(j)
    return keep
