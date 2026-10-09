"""Regression standardization after a Cox model, matching ``stdReg2::standardize_coxph``.

``measure="survival"`` averages the Breslow Cox survival over the sample with the
exposure set to each value. The variance is the sandwich of Sjölander (2016),
which includes the spread of the covariates. ``measure="rmean"`` is the
restricted mean survival time of Chen and Tsiatis (2001), with a separate Efron
Cox model in each exposure group.

The arithmetic follows stdReg2 1.0.8 (``R/coxph_methods.R`` and ``R/utils.R``).
Three places differ on purpose for ``rmean``; see ``docs/models/vs-r.md``.
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
    rows = parsed.design.row_index
    used = data[rows.tolist()]
    labels = None
    if cluster is not None:
        series = used.get_column(cluster)
        if series.null_count():
            raise ValueError(f"cluster column {cluster!r} has missing values in the rows the model uses")
        labels = series.to_numpy()
    fit = cox_ph(used, formula, ties="breslow")
    x = fit.x
    n, p = x.shape
    beta = fit.coefficients
    # coxph centers each column at its mean, except columns of only -1, 0, 1.
    center = np.where(np.all(np.isin(x, (-1.0, 0.0, 1.0)), axis=0), 0.0, x.mean(axis=0))
    risk = np.exp((x - center) @ beta)
    time = fit.time
    event = fit.event > 0

    # coxph.detail with Breslow ties and unit weights.
    event_times = np.unique(time[event])
    if not bool(np.any(event_times <= times.min())):
        raise ValueError("No events before first value in times")
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    at_risk_from = np.searchsorted(sorted_time, event_times, side="left")
    suffix_r = np.concatenate([np.cumsum(risk[order][::-1])[::-1], [0.0]])
    suffix_rx = np.vstack([np.cumsum((risk[:, None] * x)[order][::-1], axis=0)[::-1], np.zeros((1, p))])
    s0 = suffix_r[at_risk_from]
    s1 = suffix_rx[at_risk_from]
    n_event = np.array([np.sum(event & (time == t)) for t in event_times], dtype=float)
    d_hazard = n_event / s0
    var_hazard = n_event / s0**2
    risk_means = s1 / s0[:, None] - center
    cum_hazard = np.cumsum(d_hazard)
    cum_var = np.cumsum(var_hazard)

    def step(values, at):
        k = np.searchsorted(event_times, at, side="right")
        return np.where(k > 0, np.concatenate([[0.0], values])[k], 0.0)

    # sandwich(): a row whose time equals an event time takes that time's
    # increment, censored or not (R matches rows to times by value).
    row_event = np.searchsorted(event_times, time)
    on_event = (row_event < len(event_times)) & (event_times[np.minimum(row_event, len(event_times) - 1)] == time)
    row_event = np.where(on_event, row_event, 0)
    score = fit.score_contributions()
    i_coef = -fit.information / n

    counterfactual = []
    for level in levels:
        frame = used.with_columns(_constant(used, exposure, level))
        design = build_design(frame, fit.design)
        counterfactual.append(design.x)

    n_values = len(levels)
    estimates = np.zeros((len(times), n_values))
    covariances = []
    for j, t in enumerate(times):
        if t == 0:
            estimates[j] = 1.0
            covariances.append(np.zeros((n_values, n_values)))
            continue
        h_t = float(step(cum_hazard, np.array([t]))[0])
        increment = np.where(on_event & (time <= t) & (event_times[row_event] <= t), d_hazard[row_event] / n_event[row_event], 0.0)
        tmp1 = n * increment
        tmp2 = n * step(cum_var, np.minimum(t, time)) * risk
        u_hazard = tmp1 - tmp2
        i_hazard = -np.mean(np.where(on_event[:, None], risk_means[row_event], 0.0) * tmp1[:, None], axis=0)

        surv = np.zeros((n, n_values))
        pred = np.zeros((n, n_values))
        temp = np.zeros((n_values, p))
        for k, xk in enumerate(counterfactual):
            pred_k = np.exp((xk - center) @ beta)
            surv[:, k] = np.exp(-h_t * pred_k)
            pred[:, k] = pred_k
            temp[k] = np.mean((xk - center) * (pred_k * surv[:, k])[:, None], axis=0)
        est = surv.mean(axis=0)
        estimates[j] = est

        res = np.column_stack([surv - est, score, u_hazard])
        if labels is not None:
            _, inverse = np.unique(labels, return_inverse=True)
            summed = np.zeros((inverse.max() + 1, res.shape[1]))
            np.add.at(summed, inverse, res)
            res = summed
        meat = np.cov(res, rowvar=False, ddof=1)
        top = np.hstack([-np.eye(n_values), -temp * h_t, -np.mean(pred * surv, axis=0)[:, None]])
        inner = np.zeros((p + 1, p + 1))
        inner[:p, :p] = i_coef
        inner[p, :p] = i_hazard
        inner[p, p] = -1.0
        bottom = np.hstack([np.zeros((p + 1, n_values)), inner])
        bread_inv = np.linalg.inv(np.vstack([top, bottom]))
        scale = 1 / n if labels is None else res.shape[0] / n**2
        full = bread_inv @ meat @ bread_inv.T * scale
        covariances.append(full[:n_values, :n_values])
    return StandardizedSurvival(
        measure="survival",
        exposure=exposure,
        values=levels,
        times=times,
        estimates=estimates,
        covariances=covariances,
        n_obs=n,
        fits=[fit],
    )


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
    if sorted(levels) != [0, 1] or len(levels) != 2:
        raise ValueError("the restricted mean needs values [0, 1] for a 0/1 exposure")
    if not data.schema[exposure].is_numeric():
        raise ValueError(f"the exposure {exposure!r} must be numeric 0/1")
    design = parsed.design
    group_all = data.get_column(exposure).gather(design.row_index.tolist())
    keep = group_all.is_not_null().to_numpy()
    group = group_all.to_numpy()[keep].astype(float)
    if not bool(np.all((group == 0) | (group == 1))):
        raise ValueError(f"the exposure {exposure!r} must be coded 0/1")
    columns = _columns_without(design, exposure)
    if not columns:
        raise ValueError("the restricted mean needs a covariate besides the exposure")
    x = design.x[keep][:, columns]
    time = parsed.time[keep]
    event = np.asarray(parsed.event, dtype=float)[keep] > 0
    n = x.shape[0]
    a = group.astype(int)

    events = np.flatnonzero(event & (time <= tstar))
    events = events[np.argsort(time[events], kind="stable")]
    etimes = time[events]
    event_group = a[events]
    grid = np.concatenate([[0.0], etimes])
    widths_r = _rsum_widths(grid, tstar)
    at_risk = (time[:, None] >= etimes[None, :]).astype(float)
    zz = np.einsum("ij,ik->ijk", x, x)

    rmst = np.zeros(2)
    person = np.zeros((n, 2))
    variance = np.zeros(2)
    for g in (0, 1):
        sel = a == g
        # coxph(newformula, data = group g): Efron ties, no weights, one stratum.
        m = int(sel.sum())
        plain = (np.ones(m), np.zeros(m), np.zeros(m), np.array(["_"] * m))
        beta, _info, _conv, _ll = _newton(x[sel], time[sel], event[sel], *plain, "efron")
        base_t, base_h, _ = _baseline(x[sel], beta, time[sel], event[sel], *plain, "efron")
        order = np.argsort(base_t)
        cum = np.cumsum(base_h[order])
        k = np.searchsorted(base_t[order], etimes, side="right")
        hazard_at = np.where(k > 0, np.concatenate([[0.0], cum])[k], 0.0)
        risk = np.exp(x @ beta)
        expected = risk[:, None] * hazard_at[None, :]
        surv = np.exp(-expected)
        curve = np.concatenate([[1.0], surv.mean(axis=0)])
        rmst[g] = float(widths_r @ curve)
        person[:, g] = np.column_stack([np.ones(n), surv]) @ widths_r

        widths = np.diff(np.concatenate([[0.0], etimes, [tstar]]))
        area = widths[None, :] * np.column_stack([np.ones(n), surv]) * risk[:, None]
        tail = np.cumsum(area[:, ::-1], axis=1)[:, ::-1]
        h_hat = tail.mean(axis=0)[1:]

        weight = (sel * risk)[:, None] * at_risk
        ss0 = weight.mean(axis=0)
        ss1 = weight.T @ x / n
        ss2 = np.einsum("ik,ijl->kjl", weight, zz) / n
        denom = weight.sum(axis=0)
        mine = event_group == g
        with np.errstate(divide="ignore", invalid="ignore"):
            zbar = np.where(ss0[:, None] > 0, ss1 / ss0[:, None], 0.0)
            spread = ss2 / ss0[:, None, None] - np.einsum("kj,kl->kjl", zbar, zbar)
        sigma = spread[mine].sum(axis=0) / sel.sum()
        inv_denom = np.where(mine, 1 / np.where(mine, denom, 1.0), 0.0)
        cum_p = np.cumsum(inv_denom[:, None] * zbar, axis=0)
        cum_q = np.cumsum(inv_denom)
        w_curve = widths_r[None, 1:] * surv * risk[:, None]
        g_i = w_curve @ cum_p - (w_curve @ cum_q)[:, None] * x
        g_vec = np.linalg.solve(sigma, g_i.sum(axis=0)) / sel.sum()
        term_beta = sel.sum() / n * float(g_vec @ sigma @ g_vec)
        term_hazard = float(np.sum(np.where(mine, h_hat**2 / np.where(mine, ss0 * denom, 1.0), 0.0)))
        term_person = (n - 1) / n * float(np.var(person[:, g], ddof=1))
        variance[g] = (term_beta + term_hazard + term_person) / n
    covar = (n - 1) / n**2 * float(np.cov(person[:, 1], person[:, 0], ddof=1)[0, 1])
    cov01 = np.array([[variance[0], covar], [covar, variance[1]]])
    order = [0 if level == 0 else 1 for level in levels]
    return StandardizedSurvival(
        measure="rmean",
        exposure=exposure,
        values=levels,
        times=np.array([tstar]),
        estimates=rmst[order][None, :],
        covariances=[cov01[np.ix_(order, order)]],
        n_obs=n,
    )


def _rsum_widths(grid: np.ndarray, tmax: float) -> np.ndarray:
    """Weights of ``rsum(y, grid, tmax)``: rectangles up to ``tmax``, zero past it."""
    keep = grid < tmax
    out = np.zeros(len(grid))
    out[keep] = np.diff(np.concatenate([grid[keep], [tmax]]))
    return out


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
