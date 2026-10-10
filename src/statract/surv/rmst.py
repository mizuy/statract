"""Restricted mean survival time (RMST) for one or two groups, matching ``survRM2::rmst2``.

The RMST at ``tau`` is the area under the Kaplan–Meier curve from 0 to ``tau``.
The variance is the Greenwood-type sum of ``survRM2::rmst1``,

``sum over times t_i <= tau of A(t_i)^2 d_i / (Y_i (Y_i - d_i))``,

where ``A(t_i)`` is the area under the curve from ``t_i`` to ``tau``. A time
where every subject at risk fails adds 0. This equals ``se(rmean)`` from
``summary(survfit(...), rmean = tau)`` in survival 3.5 without weights. The restricted mean time lost
(RMTL) is ``tau - RMST`` with the same standard error.

The two-group contrasts follow ``rmst2``: the difference ``RMST1 - RMST0``
with a normal interval, and the ratios ``RMST1 / RMST0`` and
``RMTL1 / RMTL0`` with intervals and tests on the log scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ..models.design import ColumnRef
from .curve import _survival_columns, _tie_counts


@dataclass
class RmstResult:
    """``arms`` has one row per group, ``contrasts`` the three ``rmst2`` contrasts (two groups only)."""

    tau: float
    level: float
    arms: pl.DataFrame
    contrasts: pl.DataFrame | None
    groups: list[object]
    reference: object | None

    def frame(self) -> pl.DataFrame:
        return self.arms


def _rmst_one(time: np.ndarray, event: np.ndarray, tau: float) -> tuple[float, float]:
    """RMST and its variance for one sample, as ``survRM2::rmst1``."""
    uniq, n_risk, n_event, _n_censor = _tie_counts(time, event, np.ones(time.shape[0]))
    idx = uniq <= tau
    wk_time = np.sort(np.concatenate([uniq[idx], [tau]]))
    surv = np.cumprod(1.0 - n_event / n_risk)[idx]
    risk = n_risk[idx]
    died = n_event[idx]
    areas = np.diff(np.concatenate([[0.0], wk_time])) * np.concatenate([[1.0], surv])
    rmst = float(np.sum(areas))
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard_var = np.where(risk - died == 0, 0.0, died / (risk * (risk - died)))
    # Area after each time t_i, up to tau.
    after = np.cumsum(areas[1:][::-1])[::-1]
    var = float(np.sum(after**2 * hazard_var))
    return rmst, var


def _max_tau(time: np.ndarray, event: np.ndarray, codes: np.ndarray, k: int) -> float:
    """The largest allowed tau, which ``rmst2`` also uses as the default."""
    last = np.array([time[codes == g].max() for g in range(k)])
    # The curve is defined past a group's last time only if every subject at that time failed.
    ended = np.array([bool(np.all(event[(codes == g) & (time == last[g])] > 0)) for g in range(k)])
    shorter = last < last.max()
    return float(last.max()) if np.all(ended[shorter]) else float(last.min())


def restricted_mean_survival(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    by: ColumnRef | None = None,
    *,
    tau: float | None = None,
    reference: object = None,
    level: float = 0.95,
) -> RmstResult:
    """Unadjusted RMST per group and two-group contrasts, as ``survRM2::rmst2``.

    ``event`` is 1 for an event and 0 for censoring. ``by`` may name a column
    with one or two values. The contrasts are the other group against
    ``reference`` (``arm = 1`` against ``arm = 0`` in ``rmst2``). The default
    reference is 0 when the values are 0 and 1, otherwise the first value in
    sorted order. ``tau`` may be at most the largest allowed value, which is
    also the default, following ``rmst2``: the largest of the groups' last
    observed times when every group whose follow-up ends earlier ends with
    events (its curve is then 0), otherwise the smallest.
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    times, events, _w, groups, codes, _entry = _survival_columns(data, time, event, by, None)
    events = np.asarray(events).astype(float)
    if not np.all((events == 0) | (events == 1)):
        raise ValueError("event must be 0/1")
    if len(groups) > 2:
        raise ValueError("by must have one or two values")
    if by is not None and len(groups) == 2:
        try:
            order = sorted(range(2), key=lambda i: groups[i])
        except TypeError:
            order = [0, 1]
        if reference is not None:
            if reference not in groups:
                raise ValueError("reference must be one of the group values")
            ref = groups.index(reference)
            order = [ref, 1 - ref]
        groups = [groups[i] for i in order]
        remap = np.empty(2, dtype=np.int32)
        remap[order[0]] = 0
        remap[order[1]] = 1
        codes = remap[codes]
    elif reference is not None:
        raise ValueError("reference needs two groups")
    k = len(groups)
    allowed = _max_tau(times, events, codes, k)
    if tau is None:
        tau = allowed
    else:
        tau = float(tau)
        if tau <= 0:
            raise ValueError("tau must be positive")
        if tau > allowed:
            raise ValueError(f"tau must be at most {allowed}")
    z = float(stats.norm.ppf(0.5 + level / 2))
    rmst = np.zeros(k)
    var = np.zeros(k)
    n = np.zeros(k, dtype=np.int64)
    for g in range(k):
        sel = codes == g
        rmst[g], var[g] = _rmst_one(times[sel], events[sel], tau)
        n[g] = int(sel.sum())
    se = np.sqrt(var)
    rmtl = tau - rmst
    arms = pl.DataFrame(
        {
            "group": groups if by is not None else [None],
            "n": n,
            "rmst": rmst,
            "std_error": se,
            "conf_low": rmst - z * se,
            "conf_high": rmst + z * se,
            "rmtl": rmtl,
            "rmtl_conf_low": rmtl - z * se,
            "rmtl_conf_high": rmtl + z * se,
        },
        strict=False,
    )
    contrasts = None
    if k == 2:
        diff = rmst[1] - rmst[0]
        diff_se = float(np.sqrt(var[0] + var[1]))
        log_ratio = np.log(rmst[1]) - np.log(rmst[0])
        ratio_se = float(np.sqrt(var[1] / rmst[1] ** 2 + var[0] / rmst[0] ** 2))
        log_rmtl = np.log(rmtl[1]) - np.log(rmtl[0])
        rmtl_se = float(np.sqrt(var[1] / rmtl[1] ** 2 + var[0] / rmtl[0] ** 2))
        log_est = np.array([log_ratio, log_rmtl])
        log_se = np.array([ratio_se, rmtl_se])
        contrasts = pl.DataFrame(
            {
                "contrast": ["RMST difference", "RMST ratio", "RMTL ratio"],
                "estimate": [diff, *np.exp(log_est)],
                "std_error": [diff_se, *log_se],
                "conf_low": [diff - z * diff_se, *np.exp(log_est - z * log_se)],
                "conf_high": [diff + z * diff_se, *np.exp(log_est + z * log_se)],
                "p_value": [
                    2 * stats.norm.sf(abs(diff) / diff_se),
                    *(2 * stats.norm.sf(np.abs(log_est) / log_se)),
                ],
            }
        )
    return RmstResult(
        tau=float(tau),
        level=level,
        arms=arms,
        contrasts=contrasts,
        groups=list(groups),
        reference=groups[0] if k == 2 else None,
    )
