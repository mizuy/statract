"""Restricted mean survival time (RMST) for one or two groups.

The RMST at ``tau`` is the area under the Kaplan–Meier curve from 0 to
``tau`` (Royston and Parmar 2013; Klein and Moeschberger 2003, section 4.5).
Its variance is the Greenwood-type estimate given by Klein and Moeschberger,

``sum over event times t_i <= tau of A(t_i)^2 d_i / (Y_i (Y_i - d_i))``,

where ``A(t_i)`` is the area under the curve from ``t_i`` to ``tau``, ``d_i``
the number of events and ``Y_i`` the number at risk. A time where every
subject at risk fails adds 0. No ``n / (n - 1)`` factor is applied. The
restricted mean time lost (RMTL) is ``tau - RMST`` with the same standard
error.

For two groups the contrasts are those of Uno et al. (2014): the difference
``RMST1 - RMST0`` with a normal interval, and the ratios ``RMST1 / RMST0``
and ``RMTL1 / RMTL0`` with intervals and tests on the log scale (delta
method, the two groups independent).

References:
    Royston P, Parmar MKB. Restricted mean survival time: an alternative to
    the hazard ratio for the design and analysis of randomized trials with a
    time-to-event outcome. BMC Med Res Methodol 2013;13:152.

    Uno H, Claggett B, Tian L, et al. Moving beyond the hazard ratio in
    quantifying the between-group difference in survival analysis.
    J Clin Oncol 2014;32:2380-2385.

    Klein JP, Moeschberger ML. Survival Analysis: Techniques for Censored and
    Truncated Data, 2nd ed. Springer; 2003.

The numbers are checked against R (``summary(survfit(...), rmean = tau)`` and
``survRM2::rmst2`` run as black boxes); no R code is used here.
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
    """``arms`` has one row per group, ``contrasts`` the three two-group contrasts (two groups only)."""

    tau: float
    level: float
    arms: pl.DataFrame
    contrasts: pl.DataFrame | None
    groups: list[object]
    reference: object | None

    def frame(self) -> pl.DataFrame:
        return self.arms


def _km_area(time: np.ndarray, event: np.ndarray, tau: float) -> tuple[float, float]:
    """Area under the Kaplan–Meier curve on ``[0, tau]`` and its Greenwood-type variance."""
    uniq, at_risk, events, _ = _tie_counts(time, event, np.ones(time.shape[0], dtype=float))
    keep = uniq <= tau
    uniq, at_risk, events = uniq[keep], at_risk[keep], events[keep]
    # Survival just after each distinct time; the curve is a step function.
    survival = np.cumprod(1.0 - events / at_risk)
    # Step k covers [left[k], right[k]) at height before[k].
    left = np.concatenate(([0.0], uniq))
    right = np.concatenate((uniq, [tau]))
    before = np.concatenate(([1.0], survival))
    pieces = before * (right - left)
    total = float(pieces.sum())
    # Area from each distinct time to tau: the pieces to the right of it.
    tail = np.cumsum(pieces[::-1])[::-1][1:]
    alive = at_risk - events
    has_event = (events > 0) & (alive > 0)
    terms = np.zeros_like(tail)
    terms[has_event] = tail[has_event] ** 2 * events[has_event] / (at_risk[has_event] * alive[has_event])
    return total, float(terms.sum())


def _largest_tau(time: np.ndarray, event: np.ndarray, codes: np.ndarray, k: int) -> float:
    """The largest ``tau`` where every group's curve is known.

    A group's curve is known up to its last observed time, and beyond it
    when every subject observed at that time had the event (the curve is then
    0). The limit is the largest last time if every group that ends earlier
    ends at 0, otherwise the smallest last time.
    """
    last = np.empty(k)
    ends_at_zero = np.empty(k, dtype=bool)
    for g in range(k):
        t = time[codes == g]
        e = event[codes == g]
        last[g] = t.max()
        ends_at_zero[g] = bool(np.all(e[t == last[g]] > 0))
    longest = last.max()
    earlier = last < longest
    if np.all(ends_at_zero[earlier]):
        return float(longest)
    return float(last.min())


def _default_reference(values: list[object]) -> object:
    try:
        if len(values) == 2 and set(values) == {0, 1}:
            return next(v for v in values if v == 0)
    except TypeError:
        pass
    return min(values)


def _two_sided(z: float) -> float:
    return float(2.0 * stats.norm.sf(abs(z)))


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
    """Unadjusted RMST per group and two-group contrasts.

    The RMST is the area under the Kaplan–Meier curve up to ``tau`` (Royston
    and Parmar 2013). Its standard error is the Greenwood-type estimate of
    Klein and Moeschberger (2003, section 4.5). The contrasts are the RMST
    difference, the RMST ratio and the RMTL ratio of Uno et al. (2014); the
    ratios have intervals and tests on the log scale. Results agree with
    ``survRM2::rmst2`` in R, which the tests run as a black box.

    ``event`` is 1 for an event and 0 for censoring. ``by`` may name a column
    with one or two values. The contrasts are the other group against
    ``reference``. The default reference is 0 when the values are 0 and 1,
    otherwise the first value in sorted order. ``tau`` may be at most the
    largest allowed value, which is also the default (the rule of
    ``survRM2::rmst2``): the largest of the groups' last observed times when
    every group whose follow-up ends earlier ends with events (its curve is
    then 0), otherwise the smallest.

    ``arms`` has columns ``group``, ``n``, ``rmst``, ``std_error``,
    ``conf_low``, ``conf_high``, ``rmtl``, ``rmtl_conf_low`` and
    ``rmtl_conf_high``, with the reference group first. ``contrasts`` has
    rows ``RMST difference``, ``RMST ratio`` and ``RMTL ratio`` and columns
    ``estimate``, ``std_error`` (on the log scale for the ratios),
    ``conf_low``, ``conf_high`` and ``p_value``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    times, raw_events, _, labels, codes, _ = _survival_columns(data, time, event, by, None)
    times = np.asarray(times, dtype=float)
    if times.shape[0] == 0:
        raise ValueError("no complete rows")
    events = np.asarray(raw_events, dtype=float)
    if not np.all((events == 0) | (events == 1)):
        raise ValueError("event must be 0/1")
    if np.any(times < 0):
        raise ValueError("time must be non-negative")
    k = len(labels)
    if k not in (1, 2):
        raise ValueError("by must have one or two values")

    if by is None:
        order = [0]
        ref = None
    else:
        if reference is not None and reference not in labels:
            raise ValueError("reference must be one of the group values")
        if k == 1:
            order = [0]
            ref = None
        else:
            ref = _default_reference(labels) if reference is None else reference
            first = labels.index(ref)
            order = [first, 1 - first]
            ref = labels[first]

    allowed = _largest_tau(times, events, codes, k)
    if tau is None:
        tau_used = allowed
    else:
        tau_used = float(tau)
        if not tau_used > 0:
            raise ValueError("tau must be positive")
        if tau_used > allowed:
            raise ValueError(f"tau must be at most {allowed}")

    z = float(stats.norm.ppf(0.5 + level / 2.0))
    groups = [labels[g] for g in order]
    n = np.array([int(np.sum(codes == g)) for g in order], dtype=np.int64)
    rmst = np.empty(len(order))
    var = np.empty(len(order))
    for j, g in enumerate(order):
        rmst[j], var[j] = _km_area(times[codes == g], events[codes == g], tau_used)
    se = np.sqrt(var)
    rmtl = tau_used - rmst
    arms = pl.DataFrame(
        {
            "group": pl.Series("group", groups),
            "n": n,
            "rmst": rmst,
            "std_error": se,
            "conf_low": rmst - z * se,
            "conf_high": rmst + z * se,
            "rmtl": rmtl,
            "rmtl_conf_low": rmtl - z * se,
            "rmtl_conf_high": rmtl + z * se,
        }
    )

    contrasts = None
    if len(order) == 2:
        diff = rmst[1] - rmst[0]
        diff_se = float(np.sqrt(var[0] + var[1]))
        # Delta method: Var(log X) is about Var(X) / X^2.
        log_rmst = float(np.log(rmst[1]) - np.log(rmst[0]))
        log_rmst_se = float(np.sqrt(var[1] / rmst[1] ** 2 + var[0] / rmst[0] ** 2))
        log_rmtl = float(np.log(rmtl[1]) - np.log(rmtl[0]))
        log_rmtl_se = float(np.sqrt(var[1] / rmtl[1] ** 2 + var[0] / rmtl[0] ** 2))
        contrasts = pl.DataFrame(
            {
                "contrast": ["RMST difference", "RMST ratio", "RMTL ratio"],
                "estimate": [float(diff), float(np.exp(log_rmst)), float(np.exp(log_rmtl))],
                "std_error": [diff_se, log_rmst_se, log_rmtl_se],
                "conf_low": [
                    float(diff - z * diff_se),
                    float(np.exp(log_rmst - z * log_rmst_se)),
                    float(np.exp(log_rmtl - z * log_rmtl_se)),
                ],
                "conf_high": [
                    float(diff + z * diff_se),
                    float(np.exp(log_rmst + z * log_rmst_se)),
                    float(np.exp(log_rmtl + z * log_rmtl_se)),
                ],
                "p_value": [
                    _two_sided(diff / diff_se),
                    _two_sided(log_rmst / log_rmst_se),
                    _two_sided(log_rmtl / log_rmtl_se),
                ],
            }
        )
    return RmstResult(tau=tau_used, level=level, arms=arms, contrasts=contrasts, groups=groups, reference=ref)
