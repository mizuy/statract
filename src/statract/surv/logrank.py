"""Weighted log-rank tests, including stratified and Peto–Peto weights.

The statistic matches ``survdiff``. ``rho=0`` is the Mantel–Haenszel log-rank
test and ``rho=1`` is the Peto–Peto test. Groups are compared within ``strata``
and the increments are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ..design import ColumnRef, column_series


@dataclass(frozen=True)
class LogRankResult:
    statistic: float
    p_value: float
    df: int
    observed: np.ndarray
    expected: np.ndarray
    groups: list[str]

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "group": self.groups,
                "observed": self.observed,
                "expected": self.expected,
            }
        )


def log_rank(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    by: ColumnRef,
    strata: ColumnRef | None = None,
    *,
    rho: float = 0.0,
) -> LogRankResult:
    """k-sample weighted log-rank test."""
    time_s = column_series(data, time).cast(pl.Float64)
    event_s = column_series(data, event)
    group_s = column_series(data, by)
    stratum_s = column_series(data, strata) if strata is not None else None
    mask = time_s.is_not_null() & event_s.is_not_null() & group_s.is_not_null()
    if stratum_s is not None:
        mask = mask & stratum_s.is_not_null()
    keep = mask.to_numpy()
    kept = pl.Series(keep)
    times = time_s.filter(kept).to_numpy().astype(float)
    events = np.asarray(event_s.filter(kept).to_numpy())
    group_kept = group_s.filter(kept)
    labels = group_kept.unique(maintain_order=True).to_list()
    group_codes = group_kept.replace_strict(labels, list(range(len(labels))), return_dtype=pl.Int32).to_numpy()
    if stratum_s is None:
        strata_codes = np.zeros(len(times), dtype=np.int32)
        n_strata = 1
    else:
        stratum_kept = stratum_s.filter(kept)
        stratum_labels = stratum_kept.unique(maintain_order=True).to_list()
        strata_codes = stratum_kept.replace_strict(
            stratum_labels, list(range(len(stratum_labels))), return_dtype=pl.Int32
        ).to_numpy()
        n_strata = len(stratum_labels)
    k = len(labels)
    observed = np.zeros(k)
    expected = np.zeros(k)
    variance = np.zeros((k, k))
    for stratum in range(n_strata):
        sel = strata_codes == stratum
        o, e, v = _stratum(times[sel], events[sel], group_codes[sel], k, rho)
        observed += o
        expected += e
        variance += v
    diff = (observed - expected)[:-1]
    block = variance[:-1, :-1]
    if diff.size == 0:
        stat = 0.0
    else:
        stat = float(diff @ np.linalg.pinv(block) @ diff)
    df = k - 1
    p_value = float(stats.chi2.sf(stat, df)) if df else 1.0
    return LogRankResult(stat, p_value, df, observed, expected, [str(g) for g in labels])


def _stratum(
    time: np.ndarray,
    event: np.ndarray,
    codes: np.ndarray,
    k: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted log-rank increments for one stratum.

    ``codes`` are group indices in ``0 .. k-1``. Each row counts as one
    observation, matching an unweighted ``survdiff``.
    """
    observed = np.zeros(k)
    expected = np.zeros(k)
    variance = np.zeros((k, k))
    if len(time) == 0 or k == 0:
        return observed, expected, variance
    order = np.argsort(time, kind="mergesort")
    time = np.asarray(time, dtype=float)[order]
    died = np.asarray(event)[order].astype(float) > 0
    codes = np.asarray(codes)[order]
    _uniq, start = np.unique(time, return_index=True)
    died_f = died.astype(float)
    death = np.column_stack([np.add.reduceat(died_f * (codes == i), start) for i in range(k)])
    leave = np.column_stack([np.add.reduceat((codes == i).astype(float), start) for i in range(k)])
    cumulative = np.cumsum(leave, axis=0)
    n_at = leave.sum(axis=0) - cumulative + leave
    d = death.sum(axis=1)
    n = n_at.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.divide(d, n, out=np.zeros_like(d), where=(d > 0) & (n > 0))
    survived = np.cumprod(1.0 - hazard)
    before = np.empty_like(survived)
    before[0] = 1.0
    if len(before) > 1:
        before[1:] = survived[:-1]
    weight = before**rho
    mask = (d > 0) & (n > 0)
    if not np.any(mask):
        return observed, expected, variance
    w = weight[mask]
    deaths = death[mask]
    at_risk = n_at[mask]
    n_mask = n[mask]
    d_mask = d[mask]
    observed = (w[:, None] * deaths).sum(axis=0)
    expected = (w[:, None] * at_risk * (d_mask / n_mask)[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.divide(
            d_mask * (n_mask - d_mask),
            n_mask**2 * (n_mask - 1.0),
            out=np.zeros_like(d_mask),
            where=n_mask > 1,
        )
    weighted_factor = (w**2) * factor
    outer = at_risk.T @ (at_risk * weighted_factor[:, None])
    diagonal = (at_risk * n_mask[:, None] * weighted_factor[:, None]).sum(axis=0)
    variance = np.diag(diagonal) - outer
    return observed, expected, variance
