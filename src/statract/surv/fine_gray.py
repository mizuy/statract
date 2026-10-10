"""Fine–Gray data expansion and follow-up splitting.

``fine_gray`` builds the weighted counting-process rows that ``cox_ph`` then
fits, matching ``survival::finegray``. ``split_follow_up`` matches ``survSplit``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from ..models.design import ColumnRef, column_series
from ..models.formula import is_formula, model_matrix
from .spec import _covariate_names


def fine_gray(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef | None = None,
    cause=None,
    *,
    entry: ColumnRef | None = None,
) -> pl.DataFrame:
    """Expand a competing-risks frame into Fine–Gray counting-process rows.

    ``event`` is the cause code, with ``0`` meaning censored. ``cause`` is the
    event of interest. The result has ``fgstart``, ``fgstop``, ``fgstatus``,
    and ``fgwt`` plus the original covariate columns.

    ``time`` may be a Wilkinson formula. ``fine_gray(data, "Surv(time, status) ~ x + sex", cause=1)``
    keeps the columns named on the right-hand side. ``cause`` may also be the
    third positional argument.
    """
    if is_formula(time):
        if event is not None and cause is not None:
            raise ValueError("a Surv formula already names the time and the event; pass cause once")
        if cause is None:
            cause = event
        if cause is None:
            raise ValueError("cause is required")
        return _fine_gray_formula(data, str(time), cause, entry=entry)
    if event is None or cause is None:
        raise ValueError("time, event, and cause are required when time is a column")
    time_s = column_series(data, time)
    event_s = column_series(data, event)
    time_a = np.asarray(time_s.to_numpy(), dtype=float)
    status = np.asarray(event_s.to_numpy())
    if entry is None:
        start = np.zeros(len(time_a))
        if np.nanmin(time_a) <= 0:
            start[:] = 2 * np.nanmin(time_a) - 1
    else:
        start = np.asarray(column_series(data, entry).to_numpy(), dtype=float)
    cause_value = _as_cause(cause, status)
    status_num = _cause_codes(status)
    cause_num = _cause_codes(np.array([cause_value]))[0]
    utime = np.unique(np.concatenate([start, time_a]))
    utime.sort()
    stop_index = np.searchsorted(utime, time_a, side="right").astype(float)
    start_index = np.searchsorted(utime, start, side="right").astype(float)
    censored = status_num == 0
    stop_index = stop_index.copy()
    stop_index[~censored] -= 0.2
    g_time, g_surv = _censoring_survival(start_index, stop_index, censored)
    # Map the censoring curve back to the original time scale.
    ctime = np.array([utime[int(t) - 1] for t in g_time if t >= 1], dtype=float)
    cprob = np.array([s for t, s in zip(g_time, g_surv, strict=False) if t >= 1], dtype=float)
    maxtime = float(np.nanmax(time_a))
    ct2 = np.concatenate([ctime, [maxtime]]) if ctime.size else np.array([maxtime])
    cp2 = np.concatenate([[1.0], cprob]) if cprob.size else np.array([1.0])
    if ct2.shape[0] != cp2.shape[0]:
        # A censoring time can fall on the last observed time; keep the pairs aligned.
        m = min(ct2.shape[0], cp2.shape[0])
        ct2, cp2 = ct2[:m], cp2[:m]
    interest = np.flatnonzero(status_num == cause_num)
    event_times = np.unique(time_a[interest]) if interest.size else np.array([])
    positions = np.searchsorted(ct2, event_times, side="left") if event_times.size else np.array([], dtype=int)
    ckeep = np.zeros(ct2.shape[0], dtype=bool)
    for pos in positions:
        if 1 <= int(pos) <= ckeep.shape[0]:
            ckeep[int(pos) - 1] = True
    keep = np.concatenate([[True], ckeep])
    expand = (status_num != 0) & (status_num != cause_num)
    row, fg_start, fg_stop, wt = _expand(start, time_a, ct2, cp2, expand, keep)
    covariates = data.drop([time_s.name, event_s.name])
    if entry is not None and isinstance(entry, str) and entry in covariates.columns:
        covariates = covariates.drop(entry)
    taken = covariates[row.tolist()]
    return taken.with_columns(
        pl.Series("fgstart", fg_start),
        pl.Series("fgstop", fg_stop),
        pl.Series("fgstatus", (status_num[row] == cause_num).astype(float)),
        pl.Series("fgwt", wt),
    )


def _fine_gray_formula(data: pl.DataFrame, formula: str, cause, *, entry: ColumnRef | None) -> pl.DataFrame:
    built = model_matrix(formula, data)
    if built.surv is None:
        raise ValueError("fine_gray formulas start with Surv(time, status)")
    if built.random_effects:
        raise ValueError("random effects belong in fit_mixed, for example (1 | group)")
    if built.surv.entry is not None and entry is not None:
        raise ValueError("Surv(start, stop, status) already names the entry time")
    entry_name = built.surv.entry if built.surv.entry is not None else entry
    columns = [built.surv.time, built.surv.event]
    if isinstance(entry_name, str):
        columns.append(entry_name)
    for name in _covariate_names(built.design):
        if name not in columns:
            columns.append(name)
    for name in (*built.strata, built.cluster):
        if name is not None and name not in columns:
            columns.append(name)
    subset = data.select(columns).drop_nulls()
    return fine_gray(subset, built.surv.time, built.surv.event, cause, entry=entry_name)


def split_follow_up(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    cuts: list[float] | np.ndarray,
    *,
    entry: ColumnRef | None = None,
) -> pl.DataFrame:
    """Split each follow-up at ``cuts``, matching ``survSplit``.

    The time column becomes the end of the interval. A ``tstart`` column and
    an ``episode`` column are added. The event is kept only on the interval
    that contains it.
    """
    time_s = column_series(data, time)
    event_s = column_series(data, event)
    time_a = np.asarray(time_s.to_numpy(), dtype=float)
    event_a = np.asarray(event_s.to_numpy(), dtype=float)
    start_a = (
        np.zeros(len(time_a))
        if entry is None
        else np.asarray(column_series(data, entry).to_numpy(), dtype=float)
    )
    boundaries = np.unique(np.asarray(list(cuts), dtype=float))
    rows: list[int] = []
    tstart: list[float] = []
    tstop: list[float] = []
    status: list[float] = []
    episode: list[int] = []
    for i, (left0, right, ev) in enumerate(zip(start_a, time_a, event_a, strict=False)):
        points = [float(left0), *[float(c) for c in boundaries if left0 < c < right], float(right)]
        for k in range(len(points) - 1):
            rows.append(i)
            tstart.append(points[k])
            tstop.append(points[k + 1])
            hit = ev != 0 and points[k + 1] == float(right)
            status.append(float(ev) if hit else 0.0)
            episode.append(k + 1)
    taken = data[rows]
    return taken.with_columns(
        pl.Series("tstart", tstart),
        pl.Series(time_s.name, tstop),
        pl.Series(event_s.name, status),
        pl.Series("episode", episode),
    )


def _expand(start, stop, ctime, cprob, extend, keep):
    start_a = np.ascontiguousarray(start, dtype=np.float64)
    stop_a = np.ascontiguousarray(stop, dtype=np.float64)
    ctime_a = np.ascontiguousarray(ctime, dtype=np.float64)
    cprob_a = np.ascontiguousarray(cprob, dtype=np.float64)
    extend_a = np.ascontiguousarray(extend, dtype=np.bool_)
    keep_a = np.ascontiguousarray(keep, dtype=np.bool_)
    total = int(_expand_count(start_a, stop_a, ctime_a, cprob_a, extend_a, keep_a))
    rows = np.empty(total, dtype=np.int64)
    starts = np.empty(total, dtype=np.float64)
    ends = np.empty(total, dtype=np.float64)
    wts = np.empty(total, dtype=np.float64)
    _expand_fill(start_a, stop_a, ctime_a, cprob_a, extend_a, keep_a, rows, starts, ends, wts)
    return rows, starts, ends, wts


@njit(cache=True)
def _expand_count(start, stop, ctime, cprob, extend, keep):
    n = start.shape[0]
    ncut = cprob.shape[0]
    nkeep = keep.shape[0]
    total = 0
    for i in range(n):
        total += 1
        if not (np.isfinite(start[i]) and np.isfinite(stop[i]) and extend[i]):
            continue
        j = 0
        while j < ncut and ctime[j] < stop[i]:
            j += 1
        j += 1
        while j < ncut:
            if j < nkeep and keep[j]:
                total += 1
            j += 1
    return total


@njit(cache=True)
def _expand_fill(start, stop, ctime, cprob, extend, keep, rows, starts, ends, wts):
    n = start.shape[0]
    ncut = cprob.shape[0]
    nkeep = keep.shape[0]
    cursor = 0
    for i in range(n):
        rows[cursor] = i
        starts[cursor] = start[i]
        ends[cursor] = stop[i]
        wts[cursor] = 1.0
        if not (np.isfinite(start[i]) and np.isfinite(stop[i]) and extend[i]):
            cursor += 1
            continue
        j = 0
        while j < ncut and ctime[j] < stop[i]:
            j += 1
        if j < ncut:
            ends[cursor] = ctime[j]
        tempwt = cprob[j] if j < ncut else 1.0
        cursor += 1
        j += 1
        while j < ncut:
            if j < nkeep and keep[j]:
                rows[cursor] = i
                starts[cursor] = ctime[j - 1]
                ends[cursor] = ctime[j]
                wts[cursor] = cprob[j] / tempwt
                cursor += 1
            j += 1


def _censoring_survival(start: np.ndarray, stop: np.ndarray, censored: np.ndarray):
    """Kaplan–Meier of the censoring distribution on the (start, stop] scale."""
    times = np.unique(stop[censored])
    times.sort()
    surv = 1.0
    out_t = []
    out_s = []
    for t in times:
        risk = (start < t) & (stop >= t)
        deaths = censored & (stop == t)
        n_risk = int(risk.sum())
        n_death = int(deaths.sum())
        if n_risk > 0 and n_death > 0:
            surv *= 1.0 - n_death / n_risk
            out_t.append(float(t))
            out_s.append(surv)
    return out_t, out_s


def _cause_codes(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[i] = 0
        elif isinstance(value, (bool, np.bool_)):
            out[i] = int(value)
        else:
            try:
                out[i] = int(value)
            except (TypeError, ValueError):
                out[i] = int(float(value))
    return out


def _as_cause(cause, status: np.ndarray):
    if isinstance(cause, str) and cause in set(map(str, status.tolist())):
        # A label such as "1" still compares numerically when the column is numeric.
        try:
            return int(float(cause))
        except ValueError:
            return cause
    return cause
