"""Fine–Gray data expansion and follow-up splitting.

``fine_gray`` builds the weighted counting-process rows that ``cox_ph`` then
fits, matching ``survival::finegray``. ``split_follow_up`` matches ``survSplit``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

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

    Inputs: ``time`` (column or ``Surv(...) ~ covariates`` formula), ``event``
    (cause code, ``0`` = censored), ``cause`` (event of interest) and optional
    ``entry``. Output: a polars frame with the covariate columns plus
    ``fgstart``, ``fgstop``, ``fgstatus`` and ``fgwt``.

    A row that fails from a competing cause at T stays in the risk set after T
    with weight G(t-)/G(T-), G being the Kaplan–Meier estimate of the
    censoring distribution (Fine & Gray 1999, sec. 3; Geskus 2011, sec. 2).
    The weight is a step function, so the row is split at the jumps of G and
    pieces without a failure of interest are left out (they add nothing to
    the partial likelihood). Output is compared with R's ``finegray()``.
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
    stop = np.asarray(time_s.cast(pl.Float64).to_numpy(), dtype=float)
    codes = _event_codes(event_s)
    target = _cause_code(cause)
    dropped = {time_s.name, event_s.name}
    if entry is None:
        lowest = float(np.min(stop)) if stop.size else 0.0
        start = np.full(stop.size, 0.0 if lowest > 0 else 2 * lowest - 1)
    else:
        entry_s = column_series(data, entry)
        start = np.asarray(entry_s.cast(pl.Float64).to_numpy(), dtype=float)
        if isinstance(entry, str):
            dropped.add(entry_s.name)
    rows, fg_start, fg_stop, fg_status, fg_weight = _expand(start, stop, codes, target)
    keep = [name for name in data.columns if name not in dropped]
    out = data.select(keep)[rows] if keep else pl.DataFrame({"_": np.zeros(len(rows))}).drop("_")
    return out.with_columns(
        pl.Series("fgstart", fg_start, dtype=pl.Float64),
        pl.Series("fgstop", fg_stop, dtype=pl.Float64),
        pl.Series("fgstatus", fg_status, dtype=pl.Float64),
        pl.Series("fgwt", fg_weight, dtype=pl.Float64),
    )


def _event_codes(series: pl.Series) -> np.ndarray:
    """Integer event codes: null/NaN is 0 (censored), booleans are 0/1."""
    if series.dtype == pl.Boolean:
        return series.fill_null(False).cast(pl.Int64).to_numpy()
    if series.dtype in (pl.Utf8, pl.Categorical) or isinstance(series.dtype, pl.Enum):
        text = series.cast(pl.Utf8)
        numeric = text.cast(pl.Float64, strict=False)
        bad = numeric.is_null() & text.is_not_null()
        if bool(bad.any()):
            raise ValueError("event codes must be numbers (0 = censored)")
        series = numeric
    values = np.asarray(series.cast(pl.Float64).to_numpy(), dtype=float)
    values = np.where(np.isnan(values), 0.0, values)
    return values.astype(np.int64)


def _cause_code(cause) -> int:
    if isinstance(cause, bool | np.bool_):
        return int(cause)
    if isinstance(cause, str):
        try:
            return int(float(cause))
        except ValueError as error:
            raise ValueError(f"cause {cause!r} is not an event code") from error
    return int(cause)


def _expand(start: np.ndarray, stop: np.ndarray, codes: np.ndarray, target: int):
    """Rows of the weighted data set: source row, start, stop, status, weight."""
    censored = codes == 0
    # Kaplan–Meier of the censoring time with delayed entry. At a tie, failures
    # come first, so rows failing at c are not at risk of censoring at c.
    censor_times = np.unique(stop[censored])
    entry_sorted = np.sort(start)
    stop_sorted = np.sort(stop)
    n = stop.size
    after = n - np.searchsorted(stop_sorted, censor_times, side="right")
    entered = np.searchsorted(entry_sorted, censor_times, side="left")
    n_censored = np.bincount(np.searchsorted(censor_times, stop[censored]), minlength=censor_times.size)
    # at risk: entry < c and (time > c, or censored at c)
    at_risk = after + n_censored - (n - entered)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(at_risk > 0, 1.0 - n_censored / np.where(at_risk > 0, at_risk, 1), 1.0)
    surv_after = np.cumprod(factor)  # G just after each censoring time
    last = float(stop.max()) if n else 0.0
    breaks = censor_times
    g_after = surv_after
    if breaks.size == 0 or breaks[-1] < last:
        breaks = np.append(breaks, last)
        g_after = np.append(g_after, g_after[-1] if g_after.size else 1.0)
    target_times = np.unique(stop[codes == target])

    rows: list[int] = []
    fg_start: list[float] = []
    fg_stop: list[float] = []
    fg_status: list[float] = []
    fg_weight: list[float] = []
    for i in range(n):
        code = codes[i]
        if code == 0 or code == target:
            rows.append(i)
            fg_start.append(start[i])
            fg_stop.append(stop[i])
            fg_status.append(1.0 if code == target else 0.0)
            fg_weight.append(1.0)
            continue
        t_fail = stop[i]
        first = int(np.searchsorted(breaks, t_fail, side="left"))
        # G(T-): G after the last censoring time strictly before T.
        before = int(np.searchsorted(censor_times, t_fail, side="left"))
        g_fail = surv_after[before - 1] if before > 0 else 1.0
        rows.append(i)
        fg_start.append(start[i])
        fg_stop.append(breaks[first])
        fg_status.append(0.0)
        fg_weight.append(1.0)
        for j in range(first, breaks.size - 1):
            lo, hi = breaks[j], breaks[j + 1]
            k = int(np.searchsorted(target_times, lo, side="right"))
            if k >= target_times.size or target_times[k] > hi:
                continue
            rows.append(i)
            fg_start.append(lo)
            fg_stop.append(hi)
            fg_status.append(0.0)
            fg_weight.append(g_after[j] / g_fail)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(fg_start, dtype=float),
        np.asarray(fg_stop, dtype=float),
        np.asarray(fg_status, dtype=float),
        np.asarray(fg_weight, dtype=float),
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
