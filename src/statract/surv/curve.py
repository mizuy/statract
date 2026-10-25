"""Kaplan–Meier, Nelson–Aalen, and Aalen–Johansen curves.

Confidence intervals follow ``survfit``: ``std_error`` is the standard error of
the log survival (Kaplan–Meier) or of the cumulative hazard (Nelson–Aalen).
The default interval is the log interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numba import njit
from scipy import stats

from ..models.design import ColumnRef, column_series

_CONFIDENCE = ("log", "log-log", "plain", "logit", "arcsin")
_Z_CACHE: dict[float, float] = {}


def _norm_z(level: float) -> float:
    """Standard-normal critical value. The common 95% point is computed once."""
    z = _Z_CACHE.get(level)
    if z is None:
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        _Z_CACHE[level] = z
    return z


@dataclass
class SurvivalCurve:
    """Step-function survival or cumulative-incidence estimate."""

    table: pl.DataFrame
    kind: str
    confidence: str
    level: float

    def frame(self) -> pl.DataFrame:
        return self.table

    def at(self, times: float | list[float] | np.ndarray) -> pl.DataFrame:
        """Right-continuous value at each requested time, within each group and state.

        ``n_risk`` follows ``summary(survfit(...), times=)``: the risk set at
        the first tabled time at or after ``t``, so the subjects with
        ``time >= t`` when there is no delayed entry. It is 0 past the last time.
        """
        asked = np.atleast_1d(np.asarray(times, dtype=float))
        table = self.table
        has_group = "group" in table.columns
        has_state = "state" in table.columns
        groups = table["group"].to_list() if has_group else [None] * table.height
        states = table["state"].to_list() if has_state else [None] * table.height
        keys = np.empty(table.height, dtype=object)
        keys[:] = list(zip(groups, states, strict=True))
        key_ids, key_codes = _appearance_codes(keys)
        known = table["time"].to_numpy()
        estimate = table["estimate"].to_numpy()
        std_error = table["std_error"].to_numpy()
        low = table["conf_low"].to_numpy()
        high = table["conf_high"].to_numpy()
        n_risk = table["n_risk"].to_numpy()
        before = 1.0 if self.kind != "aalen_johansen" else 0.0
        n_ask = asked.shape[0]
        n_out = n_ask * len(key_ids)
        out_time = np.empty(n_out, dtype=float)
        out_est = np.empty(n_out, dtype=float)
        out_se = np.empty(n_out, dtype=float)
        out_low = np.empty(n_out, dtype=float)
        out_high = np.empty(n_out, dtype=float)
        out_risk = np.empty(n_out, dtype=float)
        out_group: list[object] = []
        out_state: list[object] = []
        cursor = 0
        for index, (group, state) in enumerate(key_ids):
            sel = key_codes == index
            known_k = known[sel]
            idx = np.searchsorted(known_k, asked, side="right") - 1
            valid = idx >= 0
            end = cursor + n_ask
            out_time[cursor:end] = asked
            out_est[cursor:end] = before
            out_se[cursor:end] = 0.0
            out_low[cursor:end] = before
            out_high[cursor:end] = before
            out_risk[cursor:end] = _risk_at(known_k, n_risk[sel], asked)
            if np.any(valid):
                picked = idx[valid]
                out_est[cursor:end][valid] = estimate[sel][picked]
                out_se[cursor:end][valid] = std_error[sel][picked]
                out_low[cursor:end][valid] = low[sel][picked]
                out_high[cursor:end][valid] = high[sel][picked]
            out_group.extend([group] * n_ask)
            out_state.extend([state] * n_ask)
            cursor = end
        columns: dict[str, Any] = {
            "time": out_time,
            "estimate": out_est,
            "std_error": out_se,
            "conf_low": out_low,
            "conf_high": out_high,
            "n_risk": out_risk,
        }
        if has_group:
            columns["group"] = out_group
        if has_state:
            columns["state"] = out_state
        return pl.DataFrame(columns)

    def quantile(self, p: float | list[float] = 0.5) -> pl.DataFrame:
        """Brookmeyer–Crowley quantiles of a survival curve."""
        probs = np.atleast_1d(np.asarray(p, dtype=float))
        groups = self.table["group"].unique(maintain_order=True).to_list() if "group" in self.table.columns else [None]
        rows = []
        for group in groups:
            part = self.table if group is None else self.table.filter(pl.col("group") == group)
            if "state" in part.columns:
                part = part.filter(pl.col("state") == part["state"][0])
            time = part["time"].to_numpy()
            est = part["estimate"].to_numpy()
            low = part["conf_low"].to_numpy()
            high = part["conf_high"].to_numpy()
            for prob in probs:
                target = 1 - float(prob) if self.kind == "aalen_johansen" else float(prob)
                # survival quantile: smallest t with S(t) <= prob. Incidence: smallest t with F(t) >= prob.
                if self.kind == "aalen_johansen":
                    hit = np.flatnonzero(est >= float(prob))
                else:
                    hit = np.flatnonzero(est <= float(prob))
                q = float(time[hit[0]]) if len(hit) else float("nan")
                # Brookmeyer–Crowley: times where the pointwise interval covers the target survival.
                if self.kind == "aalen_johansen":
                    cover = (low <= float(prob)) & (high >= float(prob))
                else:
                    cover = (low <= target) & (high >= target)
                covered = time[cover]
                lo = float(covered[0]) if len(covered) else float("nan")
                hi = float(covered[-1]) if len(covered) else float("nan")
                row = {"probability": float(prob), "quantile": q, "conf_low": lo, "conf_high": hi}
                if group is not None:
                    row["group"] = group
                rows.append(row)
        return pl.DataFrame(rows)


def survival_curve(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    by: ColumnRef | None = None,
    *,
    kind: str = "kaplan_meier",
    confidence: str = "log",
    level: float = 0.95,
    weights: ColumnRef | None = None,
    entry: ColumnRef | None = None,
) -> SurvivalCurve:
    """Estimate a survival or cumulative-incidence curve.

    ``event`` is 1/0 for Kaplan–Meier and Nelson–Aalen. For
    ``kind="aalen_johansen"`` it is a cause code, with 0 meaning censored.
    ``entry`` is the left-truncation time: the risk set at ``t`` is
    ``entry < t <= time``.

    For Aalen–Johansen, ``std_error`` is the infinitesimal-jackknife standard
    error of each cumulative incidence, as in a multi-state ``survfit``, and
    ``confidence`` transforms the interval the same way as for survival.
    """
    if kind not in {"kaplan_meier", "nelson_aalen", "aalen_johansen"}:
        raise ValueError("kind must be kaplan_meier, nelson_aalen, or aalen_johansen")
    if confidence not in _CONFIDENCE:
        raise ValueError(f"confidence must be one of {_CONFIDENCE}")
    times, events, w, groups, codes, entries = _survival_columns(data, time, event, by, weights, entry)
    if kind == "aalen_johansen" or groups == [None]:
        frames = []
        for index, group in enumerate(groups):
            sel = np.ones(len(times), dtype=bool) if group is None else codes == index
            table = _one_curve(
                times[sel],
                events[sel],
                w[sel],
                kind,
                confidence,
                level,
                None if entries is None else entries[sel],
            )
            if group is not None:
                table = table.with_columns(pl.lit(group).alias("group"))
            frames.append(table)
        out = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
        return SurvivalCurve(table=out, kind=kind, confidence=confidence, level=level)
    pieces = []
    for index, group in enumerate(groups):
        sel = codes == index
        columns = _curve_arrays(
            times[sel],
            events[sel],
            w[sel],
            kind,
            confidence,
            level,
            None if entries is None else entries[sel],
        )
        columns["group"] = np.full(columns["time"].shape[0], group)
        pieces.append(columns)
    merged = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0] if name != "group"}
    merged["group"] = [value for piece in pieces for value in piece["group"].tolist()]
    return SurvivalCurve(table=pl.DataFrame(merged), kind=kind, confidence=confidence, level=level)


def _appearance_codes(values: np.ndarray) -> tuple[list[object], np.ndarray]:
    """First-seen labels and integer codes. Order matches ``unique(maintain_order=True)``."""
    labels: list[object] = []
    slots: dict[object, int] = {}
    codes = np.empty(values.shape[0], dtype=np.int32)
    for i, value in enumerate(values.tolist()):
        slot = slots.get(value)
        if slot is None:
            slot = len(labels)
            slots[value] = slot
            labels.append(value)
        codes[i] = slot
    return labels, codes


def _as_float(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.Float64:
        return series.to_numpy()
    return series.cast(pl.Float64).to_numpy()


def _survival_columns(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    by: ColumnRef | None,
    weights: ColumnRef | None,
    entry: ColumnRef | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[object | None], np.ndarray, np.ndarray | None]:
    """Aligned time, event, weight, group codes, and entry after dropping nulls."""
    named = all(isinstance(ref, str) for ref in (time, event, by, weights, entry) if ref is not None)
    if named:
        time_s = data.get_column(str(time))
        event_s = data.get_column(str(event))
        weight_s = data.get_column(str(weights)) if weights is not None else None
        group_s = data.get_column(str(by)) if by is not None else None
        entry_s = data.get_column(str(entry)) if entry is not None else None
        complete = time_s.null_count() == 0 and event_s.null_count() == 0
        if weight_s is not None:
            complete = complete and weight_s.null_count() == 0
        if group_s is not None:
            complete = complete and group_s.null_count() == 0
        if entry_s is not None:
            complete = complete and entry_s.null_count() == 0
        if complete:
            times = _as_float(time_s)
            events = event_s.to_numpy()
            w = _as_float(weight_s) if weight_s is not None else np.ones(times.shape[0], dtype=float)
            entered = _as_float(entry_s) if entry_s is not None else None
            if group_s is None:
                return times, events, w, [None], np.zeros(times.shape[0], dtype=np.int32), entered
            labels, codes = _appearance_codes(group_s.to_numpy())
            return times, events, w, labels, codes, entered
    time_s = column_series(data, time).cast(pl.Float64)
    event_s = column_series(data, event)
    weight_s = column_series(data, weights).cast(pl.Float64) if weights is not None else None
    group_s = column_series(data, by) if by is not None else None
    entry_s = column_series(data, entry).cast(pl.Float64) if entry is not None else None
    mask = time_s.is_not_null() & event_s.is_not_null()
    if weight_s is not None:
        mask = mask & weight_s.is_not_null()
    if group_s is not None:
        mask = mask & group_s.is_not_null()
    if entry_s is not None:
        mask = mask & entry_s.is_not_null()
    keep = mask.to_numpy()
    if bool(keep.all()):
        times = time_s.to_numpy()
        events = event_s.to_numpy()
        w = weight_s.to_numpy() if weight_s is not None else np.ones(times.shape[0], dtype=float)
        entered = entry_s.to_numpy() if entry_s is not None else None
        gser = group_s
    else:
        kept = pl.Series(keep)
        times = time_s.filter(kept).to_numpy()
        events = event_s.filter(kept).to_numpy()
        w = weight_s.filter(kept).to_numpy() if weight_s is not None else np.ones(int(keep.sum()), dtype=float)
        entered = entry_s.filter(kept).to_numpy() if entry_s is not None else None
        gser = None if group_s is None else group_s.filter(kept)
    if gser is None:
        return times, events, np.asarray(w, dtype=float), [None], np.zeros(times.shape[0], dtype=np.int32), entered
    labels, codes = _appearance_codes(gser.to_numpy())
    return times, events, np.asarray(w, dtype=float), labels, codes, None if entered is None else np.asarray(entered, dtype=float)


def _one_curve(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    kind: str,
    confidence: str,
    level: float,
    entry: np.ndarray | None = None,
) -> pl.DataFrame:
    if kind == "aalen_johansen":
        return _aalen_johansen(time, event, weights, confidence, level, entry)
    return pl.DataFrame(_curve_arrays(time, event, weights, kind, confidence, level, entry))


def _curve_arrays(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    kind: str,
    confidence: str,
    level: float,
    entry: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if entry is None:
        uniq, n_risk_a, n_event_a, n_censor_a = _tie_counts(time, event, weights)
    else:
        uniq, n_risk_a, n_event_a, n_censor_a = _truncated_counts(time, event, weights, entry)
    if kind == "kaplan_meier":
        with np.errstate(divide="ignore", invalid="ignore"):
            hazard = np.divide(n_event_a, n_risk_a, out=np.zeros_like(n_event_a), where=n_risk_a > 0)
        survival = np.cumprod(1 - hazard)
        greenwood = np.cumsum(
            np.divide(
                n_event_a,
                n_risk_a * (n_risk_a - n_event_a),
                out=np.zeros_like(n_event_a),
                where=(n_risk_a > n_event_a) & (n_risk_a > 0),
            )
        )
        se_log = np.sqrt(np.clip(greenwood, 0, None))
        # summary.survfit reports the standard error of S, which is S * se(log S).
        std_error = survival * se_log
        low, high = _survival_interval(survival, se_log, confidence, level)
        # Once everyone at risk has the event, S is 0 and se(log S) is
        # infinite, so the standard error and the interval are undefined.
        zeroed = np.cumsum((n_event_a > 0) & (n_event_a >= n_risk_a)) > 0
        std_error[zeroed] = np.nan
        low[zeroed] = np.nan
        high[zeroed] = np.nan
        estimate = survival
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            jumps = np.divide(n_event_a, n_risk_a, out=np.zeros_like(n_event_a), where=n_risk_a > 0)
            var = np.divide(n_event_a, n_risk_a**2, out=np.zeros_like(n_event_a), where=n_risk_a > 0)
        cumulative = np.cumsum(jumps)
        std_error = np.sqrt(np.cumsum(var))
        survival = np.exp(-cumulative)
        low, high = _hazard_interval(cumulative, std_error, survival, confidence, level)
        estimate = survival
    return {
        "time": uniq,
        "n_risk": n_risk_a,
        "n_event": n_event_a,
        "n_censor": n_censor_a,
        "estimate": estimate,
        "std_error": std_error,
        "conf_low": low,
        "conf_high": high,
    }


def _risk_at(known: np.ndarray, n_risk: np.ndarray, asked: np.ndarray) -> np.ndarray:
    """Risk set at each asked time: the risk at the first tabled time >= t.

    The table lists every observed time, so no subject leaves the risk set
    between ``t`` and that time. Past the last time nobody is at risk.
    """
    out = np.zeros(asked.shape[0], dtype=float)
    if known.size == 0:
        return out
    nxt = np.searchsorted(known, asked, side="left")
    ok = nxt < known.size
    out[ok] = n_risk[nxt[ok]]
    return out


def _tie_counts(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique times and the risk, event, and censor totals at each time."""
    if len(time) == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    order = np.argsort(time, kind="mergesort")
    time = np.asarray(time, dtype=float)[order]
    died = np.asarray(event)[order].astype(float) > 0
    weights = np.asarray(weights, dtype=float)[order]
    uniq, start = np.unique(time, return_index=True)
    weighted_event = np.where(died, weights, 0.0)
    n_event = np.add.reduceat(weighted_event, start)
    weight_sum = np.add.reduceat(weights, start)
    n_censor = weight_sum - n_event
    cumulative = np.cumsum(weight_sum)
    n_risk = cumulative[-1] - cumulative + weight_sum
    return uniq, n_risk, n_event, n_censor


def _survival_interval(survival: np.ndarray, std_log: np.ndarray, confidence: str, level: float) -> tuple[np.ndarray, np.ndarray]:
    z = _norm_z(level)
    s = np.clip(survival, 1e-15, 1 - 1e-15)
    if confidence == "log":
        low = np.exp(np.log(s) - z * std_log)
        high = np.exp(np.log(s) + z * std_log)
    elif confidence == "log-log":
        se = std_log / np.log(s)
        loglog = np.log(-np.log(s))
        low = np.exp(-np.exp(loglog - z * se))
        high = np.exp(-np.exp(loglog + z * se))
    elif confidence == "plain":
        se = s * std_log
        low = s - z * se
        high = s + z * se
    elif confidence == "logit":
        logit = np.log(s / (1 - s))
        se = std_log / (1 - s)
        low = 1 / (1 + np.exp(-(logit - z * se)))
        high = 1 / (1 + np.exp(-(logit + z * se)))
    else:
        half = 0.5 * z * std_log * np.sqrt(s / (1 - s))
        angle = np.arcsin(np.sqrt(s))
        low = np.sin(np.clip(angle - half, 0, np.pi / 2)) ** 2
        high = np.sin(np.clip(angle + half, 0, np.pi / 2)) ** 2
    return np.clip(low, 0, 1), np.clip(high, 0, 1)


def _hazard_interval(
    cumulative: np.ndarray,
    std_h: np.ndarray,
    survival: np.ndarray,
    confidence: str,
    level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a cumulative-hazard interval into a survival interval."""
    z = _norm_z(level)
    if confidence == "log":
        se_log = std_h  # se(log S) = se(H) when S = exp(-H)
        return _survival_interval(survival, se_log, "log", level)
    low_h = np.clip(cumulative - z * std_h, 0, None)
    high_h = cumulative + z * std_h
    # plain hazard interval, mapped through exp(-H), so the bounds swap
    return np.exp(-high_h), np.exp(-low_h)


def _truncated_counts(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    entry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Risk, events, and censorings when the risk set is ``entry < t <= time``."""
    time = np.asarray(time, dtype=float)
    entry = np.asarray(entry, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = time > entry
    time, event, weights, entry = time[ok], np.asarray(event)[ok], weights[ok], entry[ok]
    if time.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    died = np.array([value not in (0, "0", False) and not (isinstance(value, float) and value == 0.0) for value in event])
    uniq = np.unique(time)
    n_risk = np.empty(uniq.size, dtype=float)
    n_event = np.empty(uniq.size, dtype=float)
    n_censor = np.empty(uniq.size, dtype=float)
    for index, stamp in enumerate(uniq):
        at_risk = (entry < stamp) & (time >= stamp)
        at = time == stamp
        n_risk[index] = float(weights[at_risk].sum())
        n_event[index] = float(weights[at & died].sum())
        n_censor[index] = float(weights[at & ~died].sum())
    return uniq, n_risk, n_event, n_censor


def _cause_key(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def _is_censored(value: object) -> bool:
    key = _cause_key(value)
    return key in (0, "0", False)


def _empty_aj() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "time": pl.Float64,
            "state": pl.String,
            "n_risk": pl.Float64,
            "n_event": pl.Float64,
            "n_censor": pl.Float64,
            "estimate": pl.Float64,
            "std_error": pl.Float64,
            "conf_low": pl.Float64,
            "conf_high": pl.Float64,
        }
    )


def _aalen_johansen(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    confidence: str,
    level: float,
    entry: np.ndarray | None = None,
) -> pl.DataFrame:
    """Aalen–Johansen cumulative incidence of each cause.

    The risk set at ``t`` is ``entry < t <= time``. The standard error is the
    infinitesimal jackknife (Jaeckel 1972; Glidden 2002), as in a multi-state
    ``survfit``: ``sqrt(sum_i (w_i dF/dw_i)^2)``.
    """
    time = np.asarray(time, dtype=float)
    weights = np.asarray(weights, dtype=float)
    entry = np.full(time.shape[0], -np.inf) if entry is None else np.asarray(entry, dtype=float)
    ok = np.isfinite(time) & ~np.isnan(entry) & (time > entry)
    time, event, weights, entry = time[ok], np.asarray(event)[ok], weights[ok], entry[ok]
    if time.size == 0:
        return _empty_aj()
    key_list = [_cause_key(value) for value in event.tolist()]
    causes = sorted({key for key in key_list if not _is_censored(key)})
    if not causes:
        return _empty_aj()
    code_of = {cause: index for index, cause in enumerate(causes)}
    codes = np.array([code_of.get(key, -1) for key in key_list], dtype=np.int64)
    stamps = np.unique(time)
    n_stamp = stamps.size
    n_cause = len(causes)
    # Positions 1..J are the tabled times and 0 is the start. A subject is at
    # risk at positions first..last.
    last = np.searchsorted(stamps, time, side="left") + 1
    first = np.searchsorted(stamps, entry, side="right") + 1
    diff = np.zeros(n_stamp + 2)
    np.add.at(diff, first, weights)
    np.add.at(diff, last + 1, -weights)
    n_risk = np.cumsum(diff)[1 : n_stamp + 1]
    deaths = np.zeros((n_cause, n_stamp))
    died = codes >= 0
    np.add.at(deaths, (codes[died], last[died] - 1), weights[died])
    variance, cif = _aj_ij_variance(n_risk, deaths, first.astype(np.int64), last.astype(np.int64), codes, weights)
    frames = []
    for k, cause in enumerate(causes):
        p = cif[k, 1:]
        se = np.sqrt(np.clip(variance[k], 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            se_log = np.where(p > 0, se / p, 0.0)
        low, high = _survival_interval(p, se_log, confidence, level)
        flat = p <= 0
        low[flat] = 0.0
        high[flat] = 0.0
        frames.append(
            pl.DataFrame(
                {
                    "time": stamps,
                    "state": [str(cause)] * n_stamp,
                    "n_risk": n_risk,
                    "n_event": deaths[k],
                    "n_censor": np.zeros(n_stamp),
                    "estimate": p,
                    "std_error": se,
                    "conf_low": low,
                    "conf_high": high,
                }
            )
        )
    return pl.concat(frames)


@njit(cache=True)
def _aj_ij_variance(n_risk, deaths, first, last, codes, weights):
    """Infinitesimal-jackknife variance of each cumulative incidence in O(n + J).

    See notes/aj-variance/aj-ij-variance.md for the derivation. Each subject's
    influence U_i = dF/dw_i is H - b F - q while at risk, has an extra term at
    its own time, and is alpha + c F after it leaves. The variance at each time
    is sum_i w_i^2 U_i^2, kept as running sums of the quadratic's coefficients.
    """
    n_cause, n_stamp = deaths.shape
    n = weights.shape[0]
    # Curves and the shared derivative of someone who stays at risk.
    surv = np.ones(n_stamp + 1)
    cif = np.zeros((n_cause, n_stamp + 1))
    g = np.zeros(n_stamp + 1)
    h = np.zeros((n_cause, n_stamp + 1))
    for j in range(1, n_stamp + 1):
        nr = n_risk[j - 1]
        total = 0.0
        for k in range(n_cause):
            total += deaths[k, j - 1]
        haz = total / nr if nr > 0.0 else 0.0
        surv[j] = surv[j - 1] * (1.0 - haz)
        g[j] = g[j - 1] * (1.0 - haz) + (surv[j - 1] * haz / nr if nr > 0.0 else 0.0)
        for k in range(n_cause):
            hk = deaths[k, j - 1] / nr if nr > 0.0 else 0.0
            cif[k, j] = cif[k, j - 1] + surv[j - 1] * hk
            h[k, j] = h[k, j - 1] + g[j - 1] * hk - (surv[j - 1] * hk / nr if nr > 0.0 else 0.0)
    # Running-sum pieces, as difference arrays over positions.
    in_w = np.zeros(n_stamp + 2)
    in_b = np.zeros(n_stamp + 2)
    in_bb = np.zeros(n_stamp + 2)
    in_q = np.zeros((n_cause, n_stamp + 2))
    in_qq = np.zeros((n_cause, n_stamp + 2))
    in_bq = np.zeros((n_cause, n_stamp + 2))
    out_aa = np.zeros((n_cause, n_stamp + 2))
    out_ac = np.zeros((n_cause, n_stamp + 2))
    out_cc = np.zeros(n_stamp + 2)
    at_exit = np.zeros((n_cause, n_stamp + 1))
    for i in range(n):
        a = first[i] - 1
        m = last[i]
        w2 = weights[i] * weights[i]
        b = g[a] / surv[a] if surv[a] > 0.0 else 0.0
        nr = n_risk[m - 1]
        step = surv[m - 1] / nr if nr > 0.0 else 0.0
        d_surv = g[m] - b * surv[m] - (step if codes[i] >= 0 else 0.0)
        c = d_surv / surv[m] if surv[m] > 0.0 else 0.0
        if a + 1 < m:
            in_w[a + 1] += w2
            in_w[m] -= w2
            in_b[a + 1] += w2 * b
            in_b[m] -= w2 * b
            in_bb[a + 1] += w2 * b * b
            in_bb[m] -= w2 * b * b
        out_cc[m + 1] += w2 * c * c
        for k in range(n_cause):
            q = h[k, a] - b * cif[k, a]
            d_cif = h[k, m] - b * cif[k, m] - q + (step if codes[i] == k else 0.0)
            at_exit[k, m] += w2 * d_cif * d_cif
            alpha = d_cif - c * cif[k, m]
            out_aa[k, m + 1] += w2 * alpha * alpha
            out_ac[k, m + 1] += w2 * alpha * c
            if a + 1 < m:
                in_q[k, a + 1] += w2 * q
                in_q[k, m] -= w2 * q
                in_qq[k, a + 1] += w2 * q * q
                in_qq[k, m] -= w2 * q * q
                in_bq[k, a + 1] += w2 * b * q
                in_bq[k, m] -= w2 * b * q
    variance = np.zeros((n_cause, n_stamp))
    sw = 0.0
    sb = 0.0
    sbb = 0.0
    scc = 0.0
    sq = np.zeros(n_cause)
    sqq = np.zeros(n_cause)
    sbq = np.zeros(n_cause)
    saa = np.zeros(n_cause)
    sac = np.zeros(n_cause)
    for j in range(1, n_stamp + 1):
        sw += in_w[j]
        sb += in_b[j]
        sbb += in_bb[j]
        scc += out_cc[j]
        for k in range(n_cause):
            sq[k] += in_q[k, j]
            sqq[k] += in_qq[k, j]
            sbq[k] += in_bq[k, j]
            saa[k] += out_aa[k, j]
            sac[k] += out_ac[k, j]
            hj = h[k, j]
            fj = cif[k, j]
            inside = (
                hj * hj * sw
                + fj * fj * sbb
                + sqq[k]
                - 2.0 * hj * fj * sb
                - 2.0 * hj * sq[k]
                + 2.0 * fj * sbq[k]
            )
            outside = saa[k] + 2.0 * fj * sac[k] + fj * fj * scc
            variance[k, j - 1] = inside + outside + at_exit[k, j]
    return variance, cif

