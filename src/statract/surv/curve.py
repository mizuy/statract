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
        """Right-continuous value at each requested time, within each group."""
        asked = np.atleast_1d(np.asarray(times, dtype=float))
        table = self.table
        has_group = "group" in table.columns
        has_state = "state" in table.columns
        if has_group:
            labels = table["group"].to_numpy()
            group_ids, group_codes = _appearance_codes(labels)
        else:
            group_ids = [None]
            group_codes = np.zeros(table.height, dtype=np.int32)
        known = table["time"].to_numpy()
        estimate = table["estimate"].to_numpy()
        std_error = table["std_error"].to_numpy()
        low = table["conf_low"].to_numpy()
        high = table["conf_high"].to_numpy()
        n_risk = table["n_risk"].to_numpy()
        state = table["state"].to_numpy() if has_state else None
        before = 1.0 if self.kind != "aalen_johansen" else 0.0
        n_ask = asked.shape[0]
        n_out = n_ask * len(group_ids)
        out_time = np.empty(n_out, dtype=float)
        out_est = np.empty(n_out, dtype=float)
        out_se = np.empty(n_out, dtype=float)
        out_low = np.empty(n_out, dtype=float)
        out_high = np.empty(n_out, dtype=float)
        out_risk = np.empty(n_out, dtype=float)
        out_group: list[object] | None = [] if has_group else None
        out_state: list[object] | None = [] if has_state else None
        cursor = 0
        for index, group in enumerate(group_ids):
            sel = group_codes == index
            known_g = known[sel]
            idx = np.searchsorted(known_g, asked, side="right") - 1
            valid = idx >= 0
            end = cursor + n_ask
            out_time[cursor:end] = asked
            out_est[cursor:end] = before
            out_se[cursor:end] = 0.0
            out_low[cursor:end] = before
            out_high[cursor:end] = before
            first_risk = float(n_risk[sel][0]) if np.any(sel) else 0.0
            out_risk[cursor:end] = first_risk
            if np.any(valid):
                picked = idx[valid]
                out_est[cursor:end][valid] = estimate[sel][picked]
                out_se[cursor:end][valid] = std_error[sel][picked]
                out_low[cursor:end][valid] = low[sel][picked]
                out_high[cursor:end][valid] = high[sel][picked]
                out_risk[cursor:end][valid] = n_risk[sel][picked]
            if out_group is not None:
                out_group.extend([group] * n_ask)
            if out_state is not None and state is not None:
                state_g = state[sel]
                for ok, pos in zip(valid, idx, strict=True):
                    out_state.append(state_g[int(pos)] if ok else None)
            cursor = end
        columns: dict[str, Any] = {
            "time": out_time,
            "estimate": out_est,
            "std_error": out_se,
            "conf_low": out_low,
            "conf_high": out_high,
            "n_risk": out_risk,
        }
        if out_group is not None:
            columns["group"] = out_group
        if out_state is not None:
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
    if entry is not None:
        return _aalen_johansen_truncated(time, event, weights, entry, level)
    causes = [c for c in sorted(set(np.asarray(event).tolist())) if not _is_censored(c)]
    if not causes:
        return _empty_aj()
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    event = np.asarray(event)[order]
    weights = weights[order]
    uniq = np.unique(time)
    n_risk = []
    deaths = {c: [] for c in causes}
    remaining = float(weights.sum())
    any_event = np.array([e not in (0, "0", False) for e in event])
    for t in uniq:
        at = time == t
        n_risk.append(remaining)
        for cause in causes:
            deaths[cause].append(float(weights[at & (event == cause)].sum()))
        remaining -= float(weights[at].sum())
    n_risk_a = np.asarray(n_risk, dtype=float)
    total_events = np.zeros(len(uniq))
    for cause in causes:
        total_events += np.asarray(deaths[cause])
    with np.errstate(divide="ignore", invalid="ignore"):
        overall = np.divide(total_events, n_risk_a, out=np.zeros_like(total_events), where=n_risk_a > 0)
    survival_before = np.cumprod(1 - overall)
    survival_before = np.concatenate([[1.0], survival_before[:-1]])
    frames = []
    for cause in causes:
        d = np.asarray(deaths[cause], dtype=float)
        jumps = survival_before * np.divide(d, n_risk_a, out=np.zeros_like(d), where=n_risk_a > 0)
        cif = np.cumsum(jumps)
        # Aalen variance of the cumulative incidence, Greenwood-style.
        all_haz = np.divide(total_events, n_risk_a, out=np.zeros_like(d), where=n_risk_a > 0)
        var = _aj_variance(
            np.ascontiguousarray(survival_before, dtype=np.float64),
            np.ascontiguousarray(d, dtype=np.float64),
            np.ascontiguousarray(total_events, dtype=np.float64),
            np.ascontiguousarray(n_risk_a, dtype=np.float64),
            np.ascontiguousarray(cif, dtype=np.float64),
            np.ascontiguousarray(all_haz, dtype=np.float64),
        )
        std = np.sqrt(np.clip(var, 0, None))
        z = _norm_z(level)
        low = np.clip(cif - z * std, 0, 1)
        high = np.clip(cif + z * std, 0, 1)
        frames.append(
            pl.DataFrame(
                {
                    "time": uniq,
                    "state": [str(_cause_key(cause))] * len(uniq),
                    "n_risk": n_risk_a,
                    "n_event": d,
                    "n_censor": np.zeros(len(uniq)),
                    "estimate": cif,
                    "std_error": std,
                    "conf_low": low,
                    "conf_high": high,
                }
            )
        )
    return pl.concat(frames)


@njit(cache=True)
def _aj_variance(survival_before, d, total, n_risk, cif, all_haz):
    """Variance of one cumulative incidence. The covariance piece is a prefix sum."""
    n = n_risk.shape[0]
    var = np.empty(n)
    running = 0.0
    integ = 0.0
    acc = 0.0
    cif_inc = 0.0
    for i in range(n):
        if n_risk[i] > 0.0:
            running += (survival_before[i] ** 2) * d[i] * (n_risk[i] - d[i]) / n_risk[i] ** 3
            if i > 0 and n_risk[i - 1] > 0.0 and n_risk[i - 1] != total[i - 1]:
                integ += all_haz[i - 1] / (n_risk[i - 1] - total[i - 1])
            jump = survival_before[i] * d[i] / n_risk[i]
            cif_inc += jump
            acc += cif_inc * total[i] / (n_risk[i] ** 2) * survival_before[i]
        var[i] = running + (cif[i] ** 2) * integ - 2.0 * cif[i] * acc
    return var


def _aalen_johansen_truncated(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    entry: np.ndarray,
    level: float,
) -> pl.DataFrame:
    """Aalen–Johansen on event times only. The risk set is ``entry < t <= time``."""
    time = np.asarray(time, dtype=float)
    entry = np.asarray(entry, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(time) & np.isfinite(entry) & (time > entry)
    time, event, weights, entry = time[ok], np.asarray(event)[ok], weights[ok], entry[ok]
    if time.size == 0:
        return _empty_aj()
    key_list = [_cause_key(value) for value in event.tolist()]
    causes = sorted(key for key in set(key_list) if not _is_censored(key))
    if not causes:
        return _empty_aj()
    code_of = {cause: index for index, cause in enumerate(causes)}
    codes = np.array([code_of.get(key, -1) for key in key_list], dtype=np.int32)
    stamps = np.unique(time[codes >= 0])
    n_risk = np.empty(stamps.size, dtype=float)
    deaths = {cause: np.zeros(stamps.size, dtype=float) for cause in causes}
    for index, stamp in enumerate(stamps):
        at_risk = (entry < stamp) & (time >= stamp)
        at = time == stamp
        n_risk[index] = float(weights[at_risk].sum())
        for cause in causes:
            deaths[cause][index] = float(weights[at & (codes == code_of[cause])].sum())
    total = np.zeros(stamps.size, dtype=float)
    for cause in causes:
        total += deaths[cause]
    with np.errstate(divide="ignore", invalid="ignore"):
        overall = np.divide(total, n_risk, out=np.zeros_like(total), where=n_risk > 0)
    survival_before = np.concatenate([[1.0], np.cumprod(1 - overall)[:-1]])
    z = _norm_z(level)
    frames = []
    for cause in causes:
        d = deaths[cause]
        jumps = survival_before * np.divide(d, n_risk, out=np.zeros_like(d), where=n_risk > 0)
        cif = np.cumsum(jumps)
        all_haz = np.divide(total, n_risk, out=np.zeros_like(d), where=n_risk > 0)
        var = _aj_variance(
            np.ascontiguousarray(survival_before, dtype=np.float64),
            np.ascontiguousarray(d, dtype=np.float64),
            np.ascontiguousarray(total, dtype=np.float64),
            np.ascontiguousarray(n_risk, dtype=np.float64),
            np.ascontiguousarray(cif, dtype=np.float64),
            np.ascontiguousarray(all_haz, dtype=np.float64),
        )
        std = np.sqrt(np.clip(var, 0, None))
        frames.append(
            pl.DataFrame(
                {
                    "time": stamps,
                    "state": [str(cause)] * len(stamps),
                    "n_risk": n_risk,
                    "n_event": d,
                    "n_censor": np.zeros(len(stamps)),
                    "estimate": cif,
                    "std_error": std,
                    "conf_low": np.clip(cif - z * std, 0, 1),
                    "conf_high": np.clip(cif + z * std, 0, 1),
                }
            )
        )
    return pl.concat(frames)
