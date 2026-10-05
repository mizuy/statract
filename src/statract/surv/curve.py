"""Kaplan–Meier, Nelson–Aalen, and Aalen–Johansen curves.

Confidence intervals follow ``survfit``: ``std_error`` is the standard error of
the log survival (Kaplan–Meier) or of the cumulative hazard (Nelson–Aalen).
The default interval is the log interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit
from scipy import stats

from ..design import ColumnRef, column_series

_CONFIDENCE = ("log", "log-log", "plain", "logit", "arcsin")


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
        frames = []
        groups = self.table["group"].unique(maintain_order=True).to_list() if "group" in self.table.columns else [None]
        for group in groups:
            part = self.table if group is None else self.table.filter(pl.col("group") == group)
            known = part["time"].to_numpy()
            rows = []
            for t in asked:
                idx = int(np.searchsorted(known, t, side="right") - 1)
                if idx < 0:
                    estimate = 1.0 if self.kind != "aalen_johansen" else 0.0
                    row = {
                        "time": float(t),
                        "estimate": estimate,
                        "std_error": 0.0,
                        "conf_low": estimate,
                        "conf_high": estimate,
                        "n_risk": int(part["n_risk"][0]) if part.height else 0,
                    }
                else:
                    src = part.row(idx, named=True)
                    row = {
                        "time": float(t),
                        "estimate": src["estimate"],
                        "std_error": src["std_error"],
                        "conf_low": src["conf_low"],
                        "conf_high": src["conf_high"],
                        "n_risk": src["n_risk"],
                    }
                if group is not None:
                    row["group"] = group
                if "state" in part.columns and idx >= 0:
                    row["state"] = part.row(idx, named=True)["state"]
                rows.append(row)
            frames.append(pl.DataFrame(rows))
        return pl.concat(frames, how="diagonal_relaxed")

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
) -> SurvivalCurve:
    """Estimate a survival or cumulative-incidence curve.

    ``event`` is 1/0 for Kaplan–Meier and Nelson–Aalen. For
    ``kind="aalen_johansen"`` it is a cause code, with 0 meaning censored.
    """
    if kind not in {"kaplan_meier", "nelson_aalen", "aalen_johansen"}:
        raise ValueError("kind must be kaplan_meier, nelson_aalen, or aalen_johansen")
    if confidence not in _CONFIDENCE:
        raise ValueError(f"confidence must be one of {_CONFIDENCE}")
    time_s = column_series(data, time).cast(pl.Float64)
    event_s = column_series(data, event)
    weight_s = column_series(data, weights).cast(pl.Float64) if weights is not None else None
    group_s = column_series(data, by) if by is not None else None
    mask = time_s.is_not_null() & event_s.is_not_null()
    if weight_s is not None:
        mask = mask & weight_s.is_not_null()
    if group_s is not None:
        mask = mask & group_s.is_not_null()
    keep = mask.to_numpy()
    times = time_s.filter(pl.Series(keep)).to_numpy()
    events = event_s.filter(pl.Series(keep)).to_numpy()
    w = weight_s.filter(pl.Series(keep)).to_numpy() if weight_s is not None else np.ones(times.shape[0])
    if group_s is None:
        groups: list[object | None] = [None]
        codes = np.zeros(len(times), dtype=np.int32)
    else:
        gser = group_s.filter(pl.Series(keep))
        groups = gser.unique(maintain_order=True).to_list()
        codes = gser.replace_strict(groups, list(range(len(groups))), return_dtype=pl.Int32).to_numpy()
    frames = []
    for index, group in enumerate(groups):
        sel = np.ones(len(times), dtype=bool) if group is None else codes == index
        table = _one_curve(times[sel], events[sel], np.asarray(w[sel], dtype=float), kind, confidence, level)
        if group is not None:
            table = table.with_columns(pl.lit(group).alias("group"))
        frames.append(table)
    out = pl.concat(frames, how="diagonal_relaxed")
    return SurvivalCurve(table=out, kind=kind, confidence=confidence, level=level)


def _one_curve(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    kind: str,
    confidence: str,
    level: float,
) -> pl.DataFrame:
    if kind == "aalen_johansen":
        return _aalen_johansen(time, event, weights, confidence, level)
    uniq, n_risk_a, n_event_a, n_censor_a = _tie_counts(time, event, weights)
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
    return pl.DataFrame(
        {
            "time": uniq,
            "n_risk": n_risk_a,
            "n_event": n_event_a,
            "n_censor": n_censor_a,
            "estimate": estimate,
            "std_error": std_error,
            "conf_low": low,
            "conf_high": high,
        }
    )


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
    z = float(stats.norm.ppf(0.5 + level / 2))
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
    z = float(stats.norm.ppf(0.5 + level / 2))
    if confidence == "log":
        se_log = std_h  # se(log S) = se(H) when S = exp(-H)
        return _survival_interval(survival, se_log, "log", level)
    low_h = np.clip(cumulative - z * std_h, 0, None)
    high_h = cumulative + z * std_h
    # plain hazard interval, mapped through exp(-H), so the bounds swap
    return np.exp(-high_h), np.exp(-low_h)


def _aalen_johansen(
    time: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
    confidence: str,
    level: float,
) -> pl.DataFrame:
    causes = [c for c in sorted(set(np.asarray(event).tolist())) if c not in (0, "0", False)]
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
        z = float(stats.norm.ppf(0.5 + level / 2))
        low = np.clip(cif - z * std, 0, 1)
        high = np.clip(cif + z * std, 0, 1)
        frames.append(
            pl.DataFrame(
                {
                    "time": uniq,
                    "state": [str(cause)] * len(uniq),
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
