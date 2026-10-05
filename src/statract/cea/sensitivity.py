"""Deterministic and probabilistic sensitivity analysis (model-agnostic).

Callers supply a function that maps a parameter dict to strategy cost/effect
vectors. This module does not know about disease states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from statract.cea.summarize import net_monetary_benefit

OutcomeFn = Callable[[dict[str, float]], tuple[Sequence[float], Sequence[float]]]
"""``fn(params) -> (costs, effects)`` aligned with ``strategies``."""


@dataclass(frozen=True)
class PsaResult:
    """PSA draws and strategy labels."""

    strategies: tuple[str, ...]
    cost: np.ndarray  # (n_sim, n_strat)
    effect: np.ndarray  # (n_sim, n_strat)
    params: pl.DataFrame | None = None


def run_psa(
    *,
    strategies: Sequence[str],
    param_draws: Mapping[str, Sequence[float]] | pl.DataFrame,
    evaluate: OutcomeFn,
) -> PsaResult:
    """Evaluate ``evaluate`` on each joint parameter draw.

    Parameters
    ----------
    strategies
        Strategy names (length ``k``).
    param_draws
        Mapping param → length-``n`` sequence, or a DataFrame with one row per
        simulation.
    evaluate
        Returns ``(costs, effects)`` of length ``k`` for a parameter dict.
    """
    names = tuple(str(s) for s in strategies)
    k = len(names)
    if k == 0:
        raise ValueError("strategies must be non-empty")

    if isinstance(param_draws, pl.DataFrame):
        df = param_draws
        n = df.height
        keys = df.columns
        rows = [{c: float(df[c][i]) for c in keys} for i in range(n)]
        params_df = df
    else:
        keys = list(param_draws.keys())
        if not keys:
            raise ValueError("param_draws must be non-empty")
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in param_draws.items()}
        lengths = {len(a) for a in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all param_draws sequences must have the same length")
        n = lengths.pop()
        rows = [{k: float(arrays[k][i]) for k in keys} for i in range(n)]
        params_df = pl.DataFrame({k: arrays[k] for k in keys})

    cost = np.zeros((n, k), dtype=np.float64)
    effect = np.zeros((n, k), dtype=np.float64)
    for i, params in enumerate(rows):
        c, e = evaluate(params)
        c_arr = np.asarray(c, dtype=np.float64)
        e_arr = np.asarray(e, dtype=np.float64)
        if c_arr.shape != (k,) or e_arr.shape != (k,):
            raise ValueError(f"evaluate must return cost/effect of length {k}")
        cost[i] = c_arr
        effect[i] = e_arr

    return PsaResult(strategies=names, cost=cost, effect=effect, params=params_df)


def ce_plane(
    psa: PsaResult,
    *,
    comparator: str,
) -> pl.DataFrame:
    """Incremental cost and effect vs a comparator for each PSA draw."""
    if comparator not in psa.strategies:
        raise ValueError(f"comparator {comparator!r} not in strategies")
    j0 = psa.strategies.index(comparator)
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(psa.strategies):
        if j == j0:
            continue
        for i in range(psa.cost.shape[0]):
            rows.append(
                {
                    "sim": i,
                    "strategy": name,
                    "comparator": comparator,
                    "delta_cost": float(psa.cost[i, j] - psa.cost[i, j0]),
                    "delta_effect": float(psa.effect[i, j] - psa.effect[i, j0]),
                },
            )
    return pl.DataFrame(rows)


def ceac(
    psa: PsaResult,
    *,
    wtp: Sequence[float] | np.ndarray,
) -> pl.DataFrame:
    """Cost-effectiveness acceptability curve: P(max NMB) by WTP.

    Draws with a non-finite cost or effect are left out of the probability.
    If every draw is non-finite, ``prob_ce`` is NaN and the first strategy is
    not treated as best.
    """
    wtps = np.asarray(wtp, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for w in wtps:
        nmb = psa.effect * float(w) - psa.cost  # (n_sim, k)
        valid = np.isfinite(nmb).all(axis=1)
        best = np.argmax(nmb[valid], axis=1) if np.any(valid) else np.array([], dtype=int)
        for j, name in enumerate(psa.strategies):
            rows.append(
                {
                    "wtp": float(w),
                    "strategy": name,
                    "prob_ce": float(np.mean(best == j)) if best.size else float("nan"),
                },
            )
    return pl.DataFrame(rows)


def evpi(
    psa: PsaResult,
    *,
    wtp: Sequence[float] | np.ndarray,
) -> pl.DataFrame:
    """Expected value of perfect information per WTP threshold."""
    wtps = np.asarray(wtp, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for w in wtps:
        nmb = psa.effect * float(w) - psa.cost
        # EVPI = E[max_j NMB_j] - max_j E[NMB_j]
        ev_perfect = float(np.mean(np.max(nmb, axis=1)))
        ev_best = float(np.max(np.mean(nmb, axis=0)))
        rows.append({"wtp": float(w), "evpi": ev_perfect - ev_best})
    return pl.DataFrame(rows)


def one_way_dsa(
    *,
    strategies: Sequence[str],
    base_params: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]],
    evaluate: OutcomeFn,
    outcome: str = "nmb",
    wtp: float | None = None,
    comparator: str | None = None,
) -> pl.DataFrame:
    """One-way deterministic sensitivity analysis.

    Parameters
    ----------
    outcome
        ``\"nmb\"`` (requires ``wtp``): NMB of the best strategy, or of
        ``comparator`` if that name is set.
        ``\"delta_nmb\"`` (requires ``wtp`` and ``comparator``): incremental NMB
        of the first other strategy vs comparator.
        ``\"icer\"`` (requires ``comparator``): ICER of the first other
        strategy vs comparator.
    """
    names = [str(s) for s in strategies]
    if not names:
        raise ValueError("strategies must be non-empty")
    base = {str(k): float(v) for k, v in base_params.items()}
    rows: list[dict[str, Any]] = []

    def _outcome(params: dict[str, float]) -> float:
        c, e = evaluate(params)
        c_arr = np.asarray(c, dtype=np.float64)
        e_arr = np.asarray(e, dtype=np.float64)
        if outcome == "nmb":
            if wtp is None:
                raise ValueError("wtp is required when outcome='nmb'")
            nmb = net_monetary_benefit(c_arr, e_arr, wtp=wtp)
            # Report NMB of the best strategy (or of comparator if set)
            if comparator is not None:
                j = names.index(comparator)
                return float(nmb[j])
            return float(np.max(nmb))
        if outcome == "delta_nmb":
            if wtp is None:
                raise ValueError("wtp is required when outcome='delta_nmb'")
            if comparator is None:
                raise ValueError("comparator is required when outcome='delta_nmb'")
            j0 = names.index(comparator)
            others = [j for j in range(len(names)) if j != j0]
            if not others:
                raise ValueError("need at least two strategies for delta_nmb DSA")
            j = others[0]
            nmb = net_monetary_benefit(c_arr, e_arr, wtp=wtp)
            return float(nmb[j] - nmb[j0])
        if outcome == "icer":
            if comparator is None:
                raise ValueError("comparator is required when outcome='icer'")
            j0 = names.index(comparator)
            # First other strategy
            others = [j for j in range(len(names)) if j != j0]
            if not others:
                raise ValueError("need at least two strategies for ICER DSA")
            j = others[0]
            de = float(e_arr[j] - e_arr[j0])
            dc = float(c_arr[j] - c_arr[j0])
            if abs(de) < 1e-15:
                return float("nan")
            return dc / de
        raise ValueError(f"unknown outcome {outcome!r}")

    base_val = _outcome(base)
    for param, (lo, hi) in ranges.items():
        if param not in base:
            raise ValueError(f"parameter {param!r} not in base_params")
        low_p = dict(base)
        low_p[param] = float(lo)
        high_p = dict(base)
        high_p[param] = float(hi)
        low_v = _outcome(low_p)
        high_v = _outcome(high_p)
        rows.append(
            {
                "parameter": param,
                "base": float(base[param]),
                "low": float(lo),
                "high": float(hi),
                "outcome_base": base_val,
                "outcome_low": low_v,
                "outcome_high": high_v,
                "outcome_min": min(low_v, high_v),
                "outcome_max": max(low_v, high_v),
                "spread": abs(high_v - low_v),
            },
        )
    return pl.DataFrame(rows).sort("spread", descending=True)


def tornado_table(dsa: pl.DataFrame) -> pl.DataFrame:
    """Alias for a DSA table already sorted by spread (tornado input)."""
    if "spread" not in dsa.columns:
        raise ValueError("dsa table must contain a 'spread' column")
    return dsa.sort("spread", descending=True)
