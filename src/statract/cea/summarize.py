"""Cost-effectiveness summary tables (dampack ``calculate_icers``-style).

Model-agnostic: takes strategy costs and effects and returns ICER / dominance
status. Does not run a Markov model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

# Status labels aligned with dampack
STATUS_ND = "ND"  # non-dominated (on frontier)
STATUS_D = "D"  # strongly dominated
STATUS_ED = "ED"  # extended / weakly dominated

__all__ = [
    "STATUS_D",
    "STATUS_ED",
    "STATUS_ND",
    "calculate_icers",
    "net_monetary_benefit",
]


def net_monetary_benefit(
    cost: Sequence[float] | np.ndarray,
    effect: Sequence[float] | np.ndarray,
    *,
    wtp: float,
) -> np.ndarray:
    """NMB = effect * WTP - cost for each strategy."""
    c = np.asarray(cost, dtype=np.float64)
    e = np.asarray(effect, dtype=np.float64)
    if c.shape != e.shape:
        raise ValueError("cost and effect must have the same shape")
    return e * float(wtp) - c


def calculate_icers(
    cost: Sequence[float] | np.ndarray,
    effect: Sequence[float] | np.ndarray,
    strategies: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Incremental cost-effectiveness ratios with strong / extended dominance.

    Parameters
    ----------
    cost, effect
        One value per strategy (same length).
    strategies
        Strategy names. Defaults to ``Strategy_1``, ``Strategy_2``, ...

    Returns
    -------
    polars.DataFrame
        Columns: ``strategy``, ``cost``, ``effect``, ``incremental_cost``,
        ``incremental_effect``, ``icer``, ``status``.

        ``status`` is ``ND`` (frontier), ``D`` (strongly dominated), or ``ED``
        (extended dominated). Incremental columns and ``icer`` are filled only
        for non-dominated strategies on the cost–effect frontier (sorted by
        ascending cost).
    """
    c = np.asarray(cost, dtype=np.float64)
    e = np.asarray(effect, dtype=np.float64)
    if c.ndim != 1 or e.ndim != 1 or c.shape != e.shape:
        raise ValueError("cost and effect must be 1-d arrays of equal length")
    n = int(c.shape[0])
    if n == 0:
        raise ValueError("at least one strategy is required")
    if strategies is None:
        names = [f"Strategy_{i + 1}" for i in range(n)]
    else:
        names = [str(s) for s in strategies]
        if len(names) != n:
            raise ValueError("strategies length must match cost/effect")
        if len(set(names)) != n:
            raise ValueError("strategy names must be unique")

    # Strong dominance: another strategy has cost <= and effect >=, with at
    # least one inequality strict.
    strongly = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if c[j] <= c[i] and e[j] >= e[i] and (c[j] < c[i] or e[j] > e[i]):
                strongly[i] = True
                break

    # Candidates for the frontier: not strongly dominated.
    cand = np.where(~strongly)[0].tolist()
    # Sort by ascending cost, then descending effect (tie-break).
    cand.sort(key=lambda i: (c[i], -e[i]))

    # Build frontier by successively removing extended dominated strategies.
    # A strategy is ED if the ICER to the next better strategy is higher than
    # the ICER from the previous frontier strategy to that next strategy
    # (i.e. it lies below the line connecting cheaper and more effective ND).
    frontier: list[int] = []
    for idx in cand:
        frontier.append(idx)
        while len(frontier) >= 3:
            a, b, d = frontier[-3], frontier[-2], frontier[-1]
            de_ab = e[b] - e[a]
            dc_ab = c[b] - c[a]
            de_bd = e[d] - e[b]
            dc_bd = c[d] - c[b]
            # Flat / non-increasing effect segments: drop middle if not improving.
            if de_ab <= 0:
                frontier.pop(-2)
                continue
            if de_bd <= 0:
                # d not better than b on effect; keep cheaper of b vs d.
                if c[d] >= c[b]:
                    frontier.pop()
                else:
                    frontier.pop(-2)
                continue
            icer_ab = dc_ab / de_ab
            icer_bd = dc_bd / de_bd
            if icer_bd < icer_ab:
                # b is extended dominated
                frontier.pop(-2)
            else:
                break

    on_frontier = set(frontier)
    status = np.full(n, STATUS_D, dtype=object)
    for i in range(n):
        if strongly[i]:
            status[i] = STATUS_D
        elif i in on_frontier:
            status[i] = STATUS_ND
        else:
            status[i] = STATUS_ED

    # Incremental quantities along frontier order (ascending cost).
    incr_cost = np.full(n, np.nan, dtype=np.float64)
    incr_effect = np.full(n, np.nan, dtype=np.float64)
    icer = np.full(n, np.nan, dtype=np.float64)
    if frontier:
        first = frontier[0]
        incr_cost[first] = np.nan
        incr_effect[first] = np.nan
        icer[first] = np.nan
        for prev, cur in zip(frontier[:-1], frontier[1:]):
            dc = c[cur] - c[prev]
            de = e[cur] - e[prev]
            incr_cost[cur] = dc
            incr_effect[cur] = de
            if abs(de) < 1e-15:
                icer[cur] = np.nan
            else:
                icer[cur] = dc / de

    # Row order: frontier (cost ascending), then ED, then D (each by cost).
    order: list[int] = list(frontier)
    rest_ed = sorted(
        [i for i in range(n) if status[i] == STATUS_ED],
        key=lambda i: (c[i], -e[i]),
    )
    rest_d = sorted(
        [i for i in range(n) if status[i] == STATUS_D],
        key=lambda i: (c[i], -e[i]),
    )
    order.extend(rest_ed)
    order.extend(rest_d)

    return pl.DataFrame(
        {
            "strategy": [names[i] for i in order],
            "cost": c[order],
            "effect": e[order],
            "incremental_cost": incr_cost[order],
            "incremental_effect": incr_effect[order],
            "icer": icer[order],
            "status": [str(status[i]) for i in order],
        },
    )
