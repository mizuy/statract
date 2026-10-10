"""Minimum-cost flow for full and optimal matching.

The network is the one ``optmatch::fmatch`` builds. Treated units are the
upstream nodes, controls the downstream nodes, and two bookkeeping nodes
absorb the ratio constraints. Distance arcs have capacity 1. A positive flow
on a distance arc puts that treated unit and that control in the same subclass.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

_INF = 1e100


class _Edge:
    __slots__ = ("src", "dst", "cap", "cost", "rev")

    def __init__(self, src: int, dst: int, cap: float, cost: float, rev: _Edge | None) -> None:
        self.src = src
        self.dst = dst
        self.cap = cap
        self.cost = cost
        self.rev = rev


def full_match_edges(
    cost: np.ndarray,
    *,
    min_controls: float = 0.0,
    max_controls: float = math.inf,
    mean_controls: float | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Edges of an optimal full match, and the sum of their original costs.

    ``cost`` has one row per treated unit and one column per control.
    ``min_controls=0`` and ``max_controls=inf`` is unrestricted full matching.
    Setting both bounds and ``mean_controls`` to the same integer is 1:k optimal
    matching.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("full matching needs a treated-by-control distance matrix")
    n_t, n_c = cost.shape
    if n_t == 0 or n_c == 0:
        return [], 0.0
    finite = np.isfinite(cost)
    if not np.any(finite):
        return [], 0.0
    # optmatch refuses nonpositive discrepancies when a treated unit may take
    # several controls. A tie is the closest legal match, so it sits just
    # above zero before the integer scaling.
    usable = np.array(cost, dtype=float, copy=True)
    usable[~finite] = np.inf
    positive = usable[np.isfinite(usable) & (usable > 0)]
    floor = float(positive.min()) * 1e-6 if positive.size else 1e-8
    usable[np.isfinite(usable) & (usable <= 0)] = floor

    max_cpt, min_cpt, flipped = _ratio_bounds(n_t, n_c, min_controls, max_controls)
    if flipped:
        usable = usable.T
        n_t, n_c = n_c, n_t
    if mean_controls is None:
        fraction = 1.0
    else:
        fraction = mean_controls * n_t / n_c
        if fraction > 1:
            raise ValueError("mean_controls cannot exceed the control/treated ratio")
    max_row = int(math.ceil(1 / min_cpt))
    max_col = int(math.ceil(max_cpt))
    min_col = max(1, int(math.floor(min_cpt)))
    if min_col > 1:
        max_row = 1
    n_match = int(round(n_c * fraction))
    if _infeasible(n_t, n_c, max_row, max_col, min_col, n_match):
        return [], 0.0
    scaled = _scale_costs(usable, n_t + n_c)
    edges = _selected_edges(scaled, n_t, n_c, max_row, max_col, min_col, n_match)
    if flipped:
        edges = [(j, i) for i, j in edges]
        n_t, n_c = n_c, n_t
        usable = usable.T
    total = float(sum(cost[i, j] for i, j in edges))
    return edges, total


def _ratio_bounds(n_t: int, n_c: int, min_controls: float, max_controls: float):
    """MatchIt's translation of min/max controls, including the flipped case."""
    if max_controls > 0.5 or min_controls == 0:
        return min(max_controls, n_c), max(min_controls, 1 / n_t), False
    return min(1 / min_controls, n_c), max(1 / max_controls, 1 / n_t), True


def _infeasible(n_t: int, n_c: int, max_row: int, max_col: int, min_col: int, n_match: int) -> bool:
    if max_row > 1 and n_t / max_row > n_match:
        return True
    if max_row == 1 and n_t * min_col > n_match:
        return True
    return n_t * max_col < n_match


def _scale_costs(cost: np.ndarray, total_n: int) -> np.ndarray:
    """Integer costs at ``optmatch``'s default tolerance of 0.001."""
    finite = cost[np.isfinite(cost)]
    span = float(finite.max()) if finite.size else 1.0
    integer_max = 2147483647
    epsilon_floor = span / (integer_max / 64 - 2)
    tolerance = 0.001 * total_n
    epsilon = max(epsilon_floor, tolerance / max(total_n - 2, 1))
    scaled = np.ceil(0.5 + cost / epsilon)
    scaled[~np.isfinite(cost)] = np.inf
    return scaled


def _selected_edges(
    cost: np.ndarray,
    n_t: int,
    n_c: int,
    max_row: int,
    max_col: int,
    min_col: int,
    n_match: int,
) -> list[tuple[int, int]]:
    end = n_t + n_c
    sink = end + 1
    source = sink + 1
    target = source + 1
    n_nodes = target + 1
    graph: list[list[_Edge]] = [[] for _ in range(n_nodes)]

    def add(src: int, dst: int, cap: float, weight: float) -> None:
        if cap <= 0:
            return
        forward = _Edge(src, dst, cap, weight, None)
        backward = _Edge(dst, src, 0.0, -weight, forward)
        forward.rev = backward
        graph[src].append(forward)
        graph[dst].append(backward)

    for i in range(n_t):
        for j in range(n_c):
            weight = cost[i, j]
            if np.isfinite(weight):
                add(i, n_t + j, 1.0, float(weight))
    for i in range(n_t):
        add(i, end, max_col - min_col, 0.0)
        add(source, i, max_col, 0.0)
    for j in range(n_c):
        add(n_t + j, end, max_row - 1, 0.0)
        add(n_t + j, sink, 1.0, 0.0)
    add(end, target, max_col * n_t - n_match, 0.0)
    add(sink, target, n_match, 0.0)

    _successive_shortest_paths(graph, source, target, n_nodes)
    selected: list[tuple[int, int]] = []
    for i in range(n_t):
        for edge in graph[i]:
            # Forward distance arcs start at capacity 1 and carry a positive cost.
            # Residual capacity 0 means the arc was used.
            if edge.cost > 0 and edge.cap < 0.5 and n_t <= edge.dst < n_t + n_c:
                selected.append((i, edge.dst - n_t))
    return selected


def _successive_shortest_paths(graph: list[list[_Edge]], source: int, target: int, n_nodes: int) -> None:
    potential = np.zeros(n_nodes)
    demand = sum(edge.cap for edge in graph[source])
    sent = 0.0
    while sent < demand - 1e-8:
        dist = np.full(n_nodes, _INF)
        parent: list[_Edge | None] = [None] * n_nodes
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            current, node = heapq.heappop(heap)
            if current > dist[node] + 1e-9:
                continue
            for edge in graph[node]:
                if edge.cap <= 1e-12:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.dst]
                nxt = current + reduced
                if nxt + 1e-9 < dist[edge.dst]:
                    dist[edge.dst] = nxt
                    parent[edge.dst] = edge
                    heapq.heappush(heap, (nxt, edge.dst))
        if parent[target] is None:
            break
        for node in range(n_nodes):
            if dist[node] < _INF / 2:
                potential[node] += dist[node]
        bottleneck = _INF
        node = target
        while node != source:
            edge = parent[node]
            assert edge is not None
            bottleneck = min(bottleneck, edge.cap)
            node = edge.src
        node = target
        while node != source:
            edge = parent[node]
            assert edge is not None
            edge.cap -= bottleneck
            assert edge.rev is not None
            edge.rev.cap += bottleneck
            node = edge.src
        sent += bottleneck
