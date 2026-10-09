"""Conditional inference trees in the sense of partykit::ctree.

Variable selection uses the quadratic form of the conditional linear
statistic, with a Šidák adjustment across the covariates at the node (the
adjustment partykit calls Bonferroni). The split then maximises the two-sample
quadratic statistic. Defaults match ``ctree_control``: alpha 0.05, minsplit
20, minbucket 7, minprob 0.01, splittry 2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import chi2

from .design import ColumnRef, column_series


@dataclass
class _Split:
    column: str
    break_at: float | None = None
    left_levels: tuple[str, ...] | None = None
    right_levels: tuple[str, ...] | None = None


@dataclass
class _Node:
    n: int
    mean: float
    terms: list[str]
    statistic: np.ndarray
    p_value: np.ndarray
    split: _Split | None = None
    left: _Node | None = None
    right: _Node | None = None


@dataclass
class ConditionalTree:
    """A fitted conditional inference tree. Predictions are node means."""

    root: _Node
    predictors: list[str]
    columns: dict[str, np.ndarray]
    outcome: str
    y: np.ndarray

    def predict(self, data: pl.DataFrame | None = None) -> np.ndarray:
        """Mean response in the terminal node of each row."""
        if data is None:
            columns = self.columns
        else:
            columns = {name: np.asarray(column_series(data, name).to_numpy()) for name in self.predictors}
        n = len(next(iter(columns.values())))
        out = np.full(n, np.nan)

        def fill(leaf: _Node, mask: np.ndarray) -> None:
            out[mask] = leaf.mean

        _route(self.root, np.ones(n, dtype=bool), columns, fill)
        return out

    def plot(self, path: Path | str | None = None, **kwargs: Any) -> Any:
        """Draw the tree like partykit's ``plot``. See :func:`plot_tree`."""
        from .tree_plot import plot_tree

        return plot_tree(self, path, **kwargs)

    def format(self, digits: int = 3) -> str:
        """Text drawing of the tree, numbered depth first like partykit's print.

        Each terminal node shows the mean response and its size.
        """
        lines: list[str] = []
        counter = [0]

        def visit(node: _Node, depth: int, label: str) -> None:
            counter[0] += 1
            prefix = "|   " * depth
            head = f"{prefix}[{counter[0]}] {label}".rstrip()
            if node.split is None or node.left is None or node.right is None:
                lines.append(f"{head}: {node.mean:.{digits}f} (n = {node.n})")
                return
            lines.append(head)
            split = node.split
            if split.break_at is not None:
                left = f"{split.column} <= {split.break_at:.{digits}g}"
                right = f"{split.column} > {split.break_at:.{digits}g}"
            else:
                left = f"{split.column} in {{{', '.join(split.left_levels or ())}}}"
                right = f"{split.column} in {{{', '.join(split.right_levels or ())}}}"
            visit(node.left, depth + 1, left)
            visit(node.right, depth + 1, right)

        visit(self.root, 0, "root")
        return "\n".join(lines) + "\n"

    def n_terminal(self) -> int:
        """Number of terminal nodes."""

        def count(node: _Node) -> int:
            if node.split is None or node.left is None or node.right is None:
                return 1
            return count(node.left) + count(node.right)

        return count(self.root)

    def tests(self) -> pl.DataFrame:
        """Quadratic statistic and Šidák-adjusted p-value for each root covariate."""
        return pl.DataFrame(
            {
                "term": self.root.terms,
                "statistic": self.root.statistic,
                "p_value": self.root.p_value,
            }
        )


def conditional_tree(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: list[str],
    *,
    alpha: float = 0.05,
    minsplit: int = 20,
    minbucket: int = 7,
    minprob: float = 0.01,
    splittry: int = 2,
) -> ConditionalTree:
    """Grow a conditional inference tree for a numeric outcome.

    Predictors are column names. Strings are unordered factors and are split
    by a binary partition of the levels. Numeric columns are split at an
    observed value, and values equal to the break go left.

    As in partykit, a split search at a node of size n needs at least
    ``max(minbucket, ceil(minprob * n))`` rows on each side, and when the most
    significant covariate has no admissible split the next ones are tried, up
    to ``splittry`` covariates in all.
    """
    if not predictors:
        raise ValueError("conditional_tree needs at least one predictor")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    y_series = column_series(data, outcome)
    keep = y_series.is_not_null().to_numpy()
    for name in predictors:
        keep &= column_series(data, name).is_not_null().to_numpy()
    y = np.asarray(y_series.filter(pl.Series(keep)).to_numpy(), dtype=float)
    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("the outcome needs at least two finite observations")
    columns = {
        name: np.asarray(column_series(data, name).filter(pl.Series(keep)).to_numpy()) for name in predictors
    }
    factors = {name for name, values in columns.items() if values.dtype.kind in {"U", "O", "S"}}
    root = _grow(y, columns, factors, alpha, minsplit, minbucket, minprob, splittry)
    return ConditionalTree(root=root, predictors=list(predictors), columns=columns, outcome=y_series.name, y=y)


def _grow(y, columns, factors, alpha, minsplit, minbucket, minprob, splittry) -> _Node:
    node = _Node(n=int(y.size), mean=float(np.mean(y)), terms=[], statistic=np.array([]), p_value=np.array([]))
    if y.size < minsplit:
        return node
    tested: list[tuple[str, float, float]] = []
    for name, values in columns.items():
        result = _factor_association(values, y) if name in factors else _numeric_association(values, y)
        if result is not None:
            tested.append((name, *result))
    if not tested:
        return node
    m = len(tested)
    stats = np.array([stat for _name, stat, _p in tested])
    raw = np.array([p for _name, _stat, p in tested])
    # Work on the log scale like partykit: 1 - (1 - p) ** m rounds every
    # p below about 1e-17 to 0, and the first such column would then win.
    crit = m * np.log1p(-raw)
    adjusted = -np.expm1(crit)
    node.terms = [name for name, _stat, _p in tested]
    node.statistic = stats
    node.p_value = adjusted
    # partykit treats values closer than sqrt(DBL_MIN) as ties and breaks
    # them by the larger statistic.
    ties = np.flatnonzero(np.abs(crit - crit.max()) < math.sqrt(np.finfo(float).tiny))
    if ties.size > 1:
        ranks = np.argsort(np.argsort(stats[ties], kind="stable"), kind="stable") + 1.0
        crit[ties] += ranks / (ties.size * 1000)
    logmin = math.log1p(-alpha)
    order = [int(j) for j in np.argsort(-crit, kind="stable") if crit[j] > logmin][:splittry]
    # partykit widens minbucket at large nodes during the split search only.
    search_bucket = max(minbucket, math.ceil(minprob * y.size))
    for best in order:
        name = tested[best][0]
        values = columns[name]
        if name in factors:
            split = _best_factor_split(name, values, y, search_bucket)
            if split is None:
                continue
            left_mask = np.isin(np.asarray(values).astype(str), split.left_levels)
        else:
            split = _best_numeric_split(name, np.asarray(values, dtype=float), y, search_bucket)
            if split is None:
                continue
            left_mask = np.asarray(values, dtype=float) <= split.break_at
        break
    else:
        return node
    if int(left_mask.sum()) < minbucket or int((~left_mask).sum()) < minbucket:
        return node
    node.split = split
    child = {key: value[left_mask] for key, value in columns.items()}
    node.left = _grow(y[left_mask], child, factors, alpha, minsplit, minbucket, minprob, splittry)
    child = {key: value[~left_mask] for key, value in columns.items()}
    node.right = _grow(y[~left_mask], child, factors, alpha, minsplit, minbucket, minprob, splittry)
    return node


def _route(
    node: _Node, mask: np.ndarray, columns: dict[str, np.ndarray], visit: Callable[[_Node, np.ndarray], None]
) -> None:
    """Send the rows in ``mask`` down the tree and call ``visit(leaf, mask)`` at each leaf."""
    if node.split is None or node.left is None or node.right is None:
        visit(node, mask)
        return
    values = columns[node.split.column]
    if node.split.break_at is not None:
        numeric = np.asarray(values, dtype=float)
        finite = np.isfinite(numeric)
        go_left = finite & (numeric <= node.split.break_at)
        go_right = finite & ~go_left
    else:
        labels = np.asarray(values).astype(str)
        go_left = np.isin(labels, node.split.left_levels)
        go_right = np.isin(labels, node.split.right_levels)
    _route(node.left, mask & go_left, columns, visit)
    _route(node.right, mask & go_right, columns, visit)


def _numeric_association(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    ss_x = float(np.dot(xc, xc))
    ss_y = float(np.dot(yc, yc))
    if ss_x <= 1e-12 or ss_y <= 1e-12:
        return None
    var = ss_x * ss_y / (len(y) - 1)
    stat = float(np.dot(xc, yc) ** 2 / var)
    return stat, float(chi2.sf(stat, 1))


def _factor_association(values: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    labels = [str(level) for level in np.unique(np.asarray(values).astype(str))]
    if len(labels) < 2:
        return None
    dummies = np.column_stack([(np.asarray(values).astype(str) == level).astype(float) for level in labels])
    xc = dummies - dummies.mean(axis=0)
    yc = y - y.mean()
    ss_y = float(np.dot(yc, yc))
    if ss_y <= 1e-12:
        return None
    cov = (ss_y / (len(y) - 1)) * (xc.T @ xc)
    diff = xc.T @ yc
    rank = int(np.linalg.matrix_rank(cov, tol=1e-8))
    if rank < 1:
        return None
    stat = float(diff @ np.linalg.pinv(cov) @ diff)
    return stat, float(chi2.sf(stat, rank))


def _best_numeric_split(name: str, x: np.ndarray, y: np.ndarray, minbucket: int) -> _Split | None:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    n = len(y)
    ss = float(np.dot(y - y.mean(), y - y.mean()))
    if ss <= 1e-12:
        return None
    csum = np.cumsum(ys)
    best = -1.0
    break_at = None
    mean = float(y.mean())
    for i in range(minbucket, n - minbucket + 1):
        if xs[i - 1] == xs[i]:
            continue
        n_left = i
        n_right = n - i
        var = n_left * n_right / (n * (n - 1)) * ss
        if var <= 1e-12:
            continue
        stat = (csum[i - 1] - n_left * mean) ** 2 / var
        if stat > best:
            best = float(stat)
            break_at = float(xs[i - 1])
    if break_at is None:
        return None
    return _Split(column=name, break_at=break_at)


def _best_factor_split(name: str, values: np.ndarray, y: np.ndarray, minbucket: int) -> _Split | None:
    labels = [str(level) for level in np.unique(np.asarray(values).astype(str))]
    k = len(labels)
    if k < 2:
        return None
    text = np.asarray(values).astype(str)
    best = -1.0
    chosen: tuple[str, ...] | None = None
    n = len(y)
    ss = float(np.dot(y - y.mean(), y - y.mean()))
    mean = float(y.mean())
    # The first level stays on the left, which removes the reflected partition.
    others = labels[1:]
    for mask in range(1, 2 ** (k - 1)):
        right = tuple(level for bit, level in enumerate(others) if mask & (1 << bit))
        left_mask = ~np.isin(text, right)
        n_left = int(left_mask.sum())
        n_right = n - n_left
        if n_left < minbucket or n_right < minbucket:
            continue
        var = n_left * n_right / (n * (n - 1)) * ss
        if var <= 1e-12:
            continue
        stat = (float(y[left_mask].sum()) - n_left * mean) ** 2 / var
        if stat > best:
            best = float(stat)
            chosen = right
    if chosen is None:
        return None
    left = tuple(level for level in labels if level not in chosen)
    return _Split(column=name, left_levels=left, right_levels=chosen)
