"""Draw a conditional inference tree like partykit's ``plot(ctree)``.

Inner nodes are ellipses with the split variable and its adjusted p-value.
Edges carry the split rule. Terminal nodes are panels at the bottom: a
boxplot of the response, or a stacked bar of the 0/1 share when the response
is binary. Nodes are numbered depth first, as in ``ConditionalTree.format``.
Pure matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from . import _mpl as _mpl  # noqa: F401
from .tree import _Node, _route

if TYPE_CHECKING:
    from .tree import ConditionalTree

TerminalPanel = Literal["auto", "boxplot", "barplot"]

_INK = "#111111"
_MUTED = "#555555"
_FILL = "#0073C2"
_FILL_LIGHT = "#d9e6f2"
_LEAF_IN = 1.05
_LEVEL_IN = 1.05
_LEAF_GAP_IN = 0.45
_PANEL_IN = 1.7
_HEADER_IN = 0.3


@dataclass
class _Placed:
    node: _Node
    id: int
    depth: int
    x: float = 0.0
    leaf_index: int | None = None
    left: _Placed | None = None
    right: _Placed | None = None


def _is_leaf(node: _Node) -> bool:
    return node.split is None or node.left is None or node.right is None


def _place(root: _Node) -> tuple[_Placed, list[_Placed], list[_Placed]]:
    """Number nodes depth first; leaves get x = 0, 1, ...; parents sit midway."""
    counter = [0]
    leaves: list[_Placed] = []
    inner: list[_Placed] = []

    def visit(node: _Node, depth: int) -> _Placed:
        counter[0] += 1
        placed = _Placed(node=node, id=counter[0], depth=depth)
        if _is_leaf(node):
            placed.leaf_index = len(leaves)
            placed.x = float(len(leaves))
            leaves.append(placed)
            return placed
        assert node.left is not None and node.right is not None
        placed.left = visit(node.left, depth + 1)
        placed.right = visit(node.right, depth + 1)
        placed.x = (placed.left.x + placed.right.x) / 2
        inner.append(placed)
        return placed

    return visit(root, 0), leaves, inner


def _format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _edge_labels(node: _Node, digits: int) -> tuple[str, str]:
    split = node.split
    assert split is not None
    if split.break_at is not None:
        cut = f"{split.break_at:.{digits}g}"
        return f"<= {cut}", f"> {cut}"
    return ", ".join(split.left_levels or ()), ", ".join(split.right_levels or ())


def plot_tree(
    tree: ConditionalTree,
    path: Path | str | None = None,
    *,
    terminal: TerminalPanel = "auto",
    digits: int = 3,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
    also_pdf: bool = False,
) -> Path | Any:
    """Draw a fitted ``conditional_tree`` in the layout of partykit's ``plot``.

    ``terminal="auto"`` uses a stacked bar when the outcome is 0/1 and a
    boxplot otherwise. With ``path`` the figure is saved and the path returned;
    without it the matplotlib ``Figure`` is returned.
    """
    _root, leaves, inner = _place(tree.root)
    y = np.asarray(tree.y, dtype=float)
    binary = bool(np.isin(y, (0.0, 1.0)).all())
    if terminal == "auto":
        terminal = "barplot" if binary else "boxplot"
    if terminal == "barplot" and not binary:
        raise ValueError("terminal='barplot' needs a 0/1 outcome")

    leaf_y: dict[int, np.ndarray] = {}

    def collect(leaf: _Node, mask: np.ndarray) -> None:
        leaf_y[id(leaf)] = y[mask]

    _route(tree.root, np.ones(y.size, dtype=bool), tree.columns, collect)

    n_leaves = len(leaves)
    n_levels = max((p.depth for p in inner), default=-1) + 1
    title_in = 0.35 if title else 0.0
    width = max(3.0, n_leaves * _LEAF_IN)
    height = title_in + n_levels * _LEVEL_IN + _LEAF_GAP_IN + _HEADER_IN + _PANEL_IN + 0.3
    if figsize is not None:
        width, height = figsize
    fig = plt.figure(figsize=(width, height))

    # Tree canvas in inches so ellipses and edges keep their shape.
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    def x_in(x: float) -> float:
        return (x + 0.5) * width / n_leaves

    panel_bottom = 0.3
    panel_top = panel_bottom + _PANEL_IN
    leaf_top = panel_top + _HEADER_IN
    top = height - title_in

    def y_in(depth: int) -> float:
        return top - (depth + 0.5) * _LEVEL_IN

    def anchor(p: _Placed) -> tuple[float, float]:
        return (x_in(p.x), leaf_top) if p.leaf_index is not None else (x_in(p.x), y_in(p.depth))

    text_kw = dict(ha="center", va="center", color=_INK, fontsize=8)
    for p in inner:
        assert p.left is not None and p.right is not None
        x0, y0 = anchor(p)
        labels = _edge_labels(p.node, digits)
        for child, label in ((p.left, labels[0]), (p.right, labels[1])):
            x1, y1 = anchor(child)
            ax.plot([x0, x1], [y0, y1], color=_MUTED, linewidth=0.8, zorder=1)
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                label,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"),
                zorder=3,
                **{**text_kw, "fontsize": 7.5},
            )
        assert p.node.split is not None
        p_value = float(np.min(p.node.p_value))
        ax.text(
            x0,
            y0,
            f"{p.node.split.column}\n{_format_p(p_value)}",
            bbox=dict(boxstyle="ellipse,pad=0.25", facecolor="white", edgecolor=_INK, linewidth=0.8),
            zorder=4,
            **text_kw,
        )
        ax.annotate(
            str(p.id),
            (x0, y0),
            xytext=(0, 21),
            textcoords="offset points",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=_INK, linewidth=0.6),
            zorder=5,
            **{**text_kw, "fontsize": 7},
        )

    if title:
        ax.text(width / 2, height - 0.12, title, ha="center", va="top", fontsize=10, color=_INK)

    # Shared response scale across terminal panels.
    if terminal == "boxplot":
        lo, hi = float(np.min(y)), float(np.max(y))
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        ylim = (lo - pad, hi + pad)
    else:
        ylim = (0.0, 1.0)

    slot = width / n_leaves
    panel_w = 0.62 * slot
    for i, p in enumerate(leaves):
        values = leaf_y[id(p.node)]
        cx = x_in(p.x)
        ax.text(cx, panel_top + 0.12, f"Node {p.id} (n = {values.size})", **{**text_kw, "fontsize": 7.5})
        left = (cx - panel_w / 2) / width
        pax = fig.add_axes((left, panel_bottom / height, panel_w / width, _PANEL_IN / height))
        pax.set_ylim(*ylim)
        pax.tick_params(axis="y", labelsize=7, length=2, labelleft=i == 0)
        for spine in pax.spines.values():
            spine.set_linewidth(0.6)
        if terminal == "boxplot":
            pax.boxplot(
                values,
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=_FILL_LIGHT, edgecolor=_INK, linewidth=0.7),
                medianprops=dict(color=_INK, linewidth=1.2),
                whiskerprops=dict(color=_INK, linewidth=0.7, linestyle="--"),
                capprops=dict(color=_INK, linewidth=0.7),
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none", markeredgecolor=_MUTED),
            )
            pax.set_xlim(0.5, 1.5)
        else:
            share = float(values.mean()) if values.size else 0.0
            pax.bar([0], [share], width=1.0, color=_FILL, edgecolor=_INK, linewidth=0.6)
            pax.bar([0], [1 - share], bottom=[share], width=1.0, color=_FILL_LIGHT, edgecolor=_INK, linewidth=0.6)
            pax.set_xlim(-0.5, 0.5)
            if share >= 0.15:
                pax.text(0, share - 0.03, f"{share:.2f}", ha="center", va="top", fontsize=7, color="white")
            else:
                pax.text(0, share + 0.03, f"{share:.2f}", ha="center", va="bottom", fontsize=7, color=_INK)
        pax.set_xticks([])
        if i == 0:
            pax.set_ylabel(tree.outcome, fontsize=7.5, color=_INK)

    if path is None:
        return fig
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, Any] = dict(dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out, **save_kw)
    if also_pdf:
        fig.savefig(out.with_suffix(".pdf"), format="pdf", **save_kw)
    plt.close(fig)
    return out
