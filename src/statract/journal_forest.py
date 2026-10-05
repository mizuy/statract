"""Grouped / multi-panel HR forest layout (matplotlib).

Canonical coefficient forests (GLM OR / Cox HR / OLS) use
``statract.plot_forest``. This module keeps the layout-first renderer for
prepared tables that need n/event columns, group headers, or stacked panels.

``save_prepared_hr_forest`` is a compatibility adapter: simple frames go through
``plot_forest``; grouped/panel frames stay on this renderer. Saving uses a
local ``rc_context`` (Helvetica / ``pdf.fonttype=42``) and restores whatever
rc was active — it does not import ``endolab.mpl``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

RowKind = Literal["data", "group_header", "gap"]
TextAlign = Literal["left", "center", "right"]
FigureWidthPreset = Literal["single", "double"]

SINGLE_COLUMN_WIDTH_IN = 3.15
DOUBLE_COLUMN_WIDTH_IN = 6.5
FIGURE_WIDTH_PRESETS: dict[FigureWidthPreset, float] = {
    "single": SINGLE_COLUMN_WIDTH_IN,
    "double": DOUBLE_COLUMN_WIDTH_IN,
}
DEFAULT_FIGURE_WIDTH: FigureWidthPreset = "double"
WIDTH_IN = FIGURE_WIDTH_PRESETS[DEFAULT_FIGURE_WIDTH]
FONT_PT = 9.0
FONT_PT_NUM = 10.0
FONT_PT_AXIS = 9.0
ROW_HEIGHT_PT = 14.0
GROUP_HEADER_HEIGHT_PT = 12.0
GROUP_GAP_PT = 8.0
QUARTILE_LABEL_INDENT = "  "
HEADER_HEIGHT_PT = 12.0
AXIS_HEIGHT_PT = 10.0
PANEL_GAP_PT = 3.0
DPI = 600
PT_PER_IN = 72.0


def _forest_rc() -> dict[str, object]:
    """Fresh rcParams so Matplotlib cannot mutate the module-level fallback list."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def pt_to_in(pt: float) -> float:
    return pt / PT_PER_IN


def resolve_figure_width(
    width_in: float | None = None,
    *,
    preset: FigureWidthPreset | None = None,
) -> float:
    if width_in is not None:
        return width_in
    if preset is not None:
        return FIGURE_WIDTH_PRESETS[preset]
    return FIGURE_WIDTH_PRESETS[DEFAULT_FIGURE_WIDTH]


@dataclass(frozen=True)
class ForestTextCol:
    """HR プロット右側などに追加するテキスト列。"""

    key: str
    title: str
    ha: TextAlign = "right"
    min_width_pt: float = 14.0
    max_width_frac: float = 0.10


@dataclass(frozen=True)
class ForestRow:
    label: str
    n: int
    event: int
    hr: float
    conf_low: float
    conf_high: float
    desc: str
    pvalue: str
    is_reference: bool = False
    row_kind: RowKind = "data"
    sub_label: str = ""
    extras: tuple[tuple[str, str], ...] = ()
    plot_hr: bool = True


def _row_height_pt(row: ForestRow) -> float:
    if row.row_kind == "gap":
        return GROUP_GAP_PT
    if row.row_kind == "group_header":
        return GROUP_HEADER_HEIGHT_PT
    return ROW_HEIGHT_PT


def _data_rows(rows: Sequence[ForestRow]) -> list[ForestRow]:
    return [r for r in rows if r.row_kind == "data"]


@dataclass(frozen=True)
class ForestPanelSpec:
    panel_id: str
    rows: tuple[ForestRow, ...]
    x_lim: tuple[float, float]
    show_headers: bool
    panel_label: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def has_header_band(self) -> bool:
        return self.show_headers or bool(self.panel_label)

    @property
    def height_in(self) -> float:
        header = pt_to_in(HEADER_HEIGHT_PT) if self.has_header_band else 0.0
        rows_h = sum(pt_to_in(_row_height_pt(r)) for r in self.rows)
        return header + rows_h + pt_to_in(AXIS_HEIGHT_PT)


@dataclass(frozen=True)
class ForestLayout:
    width_in: float
    height_in: float
    panels: tuple[ForestPanelSpec, ...]
    col_left: dict[str, float]
    col_right: dict[str, float]
    forest_x: tuple[float, float]
    show_n_event: bool = True
    show_pvalue: bool = True
    sub_label_col: bool = False
    sub_label_title: str = ""
    label_title: str = ""
    extra_cols: tuple[ForestTextCol, ...] = ()


def _row_extra(row: ForestRow, key: str) -> str:
    for col_key, value in row.extras:
        if col_key == key:
            return value
    return ""


def _text_x(col_left: dict[str, float], col_right: dict[str, float], key: str, *, ha: TextAlign) -> float:
    if ha == "left":
        return col_left[key]
    if ha == "right":
        return col_right[key]
    return _col_center(col_left, col_right, key)


def _compute_xlim(rows: Sequence[ForestRow], *, pad_factor: float = 0.08, min_pad: float = 0.005) -> tuple[float, float]:
    plottable = [r for r in rows if r.row_kind == "data" and r.plot_hr and not r.is_reference]
    if not plottable:
        return 0.90, 1.10
    lo = min([1.0] + [r.conf_low for r in plottable])
    hi = max([1.0] + [r.conf_high for r in plottable])
    pad = max((hi - lo) * pad_factor, min_pad)
    return lo - pad, hi + pad


def _measure_text_width_in(texts: Sequence[str], *, font_pt: float = FONT_PT) -> float:
    if not texts:
        return 0.0
    fig = Figure(figsize=(2, 1), dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()
    widths: list[float] = []
    for text in texts:
        if not text:
            continue
        artist = fig.text(0, 0, text, fontsize=font_pt, family="sans-serif")
        bbox = artist.get_window_extent(renderer=renderer)
        widths.append(bbox.width / fig.dpi)
        artist.remove()
    return max(widths) if widths else 0.0


def _compute_column_layout(
    rows: Sequence[ForestRow],
    *,
    width_in: float = WIDTH_IN,
    show_n_event: bool = True,
    show_pvalue: bool = True,
    sub_label_col: bool = False,
    sub_label_title: str = "",
    extra_cols: Sequence[ForestTextCol] = (),
) -> tuple[dict[str, float], dict[str, float], tuple[float, float]]:
    data = _data_rows(rows)
    labels = [r.label for r in rows if r.row_kind != "gap"]
    descs = [r.desc for r in data]

    pad = pt_to_in(4)
    gap = pt_to_in(3)
    w_label = min(max(_measure_text_width_in(labels, font_pt=FONT_PT) + pad, pt_to_in(32)), width_in * 0.40)
    w_sub = 0.0
    if sub_label_col:
        sub_labels = [r.sub_label for r in rows if r.row_kind != "gap"]
        sub_title = sub_label_title or " "
        w_sub = min(
            max(_measure_text_width_in([*sub_labels, sub_title], font_pt=FONT_PT) + pad, pt_to_in(20)),
            width_in * 0.18,
        )
    w_desc = min(max(_measure_text_width_in([*descs, "HR (95% CI)"], font_pt=FONT_PT_NUM) + pad, pt_to_in(28)), width_in * 0.28)
    w_pval = 0.0
    if show_pvalue:
        pvals = [r.pvalue for r in data]
        w_pval = min(max(_measure_text_width_in([*pvals, "P value"], font_pt=FONT_PT_NUM) + pad, pt_to_in(16)), width_in * 0.14)

    w_n = w_event = 0.0
    n_gaps = 2
    if sub_label_col:
        n_gaps += 1
    if show_pvalue:
        n_gaps += 1
    if show_n_event:
        ns = [str(r.n) for r in data]
        events = [str(r.event) for r in data]
        w_n = min(max(_measure_text_width_in([*ns, "n"], font_pt=FONT_PT_NUM) + pad, pt_to_in(14)), width_in * 0.09)
        w_event = min(max(_measure_text_width_in([*events, "event"], font_pt=FONT_PT_NUM) + pad, pt_to_in(16)), width_in * 0.10)
        n_gaps += 2

    extra_widths: dict[str, float] = {}
    for col in extra_cols:
        texts = [_row_extra(r, col.key) for r in data]
        extra_widths[col.key] = min(
            max(
                _measure_text_width_in([*texts, col.title], font_pt=FONT_PT_NUM) + pad,
                pt_to_in(col.min_width_pt),
            ),
            width_in * col.max_width_frac,
        )
        n_gaps += 1

    used = w_label + w_sub + w_n + w_event + w_desc + w_pval + sum(extra_widths.values()) + gap * n_gaps
    left_margin = pt_to_in(2)
    right_margin = pt_to_in(8)
    min_forest = width_in * 0.10
    budget = width_in - left_margin - right_margin - min_forest
    if used > budget and budget > 0:
        scale = budget / used
        w_label *= scale
        w_desc *= scale
        if show_pvalue:
            w_pval *= scale
        if show_n_event:
            w_n *= scale
            w_event *= scale
        for key in extra_widths:
            extra_widths[key] *= scale
        used = w_label + w_sub + w_n + w_event + w_desc + w_pval + sum(extra_widths.values()) + gap * n_gaps
    w_forest = max(width_in - left_margin - right_margin - used, min_forest)

    x = left_margin
    left: dict[str, float] = {}
    right: dict[str, float] = {}
    left["label"] = x / width_in
    x += w_label
    right["label"] = x / width_in
    x += gap
    if sub_label_col:
        left["sub_label"] = x / width_in
        x += w_sub
        right["sub_label"] = x / width_in
        x += gap
    if show_n_event:
        left["n"] = x / width_in
        x += w_n
        right["n"] = x / width_in
        x += gap
        left["event"] = x / width_in
        x += w_event
        right["event"] = x / width_in
        x += gap
    left["forest"] = x / width_in
    x += w_forest
    right["forest"] = x / width_in
    x += gap
    left["desc"] = x / width_in
    x += w_desc
    right["desc"] = x / width_in
    x += gap
    if show_pvalue:
        left["pvalue"] = x / width_in
        x += w_pval
        right["pvalue"] = min(x / width_in, (width_in - right_margin) / width_in)
        x += gap
    for col in extra_cols:
        left[col.key] = x / width_in
        x += extra_widths[col.key]
        right[col.key] = min(x / width_in, 0.995)
        x += gap

    return left, right, (left["forest"], right["forest"])


def build_forest_layout_panels(
    panel_rows: Sequence[tuple[str, Sequence[ForestRow]]],
    *,
    width_in: float = WIDTH_IN,
    panel_labels: dict[str, str] | None = None,
    shared_x_lim: bool = False,
    show_n_event: bool = True,
    show_pvalue: bool = True,
    sub_label_col: bool = False,
    sub_label_title: str = "",
    label_title: str = "",
    extra_cols: Sequence[ForestTextCol] = (),
) -> ForestLayout:
    all_rows = [r for _, rs in panel_rows for r in rs]
    if not all_rows:
        msg = "panel_rows must contain at least one row"
        raise ValueError(msg)

    extra_cols_tuple = tuple(extra_cols)
    data_rows = _data_rows(all_rows)
    col_left, col_right, forest_x = _compute_column_layout(
        all_rows,
        width_in=width_in,
        show_n_event=show_n_event,
        show_pvalue=show_pvalue,
        sub_label_col=sub_label_col,
        sub_label_title=sub_label_title,
        extra_cols=extra_cols_tuple,
    )
    global_x_lim = _compute_xlim(data_rows) if shared_x_lim else None
    panels: list[ForestPanelSpec] = []
    for i, (pid, rs) in enumerate(panel_rows):
        if not rs:
            continue
        panel_data = _data_rows(rs)
        panels.append(
            ForestPanelSpec(
                panel_id=pid,
                rows=tuple(rs),
                x_lim=global_x_lim if global_x_lim is not None else _compute_xlim(panel_data),
                show_headers=i == 0,
                panel_label=(panel_labels or {}).get(pid),
            )
        )

    gap_in = pt_to_in(PANEL_GAP_PT) * max(len(panels) - 1, 0)
    height_in = sum(p.height_in for p in panels) + gap_in
    return ForestLayout(
        width_in=width_in,
        height_in=height_in,
        panels=tuple(panels),
        col_left=col_left,
        col_right=col_right,
        forest_x=forest_x,
        show_n_event=show_n_event,
        show_pvalue=show_pvalue,
        sub_label_col=sub_label_col,
        sub_label_title=sub_label_title,
        label_title=label_title,
        extra_cols=extra_cols_tuple,
    )


def _col_center(col_left: dict[str, float], col_right: dict[str, float], key: str) -> float:
    return (col_left[key] + col_right[key]) / 2


def _row_centers_from_top(rows: Sequence[ForestRow]) -> list[float]:
    """各 row の中心位置（パネル行領域上端からの inch）。"""
    y = 0.0
    centers: list[float] = []
    for row in rows:
        h = pt_to_in(_row_height_pt(row))
        centers.append(y + h / 2)
        y += h
    return centers


def _forest_y_positions(rows: Sequence[ForestRow]) -> tuple[float, dict[int, float]]:
    """forest 軸用: 全行高合計と row index → y（下原点）。"""
    total_h = sum(pt_to_in(_row_height_pt(r)) for r in rows)
    y_from_top = 0.0
    positions: dict[int, float] = {}
    for i, row in enumerate(rows):
        h = pt_to_in(_row_height_pt(row))
        center_from_top = y_from_top + h / 2
        positions[i] = total_h - center_from_top
        y_from_top += h
    return total_h, positions


def _draw_panel(
    fig: Figure,
    panel: ForestPanelSpec,
    *,
    layout: ForestLayout,
    panel_top_in: float,
) -> float:
    header_h = pt_to_in(HEADER_HEIGHT_PT) if panel.has_header_band else 0.0
    axis_h = pt_to_in(AXIS_HEIGHT_PT)
    panel_bottom_in = panel_top_in - panel.height_in
    h = layout.height_in
    row_centers = _row_centers_from_top(panel.rows)

    if panel.has_header_band:
        y_header = (panel_top_in - header_h * 0.5) / h
        fig.text(
            layout.col_left["label"],
            y_header,
            layout.label_title or panel.panel_label or "",
            ha="left",
            va="center",
            fontsize=FONT_PT,
            transform=fig.transFigure,
        )
        if panel.show_headers:
            if layout.sub_label_col:
                fig.text(
                    _text_x(layout.col_left, layout.col_right, "sub_label", ha="left"),
                    y_header,
                    layout.sub_label_title,
                    ha="left",
                    va="center",
                    fontsize=FONT_PT,
                    transform=fig.transFigure,
                )
            if layout.show_n_event:
                fig.text(layout.col_right["n"], y_header, "n", ha="right", va="center", fontsize=FONT_PT, transform=fig.transFigure)
                fig.text(layout.col_right["event"], y_header, "event", ha="right", va="center", fontsize=FONT_PT, transform=fig.transFigure)
            fig.text(
                _col_center(layout.col_left, layout.col_right, "desc"),
                y_header,
                "HR (95% CI)",
                ha="center",
                va="center",
                fontsize=FONT_PT,
                transform=fig.transFigure,
            )
            if layout.show_pvalue:
                fig.text(layout.col_right["pvalue"], y_header, "P value", ha="right", va="center", fontsize=FONT_PT, transform=fig.transFigure)
            for col in layout.extra_cols:
                fig.text(
                    _text_x(layout.col_left, layout.col_right, col.key, ha=col.ha),
                    y_header,
                    col.title,
                    ha=col.ha,
                    va="center",
                    fontsize=FONT_PT,
                    transform=fig.transFigure,
                )

    for i, row in enumerate(panel.rows):
        if row.row_kind == "gap":
            continue
        yfig = (panel_top_in - header_h - row_centers[i]) / h
        if row.row_kind == "group_header":
            fig.text(
                layout.col_left["label"],
                yfig,
                row.label,
                ha="left",
                va="center",
                fontsize=FONT_PT,
                fontweight="bold",
                transform=fig.transFigure,
                clip_on=False,
            )
            continue
        fig.text(
            layout.col_left["label"],
            yfig,
            row.label,
            ha="left",
            va="center",
            fontsize=FONT_PT,
            transform=fig.transFigure,
            clip_on=False,
        )
        if layout.sub_label_col:
            fig.text(
                _text_x(layout.col_left, layout.col_right, "sub_label", ha="left"),
                yfig,
                row.sub_label,
                ha="left",
                va="center",
                fontsize=FONT_PT,
                transform=fig.transFigure,
                clip_on=False,
            )
        if layout.show_n_event:
            fig.text(layout.col_right["n"], yfig, str(row.n), ha="right", va="center", fontsize=FONT_PT_NUM, transform=fig.transFigure)
            fig.text(layout.col_right["event"], yfig, str(row.event), ha="right", va="center", fontsize=FONT_PT_NUM, transform=fig.transFigure)
        fig.text(
            _col_center(layout.col_left, layout.col_right, "desc"),
            yfig,
            row.desc,
            ha="center",
            va="center",
            fontsize=FONT_PT_NUM,
            transform=fig.transFigure,
        )
        if layout.show_pvalue:
            fig.text(layout.col_right["pvalue"], yfig, row.pvalue, ha="right", va="center", fontsize=FONT_PT_NUM, transform=fig.transFigure)
        for col in layout.extra_cols:
            fig.text(
                _text_x(layout.col_left, layout.col_right, col.key, ha=col.ha),
                yfig,
                _row_extra(row, col.key),
                ha=col.ha,
                va="center",
                fontsize=FONT_PT_NUM,
                transform=fig.transFigure,
            )

    forest_bottom = panel_bottom_in + axis_h
    forest_height = panel.height_in - axis_h - header_h
    ax = fig.add_axes(
        [
            layout.forest_x[0],
            forest_bottom / h,
            layout.forest_x[1] - layout.forest_x[0],
            forest_height / h,
        ]
    )

    forest_total_h, y_positions = _forest_y_positions(panel.rows)
    x_lo, x_hi = panel.x_lim
    for i, row in enumerate(panel.rows):
        if row.row_kind != "data" or not row.plot_hr:
            continue
        y = y_positions[i]
        if row.is_reference:
            ax.plot([1.0], [y], "o", color="black", markersize=4, zorder=3)
            continue
        lo = max(row.conf_low, x_lo)
        hi = min(row.conf_high, x_hi)
        ax.plot([lo, hi], [y, y], color="black", linewidth=1.0, solid_capstyle="butt", zorder=2)
        if row.conf_low < x_lo:
            ax.plot([x_lo], [y], marker="o", markersize=4, markerfacecolor="white", markeredgecolor="black", zorder=3)
        if row.conf_high > x_hi:
            ax.plot([x_hi], [y], marker="o", markersize=4, markerfacecolor="white", markeredgecolor="black", zorder=3)
        hr = min(max(row.hr, x_lo), x_hi)
        ax.plot([hr], [y], "o", color="black", markersize=4, zorder=4)

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, forest_total_h)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT_PT_AXIS, length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    return panel_bottom_in


def render_hr_forest(layout: ForestLayout) -> Figure:
    fig = plt.figure(figsize=(layout.width_in, layout.height_in), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    y_top = layout.height_in
    for i, panel in enumerate(layout.panels):
        y_top = _draw_panel(fig, panel, layout=layout, panel_top_in=y_top)
        if i < len(layout.panels) - 1:
            y_top -= pt_to_in(PANEL_GAP_PT)

    return fig


def save_hr_forest_figure(
    layout: ForestLayout,
    out_path: Path,
    *,
    dpi: int = DPI,
    also_pdf: bool = True,
) -> list[Path]:
    out_path = out_path.resolve()
    stem = out_path.with_suffix("") if out_path.suffix else out_path
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    with plt.rc_context(_forest_rc()):
        fig = render_hr_forest(layout)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches=None, pad_inches=0)
        written = [png_path]
        if also_pdf:
            fig.savefig(pdf_path, format="pdf", facecolor="white", edgecolor="none", bbox_inches=None, pad_inches=0)
            written.append(pdf_path)
        plt.close(fig)
    return written


def rows_from_prepared(df: pl.DataFrame) -> list[ForestRow]:
    out: list[ForestRow] = []
    for rec in df.iter_rows(named=True):
        is_ref = str(rec.get("desc", "")) == "Reference"
        out.append(
            ForestRow(
                label=str(rec["name"]),
                n=int(rec["n"]),
                event=int(rec["event"]),
                hr=float(rec["estimate"]),
                conf_low=float(rec["conf.low"]),
                conf_high=float(rec["conf.high"]),
                desc=str(rec["desc"]),
                pvalue=str(rec.get("pvalue", "") or ""),
                is_reference=is_ref,
            )
        )
    return out


def _group_header_row(label: str) -> ForestRow:
    return ForestRow(
        label=label,
        n=0,
        event=0,
        hr=1.0,
        conf_low=1.0,
        conf_high=1.0,
        desc="",
        pvalue="",
        row_kind="group_header",
    )


def _gap_row() -> ForestRow:
    return ForestRow(
        label="",
        n=0,
        event=0,
        hr=1.0,
        conf_low=1.0,
        conf_high=1.0,
        desc="",
        pvalue="",
        row_kind="gap",
    )


def _indent_quartile_data_rows(rows: list[ForestRow]) -> list[ForestRow]:
    out: list[ForestRow] = []
    for row in rows:
        if row.row_kind == "data" and row.label in ("Q1", "Q2", "Q3", "Q4"):
            out.append(replace(row, label=f"{QUARTILE_LABEL_INDENT}{row.label}"))
        else:
            out.append(row)
    return out


def rows_from_prepared_named_groups(
    df: pl.DataFrame,
    group_specs: Sequence[tuple[str | None, Sequence[str]]],
) -> list[ForestRow]:
    """Named group headers + ordered outcome labels (``name`` column) per group."""
    if df.is_empty():
        return []
    rows: list[ForestRow] = []
    for gi, (group_label, labels) in enumerate(group_specs):
        if group_label:
            rows.append(_group_header_row(group_label))
        for label in labels:
            sub = df.filter(pl.col("name") == label)
            rows.extend(rows_from_prepared(sub))
        if gi < len(group_specs) - 1:
            rows.append(_gap_row())
    return rows


def rows_from_prepared_grouped(
    df: pl.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[str],
) -> list[ForestRow]:
    if df.is_empty() or group_col not in df.columns:
        return rows_from_prepared(df)

    present = [g for g in group_order if not df.filter(pl.col(group_col) == g).is_empty()]
    rows: list[ForestRow] = []
    for gi, group_id in enumerate(present):
        sub = df.filter(pl.col(group_col) == group_id).sort("name")
        rows.append(_group_header_row(group_id))
        rows.extend(_indent_quartile_data_rows(rows_from_prepared(sub)))
        if gi < len(present) - 1:
            rows.append(_gap_row())
    return rows


def panel_rows_from_prepared(
    df: pl.DataFrame,
    *,
    panel_col: str,
    panel_order: Sequence[str],
    group_col: str | None = None,
    group_order: Sequence[str] | None = None,
) -> list[tuple[str, list[ForestRow]]]:
    panels: list[tuple[str, list[ForestRow]]] = []
    for panel_id in panel_order:
        sub = df.filter(pl.col(panel_col) == panel_id)
        if sub.is_empty():
            continue
        if group_col and group_col in sub.columns and group_order:
            rows = rows_from_prepared_grouped(sub, group_col=group_col, group_order=group_order)
        else:
            rows = rows_from_prepared(sub)
        panels.append((panel_id, rows))
    return panels


def save_prepared_hr_forest(
    df: pl.DataFrame,
    out_path: Path,
    *,
    panel_col: str | None = None,
    panel_order: Sequence[str] | None = None,
    panel_labels: dict[str, str] | None = None,
    group_col: str | None = None,
    group_order: Sequence[str] | None = None,
    shared_x_lim: bool = False,
    width_in: float | None = None,
    figure_width: FigureWidthPreset | None = None,
    dpi: int = DPI,
    also_pdf: bool = True,
) -> list[Path]:
    """Save a prepared HR table. Prefer ``plot_forest`` for tidy GLM/Cox frames.

    Simple single-panel frames are drawn with ``plot_forest(..., layout="table",
    style="bw")``. Grouped or multi-panel frames still use the layout-first
    renderer in this module.
    """
    if df.is_empty():
        return []

    uses_journal = bool(
        (panel_col and panel_col in df.columns and panel_order)
        or (group_col and group_col in df.columns and group_order)
    )
    if not uses_journal:
        from .forest import plot_forest

        extra = tuple(c for c in ("n", "event") if c in df.columns)
        png = Path(
            plot_forest(
                df,
                Path(out_path),
                term="name" if "name" in df.columns else None,
                estimate="estimate" if "estimate" in df.columns else None,
                conf_low="conf.low" if "conf.low" in df.columns else None,
                conf_high="conf.high" if "conf.high" in df.columns else None,
                drop_intercept=False,
                log_scale=True,
                null_value=1.0,
                xlabel="Hazard ratio",
                layout="table",
                style="bw",
                extra_cols=extra,
                dpi=dpi,
                also_pdf=also_pdf,
            )
        )
        written = [png]
        pdf = png.with_suffix(".pdf")
        if also_pdf and pdf.exists() and pdf not in written:
            written.append(pdf)
        return written

    w = resolve_figure_width(width_in, preset=figure_width)
    if panel_col and panel_col in df.columns and panel_order:
        panel_rows = panel_rows_from_prepared(
            df,
            panel_col=panel_col,
            panel_order=panel_order,
            group_col=group_col,
            group_order=group_order,
        )
        layout = build_forest_layout_panels(
            panel_rows,
            width_in=w,
            panel_labels=panel_labels,
            shared_x_lim=shared_x_lim,
        )
    else:
        layout = build_forest_layout_panels(
            [("__single__", rows_from_prepared_grouped(df, group_col=group_col, group_order=group_order))],
            width_in=w,
            panel_labels=panel_labels,
            shared_x_lim=shared_x_lim,
        )

    return save_hr_forest_figure(layout, out_path, dpi=dpi, also_pdf=also_pdf)


def save_forest_panel_rows(
    panel_rows: Sequence[tuple[str, Sequence[ForestRow]]],
    out_path: Path,
    *,
    shared_x_lim: bool = False,
    show_n_event: bool = True,
    show_pvalue: bool = True,
    sub_label_col: bool = False,
    sub_label_title: str = "",
    label_title: str = "",
    extra_cols: Sequence[ForestTextCol] = (),
    panel_labels: dict[str, str] | None = None,
    width_in: float | None = None,
    figure_width: FigureWidthPreset | None = None,
    dpi: int = DPI,
    also_pdf: bool = True,
) -> list[Path]:
    if not panel_rows or all(not rs for _, rs in panel_rows):
        return []
    w = resolve_figure_width(width_in, preset=figure_width)
    layout = build_forest_layout_panels(
        panel_rows,
        width_in=w,
        panel_labels=panel_labels,
        shared_x_lim=shared_x_lim,
        show_n_event=show_n_event,
        show_pvalue=show_pvalue,
        sub_label_col=sub_label_col,
        sub_label_title=sub_label_title,
        label_title=label_title,
        extra_cols=extra_cols,
    )
    return save_hr_forest_figure(layout, out_path, dpi=dpi, also_pdf=also_pdf)


gap_row = _gap_row


def quartile_group_rows(df: pl.DataFrame, *, group_label: str, group_col: str = "qi_label") -> list[ForestRow]:
    """グループ見出し + Q1–Q4 行（`group_col` でフィルタ）。"""
    if df.is_empty() or group_col not in df.columns:
        return []
    sub = df.filter(pl.col(group_col) == group_label).sort("name")
    if sub.is_empty():
        return []
    return [_group_header_row(group_label), *rows_from_prepared(sub)]
