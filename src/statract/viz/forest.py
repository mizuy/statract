"""Forest plots for coefficient / OR / HR tables.

Accepts ``Fit.tidy`` / Cox ``tidy`` frames (with or without ``exponentiate=True``)
and mixed-model ``tidy(exponentiate=True)`` frames. Pure matplotlib — no R / rpy2.

Default layout joins a text table (term, estimate/CI, p) with the forest panel,
inspired by the column layout of ``forest.R`` / classic forest tables.

``style="color"`` (screen/docs default) uses the same JCO blue as Kaplan–Meier;
``style="bw"`` is a print-journal grayscale mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import polars as pl

_INTERCEPT_NAMES = frozenset(
    {
        "(Intercept)",
        "Intercept",
        "intercept",
        "(intercept)",
    }
)

ForestLayout = Literal["table", "points"]
ForestPlotStyle: TypeAlias = Literal["color", "bw"]

# Match plot_survival JCO blue so GLM/Cox figures sit next to KM on the same page.
_FOREST_COLOR = "#0073C2"
_FOREST_INK = "#111111"
_FOREST_MUTED = "#555555"
_ROW_BAND_COLOR = "#f4f4f4"
_ROW_BAND_BW = "#efefef"
_FONTSIZE = 8.5
_HEADER_SIZE = 8.5
_AXIS_SIZE = 8.0
_ROW_IN = 0.22
_HEADER_IN = 0.24
_AXIS_IN = 0.36
_TITLE_IN = 0.28
_COL_PAD_IN = 0.05
_FOREST_PANEL_IN = 3.20
_P_COL_IN = 0.50
_EST_COL_IN = 1.22
_FOREST_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _pick_columns(
    frame: pl.DataFrame,
    *,
    term: str | None,
    estimate: str | None,
    conf_low: str | None,
    conf_high: str | None,
) -> tuple[str, str, str, str, bool]:
    """Resolve term / estimate / CI columns and whether values are ratios."""
    cols = set(frame.columns)
    term_col = term
    if term_col is None:
        for candidate in ("term", "variable", "name"):
            if candidate in cols:
                term_col = candidate
                break
    if term_col is None or term_col not in cols:
        raise ValueError("tidy frame needs a term column (term / variable / name)")

    if estimate is not None or conf_low is not None or conf_high is not None:
        est = estimate or "estimate"
        lo = conf_low or "conf_low"
        hi = conf_high or "conf_high"
        for name in (est, lo, hi):
            if name not in cols:
                raise ValueError(f"column {name!r} not in tidy frame")
        ratio = (
            est.startswith("exp_")
            or est in {"oddsratio", "hr", "or"}
            or lo in {"conf.low", "exp_conf_low"}
            or hi in {"conf.high", "exp_conf_high"}
        )
        return term_col, est, lo, hi, ratio

    if {"exp_estimate", "exp_conf_low", "exp_conf_high"} <= cols:
        return term_col, "exp_estimate", "exp_conf_low", "exp_conf_high", True
    if {"oddsratio", "lo", "hi"} <= cols:
        return term_col, "oddsratio", "lo", "hi", True
    if {"estimate", "conf.low", "conf.high"} <= cols:
        return term_col, "estimate", "conf.low", "conf.high", True
    if {"estimate", "conf_low", "conf_high"} <= cols:
        return term_col, "estimate", "conf_low", "conf_high", False
    raise ValueError(
        "tidy frame needs estimate/CI columns "
        "(exp_estimate/exp_conf_* or estimate/conf_* or oddsratio/lo/hi)"
    )


def _format_estimate(est: float, lo: float, hi: float, *, digits: int = 2) -> str:
    if not np.isfinite(est):
        return "—"
    lo_s = f"{lo:.{digits}f}" if np.isfinite(lo) else "—"
    hi_s = f"{hi:.{digits}f}" if np.isfinite(hi) else "—"
    return f"{est:.{digits}f} ({lo_s}–{hi_s})"


def _format_pvalue(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _estimate_header(xlabel: str, *, is_ratio: bool) -> str:
    lower = xlabel.lower()
    if "hazard" in lower:
        return "HR (95% CI)"
    if "odds" in lower:
        return "OR (95% CI)"
    if is_ratio:
        return "Estimate (95% CI)"
    return "Estimate (95% CI)"


def _prepare_frame(
    tidy: pl.DataFrame,
    *,
    term: str | None,
    estimate: str | None,
    conf_low: str | None,
    conf_high: str | None,
    drop_intercept: bool,
    extra_cols: Sequence[str] = (),
) -> tuple[pl.DataFrame, bool]:
    term_col, est_col, lo_col, hi_col, is_ratio = _pick_columns(
        tidy,
        term=term,
        estimate=estimate,
        conf_low=conf_low,
        conf_high=conf_high,
    )
    exprs = [
        pl.col(term_col).cast(pl.Utf8).alias("term"),
        pl.col(est_col).cast(pl.Float64).alias("estimate"),
        pl.col(lo_col).cast(pl.Float64).alias("conf_low"),
        pl.col(hi_col).cast(pl.Float64).alias("conf_high"),
    ]
    if "p_value" in tidy.columns:
        exprs.append(pl.col("p_value").cast(pl.Float64).alias("p_value"))
    elif "pvalue" in tidy.columns:
        exprs.append(pl.col("pvalue").cast(pl.Utf8).alias("pvalue"))
    seen = {"term", "estimate", "conf_low", "conf_high", "p_value", "pvalue"}
    for col in extra_cols:
        if col not in tidy.columns:
            raise ValueError(f"extra column {col!r} not in tidy frame")
        alias = col
        if alias in seen:
            continue
        exprs.append(pl.col(col).cast(pl.Utf8).alias(alias))
        seen.add(alias)
    frame = tidy.select(exprs)
    if drop_intercept:
        frame = frame.filter(~pl.col("term").is_in(list(_INTERCEPT_NAMES)))
    if frame.height == 0:
        raise ValueError("no terms left to plot after dropping intercept")
    return frame, is_ratio


def _style_spec(style: ForestPlotStyle) -> dict[str, str]:
    if style == "bw":
        return {
            "marker": _FOREST_INK,
            "error": _FOREST_INK,
            "ref": _FOREST_INK,
            "ink": _FOREST_INK,
            "band": _ROW_BAND_BW,
        }
    return {
        "marker": _FOREST_COLOR,
        "error": _FOREST_COLOR,
        "ref": _FOREST_MUTED,
        "ink": _FOREST_INK,
        "band": _ROW_BAND_COLOR,
    }


def _auto_xlim(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    ref: float,
    use_log: bool,
) -> tuple[float, float]:
    finite_lo = lo[np.isfinite(lo)]
    finite_hi = hi[np.isfinite(hi)]
    x0 = float(np.min(finite_lo)) if finite_lo.size else ref
    x1 = float(np.max(finite_hi)) if finite_hi.size else ref
    x0 = min(x0, ref)
    x1 = max(x1, ref)
    if use_log:
        x0 = max(x0, 1e-8)
        if x1 <= 0:
            x1 = max(ref, 1.0)
        log0, log1 = np.log(x0), np.log(x1)
        pad = max((log1 - log0) * 0.04, 0.05)
        return float(np.exp(log0 - pad)), float(np.exp(log1 + pad))
    span = x1 - x0
    pad = max(span * 0.05, 0.04 * max(abs(x0), abs(x1), 1.0))
    return x0 - pad, x1 + pad


def _ratio_ticks(x0: float, x1: float) -> list[float]:
    candidates: list[float] = []
    for exp in range(-4, 6):
        for mult in (1.0, 2.0, 5.0):
            candidates.append(mult * 10.0**exp)
    candidates.extend([0.25, 4.0, 25.0, 40.0])
    ticks = [t for t in candidates if x0 * 0.98 <= t <= x1 * 1.02]
    if x0 < 1.0 < x1 and 1.0 not in ticks:
        ticks.append(1.0)
    ticks = sorted({float(t) for t in ticks})
    if len(ticks) > 7:
        preferred = {0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0}
        ticks = [t for t in ticks if t in preferred or abs(np.log10(t) % 1.0) < 1e-9]
    if not ticks:
        ticks = [x0, 1.0, x1] if x0 < 1.0 < x1 else [x0, x1]
    return ticks


def _format_ratio_tick(value: float, _pos: float) -> str:
    if value <= 0:
        return ""
    if abs(value - round(value)) < 1e-9 and abs(value) >= 1:
        return str(int(round(value)))
    text = f"{value:g}"
    return text


def _approx_text_in(text: str, *, fontsize: float, weight: str = "normal") -> float:
    """Rough DejaVu Sans width in inches (enough to pack columns without a renderer)."""
    em = fontsize / 72.0
    factor = 0.56 if weight == "bold" else 0.52
    return max(len(text) * factor * em, 0.12)


def _table_column_inches(
    labels: list[str],
    est_texts: list[str],
    p_texts: list[str],
    est_header: str,
) -> tuple[float, float, float]:
    term_in = max(
        _approx_text_in("Term", fontsize=_HEADER_SIZE, weight="bold"),
        *(_approx_text_in(s, fontsize=_FONTSIZE) for s in labels),
    )
    est_in = max(
        _EST_COL_IN,
        _approx_text_in(est_header, fontsize=_HEADER_SIZE, weight="bold"),
        *(_approx_text_in(s, fontsize=_FONTSIZE) for s in est_texts),
    )
    p_in = max(
        _P_COL_IN,
        _approx_text_in("P", fontsize=_HEADER_SIZE, weight="bold"),
        *(_approx_text_in(s, fontsize=_FONTSIZE) for s in p_texts),
    )
    return min(2.20, term_in + 0.08), est_in + 0.04, max(0.50, p_in)


def _table_ylim(n: int, header_y: float | None) -> tuple[float, float]:
    if header_y is None:
        return -0.42, n - 0.50
    return -0.42, header_y + 0.18


def _row_bands(ax: Any, y: np.ndarray, *, color: str, header_y: float | None) -> None:
    ax.set_ylim(*_table_ylim(len(y), header_y))
    for i, yi in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=color, zorder=0, linewidth=0)


def plot_forest(
    tidy: pl.DataFrame,
    path: Path | str | None = None,
    *,
    term: str | None = None,
    estimate: str | None = None,
    conf_low: str | None = None,
    conf_high: str | None = None,
    null_value: float | None = None,
    drop_intercept: bool = True,
    log_scale: bool | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    layout: ForestLayout = "table",
    style: ForestPlotStyle = "color",
    estimate_digits: int = 2,
    extra_cols: Sequence[str] = (),
    dpi: int = 180,
    also_pdf: bool = False,
) -> Path | Any:
    """Draw a publication-ready forest plot from a tidy coefficient table.

    This is the canonical statistical forest (GLM OR / Cox HR / OLS). Gallery
    figures use ``layout="table"``; print journals use ``style="bw"``.

    Default ``layout="table"`` joins term / estimate (CI) / p-value text columns
    with the forest panel (table + forest as one figure). Use ``layout="points"``
    for a points-only panel (y-tick labels only). ``style="color"`` is the
    screen/docs default; ``style="bw"`` is grayscale for print journals.

    Args:
        tidy: Coefficient table, typically ``fit.tidy()`` or
            ``fit.tidy(exponentiate=True)``. Also accepts experimental OR tables
            summaries with ``oddsratio`` / ``lo`` / ``hi``, and prepared HR
            tables with ``name`` / ``estimate`` / ``conf.low`` / ``conf.high``.
        path: If set, save a PNG and return that path; otherwise return the
            matplotlib ``Figure``.
        term: Label column. Auto-detects ``term``, ``variable``, or ``name``.
        estimate: Point-estimate column. Auto-detects ``exp_estimate``,
            ``oddsratio``, or ``estimate`` when omitted.
        conf_low: Lower CI column. Auto-detects ``exp_conf_low``, ``lo``,
            ``conf.low``, or ``conf_low`` when omitted.
        conf_high: Upper CI column. Auto-detects ``exp_conf_high``, ``hi``,
            ``conf.high``, or ``conf_high`` when omitted.
        null_value: Vertical reference. Defaults to 1 for ratios, 0 otherwise.
        drop_intercept: Drop intercept rows when recognized.
        log_scale: Log x-axis for ratio estimates (default when ratio columns
            are used).
        title: Optional plot title.
        xlabel: Optional x-axis label.
        xlim: Optional x-axis limits.
        figsize: Figure size; height scales with the number of terms when omitted.
        layout: ``"table"`` (default) or ``"points"``.
        style: ``"color"`` (screen/docs default) or ``"bw"`` (grayscale print).
        estimate_digits: Digits for the estimate (CI) text column.
        extra_cols: Optional extra tidy columns drawn as text in the table
            (for example ``n`` / ``event``).
        dpi: PNG resolution when ``path`` is set.
        also_pdf: Also write ``path`` with a ``.pdf`` suffix. The return value
            is still the PNG path.

    Returns:
        ``Path`` when ``path`` is given, otherwise a matplotlib ``Figure``.
    """
    if tidy.height == 0:
        raise ValueError("tidy frame is empty")
    if layout not in {"table", "points"}:
        raise ValueError("layout must be 'table' or 'points'")
    if style not in {"color", "bw"}:
        raise ValueError('style must be "color" or "bw"')

    extra_cols = tuple(extra_cols)
    frame, is_ratio = _prepare_frame(
        tidy,
        term=term,
        estimate=estimate,
        conf_low=conf_low,
        conf_high=conf_high,
        drop_intercept=drop_intercept,
        extra_cols=extra_cols,
    )

    labels = frame["term"].to_list()
    est = np.asarray(frame["estimate"].to_list(), dtype=float)
    lo = np.asarray(frame["conf_low"].to_list(), dtype=float)
    hi = np.asarray(frame["conf_high"].to_list(), dtype=float)
    if "p_value" in frame.columns:
        pvals = frame["p_value"].to_list()
        p_as_text = False
    elif "pvalue" in frame.columns:
        pvals = frame["pvalue"].to_list()
        p_as_text = True
    else:
        pvals = [None] * len(labels)
        p_as_text = False
    y = np.arange(len(labels))[::-1]

    use_log = is_ratio if log_scale is None else log_scale
    ref = (1.0 if is_ratio else 0.0) if null_value is None else float(null_value)
    if xlabel is None:
        xlabel = "Odds / hazard ratio" if is_ratio else "Coefficient"

    n = len(labels)
    colors = _style_spec(style)
    header_y = (n - 0.22) if layout == "table" else None
    resolved_xlim = xlim if xlim is not None else _auto_xlim(lo, hi, ref=ref, use_log=use_log)
    sep_y = float(np.max(y) + 0.5)

    with plt.rc_context(_FOREST_RC):
        if layout == "points":
            if figsize is None:
                figsize = (6.4, max(1.4, _ROW_IN * n + _AXIS_IN + 0.35))
            fig, ax = plt.subplots(figsize=figsize, facecolor="white")
            fig.patch.set_facecolor("white")
            _draw_forest_panel(
                ax,
                est,
                lo,
                hi,
                y,
                labels,
                ref=ref,
                use_log=use_log,
                xlim=resolved_xlim,
                xlabel=xlabel,
                colors=colors,
                header_y=None,
            )
            if title:
                ax.set_title(title, fontsize=10, color=colors["ink"], pad=4)
            fig.tight_layout(pad=0.25)
        else:
            est_header = _estimate_header(xlabel, is_ratio=is_ratio)
            est_texts = [
                _format_estimate(float(est[i]), float(lo[i]), float(hi[i]), digits=estimate_digits)
                for i in range(n)
            ]
            if p_as_text:
                p_texts = [
                    "—" if pvals[i] is None or str(pvals[i]).strip() == "" else str(pvals[i])
                    for i in range(n)
                ]
            else:
                p_texts = [
                    _format_pvalue(None if pvals[i] is None else float(pvals[i])) for i in range(n)
                ]
            extra_blocks: list[tuple[str, list[str], float]] = []
            for col in extra_cols:
                texts = ["" if v is None else str(v) for v in frame[col].to_list()]
                width = max(
                    0.35,
                    _approx_text_in(col, fontsize=_HEADER_SIZE, weight="bold"),
                    *(_approx_text_in(s, fontsize=_FONTSIZE) for s in texts),
                )
                extra_blocks.append((col, texts, width))
            extra_in = sum(w + _COL_PAD_IN for _, _, w in extra_blocks)
            term_in, est_in, p_in = _table_column_inches(labels, est_texts, p_texts, est_header)
            table_in = term_in + _COL_PAD_IN + extra_in + est_in + _COL_PAD_IN + p_in
            if figsize is None:
                extra = _TITLE_IN if title else 0.02
                figsize = (
                    table_in + _FOREST_PANEL_IN + 0.28,
                    max(0.95, _ROW_IN * n + _HEADER_IN + _AXIS_IN + extra),
                )
            fig = plt.figure(figsize=figsize, facecolor="white")
            fig.patch.set_facecolor("white")
            fig_h = float(figsize[1])
            top = 1.0 - ((_TITLE_IN / fig_h) if title else 0.02)
            bottom = min(0.18, _AXIS_IN / fig_h)
            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[table_in, _FOREST_PANEL_IN],
                wspace=0.04,
                left=0.02,
                right=0.985,
                top=max(0.78, top - 0.01),
                bottom=max(0.10, bottom),
            )
            ax_tab = fig.add_subplot(gs[0, 0])
            ax_forest = fig.add_subplot(gs[0, 1], sharey=ax_tab)
            ax_tab.set_xlim(0.0, 1.0)
            ax_tab.set_facecolor("white")
            ax_tab.axis("off")
            _row_bands(ax_tab, y, color=colors["band"], header_y=header_y)
            cursor = term_in + _COL_PAD_IN
            columns: list[tuple[float, str, str, Any]] = [
                (0.0, "left", "Term", lambda i: labels[i]),
            ]
            for header, texts, width in extra_blocks:
                columns.append((cursor / table_in, "left", header, lambda i, t=texts: t[i]))
                cursor += width + _COL_PAD_IN
            columns.append(((cursor + est_in / 2.0) / table_in, "center", est_header, lambda i: est_texts[i]))
            columns.append((1.0, "right", "P", lambda i: p_texts[i]))
            ax_tab.plot(
                [0.0, 1.0],
                [sep_y, sep_y],
                color=colors["ink"],
                lw=0.7,
                clip_on=False,
                zorder=2,
            )
            for hx, ha, header, getter in columns:
                ax_tab.text(
                    hx,
                    header_y,
                    header,
                    va="center",
                    ha=ha,
                    fontsize=_HEADER_SIZE,
                    fontweight="bold",
                    color=colors["ink"],
                    clip_on=False,
                )
                for i in range(n):
                    ax_tab.text(
                        hx,
                        y[i],
                        getter(i),
                        va="center",
                        ha=ha,
                        fontsize=_FONTSIZE,
                        color=colors["ink"],
                        clip_on=False,
                    )
            _draw_forest_panel(
                ax_forest,
                est,
                lo,
                hi,
                y,
                labels=None,
                ref=ref,
                use_log=use_log,
                xlim=resolved_xlim,
                xlabel=xlabel,
                colors=colors,
                header_y=header_y,
            )
            if title:
                fig.text(
                    0.5,
                    0.995,
                    title,
                    ha="center",
                    va="top",
                    fontsize=10,
                    color=colors["ink"],
                )

        if path is None:
            return fig
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_kw = dict(
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(out, **save_kw)
        if also_pdf:
            fig.savefig(out.with_suffix(".pdf"), format="pdf", **save_kw)
        plt.close(fig)
        return out


def _draw_forest_panel(
    ax: Any,
    est: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    labels: list[str] | None,
    *,
    ref: float,
    use_log: bool,
    xlim: tuple[float, float],
    xlabel: str,
    colors: dict[str, str],
    header_y: float | None,
) -> None:
    ax.set_facecolor("white")
    _row_bands(ax, y, color=colors["band"], header_y=header_y)
    x0, x1 = xlim
    if use_log:
        ax.set_xscale("log")
    ax.set_xlim(x0, x1)

    ax.axvline(ref, color=colors["ref"], linestyle="--", linewidth=0.9, zorder=1)
    if header_y is not None:
        sep_y = float(np.max(y) + 0.5)
        ax.plot([x0, x1], [sep_y, sep_y], color=colors["ink"], lw=0.7, zorder=1)

    for i, yi in enumerate(y):
        _draw_interval(
            ax,
            float(est[i]),
            float(lo[i]),
            float(hi[i]),
            float(yi),
            xlim=(x0, x1),
            color=colors["error"],
            marker_color=colors["marker"],
        )

    ax.set_yticks(y)
    if labels is None:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticklabels(labels, fontsize=_FONTSIZE, color=colors["ink"])
        ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel(xlabel, fontsize=_FONTSIZE, color=colors["ink"], labelpad=2)
    ax.tick_params(axis="x", colors=colors["ink"], labelsize=_AXIS_SIZE, length=3.0, width=0.7, pad=1.5)
    if use_log:
        ticks = _ratio_ticks(x0, x1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_ratio_tick))
        ax.xaxis.set_minor_locator(FixedLocator([]))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(colors["ink"])
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_ylim(*_table_ylim(len(y), header_y))


def _draw_interval(
    ax: Any,
    est: float,
    lo: float,
    hi: float,
    y: float,
    *,
    xlim: tuple[float, float],
    color: str,
    marker_color: str,
) -> None:
    x0, x1 = xlim
    if not np.isfinite(est):
        return
    lo_v = lo if np.isfinite(lo) else est
    hi_v = hi if np.isfinite(hi) else est
    left_clip = lo_v < x0
    right_clip = hi_v > x1
    a = min(max(lo_v, x0), x1)
    b = max(min(hi_v, x1), x0)
    ax.plot([a, b], [y, y], color=color, linewidth=1.2, solid_capstyle="butt", zorder=2, clip_on=False)
    if left_clip:
        ax.plot(x0, y, marker="<", color=color, markersize=5.0, zorder=3, clip_on=False)
    if right_clip:
        ax.plot(x1, y, marker=">", color=color, markersize=5.0, zorder=3, clip_on=False)
    mx = min(max(est, x0), x1)
    ax.plot(
        mx,
        y,
        marker="s",
        color=marker_color,
        markersize=5.2,
        markeredgewidth=0.0,
        zorder=4,
        clip_on=False,
    )
