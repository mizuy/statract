"""Survival analysis utilities for Polars DataFrames.

Kaplan–Meier plotting and related helpers sit on the in-house curve
(``statract.surv.survival_curve``), not on an external fitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import polars as pl
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..surv.curve import survival_curve
from ..surv.logrank import log_rank

# Type aliases
ColumnRef: TypeAlias = str | pl.Expr
SurvivalPlotStyle: TypeAlias = Literal["color", "bw"]

# survminer / ggsci pal_jco — high-contrast on white, print-safe enough for screens.
_KM_COLORS = (
    "#0073C2",
    "#EFC000",
    "#868686",
    "#CD534C",
    "#7AA6DC",
    "#003C67",
    "#8F7700",
    "#A73030",
)
_KM_BW_GRAYS = ("#000000", "#4A4A4A", "#7A7A7A", "#A0A0A0")
_KM_BW_LINESTYLES = ("-", "--", "-.", ":")
_KM_BW_CENSORS: tuple[dict[str, Any], ...] = (
    {"marker": "+", "ms": 8, "mew": 1.15},
    {"marker": "x", "ms": 7, "mew": 1.15},
    {"marker": "o", "ms": 4.5, "mew": 1.05, "fillstyle": "none"},
    {"marker": "s", "ms": 4.0, "mew": 1.05, "fillstyle": "none"},
)


@dataclass
class KmPlotCurve:
    """Step KM (or 1−KM) series plus follow-up times for number-at-risk."""

    label: str
    durations: np.ndarray
    time: np.ndarray
    survival: np.ndarray
    conf_low: np.ndarray
    conf_high: np.ndarray
    n_censor: np.ndarray
    entry: np.ndarray | None = None

    @property
    def _label(self) -> str:
        return self.label


def _col(ref: ColumnRef) -> pl.Expr:
    return pl.col(ref) if isinstance(ref, str) else ref


def _at_risk_at_time(curve: Any, t: float) -> int:
    """Subjects still under follow-up at time ``t`` (``T_i >= t``, with delayed entry)."""
    durations = np.asarray(curve.durations, dtype=float)
    entry = getattr(curve, "entry", None)
    if entry is None:
        return int(np.sum(durations >= t))
    entry_arr = np.asarray(entry, dtype=float)
    return int(np.sum((entry_arr <= t) & (durations >= t)))


def _nice_candidates(raw: float) -> list[float]:
    if raw <= 0:
        return [1.0]
    magnitude = 10 ** np.floor(np.log10(raw))
    out = [float(c * magnitude) for c in (1.0, 2.0, 2.5, 5.0, 10.0)]
    return out


def _nice_step(span: float, target_ticks: int) -> float:
    """Round step covering ``span`` with about ``target_ticks`` intervals.

    Prefer the 1–2–5 candidate at or just above ``span / target``, but step
    down one nice size when the larger step would leave the axis sparse
    (month-scale follow-up of ~80 with target 8 would otherwise jump to 20).
    """
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    if raw <= 0:
        return 1.0
    candidates = _nice_candidates(raw)
    chosen = candidates[-1]
    prev = candidates[0]
    for step in candidates:
        if step >= raw:
            chosen = step
            break
        prev = step
    n_with_chosen = span / chosen
    if chosen > prev and n_with_chosen < max(target_ticks, 1) * 0.6:
        return float(prev)
    return float(chosen)


def default_at_risk_xticks(time_max: float, *, n_ticks: int = 8) -> np.ndarray:
    """Choose survminer-style time breaks (nice 1–2–5 steps, ~6–10 ticks).

    Counts in the number-at-risk table use these same breaks. The last tick is
    a round step (possibly just past follow-up), not the raw ``time_max``
    (which produced sparse axes like ``0, 1000, 2000, 3000, 3329``).
    """
    tmax = float(max(time_max, 0.0))
    if tmax == 0.0:
        return np.asarray([0.0])
    step = _nice_step(tmax, max(n_ticks - 1, 1))
    # One extra half-step so a round boundary just past tmax is kept (3500, not 3329).
    ticks = np.arange(0.0, tmax + step * 0.51, step)
    if ticks.size == 0:
        ticks = np.asarray([0.0, tmax])
    if ticks.size > 10:
        step = _nice_step(tmax, 6)
        ticks = np.arange(0.0, tmax + step * 0.51, step)
    return np.asarray(ticks, dtype=float)


def _cycle(values: tuple[Any, ...] | tuple[dict[str, Any], ...], n: int) -> list[Any]:
    return [values[i % len(values)] for i in range(n)]


def _km_group_styles(style: SurvivalPlotStyle, n: int) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    if style == "bw":
        colors = _cycle(_KM_BW_GRAYS, n)
        linestyles = _cycle(_KM_BW_LINESTYLES, n)
        censors = _cycle(_KM_BW_CENSORS, n)
        return colors, linestyles, censors
    colors = _cycle(_KM_COLORS, n)
    linestyles = ["-"] * n
    censors = [{"marker": "+", "ms": 7, "mew": 1.1}] * n
    return colors, linestyles, censors


def _apply_survminer_axes(ax: Any, *, cdf: bool) -> None:
    """Classic white panel: no top/right spines, y in [0, 1], readable ticks."""
    ax.set_facecolor("white")
    fig = ax.figure
    if fig is not None:
        fig.patch.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#222222", labelsize=9, length=3.5, width=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Cumulative incidence" if cdf else "Survival probability")
    ax.grid(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_title("")
        for text in legend.get_texts():
            text.set_fontsize(9)


def _new_km_axes(*, n_groups: int, at_risk_counts: bool) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    extra = (1.0 + 0.12 * n_groups) if at_risk_counts else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 4.9 + extra), facecolor="white")
    fig.patch.set_facecolor("white")
    # Risk panel is appended from this axes (same spine x0/x1). Returning
    # ``None`` keeps a single layout path in ``add_at_risk_counts``.
    return ax, None


def _format_time_tick(value: float, _pos: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _padded_xlim(xticks: list[float] | np.ndarray, time_max: float) -> tuple[float, float]:
    """Inset the first/last ticks from the spines so centered NAR digits at t=0 fit.

    This is axis expansion (ggplot-style), not a fudge between counts and ticks:
    both the KM ticks and the NAR text stay on the same data x.
    """
    left = min(0.0, float(np.min(xticks)))
    right = float(max(np.max(xticks), time_max))
    span = max(right - left, 1e-9)
    return left - 0.045 * span, right + 0.03 * span


def _sync_xlim(src: Any, dest: Any) -> None:
    dest.set_xlim(src.get_xlim())

    def _on_xlim(_event: Any) -> None:
        if dest.get_xlim() != src.get_xlim():
            dest.set_xlim(src.get_xlim())

    src.callbacks.connect("xlim_changed", _on_xlim)


def _append_risk_axes(ax: Any, n_groups: int) -> Any:
    """Panel under ``ax`` with the same display width (aligned spines), not twiny."""
    divider = make_axes_locatable(ax)
    size_in = max(0.7, 0.32 * n_groups + 0.28)
    # Do not sharex: sharing hides the KM tick labels when the table turns them off.
    return divider.append_axes("bottom", size=size_in, pad=0.55)


def add_at_risk_counts(
    *curves: Any,
    ax: Any = None,
    xticks: np.ndarray | list[float] | None = None,
    ypos: float = -0.6,  # noqa: ARG001 — kept for call-site compatibility
    risk_ax: Any = None,
    colors: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Draw an at-risk table under a KM/CIF plot.

    Counts are placed in **data x** of an axes that ``sharex`` the KM curve
    (GridSpec or ``make_axes_locatable``). Numbers cannot drift from the curve
    ticks: both use the same x-scale and the same display mapping. Group
    names are y-tick labels in the left margin (not data-x text at ``-0.02``,
    which collapsed onto t=0 for day-scale follow-up).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    if not curves:
        return ax
    if xticks is None:
        min_time, max_time = ax.get_xlim()
        xticks = [float(t) for t in ax.get_xticks() if min_time <= float(t) <= max_time]
        if not xticks:
            xmax = max(float(np.max(np.asarray(c.durations, dtype=float))) for c in curves)
            xticks = list(default_at_risk_xticks(xmax))
    else:
        xticks = [float(t) for t in xticks]

    n = len(curves)
    if risk_ax is None:
        risk_ax = _append_risk_axes(ax, n_groups=n)

    palette = list(colors) if colors is not None else ["#222222"] * n
    if len(palette) < n:
        palette = _cycle(tuple(palette) or ("#222222",), n)

    risk_ax.set_facecolor("white")
    _sync_xlim(ax, risk_ax)
    risk_ax.xaxis.set_major_locator(FixedLocator(xticks))
    risk_ax.xaxis.set_major_formatter(FuncFormatter(_format_time_tick))
    risk_ax.tick_params(axis="x", labelbottom=False, length=0, width=0)
    risk_ax.tick_params(axis="y", length=0, labelsize=8, pad=8)
    for spine in risk_ax.spines.values():
        spine.set_visible(False)
    risk_ax.set_ylim(n - 0.5, -0.5)
    risk_ax.set_yticks(list(range(n)))
    labels = [str(c._label) for c in curves]
    risk_ax.set_yticklabels(labels)
    for tick_label, color in zip(risk_ax.get_yticklabels(), palette, strict=True):
        tick_label.set_color(color)
    risk_ax.set_ylabel("At risk", fontsize=9, labelpad=10)

    count_texts: list[Any] = []
    for i, curve in enumerate(curves):
        for tick in xticks:
            artist = risk_ax.text(
                tick,
                i,
                str(_at_risk_at_time(curve, tick)),
                ha="center",
                va="center",
                fontsize=8,
                color=palette[i],
                clip_on=True,
            )
            count_texts.append(artist)

    ax._statract_at_risk_ax = risk_ax  # type: ignore[attr-defined]
    ax._statract_at_risk_xticks = np.asarray(xticks, dtype=float)  # type: ignore[attr-defined]
    ax._statract_at_risk_texts = count_texts  # type: ignore[attr-defined]
    return ax


def timedelta_to_years(expr: pl.Expr) -> pl.Expr:
    """Convert timedelta expression to years.

    Converts Polars Duration to years using average year length (365.242 days).

    Examples:
        >>> import polars as pl
        >>> from statract import timedelta_to_years
        >>>
        >>> df = pl.DataFrame({
        ...     "duration": pl.Series([365, 730, 1095], dtype=pl.Duration("ns"))
        ... })
        >>> df.select(timedelta_to_years(pl.col("duration")).alias("years"))
        shape: (3, 1)
        ┌───────┐
        │ years │
        │ ---   │
        │ f64   │
        ╞═══════╡
        │ ~1.0  │
        │ ~2.0  │
        │ ~3.0  │
        └───────┘
    """
    return (expr.dt.total_days() / 365.242).clip(0)


def cumulative_survival_ci(
    time: ColumnRef,
    status: ColumnRef,
    *,
    survive_at: float = 5.0,
    cdf: bool = False,
) -> pl.Expr:
    """Calculate cumulative survival/incidence rate with confidence interval.

    Returns a struct containing rate, lo, hi, and summary string.

    Examples:
        >>> import polars as pl
        >>> from statract import cumulative_survival_ci
        >>>
        >>> # Survival analysis data
        >>> df = pl.DataFrame({
        ...     "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "status": [True, True, False, True, False]  # True = event occurred
        ... })
        >>>
        >>> # Calculate 5-year survival rate
        >>> result = df.select(
        ...     cumulative_survival_ci("time", "status", survive_at=5.0).alias("survival")
        ... )
        >>> print(result.select(
        ...     pl.col("survival").struct.field("rate"),
        ...     pl.col("survival").struct.field("lo"),
        ...     pl.col("survival").struct.field("hi")
        ... ))
        >>>
        >>> # Calculate cumulative incidence (cdf=True)
        >>> result = df.select(
        ...     cumulative_survival_ci("time", "status", survive_at=5.0, cdf=True).alias("incidence")
        ... )
    """
    """Return an expression for cumulative survival/incidence at a time point as a struct.

    Uses ``survival_curve`` inside `map_batches`, and returns:
    Struct{rate: f64, lo: f64, hi: f64, summary: str}
    """

    out_dtype = pl.Struct(
        [
            pl.Field("rate", pl.Float64),
            pl.Field("lo", pl.Float64),
            pl.Field("hi", pl.Float64),
            pl.Field("summary", pl.String),
        ]
    )

    t_expr = _col(time).alias("_t")
    e_expr = _col(status).alias("_e")

    # Ensure the expression depends on df via selecting columns; compute via map_batches.
    return pl.struct([t_expr, e_expr]).map_batches(
        lambda s: _km_ci_batch(s, survive_at=survive_at, cdf=cdf, out_dtype=out_dtype),
        return_dtype=out_dtype,
    )


def _km_ci_batch(s: pl.Series, *, survive_at: float, cdf: bool, out_dtype: pl.DataType) -> pl.Series:
    # s is a Series of struct with fields _t and _e
    t_raw = np.asarray(s.struct.field("_t"))
    e_raw = np.asarray(s.struct.field("_e"))

    # Drop nulls (keep rows where both are non-null).
    mask = np.array(
        [ti is not None and ei is not None for ti, ei in zip(t_raw, e_raw, strict=False)],
    )
    t = np.asarray(t_raw[mask], dtype=float)
    e = np.asarray(e_raw[mask])

    if t.size == 0:
        return pl.Series(
            [{"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "summary": ""}],
            dtype=out_dtype,
        )

    curve = survival_curve(pl.DataFrame({"t": t, "e": e}), "t", "e")
    at = curve.at(survive_at)
    rate = float(at["estimate"][0])
    lo_v = float(at["conf_low"][0])
    hi_v = float(at["conf_high"][0])
    if cdf:
        rate, lo_v, hi_v = 1.0 - rate, 1.0 - hi_v, 1.0 - lo_v
        if lo_v > hi_v:
            lo_v, hi_v = hi_v, lo_v
    summary = f"{rate:.1%} ({lo_v:.1%}-{hi_v:.1%})"
    return pl.Series([{"rate": rate, "lo": lo_v, "hi": hi_v, "summary": summary}], dtype=out_dtype)


def log_rank_pvalue(time: ColumnRef, status: ColumnRef, group: ColumnRef) -> pl.Expr:
    """Return an expression for log-rank test p-value between two groups.

    Compares survival curves between two groups using the in-house log-rank
    test inside `map_batches`. Returns a scalar float expression.

    Args:
        time: Time column expression (column name or pl.Expr)
        status: Event status column expression (True = event occurred)
        group: Group column expression (must have exactly 2 groups)

    Returns:
        Polars Expr representing the p-value

    Examples:
        >>> import polars as pl
        >>> from statract import log_rank_pvalue
        >>>
        >>> df = pl.DataFrame({
        ...     "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "status": [True, True, False, True, False],
        ...     "group": ["A", "A", "A", "B", "B"]
        ... })
        >>> pvalue = df.select(log_rank_pvalue("time", "status", "group").alias("p")).item()
        >>> print(f"Log-rank p-value: {pvalue}")
    """
    t_expr = _col(time).alias("_t")
    e_expr = _col(status).alias("_e")
    g_expr = _col(group).alias("_g")

    return pl.struct([t_expr, e_expr, g_expr]).map_batches(
        lambda s: _logrank_batch(s),
        return_dtype=pl.Float64,
        returns_scalar=True,
    )


def _logrank_batch(s: pl.Series) -> float:
    """Helper for log_rank_pvalue: compute log-rank p-value from struct Series."""
    # s is a Series of struct with fields _t, _e, _g
    t_raw = np.asarray(s.struct.field("_t"))
    e_raw = np.asarray(s.struct.field("_e"))
    g_raw = np.asarray(s.struct.field("_g"))

    # Drop nulls (keep rows where all are non-null).
    mask = np.array(
        [ti is not None and ei is not None and gi is not None for ti, ei, gi in zip(t_raw, e_raw, g_raw, strict=False)],
    )
    t = np.asarray(t_raw[mask], dtype=float)
    e = np.asarray(e_raw[mask])
    g = np.asarray(g_raw[mask])

    if t.size == 0:
        return float("nan")

    groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError(f"Log rank test compares two groups. cardinality is not 2, but {len(groups)}.")

    frame = pl.DataFrame({"t": t, "e": e, "g": g})
    return float(log_rank(frame, "t", "e", "g").p_value)


def _km_from_group(df: pl.DataFrame, time_col: str, status_col: str, label: str) -> KmPlotCurve:
    fitted = survival_curve(df, time_col, status_col)
    table = fitted.frame()
    return KmPlotCurve(
        label=str(label),
        durations=np.asarray(df[time_col].to_numpy(), dtype=float),
        time=np.asarray(table["time"].to_numpy(), dtype=float),
        survival=np.asarray(table["estimate"].to_numpy(), dtype=float),
        conf_low=np.asarray(table["conf_low"].to_numpy(), dtype=float),
        conf_high=np.asarray(table["conf_high"].to_numpy(), dtype=float),
        n_censor=np.asarray(table["n_censor"].to_numpy(), dtype=float),
    )


def _step_coords(
    time: np.ndarray,
    values: np.ndarray,
    *,
    origin: float,
    time_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.size == 0:
        t = np.asarray([0.0, time_max], dtype=float)
        v = np.asarray([origin, origin], dtype=float)
        return t, v
    if t[0] > 0.0:
        t = np.concatenate([[0.0], t])
        v = np.concatenate([[origin], v])
    if t[-1] < time_max:
        t = np.concatenate([t, [time_max]])
        v = np.concatenate([v, [v[-1]]])
    return t, v


def _plot_km_curve(
    ax: Any,
    curve: KmPlotCurve,
    *,
    cdf: bool,
    time_max: float,
    color: str,
    linestyle: str,
    censor_styles: dict[str, Any],
    ci_show: bool,
    show_censors: bool,
    ci_alpha: float,
    lw: float,
    extra: dict[str, Any],
) -> None:
    origin = 0.0 if cdf else 1.0
    if cdf:
        est = 1.0 - curve.survival
        lo = 1.0 - curve.conf_high
        hi = 1.0 - curve.conf_low
    else:
        est = curve.survival
        lo = curve.conf_low
        hi = curve.conf_high
    t, y = _step_coords(curve.time, est, origin=origin, time_max=time_max)
    _, y_lo = _step_coords(curve.time, lo, origin=origin, time_max=time_max)
    _, y_hi = _step_coords(curve.time, hi, origin=origin, time_max=time_max)
    step_kw = dict(extra)
    ax.step(
        t,
        y,
        where="post",
        color=color,
        linestyle=linestyle,
        lw=lw,
        label=curve.label,
        **step_kw,
    )
    if ci_show:
        ax.fill_between(
            t,
            y_lo,
            y_hi,
            step="post",
            color=color,
            alpha=ci_alpha,
            linewidth=0,
            zorder=0,
            label="_nolegend_",
        )
    if show_censors and curve.time.size:
        mask = np.asarray(curve.n_censor) > 0
        if np.any(mask):
            ax.plot(
                curve.time[mask],
                est[mask],
                linestyle="None",
                color=color,
                label="_nolegend_",
                **censor_styles,
            )


def plot_survival(
    data: pl.DataFrame,
    time: ColumnRef | None = None,
    status: ColumnRef | None = None,
    stime: ColumnRef | None = None,
    hue: ColumnRef | None = None,
    cdf: bool = False,
    at_risk_counts: bool = True,
    ax: Any = None,
    labels: list[str] | dict[str, str] | None = None,
    style: SurvivalPlotStyle = "color",
    **kwargs,
) -> Any:
    """Plot Kaplan–Meier curves with publication-ready survminer-like defaults.

    A single call draws CI bands, censor marks, a shared-x number-at-risk
    table, and a clean white theme. ``style="color"`` (default) uses a
    high-contrast palette; ``style="bw"`` is a print-journal grayscale mode
    that distinguishes groups by linetype and censor marker, not color alone.

    Estimates come from ``survival_curve`` (in-house Kaplan–Meier).

    Args:
        data: DataFrame containing survival data
        time: Time column expression (column name or pl.Expr). Required if stime is None.
        status: Event status column expression (True = event occurred). Required if stime is None.
        stime: Survival time column expression (positive = event, negative = censored).
               Alternative to time/status. If specified, time and status must be None.
        hue: Group column expression for stratified plots
        cdf: If True, plot cumulative incidence instead of survival
        at_risk_counts: If True (default), add a number-at-risk table below the
            plot (survminer-style). Gallery / example KM figures should keep
            this enabled.
        ax: Matplotlib axes (if None, creates a constrained figure with a
            matching risk-table panel)
        labels: Labels for groups (list or dict mapping group values to labels)
        style: ``"color"`` (screen/docs default) or ``"bw"`` (grayscale print)
        **kwargs: Plot overrides (``ci_show``, ``show_censors``, ``censor_styles``,
            ``ci_alpha``, ``lw``, plus matplotlib ``step`` kwargs). Explicit
            kwargs override the publication defaults.

    Returns:
        Matplotlib axes object (the KM panel)

    Examples:
        >>> import polars as pl
        >>> from statract import plot_survival
        >>> import matplotlib.pyplot as plt
        >>>
        >>> df = pl.DataFrame({
        ...     "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "status": [True, True, False, True, False],
        ...     "group": ["A", "A", "A", "B", "B"]
        ... })
        >>>
        >>> # Plot survival curves by group
        >>> ax = plot_survival(df, time="time", status="status", hue="group")
        >>> plt.show()
        >>>
        >>> # Plot cumulative incidence
        >>> ax = plot_survival(df, time="time", status="status", hue="group", cdf=True)
        >>> plt.show()
        >>>
        >>> # Black-and-white journal figure
        >>> ax = plot_survival(df, time="time", status="status", hue="group", style="bw")
    """
    if style not in {"color", "bw"}:
        raise ValueError('style must be "color" or "bw"')

    df = data
    time_col = "time_"
    status_col = "status_"
    hue_col = "hue_" if hue is not None else None

    if stime is not None:
        if time is not None or status is not None:
            raise ValueError(
                f"time and status cannot be specified when stime is specified: {time=!r}, {status=!r}, {stime=!r}"
            )
        stime_expr = _col(stime).alias("_stime")
        df = df.with_columns(stime_expr)
        df = df.with_columns(
            pl.col("_stime").abs().alias(time_col),
            (pl.col("_stime") > 0).alias(status_col),
        )
    else:
        if time is None or status is None:
            raise ValueError("time and status must be specified when stime is not specified")
        df = df.with_columns(_col(time).alias(time_col), _col(status).alias(status_col))

    if hue is not None:
        df = df.with_columns(_col(hue).alias(hue_col))

    if labels is not None:
        assert hue is not None
        assert df[hue_col].n_unique() == len(labels)

    curves: list[KmPlotCurve] = []
    if hue is None:
        curves.append(_km_from_group(df, time_col, status_col, "All"))
    else:
        for i, item in enumerate(df.group_by(hue_col, maintain_order=True)):
            group_df = item[1] if isinstance(item, tuple) else item
            name = group_df[hue_col][0]
            if isinstance(labels, list):
                label = labels[i]
            elif isinstance(labels, dict):
                label = labels.get(name)
            else:
                label = name
            curves.append(_km_from_group(group_df, time_col, status_col, label))

    risk_ax = None
    created_fig = ax is None
    if created_fig:
        ax, risk_ax = _new_km_axes(n_groups=len(curves), at_risk_counts=at_risk_counts)

    colors, linestyles, censor_cycle = _km_group_styles(style, len(curves))
    plot_defaults: dict[str, Any] = {
        "ci_show": True,
        "show_censors": True,
        "ci_alpha": 0.22 if style == "color" else 0.14,
        "lw": 1.8,
    }
    plot_defaults.update(kwargs)
    ci_show = bool(plot_defaults.pop("ci_show"))
    show_censors = bool(plot_defaults.pop("show_censors"))
    ci_alpha = float(plot_defaults.pop("ci_alpha"))
    lw = float(plot_defaults.pop("lw"))
    override_censor = plot_defaults.pop("censor_styles", None)

    time_max = float(df[time_col].max())
    for i, curve in enumerate(curves):
        censor_styles = {**censor_cycle[i], "markerfacecolor": "none"}
        if override_censor:
            censor_styles.update(override_censor)
        extra = dict(plot_defaults)
        extra.pop("color", None)
        extra.pop("linestyle", None)
        _plot_km_curve(
            ax,
            curve,
            cdf=cdf,
            time_max=time_max,
            color=plot_defaults.get("color", colors[i]),
            linestyle=plot_defaults.get("linestyle", linestyles[i]),
            censor_styles=censor_styles,
            ci_show=ci_show,
            show_censors=show_censors,
            ci_alpha=ci_alpha,
            lw=lw,
            extra=extra,
        )

    xticks = default_at_risk_xticks(time_max)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_time_tick))
    ax.set_xlim(*_padded_xlim(xticks, time_max))
    ax.tick_params(axis="x", labelbottom=True, length=3.5)
    ax.set_xlabel("Time")
    ax.legend()
    _apply_survminer_axes(ax, cdf=cdf)

    if at_risk_counts:
        add_at_risk_counts(
            *curves,
            ax=ax,
            xticks=xticks,
            risk_ax=risk_ax,
            colors=colors,
        )

    return ax


