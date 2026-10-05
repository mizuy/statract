"""Cox / KM diagnostic plots inspired by survminer (Python-native matplotlib).

Supported cleanly with current ``CoxFit.residuals`` / KM APIs:

- ``schoenfeld`` / ``scaled_schoenfeld`` vs time (per term; optional lowess)
- ``loglog`` complementary log-log KM curves by group
- ``dfbeta`` / ``dfbetas`` index plots
- ``martingale`` / ``deviance`` vs linear predictor (or a covariate)

Gaps vs survminer (documented, not bridged via R):

- No interactive ``ggcoxdiagnostics`` faceting / theme system
- No built-in outlier labeling / cook's distance panel
- Stratified Cox Schoenfeld panels do not draw per-stratum smooths separately
- Fine–Gray / weighted Cox PH residual plots are not specialized
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .cox import CoxFit, _schoenfeld_times
from .curve import survival_curve

DiagnosticKind = Literal[
    "schoenfeld",
    "scaled_schoenfeld",
    "loglog",
    "dfbeta",
    "dfbetas",
    "martingale",
    "deviance",
]

_RESIDUAL_KINDS = frozenset(
    {
        "schoenfeld",
        "scaled_schoenfeld",
        "dfbeta",
        "dfbetas",
        "martingale",
        "deviance",
    }
)


def _maybe_lowess(x: np.ndarray, y: np.ndarray, *, frac: float = 2 / 3) -> tuple[np.ndarray, np.ndarray] | None:
    if x.size < 5:
        return None
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        return None
    sm = lowess(ys, xs, frac=frac, return_sorted=True)
    return sm[:, 0], sm[:, 1]


def plot_loglog(
    data: pl.DataFrame,
    time: str,
    status: str,
    *,
    by: str,
    ax: Any = None,
    path: Path | str | None = None,
) -> Any:
    """Complementary log-log plot of KM curves stratified by ``by``."""
    curve = survival_curve(data, time, status, by=by)
    frame = curve.frame()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.5))
    groups = frame["group"].unique(maintain_order=True).to_list() if "group" in frame.columns else [None]
    for group in groups:
        part = frame if group is None else frame.filter(pl.col("group") == group)
        t = np.asarray(part["time"].to_list(), dtype=float)
        s = np.clip(np.asarray(part["estimate"].to_list(), dtype=float), 1e-12, 1 - 1e-12)
        mask = (t > 0) & np.isfinite(s)
        if not np.any(mask):
            continue
        y = np.log(-np.log(s[mask]))
        x = np.log(t[mask])
        label = "All" if group is None else str(group)
        ax.step(x, y, where="post", label=label)
    ax.set_xlabel("log(time)")
    ax.set_ylabel("log(-log(S(t)))")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        return out
    return ax


def plot_cox_residuals(
    fit: CoxFit,
    kind: DiagnosticKind = "scaled_schoenfeld",
    *,
    terms: Iterable[str] | None = None,
    path: Path | str | None = None,
    lowess: bool = True,
    max_panels: int = 9,
) -> Path | Any:
    """Plot Cox residual diagnostics for one residual ``kind``.

    For Schoenfeld / dfbeta families, one panel per term (capped by
    ``max_panels``). Martingale / deviance use a single panel vs the linear
    predictor.
    """
    if kind == "loglog":
        raise ValueError("use plot_loglog(...) for complementary log-log KM plots")
    if kind not in _RESIDUAL_KINDS:
        raise ValueError(f"kind must be one of {sorted(_RESIDUAL_KINDS | {'loglog'})}")

    resid = fit.residuals(kind if kind != "scaled_schoenfeld" else "scaled_schoenfeld")
    names = list(fit.names)
    if kind in {"martingale", "deviance"}:
        lp = fit.predict(kind="linear_predictor")
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(lp, resid, s=12, alpha=0.65, color="#1f4e79", edgecolors="none")
        if lowess:
            sm = _maybe_lowess(np.asarray(lp, dtype=float), np.asarray(resid, dtype=float))
            if sm is not None:
                ax.plot(sm[0], sm[1], color="#c0392b", linewidth=1.5)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Linear predictor")
        ax.set_ylabel(kind)
        ax.set_title(f"Cox {kind} residuals")
        fig.tight_layout()
        if path is None:
            return fig
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    if resid.ndim == 1:
        resid = resid.reshape(-1, 1)
    if terms is None:
        use_names = names[:max_panels]
        cols = list(range(len(use_names)))
    else:
        use_names = []
        cols = []
        for name in terms:
            if name not in names:
                raise ValueError(f"unknown term {name!r}; known: {names}")
            use_names.append(name)
            cols.append(names.index(name))
        use_names = use_names[:max_panels]
        cols = cols[:max_panels]

    n = len(use_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    if kind in {"schoenfeld", "scaled_schoenfeld"}:
        x = np.asarray(_schoenfeld_times(fit), dtype=float)
        xlab = "Time"
    else:
        x = np.arange(resid.shape[0], dtype=float)
        xlab = "Observation index"

    for i, (name, col) in enumerate(zip(use_names, cols, strict=True)):
        ax = axes[i // ncols][i % ncols]
        y = np.asarray(resid[:, col], dtype=float)
        ax.scatter(x, y, s=12, alpha=0.65, color="#1f4e79", edgecolors="none")
        if lowess:
            sm = _maybe_lowess(x, y)
            if sm is not None:
                ax.plot(sm[0], sm[1], color="#c0392b", linewidth=1.5)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel(kind)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Cox {kind} diagnostics", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path is None:
        return fig
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def write_cox_diagnostic_suite(
    fit: CoxFit,
    out_dir: Path | str,
    *,
    data: pl.DataFrame | None = None,
    time: str | None = None,
    status: str | None = None,
    by: str | None = None,
    stem: str = "cox",
) -> list[Path]:
    """Write a small suite of diagnostic PNGs under ``out_dir``.

    Always writes scaled Schoenfeld, dfbeta, martingale, and deviance plots.
    When ``data`` / ``time`` / ``status`` / ``by`` are given, also writes a
    log-log KM plot.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for kind, suffix in (
        ("scaled_schoenfeld", "schoenfeld"),
        ("dfbeta", "dfbeta"),
        ("martingale", "martingale"),
        ("deviance", "deviance"),
    ):
        path = out / f"{stem}_{suffix}.png"
        plot_cox_residuals(fit, kind=kind, path=path)  # type: ignore[arg-type]
        written.append(path)
    if data is not None and time is not None and status is not None and by is not None:
        path = out / f"{stem}_loglog.png"
        plot_loglog(data, time, status, by=by, path=path)
        written.append(path)
    return written
