"""Calibration, Brier score, and decision-curve analysis."""

from __future__ import annotations

from pathlib import Path

from . import _mpl as _mpl  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    n_bins: int = 10,
    model: str = "model",
    split: str = "validation",
) -> pl.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if y.size == 0:
        return pl.DataFrame(
            schema={
                "model": pl.String,
                "split": pl.String,
                "bin": pl.Int64,
                "n": pl.Int64,
                "pred_mean": pl.Float64,
                "obs_rate": pl.Float64,
            },
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "model": model,
                "split": split,
                "bin": i + 1,
                "n": n,
                "pred_mean": float(p[mask].mean()),
                "obs_rate": float(y[mask].mean()),
            },
        )
    return pl.DataFrame(rows)


def net_benefit(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """Decision-curve net benefit for predicting the positive class."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    n = y.size
    if n == 0:
        return float("nan")
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    w = threshold / (1.0 - threshold) if threshold < 1.0 else np.inf
    return tp / n - fp / n * w


def decision_curve_table(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    model: str = "model",
) -> pl.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 10)
    y = np.asarray(y_true, dtype=int)
    prevalence = float(y.mean()) if y.size else 0.0
    rows: list[dict[str, object]] = []
    for t in thresholds:
        t = float(t)
        nb_model = net_benefit(y, prob, t)
        nb_all = prevalence - (1.0 - prevalence) * (t / (1.0 - t)) if t < 1 else float("nan")
        rows.append(
            {
                "threshold": t,
                "model": model,
                "net_benefit": nb_model,
                "treat_all": nb_all,
                "treat_none": 0.0,
            },
        )
    return pl.DataFrame(rows)


def plot_calibration(table: pl.DataFrame, path: Path, *, title: str = "Calibration") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    for model in table["model"].unique(maintain_order=True).to_list():
        part = table.filter(pl.col("model") == model).sort("pred_mean")
        ax.plot(part["pred_mean"], part["obs_rate"], marker="o", label=str(model))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dca(table: pl.DataFrame, path: Path, *, title: str = "Decision curve") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in table["model"].unique(maintain_order=True).to_list():
        part = table.filter(pl.col("model") == model).sort("threshold")
        ax.plot(part["threshold"], part["net_benefit"], label=str(model))
    if "treat_all" in table.columns:
        part = table.filter(pl.col("model") == table["model"][0]).sort("threshold")
        ax.plot(part["threshold"], part["treat_all"], linestyle="--", color="gray", label="treat all")
        ax.plot(part["threshold"], part["treat_none"], linestyle=":", color="black", label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_probability_artifacts(
    models: list[dict],
    out_dir: Path,
    *,
    n_bins: int = 10,
    dca_thresholds: np.ndarray | None = None,
    positive_label: str = "positive class",
) -> list[Path]:
    """Write calibration / Brier / DCA tables and figures from model dicts.

    Each model dict needs ``prob_val``, ``y_val``, and ``name``.
    """
    from .reporting import write_csv_companion

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cal_parts: list[pl.DataFrame] = []
    dca_parts: list[pl.DataFrame] = []
    brier_rows: list[dict[str, object]] = []
    for m in models:
        y = np.asarray(m["y_val"], dtype=float)
        p = np.asarray(m["prob_val"], dtype=float)
        name = str(m["name"])
        cal_parts.append(
            calibration_table(y, p, n_bins=n_bins, model=name, split="validation"),
        )
        dca_parts.append(
            decision_curve_table(y, p, thresholds=dca_thresholds, model=name),
        )
        brier_rows.append(
            {
                "model": name,
                "split": "validation",
                "brier": brier_score(y, p),
                "n": int(y.size),
            },
        )
    cal = pl.concat(cal_parts, how="vertical_relaxed") if cal_parts else pl.DataFrame()
    dca = pl.concat(dca_parts, how="vertical_relaxed") if dca_parts else pl.DataFrame()
    brier = pl.DataFrame(brier_rows)

    for stem, df in (
        ("table_calibration", cal),
        ("table_dca", dca),
        ("table_brier", brier),
    ):
        path = out_dir / f"{stem}.csv"
        df.write_csv(path)
        written.extend([path, write_csv_companion(path)])

    if not cal.is_empty():
        written.append(plot_calibration(cal, out_dir / "fig_calibration.png"))
    if not dca.is_empty():
        written.append(plot_dca(dca, out_dir / "fig_dca.png"))

    lines = [
        f"Validation-set probability evaluation (positive = {positive_label}).",
        "",
    ]
    for row in brier_rows:
        lines.append(f"- `{row['model']}`: Brier = {row['brier']:.4f} (n={row['n']})")
    text_path = out_dir / "text_calibration.md"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(text_path)
    dca_md = out_dir / "text_dca.md"
    dca_md.write_text(
        "Decision curve: thresholds are a clinically plausible interval for the "
        "positive-class prediction (default 0.05–0.50). treat-all / treat-none are included.\n",
        encoding="utf-8",
    )
    written.append(dca_md)
    return written
