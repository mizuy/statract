"""Draw the wall-clock and speed-ratio charts from the published comparison table.

Other tasks keep the seconds already stored in the CSV. This script does not
remeasure them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import japanize_matplotlib  # noqa: F401  # registers the Japanese font

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs" / "stat" / "benchmarks" / "comparison.csv"
OUT = CSV_PATH.parent

PYTHON = "#1C4D7C"
R_COLOR = "#21A6B6"
LARGE = {"adult-10000", "bike-10000", "support-large", "star-large", "star-rs-large"}
SMALL = {"adult-1000", "bike-1000", "support-1000", "star-1000", "star-rs-1000"}
LABELS = {
    "ols": "OLS",
    "ols-w": "OLS (weighted)",
    "glm-bin": "GLM binomial",
    "glm-gamma": "GLM gamma",
    "hc": "sandwich HC0–HC5",
    "wald": "Wald",
    "lr": "likelihood ratio",
    "bp": "Breusch–Pagan",
    "reset": "RESET",
    "m-logit": "nearest neighbor logit",
    "m-mah": "nearest neighbor Mahalanobis",
    "m-exact": "exact match",
    "m-cem": "CEM",
    "glm-pois": "GLM Poisson",
    "nw": "Newey–West",
    "dw": "Durbin–Watson",
    "bg": "Breusch–Godfrey",
    "gam-8": "GAM k=8",
    "gam-10": "GAM k=10",
    "km": "Kaplan–Meier",
    "km-sex": "KM (group)",
    "na": "Nelson–Aalen",
    "lrk": "log-rank",
    "cox-e": "Cox Efron",
    "cox-b": "Cox Breslow",
    "cox-s": "Cox strata",
    "aft-w": "AFT Weibull",
    "aft-ln": "AFT lognormal",
    "aft-ex": "AFT exponential",
    "cl1": "cluster 1-way",
    "cl2": "cluster 2-way",
    "lmm-ri": "LMM random intercept REML",
    "lmm-ml": "LMM ML",
    "lmm-rs": "LMM random slope",
}


def _tasks(rows: list[dict[str, str]], slices: set[str]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        if row["slice"] not in slices:
            continue
        key = (row["task"], row["slice"])
        slot = grouped.get(key)
        if slot is None:
            slot = {"task": row["task"], "py": None, "r": None, "failed": False}
            grouped[key] = slot
            order.append(key)
        passed = row["passed"] == "True"
        slot["failed"] = slot["failed"] or not passed
        if slot["py"] is None and row["python_s"] and row["r_s"]:
            slot["py"] = float(row["python_s"])
            slot["r"] = float(row["r_s"])
    tasks = []
    for key in order:
        slot = grouped[key]
        if slot["py"] is None or slot["r"] is None or slot["r"] == 0.0:
            continue
        label = LABELS.get(slot["task"], slot["task"])
        if slot["failed"]:
            label = f"{label}（tol外）"
        tasks.append({**slot, "label": label})
    return tasks


def _speed(tasks: list[dict], path: Path) -> None:
    labels = [task["label"] for task in tasks]
    y = np.arange(len(tasks))
    height = 0.38
    fig_h = max(8.0, 0.42 * len(tasks) + 1.4)
    fig, ax = plt.subplots(figsize=(12.4, fig_h), dpi=120)
    ax.barh(y - height / 2, [task["py"] * 1000 for task in tasks], height=height, color=PYTHON, label="Python")
    ax.barh(y + height / 2, [task["r"] * 1000 for task in tasks], height=height, color=R_COLOR, label="R")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("壁時計（ms）")
    ax.legend(loc="lower right")
    ax.grid(axis="x", which="both", color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _ratio_tick(k: float) -> str:
    k = int(k)
    if k == 0:
        return "等速"
    factor = 2 ** abs(k)
    side = "R" if k > 0 else "Python"
    return f"{side}\n{factor:.0f}倍"


def _ratio(tasks: list[dict], path: Path) -> None:
    labels = [task["label"] for task in tasks]
    values = np.array([np.log2(task["r"] / task["py"]) for task in tasks])
    y = np.arange(len(tasks))
    fig_h = max(8.0, 0.42 * len(tasks) + 1.6)
    fig, ax = plt.subplots(figsize=(12.6, fig_h), dpi=120)
    colors = [R_COLOR if value >= 0 else PYTHON for value in values]
    bars = ax.barh(y, values, color=colors, height=0.72)
    for bar, task in zip(bars, tasks, strict=True):
        if task["failed"]:
            bar.set_hatch("///")
            bar.set_edgecolor(PYTHON)
            bar.set_linewidth(0.6)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    lo = int(np.floor(values.min()))
    hi = int(np.ceil(values.max()))
    ticks = np.arange(lo, hi + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_ratio_tick(tick) for tick in ticks], fontsize=8)
    ax.set_xlabel("log2(R の壁時計 / Python の壁時計)")
    ax.legend(
        handles=[
            Patch(facecolor=R_COLOR, label="R が遅い"),
            Patch(facecolor=PYTHON, label="Python が遅い"),
            Patch(facecolor="white", edgecolor=PYTHON, hatch="///", label="tol外"),
        ],
        loc="lower right",
    )
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    for name, slices in (("large", LARGE), ("small", SMALL)):
        tasks = _tasks(rows, slices)
        _speed(tasks, OUT / f"speed-{name}.png")
        _ratio(tasks, OUT / f"ratio-{name}.png")
        print(name, len(tasks), "tasks")


if __name__ == "__main__":
    main()
