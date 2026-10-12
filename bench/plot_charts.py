"""Draw the speed-ratio charts from the published comparison table.

This script reads the seconds stored in the CSV. It does not remeasure them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import japanize_matplotlib  # noqa: F401  # registers the Japanese font

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "models" / "benchmarks" / "comparison.csv"
OUT = CSV_PATH.parent

PYTHON = "#1C4D7C"
R_COLOR = "#21A6B6"
LARGE = {"adult-10000", "bike-10000", "support-large", "support-mi-large", "star-large", "star-rs-large"}
SMALL = {"adult-1000", "bike-1000", "support-1000", "support-mi-1000", "star-1000", "star-rs-1000"}
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
    "ttest": "t test",
    "wilcox": "Wilcoxon rank sum",
    "prop": "prop test",
    "padj": "p.adjust",
    "roc": "ROC AUC + DeLong",
    "val-lrm": "validate lrm",
    "cal-lrm": "calibrate lrm",
    "val-cph": "validate cph",
    "cal-cph": "calibrate cph",
    "std-cox": "Cox standardization",
    "cuminc": "cumulative incidence",
    "crr": "Fine–Gray crr",
    "gamm": "GAMM (gamm4)",
    "glmm-pois": "GLMM Poisson",
    "glmm-nb": "GLMM negative binomial",
    "ctree": "conditional tree",
    "mice": "MICE + pool",
}


def _tasks(rows: list[dict[str, str]], slices: set[str]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        if row["slice"] not in slices:
            continue
        slot = grouped.setdefault(row["task"], {"py": None, "r": None, "failed": False})
        slot["failed"] = slot["failed"] or row["passed"] != "True"
        if slot["py"] is None and row["python_s"] and row["r_s"]:
            slot["py"] = float(row["python_s"])
            slot["r"] = float(row["r_s"])
    return {task: slot for task, slot in grouped.items() if slot["py"] and slot["r"]}


def _ratio_tick(k: float) -> str:
    k = int(k)
    if k == 0:
        return "等速"
    factor = 2 ** abs(k)
    side = "Python" if k > 0 else "R"
    return f"{side} が\n{factor:.0f}倍速い"


def _ratio(large: dict[str, dict], small: dict[str, dict], path: Path) -> None:
    order = [task for task in LABELS if task in large or task in small]
    y = np.arange(len(order))
    height = 0.4
    fig_h = max(8.0, 0.5 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(12.6, fig_h), dpi=120)
    values = []
    for offset, slices, color in ((-height / 2, large, PYTHON), (height / 2, small, R_COLOR)):
        for i, task in enumerate(order):
            slot = slices.get(task)
            if slot is None:
                continue
            value = np.log2(slot["r"] / slot["py"])
            values.append(value)
            bar = ax.barh(y[i] + offset, value, height=height, color=color)[0]
            if slot["failed"]:
                bar.set_hatch("///")
                bar.set_edgecolor("white")
                bar.set_linewidth(0.6)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[task] for task in order], fontsize=9)
    ax.margins(y=0.01)
    ax.invert_yaxis()
    ticks = np.arange(int(np.floor(min(values))), int(np.ceil(max(values))) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_ratio_tick(tick) for tick in ticks], fontsize=8)
    ax.set_xlabel("log2(R の壁時計 / Python の壁時計)")
    ax.legend(
        handles=[
            Patch(facecolor=PYTHON, label="大きい側（約 10,000 行）"),
            Patch(facecolor=R_COLOR, label="1,000 行側"),
            Patch(facecolor="0.6", edgecolor="white", hatch="///", label="tol外"),
        ],
        loc="lower right",
    )
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    large = _tasks(rows, LARGE)
    small = _tasks(rows, SMALL)
    _ratio(large, small, OUT / "ratio.png")
    print(len(large), "large tasks,", len(small), "small tasks")


if __name__ == "__main__":
    main()
