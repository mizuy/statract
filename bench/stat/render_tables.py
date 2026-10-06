"""Print the large- and small-slice tables of docs/models/benchmarks.md from the comparison CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs" / "models" / "benchmarks" / "comparison.csv"

LARGE = ("adult-10000", "bike-10000", "support-large", "star-large", "star-rs-large")
SMALL = ("adult-1000", "bike-1000", "support-1000", "star-1000", "star-rs-1000")

# Task order, label, and API column of the page.
TASKS = [
    ("ols", "OLS", "`fit_ols` / `lm`"),
    ("ols-w", "OLS（重み）", "`fit_ols(weights=)` / `lm`"),
    ("glm-bin", "GLM 二項", "`fit_glm(binomial)` / `glm`"),
    ("glm-gamma", "GLM ガンマ", "`fit_glm(gamma)` / `glm`"),
    ("hc", "sandwich HC0–HC5", "`hc_covariance` / `vcovHC`"),
    ("wald", "Wald", "`wald_test` / `waldtest`"),
    ("lr", "尤度比", "`likelihood_ratio_test` / `lrtest`"),
    ("bp", "Breusch–Pagan", "`breusch_pagan_test` / `bptest`"),
    ("reset", "RESET", "`ramsey_reset_test` / `resettest`"),
    ("m-logit", "最近傍 logit", "`match_sample` / `matchit`"),
    ("m-mah", "最近傍 マハラノビス", '`distance="mahalanobis"`'),
    ("m-exact", "完全一致", '`method="exact"`'),
    ("m-cem", "CEM", '`method="cem"`'),
    ("glm-pois", "GLM ポアソン", "`fit_glm(poisson)` / `glm`"),
    ("nw", "Newey–West", "`newey_west_covariance` / `NeweyWest`"),
    ("dw", "Durbin–Watson", "`durbin_watson_test` / `dwtest`"),
    ("bg", "Breusch–Godfrey", "`breusch_godfrey_test` / `bgtest`"),
    ("gam-8", "GAM k=8", "`gam` / `mgcv::gam`"),
    ("gam-10", "GAM k=10", "同上"),
    ("km", "Kaplan–Meier", "`survival_curve` / `survfit`"),
    ("km-sex", "KM（群）", "`survival_curve(by=)`"),
    ("na", "Nelson–Aalen", '`kind="nelson_aalen"`'),
    ("lrk", "log-rank", "`log_rank` / `survdiff`"),
    ("cox-e", "Cox Efron", "`cox_ph` / `coxph`"),
    ("cox-b", "Cox Breslow", '`ties="breslow"`'),
    ("cox-s", "Cox 層", "`strata=`"),
    ("aft-w", "AFT Weibull", "`accelerated_failure` / `survreg`"),
    ("aft-ln", "AFT lognormal", "同上"),
    ("aft-ex", "AFT exponential", "同上"),
    ("cl1", "クラスタ 1-way", "`cluster_covariance` / `vcovCL`"),
    ("cl2", "クラスタ 2-way", "同上 2 列"),
    ("lmm-ri", "LMM 変量切片 REML", "`fit_mixed` / `lmer`"),
    ("lmm-ml", "LMM ML", '`method="ml"`'),
    ("lmm-rs", "LMM 変量傾き", "`slopes=`"),
]


def _seconds(value: float) -> str:
    if value == 0:
        return "0 s"
    ms = value * 1e3
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 100:
        return f"{ms:.2f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{value:.3f} s"


# Tasks whose tolerance is looser than vs-r.md. The reason is in benchmark-plan.md.
RELAXED = {"cl1", "cl2", "lmm-ri", "lmm-ml", "lmm-rs"}


def _note(text: str) -> str:
    tie = re.fullmatch(r"(\d+) pairs take a control with identical covariates", text)
    if tie:
        return f"{tie.group(1)} 組は共変量が同じ別の対照（タイの破り方）"
    return text


def _verdict(task: str, rows: list[dict[str, str]]) -> str:
    failed = [row for row in rows if row["passed"] != "True"]
    if not failed:
        notes = [_note(row["note"]) for row in rows if row["note"]]
        if task in RELAXED:
            notes.append("緩めた許容差")
        text = f"一致（{len(rows)}）"
        return text + ("。" + "、".join(notes) if notes else "")
    bits = []
    for row in failed:
        if row["quantity"].endswith("pairs"):
            counts = re.match(r"python (\d+) r \d+, (\d+) shared", row["note"])
            shared = f"{counts.group(1)} 組中 {counts.group(2)} 組が同じ" if counts else row["note"]
            bits.append(f"組が一致しない（{shared}）。LAPACK のビルドに依存")
        else:
            bits.append(f"{row['quantity']} {row['criterion']} {float(row['error']):.3e}")
    return f"tol 外（{len(failed)}/{len(rows)}）。" + "、".join(bits)


def _table(rows: list[dict[str, str]], slices: tuple[str, ...], with_api: bool) -> list[str]:
    head = "| 手法 | API | n | Python | R | 比 | vs R |" if with_api else "| 手法 | n | Python | R | 比 | vs R |"
    rule = "|------|-----|--:|------:|--:|---:|------|" if with_api else "|------|--:|------:|--:|---:|------|"
    lines = [head, rule]
    for task, label, api in TASKS:
        picked = [row for row in rows if row["task"] == task and row["slice"] in slices]
        if not picked:
            continue
        first = picked[0]
        py_s = float(first["python_s"])
        r_s = float(first["r_s"])
        ratio = f"{py_s / r_s:.3f}" if r_s else "—"
        verdict = _verdict(task, picked)
        if not r_s:
            verdict += "。R のタイマーが 0"
        cells = [label] + ([api] if with_api else []) + [first["n"], _seconds(py_s), _seconds(r_s), ratio, verdict]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    with CSV_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("\n".join(_table(rows, LARGE, with_api=True)))
    print()
    print("\n".join(_table(rows, SMALL, with_api=False)))


if __name__ == "__main__":
    main()
