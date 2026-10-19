"""Chapter 17 of the theory text: metrics for diagnosis.

Part V uses its own examples instead of the bleeding data of chapters 1-15.

A fecal immunochemical test (FIT, hemoglobin in ng/mL) for colorectal cancer
in 50,000 screening participants, all of whom have a colonoscopy as the
reference standard. A guaiac test (gFOBT) is done on the same people for
comparison. The data are simulated, so the truth is known. Asset names keep the old chapter
number (``ch16_*``).

Run from the repo root::

    uv run python examples/theory_bleeding/diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl
from scipy import stats

# theory_bleeding runs its pipeline only under ``__main__``; importing it is safe.
from theory_bleeding import BLUE, GRAY, RED, _expit, _write_csv

from statract import (
    binary_perf,
    mcnemar_test,
    proportion_ci,
    roc_curve,
    roc_test,
    threshold_tradeoff,
)

CUTOFF = 100  # ng/mL, positive when f-Hb >= CUTOFF
Z = float(stats.norm.ppf(0.975))
GREEN = "#3B8C4A"


# ---------------------------------------------------------------- simulation


def simulate_fit(n: int, seed: int, cancer_shift: float = 0.0) -> pl.DataFrame:
    """Screening participants with FIT, gFOBT and the true colonoscopy finding.

    ``cancer_shift`` raises the log odds of cancer (a symptomatic clinic).
    Cancer stage and side change how much the tumour bleeds; right-sided cancers
    bleed less and are also missed more often by colonoscopy.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(50, 80, n)
    male = rng.binomial(1, 0.5, n)
    cancer = rng.uniform(size=n) < _expit(
        -5.5 + cancer_shift + 0.06 * (age - 65) + 0.3 * male
    )
    adenoma = ~cancer & (
        rng.uniform(size=n) < _expit(-2.8 + 0.04 * (age - 65) + 0.4 * male)
    )
    stage = np.where(cancer, rng.choice([1, 2, 3], n, p=[0.5, 0.25, 0.25]), 0)
    right = cancer & (rng.uniform(size=n) < 0.35)
    # Log hemoglobin in the stool: none < advanced adenoma < cancer.
    mu = np.where(
        cancer,
        5.3 + 0.8 * (stage >= 2) + 0.4 * (stage == 3) - 0.8 * right,
        np.where(adenoma, 3.0, 2.0),
    )
    sd = np.where(cancer, 1.4, 1.5)
    latent = mu + sd * rng.normal(size=n)
    fhb = np.minimum(np.round(np.exp(latent)), 1000.0)  # assay upper limit
    # gFOBT reacts only to more bleeding, plus dietary false positives.
    gfobt = (latent + rng.normal(0, 0.8, n) > 6.0) | (rng.uniform(size=n) < 0.02)
    # Colonoscopy misses 2% of left-sided and 10% of right-sided cancers.
    miss = cancer & (rng.uniform(size=n) < np.where(right, 0.10, 0.02))
    return pl.DataFrame(
        {
            "age": age,
            "male": male,
            "cancer": cancer.astype(int),
            "adenoma": adenoma.astype(int),
            "stage": stage,
            "right": right.astype(int),
            "fhb": fhb,
            "gfobt": gfobt.astype(int),
            "colonoscopy": (cancer & ~miss).astype(int),
        }
    )


# ---------------------------------------------------------------- helpers


def _wilson(hits: int, total: int) -> tuple[float, float]:
    """Wilson score interval, via statract's proportion_ci."""
    d = pl.DataFrame({"x": [1] * hits + [0] * (total - hits)})
    ci = d.select(proportion_ci(pl.col("x"), method="wilson").alias("ci"))["ci"][0]
    return float(ci["lo"]), float(ci["hi"])


def _lr_ci(a: int, n1: int, b: int, n0: int) -> tuple[float, float, float]:
    """Ratio of two proportions a/n1 over b/n0 with the log method (Simel 1991)."""
    lr = (a / n1) / (b / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
    return lr, lr * np.exp(-Z * se), lr * np.exp(Z * se)


def _metrics(y: np.ndarray, positive: np.ndarray, setting: str) -> dict:
    """2x2 table, sensitivity, specificity, predictive values and likelihood ratios."""
    b = binary_perf(y, positive.astype(int))
    tp, fp, tn, fn = b["tp"], b["fp"], b["tn"], b["fn"]
    se_lo, se_hi = _wilson(tp, tp + fn)
    sp_lo, sp_hi = _wilson(tn, tn + fp)
    lrp, lrp_lo, lrp_hi = _lr_ci(tp, tp + fn, fp, fp + tn)
    lrn, lrn_lo, lrn_hi = _lr_ci(fn, tp + fn, tn, fp + tn)
    return {
        "setting": setting,
        "n": b["n"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence": (tp + fn) / b["n"],
        "sensitivity": b["sensitivity"],
        "sensitivity_lo": se_lo,
        "sensitivity_hi": se_hi,
        "specificity": b["specificity"],
        "specificity_lo": sp_lo,
        "specificity_hi": sp_hi,
        "ppv": b["ppv"],
        "npv": b["npv"],
        "lr_pos": lrp,
        "lr_pos_lo": lrp_lo,
        "lr_pos_hi": lrp_hi,
        "lr_neg": lrn,
        "lr_neg_lo": lrn_lo,
        "lr_neg_hi": lrn_hi,
    }


def _ppv_npv(
    sens: float, spec: float, prev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    return ppv, npv


def _roc_axes(ax, title: str) -> None:
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- two classes


def diagnostic_metrics(out: Path) -> None:
    """Write ch16_*.csv and figures/ch16_*.png into ``out``."""
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    screen = simulate_fit(50_000, seed=1)
    clinic = simulate_fit(5_000, seed=2, cancer_shift=1.7)
    y = screen["cancer"].to_numpy()
    fhb = screen["fhb"].to_numpy()
    pos = fhb >= CUTOFF

    # 17.1-17.3 and 17.5: the 2x2 table on 50,000 people, on the first 5000,
    # and in a clinic for people with symptoms.
    rows = [
        _metrics(y, pos, "50,000 screening participants"),
        _metrics(y[:5000], pos[:5000], "First 5000 participants"),
        _metrics(
            clinic["cancer"].to_numpy(),
            clinic["fhb"].to_numpy() >= CUTOFF,
            "Symptomatic clinic",
        ),
    ]
    table = pl.DataFrame(rows)
    base = rows[0]
    ppv_bayes, npv_bayes = _ppv_npv(
        base["sensitivity"], base["specificity"], np.array([rows[2]["prevalence"]])
    )
    table = table.with_columns(
        pl.Series("ppv_bayes", [base["ppv"], base["ppv"], float(ppv_bayes[0])]),
        pl.Series("npv_bayes", [base["npv"], base["npv"], float(npv_bayes[0])]),
    )
    _write_csv(out, "ch16_table", table)

    # Likelihood ratios for bands of f-Hb.
    bands = [
        (0, 20, "< 20"),
        (20, 100, "20-99"),
        (100, 400, "100-399"),
        (400, 1e9, ">= 400"),
    ]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    band_rows = []
    for lo, hi, label in bands:
        m = (fhb >= lo) & (fhb < hi)
        a, b = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        lr, lr_lo, lr_hi = _lr_ci(a, n1, b, n0)
        band_rows.append(
            {
                "band": label,
                "lower": float(lo),
                "upper": float(min(hi, 1000)),
                "n": int(m.sum()),
                "cancer": a,
                "no_cancer": b,
                "share_of_cancer": a / n1,
                "share_of_no_cancer": b / n0,
                "lr": lr,
                "lr_lo": lr_lo,
                "lr_hi": lr_hi,
                "observed_rate": a / (a + b),
            }
        )
    _write_csv(out, "ch16_bands", pl.DataFrame(band_rows))

    # 17.4: ROC curve, cut-offs and the Youden index.
    roc = roc_curve(y, fhb)
    auc_lo, auc, auc_hi = roc.ci_auc()
    best = roc.coords("best", ret=["threshold", "sensitivity", "specificity"])
    trade = threshold_tradeoff(
        y, fhb, thresholds=np.array([20.0, 50.0, 100.0, 150.0, 400.0])
    )
    points = pl.concat(
        [
            trade.select(
                pl.lit("fixed").alias("kind"),
                "threshold",
                "sensitivity",
                "specificity",
                "ppv",
            ),
            best.select(
                pl.lit("youden").alias("kind"),
                "threshold",
                "sensitivity",
                "specificity",
            ).with_columns(pl.lit(None, dtype=pl.Float64).alias("ppv")),
        ]
    ).with_columns(
        (pl.col("sensitivity") + pl.col("specificity") - 1).alias("youden_j"),
        pl.lit(auc).alias("auc"),
        pl.lit(auc_lo).alias("auc_lo"),
        pl.lit(auc_hi).alias("auc_hi"),
    )
    # Share of people sent to colonoscopy at each cut-off.
    pos_rate = [float(np.mean(fhb >= t)) for t in points["threshold"].to_list()]
    points = points.with_columns(pl.Series("positive_rate", pos_rate))
    _write_csv(out, "ch16_roc_points", points)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(
        1 - roc.specificities,
        roc.sensitivities,
        color=BLUE,
        label=f"FIT (AUC {auc:.3f})",
    )
    fixed = points.filter(pl.col("kind") == "fixed")
    ax.scatter(
        1 - fixed["specificity"], fixed["sensitivity"], color=BLUE, zorder=3, s=22
    )
    for t, se, sp in fixed.select(
        "threshold", "sensitivity", "specificity"
    ).iter_rows():
        ax.annotate(
            f"{t:.0f}",
            (1 - sp, se),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=9,
        )
    yb = points.filter(pl.col("kind") == "youden").row(0, named=True)
    x0 = 1 - yb["specificity"]
    ax.plot([x0, x0], [x0, yb["sensitivity"]], color=RED, lw=1.5)
    ax.scatter(
        [x0],
        [yb["sensitivity"]],
        color=RED,
        zorder=4,
        s=30,
        label=f"Largest Youden index ({yb['threshold']:.0f} ng/mL)",
    )
    _roc_axes(ax, "ROC curve of FIT (labels: cut-off in ng/mL)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_roc.png", dpi=180)
    plt.close(fig)

    # PPV and NPV against prevalence, with sensitivity and specificity fixed.
    prev = np.linspace(0.001, 0.2, 500)
    ppv, npv = _ppv_npv(base["sensitivity"], base["specificity"], prev)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(prev, ppv, color=RED, label="Positive predictive value")
    ax.plot(prev, npv, color=BLUE, label="Negative predictive value")
    for pr, ls in [(base["prevalence"], "--"), (rows[2]["prevalence"], ":")]:
        ax.axvline(pr, color=GRAY, linestyle=ls, lw=1)
        ax.annotate(
            f"{pr:.1%}",
            (pr, 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=GRAY,
        )
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Prevalence of colorectal cancer")
    ax.set_ylabel("Predictive value")
    ax.set_title(
        f"Same test (sensitivity {base['sensitivity']:.0%}, specificity {base['specificity']:.1%})"
    )
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_prevalence.png", dpi=180)
    plt.close(fig)

    # 17.6: FIT and gFOBT on the same people.
    g = screen["gfobt"].to_numpy() == 1
    cmp_rows = []
    for group, label in [(1, "sensitivity (cancer)"), (0, "specificity (no cancer)")]:
        m = y == group
        a, b = pos[m], g[m]
        if group == 0:
            a, b = ~a, ~b  # count correct negatives
        tab = np.array(
            [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
        )
        res = mcnemar_test(tab)
        cmp_rows.append(
            {
                "measure": label,
                "n": int(m.sum()),
                "fit": float(a.mean()),
                "gfobt": float(b.mean()),
                "both_correct": int(tab[0, 0]),
                "only_fit": int(tab[0, 1]),
                "only_gfobt": int(tab[1, 0]),
                "both_wrong": int(tab[1, 1]),
                "test": "McNemar (continuity correction)",
                "statistic": res.statistic,
                "p_value": res.p_value,
                "diff_lo": None,
                "diff_hi": None,
            }
        )
    roc_g = roc_curve(y, g.astype(float))
    rt = roc_test(roc, roc_g)
    cmp_rows.append(
        {
            "measure": "AUC",
            "n": int(y.size),
            "fit": rt.auc1,
            "gfobt": rt.auc2,
            "both_correct": None,
            "only_fit": None,
            "only_gfobt": None,
            "both_wrong": None,
            "test": "DeLong (paired)",
            "statistic": rt.statistic,
            "p_value": rt.p_value,
            "diff_lo": float(rt.conf_int[0]),
            "diff_hi": float(rt.conf_int[1]),
        }
    )
    _write_csv(out, "ch16_compare", pl.DataFrame(cmp_rows))

    # 17.7: pitfalls. Spectrum, two-gate design, partial verification, and an
    # imperfect reference standard.
    stage = screen["stage"].to_numpy()
    right = screen["right"].to_numpy() == 1
    adenoma = screen["adenoma"].to_numpy() == 1
    hospital_cases = (y == 1) & (stage >= 2)
    healthy = (y == 0) & ~adenoma
    two_gate = hospital_cases | healthy
    verify_rng = np.random.default_rng(16)
    verified = pos | (verify_rng.uniform(size=y.size) < 0.10)
    no_early = (y == 0) | (stage >= 2)
    no_late = (y == 0) | (stage == 1)
    no_left = (y == 0) | right
    designs = [
        ("All 50,000 participants", np.ones(y.size, dtype=bool), y),
        ("Stage I cancers only", no_late, y),
        ("Stage II-IV cancers only", no_early, y),
        ("Right-sided cancers only", no_left, y),
        ("Two-gate: stage II-IV patients vs healthy volunteers", two_gate, y),
        ("Verified only: all positives, 10% of negatives", verified, y),
        (
            "Reference = colonoscopy finding (misses some cancers)",
            np.ones(y.size, dtype=bool),
            screen["colonoscopy"].to_numpy(),
        ),
    ]
    pit_rows = []
    rocs = {}
    for name, m, ref in designs:
        r = _metrics(ref[m], pos[m], name)
        rc = roc_curve(ref[m], fhb[m])
        rocs[name] = rc
        r["auc"] = rc.auc
        pit_rows.append(r)
    _write_csv(out, "ch16_pitfalls", pl.DataFrame(pit_rows))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    styles = [
        (designs[0][0], "All participants", BLUE),
        (designs[1][0], "Stage I cancers only", GREEN),
        (designs[4][0], "Two-gate design", RED),
    ]
    for key, label, color in styles:
        rc = rocs[key]
        ax.plot(
            1 - rc.specificities,
            rc.sensitivities,
            color=color,
            label=f"{label} (AUC {rc.auc:.2f})",
        )
        r = next(row for row in pit_rows if row["setting"] == key)
        ax.scatter(
            [1 - r["specificity"]], [r["sensitivity"]], color=color, zorder=3, s=26
        )
    _roc_axes(ax, f"Same test, different study designs\n(dots: cut-off {CUTOFF} ng/mL)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_designs.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    diagnostic_metrics(_HERE / "theory_bleeding_out")
