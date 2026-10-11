"""Chapter 16 of the theory text: metrics for diagnosis.

The "test" is the chapter 9 rule: the seven-predictor model fitted on the 3000
patients, applied to 50,000 new simulated patients, positive when the
predicted risk is 5% or more. Everything else (another hospital, a small
study, a two-gate design, partial verification) is built from the same new
patients.

Run from the repo root::

    uv run python -c "from pathlib import Path; import sys; \
sys.path.insert(0, 'examples/theory_bleeding'); \
from diagnostic import diagnostic_metrics; \
diagnostic_metrics(Path('examples/theory_bleeding/theory_bleeding_out'))"
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
from theory_bleeding import (
    BLUE,
    CACHE,
    GRAY,
    PRED_FORMULA,
    RED,
    _expit,
    _test_set,
    _write_csv,
    bleed_logit,
    load_data,
)

from statract import (
    binary_perf,
    fit_glm,
    mcnemar_test,
    proportion_ci,
    roc_curve,
    roc_test,
    threshold_tradeoff,
)

THRESHOLD = 0.05
Z = float(stats.norm.ppf(0.975))


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


def diagnostic_metrics(out: Path) -> None:
    """Write ch16_*.csv and figures/ch16_*.png into ``out``."""
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load_data(CACHE)
    test = _test_set()
    y = test["bleed"].to_numpy()
    full = fit_glm(df, PRED_FORMULA, family="binomial")
    simple = fit_glm(df, "bleed ~ age + antithrombotic", family="binomial")
    p = np.asarray(full.predict(test, kind="response"))
    p_simple = np.asarray(simple.predict(test, kind="response"))
    pos = p >= THRESHOLD

    # 16.1-16.2: the 2x2 table on 50,000 patients and on a study of 1000.
    rows = [
        _metrics(y, pos, "50,000 new patients"),
        _metrics(y[:1000], pos[:1000], "Study of 1000 patients"),
    ]

    # Another hospital: same patients and effects, baseline odds x exp(0.8)
    # (the same draw as chapter 9's calibration example).
    rng = np.random.default_rng(9)
    lp_true = bleed_logit(
        *[
            test[c].to_numpy()
            for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
        ]
    )
    y_other = rng.binomial(1, _expit(lp_true + 0.8))
    rows.append(_metrics(y_other, pos, "Hospital with higher risk"))
    table = pl.DataFrame(rows)
    base = rows[0]
    ppv_bayes, npv_bayes = _ppv_npv(
        base["sensitivity"], base["specificity"], np.array([rows[2]["prevalence"]])
    )
    table = table.with_columns(
        pl.Series(
            "ppv_bayes",
            [base["ppv"], base["ppv"], float(ppv_bayes[0])],
        ),
        pl.Series(
            "npv_bayes",
            [base["npv"], base["npv"], float(npv_bayes[0])],
        ),
    )
    _write_csv(out, "ch16_table", table)

    # Likelihood ratios for bands of predicted risk.
    bands = [
        (0.0, 0.02, "< 2%"),
        (0.02, 0.05, "2-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 1.01, ">= 10%"),
    ]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    band_rows = []
    for lo, hi, label in bands:
        m = (p >= lo) & (p < hi)
        a, b = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        lr, lr_lo, lr_hi = _lr_ci(a, n1, b, n0)
        band_rows.append(
            {
                "band": label,
                "lower": lo,
                "upper": min(hi, 1.0),
                "n": int(m.sum()),
                "bleed": a,
                "no_bleed": b,
                "share_of_bleed": a / n1,
                "share_of_no_bleed": b / n0,
                "lr": lr,
                "lr_lo": lr_lo,
                "lr_hi": lr_hi,
                "observed_rate": a / (a + b),
            }
        )
    _write_csv(out, "ch16_bands", pl.DataFrame(band_rows))

    # 16.3: ROC curve, thresholds and the Youden index.
    roc = roc_curve(y, p)
    auc_lo, auc, auc_hi = roc.ci_auc()
    best = roc.coords("best", ret=["threshold", "sensitivity", "specificity"])
    trade = threshold_tradeoff(y, p, thresholds=np.array([0.02, 0.05, 0.10, 0.20]))
    points = pl.concat(
        [
            trade.select(
                pl.lit("fixed").alias("kind"),
                "threshold",
                "sensitivity",
                "specificity",
            ),
            best.select(
                pl.lit("youden").alias("kind"),
                "threshold",
                "sensitivity",
                "specificity",
            ),
        ]
    ).with_columns(
        (pl.col("sensitivity") + pl.col("specificity") - 1).alias("youden_j"),
        pl.lit(auc).alias("auc"),
        pl.lit(auc_lo).alias("auc_lo"),
        pl.lit(auc_hi).alias("auc_hi"),
    )
    _write_csv(out, "ch16_roc_points", points)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", lw=1)
    ax.plot(
        1 - roc.specificities,
        roc.sensitivities,
        color=BLUE,
        label=f"Seven predictors (AUC {auc:.3f})",
    )
    fixed = points.filter(pl.col("kind") == "fixed")
    ax.scatter(
        1 - fixed["specificity"], fixed["sensitivity"], color=BLUE, zorder=3, s=22
    )
    for t, se, sp in fixed.select(
        "threshold", "sensitivity", "specificity"
    ).iter_rows():
        offset = (8, 2) if t >= 0.2 else (6, -12)
        ax.annotate(
            f"{t:.0%}",
            (1 - sp, se),
            textcoords="offset points",
            xytext=offset,
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
        label=f"Largest Youden index ({yb['threshold']:.1%})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curve and thresholds")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_roc.png", dpi=180)
    plt.close(fig)

    # PPV and NPV against prevalence, with sensitivity and specificity fixed.
    prev = np.linspace(0.001, 0.5, 500)
    ppv, npv = _ppv_npv(base["sensitivity"], base["specificity"], prev)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(prev, ppv, color=RED, label="Positive predictive value")
    ax.plot(prev, npv, color=BLUE, label="Negative predictive value")
    for pr, ls in [(base["prevalence"], "--"), (rows[2]["prevalence"], ":")]:
        ax.axvline(pr, color=GRAY, linestyle=ls, lw=1)
    ax.annotate(
        f"{base['prevalence']:.1%}",
        (base["prevalence"], 0.5),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=9,
        color=GRAY,
    )
    ax.annotate(
        f"{rows[2]['prevalence']:.1%}",
        (rows[2]["prevalence"], 0.4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=9,
        color=GRAY,
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Prevalence (proportion who bleed)")
    ax.set_ylabel("Predictive value")
    ax.set_title(
        f"Same test (sensitivity {base['sensitivity']:.0%}, specificity {base['specificity']:.0%})"
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_prevalence.png", dpi=180)
    plt.close(fig)

    # 16.5: two tests on the same patients.
    pos2 = p_simple >= THRESHOLD
    cmp_rows = []
    for group, label in [(1, "sensitivity (bleed)"), (0, "specificity (no bleed)")]:
        m = y == group
        a, b = pos[m], pos2[m]
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
                "seven_predictors": float(a.mean()),
                "age_antithrombotic": float(b.mean()),
                "both_correct": int(tab[0, 0]),
                "only_seven": int(tab[0, 1]),
                "only_simple": int(tab[1, 0]),
                "both_wrong": int(tab[1, 1]),
                "test": "McNemar (continuity correction)",
                "statistic": res.statistic,
                "p_value": res.p_value,
                "diff_lo": None,
                "diff_hi": None,
            }
        )
    roc2 = roc_curve(y, p_simple)
    rt = roc_test(roc, roc2)
    cmp_rows.append(
        {
            "measure": "AUC",
            "n": int(y.size),
            "seven_predictors": rt.auc1,
            "age_antithrombotic": rt.auc2,
            "both_correct": None,
            "only_seven": None,
            "only_simple": None,
            "both_wrong": None,
            "test": "DeLong (paired)",
            "statistic": rt.statistic,
            "p_value": rt.p_value,
            "diff_lo": float(rt.conf_int[0]),
            "diff_hi": float(rt.conf_int[1]),
        }
    )
    _write_csv(out, "ch16_compare", pl.DataFrame(cmp_rows))

    # 16.6: pitfalls. Spectrum, two-gate design, partial verification.
    size = test["size_mm"].to_numpy()
    cases = (y == 1) & (size >= 20)
    controls = (y == 0) & (size < 15)
    two_gate = cases | controls
    verify_rng = np.random.default_rng(16)
    verified = pos | (verify_rng.uniform(size=y.size) < 0.10)
    designs = [
        ("All 50,000 patients", np.ones(y.size, dtype=bool)),
        ("Lesions < 20 mm", size < 20),
        ("Lesions >= 20 mm", size >= 20),
        ("Two-gate: bleeders >= 20 mm vs non-bleeders < 15 mm", two_gate),
        ("Verified only: all positives, 10% of negatives", verified),
    ]
    pit_rows = []
    rocs = {}
    for name, m in designs:
        r = _metrics(y[m], pos[m], name)
        rc = roc_curve(y[m], p[m])
        rocs[name] = rc
        r["auc"] = rc.auc
        pit_rows.append(r)
    _write_csv(out, "ch16_pitfalls", pl.DataFrame(pit_rows))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", lw=1)
    styles = [
        (designs[0][0], "All patients", BLUE, "-"),
        (designs[3][0], "Two-gate design", RED, "-"),
        (designs[4][0], "Verified patients only", GRAY, "-"),
    ]
    for key, label, color, ls in styles:
        rc = rocs[key]
        ax.plot(
            1 - rc.specificities,
            rc.sensitivities,
            color=color,
            linestyle=ls,
            label=f"{label} (AUC {rc.auc:.2f})",
        )
        r = next(row for row in pit_rows if row["setting"] == key)
        ax.scatter(
            [1 - r["specificity"]], [r["sensitivity"]], color=color, zorder=3, s=26
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Same test, different study designs\n(dots: threshold 5%)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch16_designs.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    diagnostic_metrics(_HERE / "theory_bleeding_out")
