from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Two-arm RCT with basic tests: licorice gargle and postoperative sore throat."""


import io
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from support import ProjectPath, flowchart, load_parquet_dir
from statract.report.artifacts import mermaid_flowchart, write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    binom_test,
    mcnemar_test,
    p_adjust,
    plot_forest,
    prop_test,
    t_test,
    wilcox_test,
    write_tableone_artifacts,
)
from build import TIMES

project = ProjectPath(__file__)
CACHE = project.cache
ANALYSIS_OUT = project.out

ARMS = ["Sugar", "Licorice"]
# Same two hues as the KM figures (JCO palette in statract.viz.km); markers differ too.
ARM_STYLE = {"Sugar": ("#868686", "s"), "Licorice": ("#0073C2", "o")}


def _clear_out(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    return out


def _write_csv(out: Path, stem: str, frame: pl.DataFrame) -> None:
    path = out / f"{stem}.csv"
    frame.write_csv(path)
    write_csv_companion(path, csv_link_prefix=out.name)


def _arm(cohort: pl.DataFrame, arm: str, col: str) -> pl.Series:
    return cohort.filter(pl.col("arm") == arm)[col]


def incidence_table(cohort: pl.DataFrame) -> pl.DataFrame:
    """Sore throat incidence per arm and time, with the exact (Clopper–Pearson) interval."""
    rows = []
    for t, label in TIMES.items():
        for arm in ARMS:
            sore = _arm(cohort, arm, f"sore_{t}")
            res = binom_test(int(sore.sum()), sore.len())
            rows.append(
                {
                    "time": label,
                    "arm": arm,
                    "events": int(sore.sum()),
                    "n": sore.len(),
                    "proportion": res.estimate,
                    "conf_low": res.conf_int[0],
                    "conf_high": res.conf_int[1],
                }
            )
    return pl.DataFrame(rows)


def sore_throat_tests(cohort: pl.DataFrame) -> pl.DataFrame:
    """Risk difference (licorice minus sugar) by prop_test, with Holm-adjusted p values."""
    rows = []
    for t, label in TIMES.items():
        lic = _arm(cohort, "Licorice", f"sore_{t}")
        sug = _arm(cohort, "Sugar", f"sore_{t}")
        res = prop_test([int(lic.sum()), int(sug.sum())], [lic.len(), sug.len()])
        rows.append(
            {
                "term": label,
                "estimate": res.estimate,
                "conf_low": res.conf_int[0],
                "conf_high": res.conf_int[1],
                "statistic": res.statistic,
                "p_value": res.p_value,
            }
        )
    frame = pl.DataFrame(rows)
    return frame.with_columns(pl.Series("p_holm", p_adjust(frame["p_value"], "holm")))


def pain_tests(cohort: pl.DataFrame) -> pl.DataFrame:
    """Pain score 0-10: Wilcoxon rank sum (primary) and the Welch mean difference.

    Most scores are 0, so the Hodges–Lehmann shift collapses to 0 (R gives the
    same). The mean difference from Welch's t test is the effect size shown.
    """
    rows = []
    for t, label in TIMES.items():
        lic = _arm(cohort, "Licorice", f"pain_{t}")
        sug = _arm(cohort, "Sugar", f"pain_{t}")
        w = wilcox_test(lic, sug, exact=False)
        tt = t_test(lic, sug)
        rows.append(
            {
                "term": label,
                "mean_licorice": float(lic.mean()),
                "mean_sugar": float(sug.mean()),
                "share_zero": float(((lic == 0).sum() + (sug == 0).sum()) / (lic.len() + sug.len())),
                "W": w.statistic,
                "p_value": w.p_value,
                "estimate": tt.estimate,
                "conf_low": tt.conf_int[0],
                "conf_high": tt.conf_int[1],
                "t_p_value": tt.p_value,
            }
        )
    frame = pl.DataFrame(rows)
    return frame.with_columns(pl.Series("p_holm", p_adjust(frame["p_value"], "holm")))


def paired_tests(cohort: pl.DataFrame) -> pl.DataFrame:
    """Within each arm, PACU 30 min against POD1 morning: McNemar and paired Wilcoxon."""
    rows = []
    for arm in ARMS:
        sub = cohort.filter(pl.col("arm") == arm)
        table = (
            pl.DataFrame({"early": sub["sore_pacu30min"], "late": sub["sore_pod1am"]})
            .group_by("early", "late")
            .len()
        )
        count = {(r["early"], r["late"]): r["len"] for r in table.iter_rows(named=True)}
        mc = mcnemar_test(sub["sore_pacu30min"], sub["sore_pod1am"])
        w = wilcox_test(sub["pain_pacu30min"], sub["pain_pod1am"], paired=True, exact=False)
        rows.append(
            {
                "arm": arm,
                "sore_both": count.get((True, True), 0),
                "sore_30min_only": count.get((True, False), 0),
                "sore_pod1_only": count.get((False, True), 0),
                "sore_neither": count.get((False, False), 0),
                "mcnemar_chisq": mc.statistic,
                "mcnemar_p": mc.p_value,
                "wilcoxon_V": w.statistic,
                "wilcoxon_p": w.p_value,
            }
        )
    return pl.DataFrame(rows)


def plot_incidence(inc: pl.DataFrame, path: Path) -> None:
    labels = list(TIMES.values())
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=180)
    for k, arm in enumerate(ARMS):
        color, marker = ARM_STYLE[arm]
        sub = inc.filter(pl.col("arm") == arm)
        x = [i + (k - 0.5) * 0.18 for i in range(len(labels))]
        y = (sub["proportion"] * 100).to_list()
        lo = [a - b for a, b in zip(y, (sub["conf_low"] * 100).to_list())]
        hi = [b - a for a, b in zip(y, (sub["conf_high"] * 100).to_list())]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=marker, color=color, ms=6, lw=1.5, capsize=0, label=arm)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Sore throat (%)")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.set_title("Sore throat incidence with 95% Clopper–Pearson CI", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    target: pl.DataFrame = data["target"]

    buf = io.StringIO()
    cohort = flowchart(
        target,
        {"Throat pain scores at all time points": pl.all_horizontal([pl.col(f"pain_{t}").is_not_null() for t in TIMES])},
        out=buf,
    )
    flow_text = buf.getvalue()
    (out / "text_flowchart.md").write_text("```text\n" + flow_text.rstrip() + "\n```\n", encoding="utf-8")
    (out / "mermaid_flowchart.md").write_text(
        mermaid_flowchart(flow_text, final_label="Analysis cohort"), encoding="utf-8"
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "BMI": ("bmi", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "ASA": ("asa", agg_category),
        "Mallampati": ("mallampati", agg_category),
        "Smoking": ("smoking", agg_category),
        "Surgery size": ("surgery_size", agg_category),
    }
    write_tableone_artifacts(out, "table1", df=cohort, params=params, hue="arm", add_all=True, add_smd=True)

    inc = incidence_table(cohort)
    _write_csv(out, "incidence", inc)
    plot_incidence(inc, out / "figures" / "incidence.png")

    rd = sore_throat_tests(cohort)
    _write_csv(out, "sore_throat_tests", rd)
    plot_forest(
        rd.with_columns((pl.col("estimate", "conf_low", "conf_high") * 100), pl.col("p_holm").alias("p_value")),
        out / "figures" / "risk_difference_forest.png",
        null_value=0.0,
        log_scale=False,
        title="Risk difference, licorice − sugar (Holm-adjusted P)",
        xlabel="Risk difference (percentage points)",
        estimate_digits=1,
        layout="table",
    )

    pain = pain_tests(cohort)
    _write_csv(out, "pain_tests", pain)

    paired = paired_tests(cohort)
    _write_csv(out, "paired_tests", paired)

    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
