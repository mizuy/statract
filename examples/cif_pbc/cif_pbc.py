from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1, naive KM, Aalen–Johansen CIF, and Fine–Gray for randomized PBC."""


import io
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from support import flowchart, load_parquet_dir
from statract.reporting import mermaid_flowchart
from statract.reporting import markdown_flowchart, write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    cox_ph,
    cumulative_incidence,
    fine_gray,
    fine_gray_regression,
    plot_forest,
    survival_curve,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

TIMES_Y = [365.0, 730.0, 1825.0, 3650.0]
FG_FORMULA = "Surv(time, status) ~ dp + age + sex + bili + albumin + edema + stage"
FG_COX = "Surv(fgstart, fgstop, fgstatus) ~ dp + age + sex + bili + albumin + edema + stage"


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


def _plot_cif(frame: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    death = frame.filter(pl.col("state").cast(pl.Utf8) == "2")
    groups = death["group"].unique(maintain_order=True).to_list() if "group" in death.columns else [None]
    for group in groups:
        part = death if group is None else death.filter(pl.col("group") == group)
        part = part.sort("time")
        ax.step(
            part["time"].to_list(),
            part["estimate"].to_list(),
            where="post",
            label=str(group) if group is not None else "CIF death",
        )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative incidence of liver death")
    ax.set_title("Aalen–Johansen CIF (cause = death; transplant competing)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cuminc(ci, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    frame = ci.frame()
    tests = {row["cause"]: row["p_value"] for row in ci.tests.iter_rows(named=True)}
    names = {"1": "transplant", "2": "death"}
    colors = {"D-penicillamine": "#1f77b4", "placebo": "#d62728"}
    styles = {"2": "-", "1": "--"}
    for (group, cause), part in frame.group_by(["group", "cause"], maintain_order=True):
        part = part.sort("time")
        ax.step(
            part["time"].to_list(),
            part["estimate"].to_list(),
            where="post",
            color=colors.get(group),
            linestyle=styles.get(cause, ":"),
            label=f"{group}, {names.get(cause, cause)} (Gray p={tests[cause]:.2f})",
        )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("cuminc: death and transplant by treatment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    target: pl.DataFrame = data["target"]

    buf = io.StringIO()
    cohort = flowchart(target, 
        {
            "Randomized (trt not missing)": pl.col("trt").is_not_null() & pl.col("trt_label").is_not_null(),
            "Time and status present": pl.col("time").is_not_null() & pl.col("status").is_not_null(),
        },
        out=buf,
    )
    flow_text = buf.getvalue()
    (out / "text_flowchart.md").write_text(markdown_flowchart(flow_text), encoding="utf-8")
    (out / "mermaid_flowchart.md").write_text(
        mermaid_flowchart(flow_text, final_label="Analysis cohort"),
        encoding="utf-8",
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Bilirubin (mg/dL)": ("bili", agg_median_iqr),
        "Albumin (g/dL)": ("albumin", agg_median_iqr),
        "Edema score": ("edema", agg_category),
        "Histologic stage": ("stage", agg_category),
    }
    params = {k: v for k, v in params.items() if v[0] in cohort.columns}
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort,
        params=params,
        hue="trt_label",
        add_all=True,
        add_pvalue=True,
    )

    km = survival_curve(cohort, "time", "death_naive", by="trt_label")
    _write_csv(out, "km_naive_curve", km.frame())
    _write_csv(out, "km_naive_at", km.at(TIMES_Y))

    aj = survival_curve(cohort, "time", "status", by="trt_label", kind="aalen_johansen")
    cif = aj.frame()
    _write_csv(out, "cif_curve", cif)
    death_cif = cif.filter(pl.col("state").cast(pl.Utf8) == "2")
    aj_death = type(aj)(table=death_cif, kind=aj.kind, confidence=aj.confidence, level=aj.level)
    _write_csv(out, "cif_death_at", aj_death.at(TIMES_Y))
    _plot_cif(cif, out / "figures" / "cif_death.png")

    ci = cumulative_incidence(cohort, "time", "status", by="trt_label")
    _write_csv(out, "gray_test", ci.tests)
    _write_csv(out, "cuminc_at", ci.at(TIMES_Y))
    _plot_cuminc(ci, out / "figures" / "cuminc.png")

    fg_df = cohort.filter(
        pl.col("time").is_not_null()
        & pl.col("status").is_not_null()
        & pl.col("dp").is_not_null()
        & pl.col("age").is_not_null()
        & pl.col("sex").is_not_null()
        & pl.col("bili").is_not_null()
        & pl.col("albumin").is_not_null()
        & pl.col("edema").is_not_null()
        & pl.col("stage").is_not_null()
    )
    expanded = fine_gray(fg_df, FG_FORMULA, cause=2)
    fg_fit = cox_ph(expanded, FG_COX, weights="fgwt")
    fg_tidy = fg_fit.tidy(exponentiate=True)
    _write_csv(out, "finegray_tidy", fg_tidy)
    plot_forest(
        fg_tidy,
        out / "figures" / "finegray_forest.png",
        title="Fine–Gray (subdistribution HR)",
        xlabel="Hazard ratio",
        layout="table",
    )
    crr = fine_gray_regression(fg_df, FG_FORMULA, cause=2)
    crr_tidy = crr.tidy(exponentiate=True)
    _write_csv(out, "crr_tidy", crr_tidy)
    _write_csv(out, "crr_glance", crr.glance())
    model_se = np.sqrt(np.diag(fg_fit.information_inverse()))
    compare = pl.DataFrame(
        {
            "term": crr_tidy["term"],
            "crr_estimate": crr_tidy["estimate"],
            "finegray_estimate": fg_tidy["estimate"],
            "crr_shr": crr_tidy["exp_estimate"],
            "finegray_shr": fg_tidy["exp_estimate"],
            "crr_se": crr_tidy["std_error"],
            "finegray_robust_se": fg_tidy["std_error"],
            "finegray_model_se": model_se,
        }
    )
    _write_csv(out, "crr_vs_finegray", compare)
    (out / "finegray_n.md").write_text(
        f"Fine–Gray complete-case n = {fg_df.height} (flowchart n = {cohort.height}); "
        f"expanded rows = {expanded.height}; cause of interest = death (status=2); "
        f"competing = transplant (status=1). Formula `{FG_FORMULA}`.\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
