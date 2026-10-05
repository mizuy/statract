from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1, KM/log-rank, clustered Cox (Lin–Wei sandwich) for retinopathy."""


import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from support import load_parquet_dir
from statract.reporting import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    cox_ph,
    log_rank,
    plot_forest,
    plot_survival,
    proportional_hazards_test,
    survival_curve,
    tableone,
    tableone_raw,
    write_cox_diagnostic_suite,
)
from config import ANALYSIS_OUT, CACHE
from project import project

COX_FORMULA = "Surv(futime, event) ~ trt_label + type + risk + cluster(id)"
COX_FORMULA_NAIVE = "Surv(futime, event) ~ trt_label + type + risk"
TABLEONE_PARAMS = {
    "Age at diabetes onset (years)": ("age", agg_mean_sd),
    "Diabetes type": ("type", agg_category),
    "Laser type": ("laser", agg_category),
    "Treated eye (laterality)": ("eye", agg_category),
    "Baseline risk score": ("risk", agg_mean_sd),
}


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


def _se_compare(naive: pl.DataFrame, clustered: pl.DataFrame) -> pl.DataFrame:
    n = naive.select(["term", "estimate", "std_error"]).rename({"std_error": "se_model"})
    c = clustered.select(["term", "std_error"]).rename({"std_error": "se_sandwich"})
    return n.join(c, on="term").with_columns(
        (pl.col("se_sandwich") / pl.col("se_model")).alias("se_ratio")
    )


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    cohort: pl.DataFrame = data["eyes"]
    n_id = cohort["id"].n_unique()
    (out / "n.md").write_text(
        f"Eye-level n = {cohort.height} from {n_id} patients "
        f"(exactly two eyes per id; no rows dropped).\n",
        encoding="utf-8",
    )

    tableone(
        cohort,
        TABLEONE_PARAMS,
        hue="trt_label",
        add_all=True,
        add_pvalue=False,
    )
    _write_csv(
        out,
        "table1",
        tableone_raw(
            cohort,
            TABLEONE_PARAMS,
            hue="trt_label",
            add_all=True,
            add_pvalue=False,
        ),
    )

    km = survival_curve(cohort, "futime", "event", by="trt_label")
    _write_csv(out, "km_curve", km.frame())
    _write_csv(out, "km_at", km.at([12.0, 24.0, 48.0]))

    lr = log_rank(cohort, "futime", "event", by="trt_label")
    (out / "logrank.md").write_text(
        f"log-rank statistic = {lr.statistic:.3f}, df = {lr.df}, p = {lr.p_value:.4g}\n",
        encoding="utf-8",
    )
    _write_csv(out, "logrank_counts", lr.frame())

    ax = plot_survival(cohort, time="futime", status="event", hue="trt_label")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Vision-loss-free survival")
    ax.figure.savefig(out / "figures" / "km_trt.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    naive = cox_ph(cohort, COX_FORMULA_NAIVE)
    fit = cox_ph(cohort, COX_FORMULA)
    labeled = cox_ph(
        cohort,
        "futime",
        "event",
        ["trt_label", "type", "risk"],
        cluster="id",
    )
    tidy_naive = naive.tidy(exponentiate=True)
    tidy = fit.tidy(exponentiate=True)
    _write_csv(out, "cox_tidy_naive", tidy_naive)
    _write_csv(out, "cox_tidy", tidy)
    _write_csv(out, "cox_se_compare", _se_compare(tidy_naive, tidy))
    se_match = bool(np.allclose(fit.covariance, labeled.covariance))
    (out / "cox_n.md").write_text(
        f"Cox n = {fit.n_obs} eyes / {n_id} patients. "
        f"Formula `{COX_FORMULA}` (Lin–Wei sandwich). "
        f"`cluster=` column API matches formula `cluster(id)` "
        f"(covariance allclose: {se_match}).\n",
        encoding="utf-8",
    )
    plot_forest(
        tidy,
        out / "figures" / "cox_forest.png",
        title="Cox PH, cluster-robust (HR)",
        xlabel="Hazard ratio",
        layout="table",
    )
    zph = proportional_hazards_test(fit)
    _write_csv(out, "ph_test", zph)
    global_row = zph.filter(pl.col("term") == "global")
    gp = float(global_row["p_value"][0]) if global_row.height else float("nan")
    (out / "ph_test.md").write_text(
        f"`proportional_hazards_test` global p = {gp:.4g}. Formula `{COX_FORMULA}`.\n",
        encoding="utf-8",
    )
    write_cox_diagnostic_suite(
        fit,
        out / "figures",
        data=cohort,
        time="futime",
        status="event",
        by="trt_label",
        stem="cox",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
