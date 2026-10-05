from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1, KM, Cox, PH diagnostics, Weibull AFT for Rotterdam breast cancer deaths."""


import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from support import load_parquet_dir
from statract.reporting import write_csv_companion
from statract import (
    accelerated_failure,
    agg_category,
    agg_mean_sd,
    cox_ph,
    log_rank,
    plot_forest,
    plot_survival,
    proportional_hazards_test,
    survival_curve,
    write_cox_diagnostic_suite,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

COX_FORMULA = "Surv(dtime, event) ~ hormon + age + nodes + size + grade"
AFT_FORMULA = COX_FORMULA


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


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    target: pl.DataFrame = data["target"]

    # Full sample: no rows dropped (flowchart omitted).
    cohort = target
    if cohort.filter(pl.col("dtime").is_null() | pl.col("event").is_null() | pl.col("hormon").is_null()).height:
        raise RuntimeError("unexpected missing dtime/event/hormon; restore a flowchart")
    (out / "n.md").write_text(
        f"Full sample n = {cohort.height} (no exclusion; flowchart omitted).\n",
        encoding="utf-8",
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Tumor size": ("size", agg_category),
        "Positive nodes": ("nodes", agg_mean_sd),
        "Grade": ("grade", agg_category),
        "Hormone therapy": ("hormon_label", agg_category),
        "Chemotherapy": ("chemo", agg_category),
        "Post-menopausal": ("meno", agg_category),
    }
    params = {k: v for k, v in params.items() if v[0] in cohort.columns}
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort,
        params=params,
        hue="hormon_label",
        add_all=True,
        add_pvalue=True,
    )

    km = survival_curve(cohort, "dtime", "event", by="hormon_label")
    _write_csv(out, "km_curve", km.frame())
    _write_csv(out, "km_at", km.at([365.0, 1825.0, 3650.0]))
    na = survival_curve(cohort, "dtime", "event", by="hormon_label", kind="nelson_aalen")
    _write_csv(out, "nelson_aalen_at", na.at([365.0, 1825.0, 3650.0]))

    lr = log_rank(cohort, "dtime", "event", by="hormon_label")
    (out / "logrank.md").write_text(
        f"log-rank statistic = {lr.statistic:.3f}, df = {lr.df}, p = {lr.p_value:.4g}\n",
        encoding="utf-8",
    )
    _write_csv(out, "logrank_counts", lr.frame())

    ax = plot_survival(cohort, time="dtime", status="event", hue="hormon_label")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Overall survival")
    ax.figure.savefig(out / "figures" / "km_hormon.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    cox_df = cohort.filter(
        pl.col("dtime").is_not_null()
        & pl.col("event").is_not_null()
        & pl.col("hormon").is_not_null()
        & pl.col("age").is_not_null()
        & pl.col("nodes").is_not_null()
        & pl.col("size").is_not_null()
        & pl.col("grade").is_not_null()
    )
    fit = cox_ph(cox_df, COX_FORMULA)
    cox_tidy = fit.tidy(exponentiate=True)
    _write_csv(out, "cox_tidy", cox_tidy)
    plot_forest(
        cox_tidy,
        out / "figures" / "cox_forest.png",
        title="Cox PH (HR)",
        xlabel="Hazard ratio",
        layout="table",
    )
    zph = proportional_hazards_test(fit)
    _write_csv(out, "ph_test", zph)
    global_row = zph.filter(pl.col("term") == "global")
    gp = float(global_row["p_value"][0]) if global_row.height else float("nan")
    (out / "ph_test.md").write_text(
        f"Cox complete-case n = {cox_df.height}. `proportional_hazards_test` global p = {gp:.4g}. "
        f"Formula `{COX_FORMULA}`.\n",
        encoding="utf-8",
    )
    write_cox_diagnostic_suite(
        fit,
        out / "figures",
        data=cohort,
        time="dtime",
        status="event",
        by="hormon_label",
        stem="cox",
    )

    aft = accelerated_failure(cox_df, AFT_FORMULA, distribution="weibull")
    _write_csv(out, "aft_weibull_tidy", aft.tidy(exponentiate=True))
    aft_ln = accelerated_failure(cox_df, AFT_FORMULA, distribution="lognormal")
    _write_csv(out, "aft_lognormal_tidy", aft_ln.tidy(exponentiate=True))
    (out / "aft_n.md").write_text(
        f"AFT n = {cox_df.height}. Weibull converged = {aft.converged}, scale = {aft.scale:.4g}. "
        f"Lognormal (documented family) converged = {aft_ln.converged}, scale = {aft_ln.scale:.4g}. "
        "Exponentiated coefficients are time ratios (`survreg` parameterization).\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
