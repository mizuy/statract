from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1, KM/log-rank, Cox PH + diagnostics for patient-level colon recurrence."""


import io
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from support import flowchart, load_parquet_dir
from statract.reporting import mermaid_flowchart
from statract.reporting import markdown_flowchart, write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    cox_ph,
    log_rank,
    plot_forest,
    plot_survival,
    proportional_hazards_test,
    standardize_cox,
    survival_curve,
    write_cox_diagnostic_suite,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

COX_FORMULA = "Surv(time, event) ~ rx + age + sex + nodes"
RX_LEVELS = ["Obs", "Lev", "Lev+5FU"]
STD_COLORS = {"Obs": "#0073C2", "Lev": "#EFC000", "Lev+5FU": "#CD534C"}


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
    raw: pl.DataFrame = data["raw"]

    buf = io.StringIO()
    cohort = flowchart(raw, 
        {
            "Recurrence record (etype=1, patient-level)": pl.col("etype") == 1,
            "Time and status present": pl.col("time").is_not_null() & pl.col("status").is_not_null(),
            "Treatment rx present": pl.col("rx").is_not_null(),
        },
        out=buf,
    )
    flow_text = buf.getvalue()
    (out / "text_flowchart.md").write_text(markdown_flowchart(flow_text), encoding="utf-8")
    (out / "mermaid_flowchart.md").write_text(
        mermaid_flowchart(flow_text, final_label="Analysis cohort"),
        encoding="utf-8",
    )

    rx_enum = pl.Enum(["Obs", "Lev", "Lev+5FU"])
    cohort = cohort.with_columns(
        pl.col("rx").cast(rx_enum),
        pl.when(pl.col("sex") == 1).then(pl.lit("male")).otherwise(pl.lit("female")).alias("sex_label"),
        (pl.col("status") == 1).alias("event"),
        pl.col("time").cast(pl.Float64),
        pl.col("age").cast(pl.Float64),
        pl.col("nodes").cast(pl.Float64),
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex_label", agg_category),
        "Positive nodes": ("nodes", agg_mean_sd),
        "Obstruction": ("obstruct", agg_category) if "obstruct" in cohort.columns else ("sex_label", agg_category),
        "Perforation": ("perfor", agg_category) if "perfor" in cohort.columns else ("sex_label", agg_category),
        "Adherence": ("adhere", agg_category) if "adhere" in cohort.columns else ("sex_label", agg_category),
        "Differentiation": ("differ", agg_category) if "differ" in cohort.columns else ("sex_label", agg_category),
        "Extent": ("extent", agg_category) if "extent" in cohort.columns else ("sex_label", agg_category),
    }
    if "obstruct" not in cohort.columns:
        params = {
            "Age (years)": ("age", agg_mean_sd),
            "Sex": ("sex_label", agg_category),
            "Positive nodes": ("nodes", agg_mean_sd),
        }
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort,
        params=params,
        hue="rx",
        add_all=True,
        add_pvalue=True,
    )

    km = survival_curve(cohort, "time", "event", by="rx")
    _write_csv(out, "km_curve", km.frame())
    _write_csv(out, "km_at_3y", km.at([365.0, 1095.0, 1825.0]))

    lr = log_rank(cohort, "time", "event", by="rx")
    (out / "logrank.md").write_text(
        f"log-rank statistic = {lr.statistic:.3f}, df = {lr.df}, p = {lr.p_value:.4g}\n",
        encoding="utf-8",
    )
    _write_csv(out, "logrank_counts", lr.frame())

    # Number-at-risk table is the default for plot_survival (survminer-style).
    ax = plot_survival(cohort, time="time", status="event", hue="rx")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Recurrence-free survival")
    fig_path = out / "figures" / "km_rx.png"
    ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    cox_df = cohort.filter(
        pl.col("time").is_not_null()
        & pl.col("event").is_not_null()
        & pl.col("rx").is_not_null()
        & pl.col("age").is_not_null()
        & pl.col("sex").is_not_null()
        & pl.col("nodes").is_not_null()
    )
    fit = cox_ph(cox_df, COX_FORMULA)
    tidy = fit.tidy(exponentiate=True)
    _write_csv(out, "cox_tidy", tidy)
    plot_forest(
        tidy,
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
        f"Cox complete-case n = {cox_df.height} (from flowchart n = {cohort.height}). "
        f"`proportional_hazards_test` global p = {gp:.4g}. Formula `{COX_FORMULA}`.\n",
        encoding="utf-8",
    )
    (out / "cox_n.md").write_text(
        f"Cox complete-case n = {cox_df.height} (from flowchart n = {cohort.height})\n",
        encoding="utf-8",
    )
    _standardized(out, cox_df)
    write_cox_diagnostic_suite(
        fit,
        out / "figures",
        data=cohort,
        time="time",
        status="event",
        by="rx",
        stem="cox",
    )
    print(f"wrote {out} from {project.project_root}")


def _standardized(out: Path, cox_df: pl.DataFrame) -> None:
    """Covariate-adjusted (marginal) survival by rx via Cox standardization."""
    # Curve on a 30-day grid (the first time must come after the first event).
    grid = [float(t) for t in range(30, 3001, 30)]
    curve = standardize_cox(cox_df, COX_FORMULA, values={"rx": RX_LEVELS}, times=grid)
    curve_tidy = curve.tidy()
    _write_csv(out, "std_surv_curve", curve_tidy)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
    for level in RX_LEVELS:
        part = curve_tidy.filter(pl.col("rx") == level)
        t = [0.0, *part["time"].to_list()]
        est = [1.0, *part["estimate"].to_list()]
        low = [1.0, *part["conf_low"].to_list()]
        high = [1.0, *part["conf_high"].to_list()]
        color = STD_COLORS[level]
        ax.step(t, est, where="post", color=color, lw=1.8, label=level)
        ax.fill_between(t, low, high, step="post", color=color, alpha=0.15, lw=0)
    for t in (1095, 1825):
        ax.axvline(t, color="#888888", lw=0.7, ls=":")
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Standardized recurrence-free survival")
    ax.set_title("Cox standardization (age, sex, nodes averaged over the cohort)", fontsize=10)
    ax.legend(title="rx", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out / "figures" / "std_surv_rx.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Survival and difference vs Obs at 1, 3, 5 years.
    at = standardize_cox(cox_df, COX_FORMULA, values={"rx": RX_LEVELS}, times=[365.0, 1095.0, 1825.0])
    surv = at.tidy().with_columns(pl.lit("survival").alias("measure"))
    diff = (
        at.tidy(contrast="difference", reference="Obs")
        .filter(pl.col("rx") != "Obs")
        .with_columns(pl.lit("difference vs Obs").alias("measure"))
    )
    _write_csv(out, "std_surv_at", pl.concat([surv, diff]).select("measure", pl.exclude("measure")))

    # Restricted mean survival to 5 years: 0/1 exposure, so one pair at a time.
    rows = []
    for arm in ("Lev", "Lev+5FU"):
        pair = cox_df.filter(pl.col("rx").is_in(["Obs", arm])).with_columns(
            (pl.col("rx") == arm).cast(pl.Int64).alias("trt")
        )
        rm = standardize_cox(
            pair,
            "Surv(time, event) ~ trt + age + sex + nodes",
            values={"trt": [0, 1]},
            times=1825.0,
            measure="rmean",
        )
        level = rm.tidy().with_columns(pl.when(pl.col("trt") == 1).then(pl.lit(arm)).otherwise(pl.lit("Obs")).alias("arm"))
        contrast = (
            rm.tidy(contrast="difference", reference=0)
            .filter(pl.col("trt") == 1)
            .with_columns(pl.lit(f"{arm} - Obs").alias("arm"))
        )
        rows.append(
            pl.concat([level, contrast])
            .with_columns(pl.lit(f"Obs vs {arm}").alias("comparison"))
            .select("comparison", "arm", "time", "estimate", "std_error", "conf_low", "conf_high")
        )
    _write_csv(out, "std_rmst_5y", pl.concat(rows))


if __name__ == "__main__":
    main()
