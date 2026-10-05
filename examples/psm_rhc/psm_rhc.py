from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Unmatched Table 1, nearest PS match, love plot, matched logistic for RHC."""


import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from support import load_parquet_dir
from statract.reporting import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    fit_glm,
    match_sample,
    plot_forest,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

COVARIATES = [
    "age",
    "sex",
    "race",
    "edu",
    "cat1",
    "ca",
    "aps1",
    "scoma1",
    "meanbp1",
    "hrt1",
    "resp1",
    "temp1",
    "pafi1",
    "alb1",
    "hema1",
    "bili1",
    "crea1",
    "sod1",
    "cardiohx",
    "chfhx",
    "chrpulhx",
    "dnr1",
]


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
    covs = [c for c in COVARIATES if c in target.columns]

    # Analysis frame is already complete (no exclusion; flowchart omitted).
    cohort = target
    missing = cohort.filter(
        pl.col("rhc").is_null()
        | pl.col("dth30_bin").is_null()
        | pl.any_horizontal([pl.col(c).is_null() for c in covs])
    )
    if missing.height:
        raise RuntimeError("unexpected incomplete RHC rows; restore a flowchart")
    (out / "n.md").write_text(
        f"Full sample n = {cohort.height} (no exclusion; flowchart omitted).\n",
        encoding="utf-8",
    )

    table1_params = {
        "Age": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Race": ("race", agg_category),
        "Education (years)": ("edu", agg_mean_sd),
        "Primary disease": ("cat1", agg_category),
        "Cancer": ("ca", agg_category),
        "APS1": ("aps1", agg_mean_sd),
        "Mean BP": ("meanbp1", agg_mean_sd),
        "DNR": ("dnr1", agg_category),
        "30-day death": ("dth30_bin", agg_mean_sd),
    }
    table1_params = {k: v for k, v in table1_params.items() if v[0] in cohort.columns}
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1_unmatched",
        df=cohort,
        params=table1_params,
        hue="rhc",
        add_all=True,
        add_pvalue=True,
    )

    matched = match_sample(
        cohort,
        "rhc",
        covs,
        method="nearest",
        distance="logit",
        order="data",
        estimand="ATT",
        ratio=1,
        replace=False,
        caliper=0.2,
    )
    balance = matched.balance()
    _write_csv(out, "balance", balance)
    fig = matched.love_plot()
    fig.savefig(out / "figures" / "love_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    frame = matched.frame()
    kept = frame.filter(pl.col("weights") > 0)
    # Same public API as unmatched: tableone(kept, params, hue="rhc", add_pvalue=True).
    write_tableone_artifacts(
        out,
        "table1_matched",
        df=kept,
        params={k: v for k, v in table1_params.items() if k != "30-day death"},
        hue="rhc",
        add_all=True,
        add_pvalue=True,
    )

    glm = fit_glm(kept, "dth30_bin ~ rhc", family="binomial", weights="weights")
    glm_tidy = glm.tidy(exponentiate=True)
    _write_csv(out, "glm_matched", glm_tidy)
    plot_forest(
        glm_tidy,
        out / "figures" / "glm_matched_forest.png",
        title="Matched logistic (OR)",
        xlabel="Odds ratio",
        layout="table",
    )
    unadj = fit_glm(cohort, "dth30_bin ~ rhc", family="binomial")
    _write_csv(out, "glm_unmatched", unadj.tidy(exponentiate=True))
    (out / "match_n.md").write_text(
        f"complete-case n = {cohort.height}; matched weights>0 n = {kept.height}; "
        f"treated matched = {kept.filter(pl.col('rhc') == 1).height}\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
