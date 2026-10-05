from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Patient Table 1, Gaussian LMM, cluster OLS, and GAM smooth of day."""


import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from support import load_parquet_dir
from statract.reporting import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    cluster_covariance,
    fit_mixed,
    fit_ols,
    gam,
    smooth,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

LMM_FORMULA = "log_bili ~ day_years + dp + (1 | id)"
OLS_FORMULA = "log_bili ~ day_years + dp"


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


def _tidy_with_cov(fit, cov: np.ndarray) -> pl.DataFrame:
    fit.covariance = cov
    return fit.tidy()


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    long: pl.DataFrame = data["long"]
    patient: pl.DataFrame = data["patient"]

    # Patient table is already the randomized baseline set (no exclusion; flowchart omitted).
    cohort_p = patient
    dropped = cohort_p.filter(
        pl.col("trt_label").is_null() | pl.col("bili").is_null() | (pl.col("bili") <= 0)
    )
    if dropped.height:
        raise RuntimeError("unexpected patient exclusions; restore a flowchart")
    (out / "n.md").write_text(
        f"Patient n = {cohort_p.height} (no exclusion; flowchart omitted).\n",
        encoding="utf-8",
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Baseline bilirubin": ("bili", agg_median_iqr),
        "Baseline albumin": ("albumin", agg_median_iqr),
        "Baseline day": ("day", agg_median_iqr),
    }
    params = {k: v for k, v in params.items() if v[0] in cohort_p.columns}
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort_p,
        params=params,
        hue="trt_label",
        add_all=True,
        add_pvalue=True,
    )

    ids = set(cohort_p["id"].to_list())
    visits = long.filter(
        pl.col("id").is_in(list(ids))
        & pl.col("log_bili").is_not_null()
        & pl.col("day_years").is_not_null()
        & pl.col("dp").is_not_null()
        & pl.col("id").is_not_null()
    )
    (out / "visit_n.md").write_text(
        f"patient n = {cohort_p.height}; longitudinal complete visits = {visits.height}; "
        f"unique id in visits = {visits['id'].n_unique()}\n",
        encoding="utf-8",
    )

    mixed = fit_mixed(visits, LMM_FORMULA)
    _write_csv(out, "lmm_fixed", mixed.tidy())
    _write_csv(out, "lmm_variance", mixed.variance_table())

    ols = fit_ols(visits, OLS_FORMULA)
    _write_csv(out, "ols_naive", ols.tidy())
    clust = cluster_covariance(ols, "id", data=visits)
    _write_csv(out, "ols_cluster", _tidy_with_cov(ols, clust))

    k = min(8, int(visits["day_years"].n_unique()))
    fitted_gam = gam(visits, "log_bili", [smooth("day_years", k=k)])
    _write_csv(out, "gam_tidy", fitted_gam.tidy())
    _write_csv(out, "gam_smooth", fitted_gam.smooth_table())
    (out / "gam_n.md").write_text(
        f"GAM is a main analysis: Gaussian `gam` + one `smooth('day_years', k={k})`. "
        "This API version has no treatment term and no random intercept "
        "(family=gaussian, basis=cr, one smooth, REML). "
        f"n visits = {visits.height}. edf = {fitted_gam.edf:.3f}.\n",
        encoding="utf-8",
    )

    grid = pl.DataFrame(
        {"day_years": np.linspace(float(visits["day_years"].min()), float(visits["day_years"].max()), 80)}
    )
    pred = fitted_gam.predict(grid)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(
        visits["day_years"].to_numpy(),
        visits["log_bili"].to_numpy(),
        s=8,
        alpha=0.15,
        color="gray",
        label="visits",
    )
    ax.plot(grid["day_years"].to_numpy(), pred, color="C0", label="GAM s(day_years)")
    ax.set_xlabel("Day (years)")
    ax.set_ylabel("log bilirubin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "gam_day.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
