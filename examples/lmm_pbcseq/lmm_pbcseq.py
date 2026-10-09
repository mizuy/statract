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
    gamm,
    smooth,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

LMM_FORMULA = "log_bili ~ day_years + dp + (1 | id)"
OLS_FORMULA = "log_bili ~ day_years + dp"
HIGH_BILI = 2.0  # mg/dL; gamm has binomial / Poisson only, so the gamm section uses bili > 2
GLMM_FORMULA = "high_bili ~ day_years + dp + (1 | id)"


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
    _gamm_section(out, visits)
    print(f"wrote {out} from {project.project_root}")


def _gamm_section(out: Path, visits: pl.DataFrame) -> None:
    """gamm (gamm4 port): s(day) + dp + (1 | id) on bili > 2, vs gam and a linear GLMM."""
    v = visits.with_columns((pl.col("bili") > HIGH_BILI).cast(pl.Int64).alias("high_bili"))
    sm = smooth("day_years", k=10, basis="tp")  # mgcv / gamm4 default s(day_years)
    fit_gamm = gamm(v, "high_bili", [sm], random="(1 | id)", predictors=["dp"], family="binomial")
    fit_gam = gam(v, "high_bili", [sm], predictors=["dp"], family="binomial")
    fit_glmm = fit_mixed(v, GLMM_FORMULA, family="binomial")
    _write_csv(out, "gamm_tidy", fit_gamm.tidy())
    _write_csv(out, "gamm_smooth", fit_gamm.smooth_table())
    _write_csv(out, "gamm_variance", fit_gamm.variance_table())

    def _row(model, tidy, edf, var):
        r = tidy.filter(pl.col("term") == "dp").row(0, named=True)
        return {
            "model": model,
            "dp_estimate": r["estimate"],
            "dp_std_error": r["std_error"],
            "dp_p_value": r["p_value"],
            "day_edf": edf,
            "id_variance": var,
        }

    compare = pl.DataFrame(
        [
            _row("gamm: s(day) + dp + (1|id)", fit_gamm.tidy(), float(fit_gamm.edf[0]),
                 float(fit_gamm.variance_table()["variance"][0])),
            _row("gam: s(day) + dp", fit_gam.tidy(), float(fit_gam.smooth_table()["edf"][0]), None),
            _row("GLMM: day + dp + (1|id)", fit_glmm.tidy(), 1.0,
                 float(fit_glmm.variance_table()["variance"][0])),
        ]
    )
    _write_csv(out, "gamm_compare", compare)
    (out / "gamm_n.md").write_text(
        f"gamm binomial: high_bili = bili > {HIGH_BILI} mg/dL; visits {v.height}, "
        f"prevalence {v['high_bili'].mean():.3f}; converged = {fit_gamm.converged}.\n",
        encoding="utf-8",
    )

    # Smooths on the log-odds scale, each centred to mean zero over the visits.
    grid = np.linspace(float(v["day_years"].min()), float(v["day_years"].max()), 100)
    pe_mm = fit_gamm.partial_effect("day_years", grid)
    pe_gam = fit_gam.partial_effect("day_years", n=100)
    slope = fit_glmm.tidy().filter(pl.col("term") == "day_years")["estimate"][0]
    lin = slope * (grid - float(v["day_years"].mean()))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for x, f, se, color, ls, label in (
        (pe_mm["day_years"], pe_mm["fit"], pe_mm["std_error"], "C0", "-", "gamm (random intercept)"),
        (pe_gam["day_years"], pe_gam["estimate"], pe_gam["std_error"], "C1", "--", "gam (no random effect)"),
    ):
        x, f, se = (np.asarray(a, dtype=float) for a in (x, f, se))
        ax.fill_between(x, f - 1.96 * se, f + 1.96 * se, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, f, color=color, linestyle=ls, linewidth=2, label=label)
    ax.plot(grid, lin, color="0.35", linestyle=":", linewidth=2, label="GLMM linear day")
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.set_xlabel("Day (years)")
    ax.set_ylabel(f"s(day) on log-odds of bili > {HIGH_BILI:g} (centred)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "gamm_day.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
