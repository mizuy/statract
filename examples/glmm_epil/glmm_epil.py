from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1 and count GLMMs (Poisson, NB, ar1, zero-inflated) for MASS::epil."""


import shutil
from pathlib import Path

import polars as pl

from support import ProjectPath, load_parquet_dir
from statract.report.artifacts import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    fit_glm,
    fit_mixed,
    glmm_random_effects,
    median_odds_ratio,
    plot_forest,
    plot_random_effects,
    write_tableone_artifacts,
)

project = ProjectPath(__file__)
CACHE = project.cache
ANALYSIS_OUT = project.out

RHS = "progabide + log_base2wk + age"

# name -> (formula, fit_mixed kwargs, number of variance parameters)
MODELS: dict[str, tuple[str, dict, int]] = {
    "Poisson GLMM, (1|subject)": (
        f"y ~ {RHS} + (1 | subject)",
        {"family": "poisson"},
        1,
    ),
    "NB GLMM, (1|subject)": (
        f"y ~ {RHS} + (1 | subject)",
        {"family": "negative_binomial"},
        1,
    ),
    "Poisson GLMM, ar1": (
        f"y ~ {RHS} + ar1(period + 0 | subject)",
        {"family": "poisson", "engine": "laplace"},
        2,
    ),
    "NB GLMM, ar1": (
        f"y ~ {RHS} + ar1(period + 0 | subject)",
        {"family": "negative_binomial", "engine": "laplace"},
        2,
    ),
    "ZINB GLMM, (1|subject)": (
        f"y ~ {RHS} + (1 | subject)",
        {"family": "negative_binomial", "zero_inflation": True},
        1,
    ),
}
MAIN = "NB GLMM, (1|subject)"


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


def _n_params(fit, n_variance: int) -> int:
    k = len(fit.coefficients) + n_variance
    if fit.theta is not None:
        k += 1
    if fit.zero_coefficients is not None:
        k += len(fit.zero_coefficients)
    return k


def _rho(fit) -> float | None:
    vt = fit.variance_table()
    if "structure" in vt.columns and (vt["structure"] == "ar1").any():
        return float(vt.filter(pl.col("structure") == "ar1")["correlation"][0])
    return None


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    visits: pl.DataFrame = data["visits"]
    patients: pl.DataFrame = data["patients"]
    if visits.height != 236 or patients.height != 59:
        raise RuntimeError(f"expected 59 x 4 = 236 rows, got {patients.height} / {visits.height}")
    (out / "n.md").write_text(
        f"{patients.height} patients x 4 visits = {visits.height} rows (no exclusion; flowchart omitted).\n",
        encoding="utf-8",
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Baseline seizures / 8 wk": ("base", agg_median_iqr),
        "Seizures, 4 visits total": ("y_total", agg_median_iqr),
        "Any seizure-free visit": ("any_zero_visit", agg_category),
    }
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=patients,
        params=params,
        hue="trt",
        add_all=True,
        add_pvalue=True,
    )

    fits = {name: fit_mixed(visits, formula, **kw) for name, (formula, kw, _) in MODELS.items()}

    # Reference only: Poisson GLM without a patient effect (ignores clustering).
    glm = fit_glm(visits, f"y ~ {RHS}", family="poisson")

    rows = []
    rr_rows = []
    for name, fit in fits.items():
        k = _n_params(fit, MODELS[name][2])
        ll = float(fit.log_likelihood)
        rows.append(
            {
                "model": name,
                "formula": MODELS[name][0],
                "df": k,
                "logLik": ll,
                "AIC": -2 * ll + 2 * k,
                "theta": fit.theta,
                "rho": _rho(fit),
                "converged": bool(fit.converged),
            }
        )
        rr = fit.tidy(exponentiate=True).filter(pl.col("term") == "progabide")
        rr_rows.append(rr.with_columns(pl.lit(name).alias("term")))
    rows.append(
        {
            "model": "Poisson GLM, no RE",
            "formula": f"y ~ {RHS}",
            "df": len(glm.coefficients),
            "logLik": float(glm.log_likelihood),
            "AIC": float(glm.aic),
            "theta": None,
            "rho": None,
            "converged": True,
        }
    )
    glm_rr = glm.tidy(exponentiate=True).filter(pl.col("term") == "progabide")
    rr_rows.append(glm_rr.with_columns(pl.lit("Poisson GLM, no RE").alias("term")))

    comparison = pl.DataFrame(rows).with_columns(
        (pl.col("AIC") - pl.col("AIC").min()).alias("delta_AIC")
    )
    _write_csv(out, "model_comparison", comparison)
    rate_ratios = pl.concat(rr_rows, how="vertical_relaxed")
    _write_csv(out, "trt_rate_ratios", rate_ratios)
    plot_forest(
        rate_ratios,
        out / "figures" / "trt_rate_ratio_forest.png",
        title="Progabide vs placebo: rate ratio by model",
        xlabel="Rate ratio (progabide / placebo)",
        layout="table",
        drop_intercept=False,
    )

    main_fit = fits[MAIN]
    main_tidy = main_fit.tidy(exponentiate=True)
    _write_csv(out, "nb_glmm_tidy", main_tidy)
    _write_csv(out, "nb_glmm_variance", main_fit.variance_table())
    plot_forest(
        main_tidy,
        out / "figures" / "nb_glmm_forest.png",
        title="NB GLMM fixed effects (rate ratio)",
        xlabel="Rate ratio",
        layout="table",
    )
    re = glmm_random_effects(main_fit, visits.select("subject"))
    _write_csv(out, "nb_glmm_random_effects", re)
    plot_random_effects(
        re,
        out / "figures" / "nb_glmm_subject_blups.png",
        title="Patient random intercepts (NB GLMM)",
        xlabel="Random intercept (log rate)",
        figsize=(6.5, 9.0),
    )
    # Same formula as the MOR, read on the log-rate scale: median rate ratio.
    mrr = median_odds_ratio(main_fit).rename({"median_odds_ratio": "median_rate_ratio"})
    _write_csv(out, "nb_glmm_mrr", mrr)

    ar1_fit = fits["NB GLMM, ar1"]
    _write_csv(out, "nb_ar1_variance", ar1_fit.variance_table())
    zi_fit = fits["ZINB GLMM, (1|subject)"]
    _write_csv(out, "zinb_zero_part", zi_fit.zero_table(exponentiate=True))

    rr_main = rate_ratios.filter(pl.col("term") == MAIN)
    (out / "glmm.md").write_text(
        f"Main model: `fit_mixed(visits, 'y ~ {RHS} + (1 | subject)', family='negative_binomial')`. "
        f"Progabide rate ratio = {float(rr_main['exp_estimate'][0]):.3f} "
        f"(95% CI {float(rr_main['exp_conf_low'][0]):.3f}-{float(rr_main['exp_conf_high'][0]):.3f}); "
        f"theta = {main_fit.theta:.2f}; patient variance = {float(mrr['variance'][0]):.3f}; "
        f"median rate ratio = {float(mrr['median_rate_ratio'][0]):.2f}.\n",
        encoding="utf-8",
    )

    print(comparison)
    print(rate_ratios.select("term", "exp_estimate", "exp_conf_low", "exp_conf_high", "p_value"))
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
