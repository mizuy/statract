from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Stabilized IPTW ATE for NHEFS quitting smoking and weight change. No ATT."""


import io
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from support import ProjectPath, flowchart, load_parquet_dir
from statract.report.artifacts import mermaid_flowchart
from statract.report.artifacts import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    fit_glm,
    fit_ols,
    hc_covariance,
    impute_chained,
    match_sample,
    plot_forest,
    pool,
    write_tableone_artifacts,
)

project = ProjectPath(__file__)
CACHE = project.cache
ANALYSIS_OUT = project.out

PS_COVS = [
    "age",
    "sex",
    "race",
    "education",
    "smokeintensity",
    "smokeyrs",
    "exercise",
    "active",
    "wt71",
]
PS_FORMULA = "qsmk ~ age + sex + race + education + smokeintensity + smokeyrs + exercise + active + wt71"
WEIGHT_CAP = 10.0
CEM_COVS = ["age", "sex", "race", "education", "wt71"]
MI_M = 20
MI_ITER = 10
MI_SEED = 20261008


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


def _tidy_hc(fit) -> pl.DataFrame:
    fit.covariance = hc_covariance(fit, kind="HC3")
    return fit.tidy()


def _iptw_ols(frame: pl.DataFrame):
    """PS GLM -> stabilized ATE weights (cap 10) -> weighted OLS with HC3 covariance."""
    ps_fit = fit_glm(frame, PS_FORMULA, family="binomial")
    ps = np.full(frame.height, np.nan)
    ps[np.asarray(ps_fit.row_index)] = np.clip(ps_fit.predict(kind="response"), 1e-6, 1 - 1e-6)
    a = frame["qsmk"].to_numpy().astype(float)
    p_a = float(np.mean(a))
    sw = np.clip(np.where(a == 1.0, p_a / ps, (1.0 - p_a) / (1.0 - ps)), 0.0, WEIGHT_CAP)
    fit = fit_ols(frame.with_columns(pl.Series("sw_trunc", sw)), "wt82_71 ~ qsmk", weights="sw_trunc")
    fit.covariance = hc_covariance(fit, kind="HC3")
    return fit


def _mi_sensitivity(out: Path, target: pl.DataFrame, covs: list[str], cc_tidy: pl.DataFrame) -> None:
    """MICE for missing wt82_71 (and any covariate), IPTW OLS per set, Rubin pooling."""
    frame = target.select(["qsmk", "wt82_71", *covs]).filter(pl.col("qsmk").is_not_null())
    mi = impute_chained(frame, m=MI_M, n_iter=MI_ITER, seed=MI_SEED)
    pooled = mi.pool(_iptw_ols)
    _write_csv(out, "mi_pooled", pooled)
    cc = cc_tidy.filter(pl.col("term") == "qsmk")
    mi_row = pooled.filter(pl.col("term") == "qsmk")
    compare = pl.DataFrame(
        {
            "analysis": ["Complete case (IPTW, HC3)", f"MICE m={MI_M} + Rubin (IPTW, HC3)"],
            "n": [int(target.filter(pl.col("wt82_71").is_not_null()).height), frame.height],
            "n_imputed": [0, int(sum(mi.missing.values()))],
            "estimate": [float(cc["estimate"][0]), float(mi_row["estimate"][0])],
            "std_error": [float(cc["std_error"][0]), float(mi_row["std_error"][0])],
            "conf_low": [float(cc["conf_low"][0]), float(mi_row["conf_low"][0])],
            "conf_high": [float(cc["conf_high"][0]), float(mi_row["conf_high"][0])],
            "fmi": [None, float(mi_row["fmi"][0])],
        }
    )
    _write_csv(out, "mi_vs_cc", compare)
    (out / "mi_note.md").write_text(
        f"impute_chained(m={MI_M}, n_iter={MI_ITER}, seed={MI_SEED}); "
        f"methods {mi.methods}; missing {mi.missing}. "
        "Imputation model: qsmk + wt82_71 + PS covariates (main effects, PMM). "
        "Each completed set refits the PS GLM, stabilized weights (cap 10), weighted OLS, HC3; "
        "pool() applies Rubin's rules with Barnard-Rubin df. Assumes MAR given these columns.\n",
        encoding="utf-8",
    )


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    target: pl.DataFrame = data["target"]
    covs = [c for c in PS_COVS if c in target.columns]

    buf = io.StringIO()
    cohort = flowchart(target, 
        {
            "Quit indicator and weight change present": pl.col("qsmk").is_not_null()
            & pl.col("wt82_71").is_not_null(),
            "Complete propensity covariates": pl.all_horizontal([pl.col(c).is_not_null() for c in covs]),
        },
        out=buf,
    )
    flow_text = buf.getvalue()
    (out / "text_flowchart.md").write_text(
        "```text\n" + flow_text.rstrip() + "\n```\n",
        encoding="utf-8",
    )
    (out / "mermaid_flowchart.md").write_text(
        mermaid_flowchart(flow_text, final_label="Analysis cohort"),
        encoding="utf-8",
    )

    params = {
        "Age": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Race": ("race", agg_category),
        "Education": ("education", agg_category),
        "Cigarettes/day (1971)": ("smokeintensity", agg_mean_sd),
        "Years smoked": ("smokeyrs", agg_mean_sd),
        "Exercise": ("exercise", agg_category),
        "Activity": ("active", agg_category),
        "Weight 1971 (kg)": ("wt71", agg_mean_sd),
        "Weight change (kg)": ("wt82_71", agg_mean_sd),
    }
    params = {k: v for k, v in params.items() if v[0] in cohort.columns}
    # tableone(...) is the Table 1 API; this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort,
        params=params,
        hue="qsmk",
        add_all=True,
        add_pvalue=True,
    )

    ps_fit = fit_glm(cohort, PS_FORMULA, family="binomial")
    _write_csv(out, "ps_glm", ps_fit.tidy(exponentiate=True))
    ps = np.clip(ps_fit.predict(kind="response"), 1e-6, 1 - 1e-6)
    # predict() uses fitted rows; align via row_index
    idx = np.asarray(ps_fit.row_index)
    ps_col = np.full(cohort.height, np.nan)
    ps_col[idx] = ps
    qsmk = cohort["qsmk"].to_numpy().astype(float)
    p_a = float(np.nanmean(qsmk))
    sw = np.where(qsmk == 1.0, p_a / ps_col, (1.0 - p_a) / (1.0 - ps_col))
    sw_trunc = np.clip(sw, 0.0, WEIGHT_CAP)
    weighted = cohort.with_columns(
        pl.Series("ps", ps_col),
        pl.Series("sw", sw),
        pl.Series("sw_trunc", sw_trunc),
    ).filter(pl.col("sw_trunc").is_not_null() & pl.col("sw_trunc").is_finite())

    wsum = weighted.select(
        pl.col("sw_trunc").min().alias("min"),
        pl.col("sw_trunc").median().alias("median"),
        pl.col("sw_trunc").mean().alias("mean"),
        pl.col("sw_trunc").max().alias("max"),
        (pl.col("sw") > WEIGHT_CAP).sum().alias("n_capped"),
        pl.len().alias("n"),
    )
    _write_csv(out, "iptw_weight_summary", wsum)
    (out / "iptw_n.md").write_text(
        f"Stabilized ATE IPTW; no `iptw()` helper (weights are computed in this script). "
        f"PS formula `{PS_FORMULA}`. Truncate `sw` at {WEIGHT_CAP:g}. "
        f"n = {weighted.height}; P(qsmk=1) = {p_a:.3f}; n weights capped = {int(wsum['n_capped'][0])}. "
        "Estimand is ATE only (no ATT).\n",
        encoding="utf-8",
    )

    unadj = fit_ols(cohort, "wt82_71 ~ qsmk")
    _write_csv(out, "ols_unadjusted", unadj.tidy())
    iptw = fit_ols(weighted, "wt82_71 ~ qsmk", weights="sw_trunc")
    _write_csv(out, "ols_iptw_model_se", iptw.tidy())
    iptw_hc = _tidy_hc(iptw)
    _write_csv(out, "ols_iptw_hc3", iptw_hc)
    plot_forest(
        iptw_hc,
        out / "figures" / "ols_iptw_forest.png",
        title="IPTW OLS (HC3)",
        xlabel="Coefficient (kg)",
        layout="table",
    )

    _mi_sensitivity(out, target, covs, iptw_hc)

    cem_note = out / "cem_sensitivity.md"
    cem_covs = [c for c in CEM_COVS if c in cohort.columns]
    try:
        matched = match_sample(cohort, "qsmk", cem_covs, method="cem")
        _write_csv(out, "cem_balance", matched.balance())
        fig = matched.love_plot()
        fig.savefig(out / "figures" / "cem_love.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        cem_note.write_text(
            "CEM is a short sensitivity of covariate balance, not a second estimand. "
            "No ATT outcome model is fit. Library `match_sample(..., method='cem')` "
            "internally uses ATT-style subclass weights; do not read this as ATE.\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        cem_note.write_text(f"CEM sensitivity skipped: {type(exc).__name__}: {exc}\n", encoding="utf-8")

    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
