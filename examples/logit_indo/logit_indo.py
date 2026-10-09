from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Table 1, site GLMM (main), and binomial GLM comparison for indo_rct."""


import shutil
from pathlib import Path

import polars as pl

from support import ProjectPath, load_parquet_dir
from statract.report.artifacts import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
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
    # Full RCT analysis set: no rows are dropped (flowchart omitted).
    cohort = target
    glm_df = cohort.filter(
        pl.col("pep").is_not_null()
        & pl.col("indomethacin").is_not_null()
        & pl.col("age").is_not_null()
        & pl.col("risk").is_not_null()
    )
    if glm_df.height != cohort.height:
        raise RuntimeError(
            f"complete-case n={glm_df.height} differs from full n={cohort.height}; "
            "flowchart would be required if anyone were excluded"
        )

    (out / "n.md").write_text(
        f"Full sample n = {glm_df.height} (no exclusion; flowchart omitted).\n",
        encoding="utf-8",
    )

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Risk score": ("risk", agg_mean_sd),
        "Gender": ("gender", agg_category),
        "Site": ("site", agg_category),
        "SOD": ("sod", agg_category) if "sod" in cohort.columns else ("site", agg_category),
        "PD stent": ("pdstent", agg_category) if "pdstent" in cohort.columns else ("site", agg_category),
        "PEP": ("outcome", agg_category),
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

    glmm_rhs = "indomethacin + age + risk"
    if "sod_yes" in glm_df.columns and glm_df["sod_yes"].null_count() == 0:
        glmm_rhs += " + sod_yes"
    if "pdstent_yes" in glm_df.columns and glm_df["pdstent_yes"].null_count() == 0:
        glmm_rhs += " + pdstent_yes"
    glmm_formula = f"pep ~ {glmm_rhs} + (1 | site)"

    note = out / "glmm.md"
    mixed = fit_mixed(glm_df, glmm_formula, family="binomial")
    tidy = mixed.tidy(exponentiate=True)
    re = glmm_random_effects(mixed, glm_df.select("site"))
    mor = median_odds_ratio(mixed)
    _write_csv(out, "glmm_tidy", tidy)
    _write_csv(out, "glmm_random_effects", re)
    _write_csv(out, "glmm_mor", mor)
    plot_forest(
        tidy,
        out / "figures" / "glmm_fixed_forest.png",
        title="Site GLMM fixed effects (OR)",
        xlabel="Odds ratio",
        layout="table",
    )
    plot_random_effects(
        re,
        out / "figures" / "glmm_site_blups.png",
        title="Site random intercepts (BLUP)",
        xlabel="Random intercept (logit)",
    )
    mor_value = float(mor["median_odds_ratio"][0])
    var = float(mor["variance"][0])
    note.write_text(
        "Primary analysis is `fit_mixed(..., family='binomial')` (lme-python / lme-rs) "
        f"on the full sample (n={glm_df.height}, formula `{glmm_formula}`). "
        f"Cluster variance = {var:.4f}; median odds ratio (MOR) = {mor_value:.3f}. "
        "MOR is `statract.median_odds_ratio` (Larsen et al.). "
        "Site BLUPs are `glmm_random_effects` / `plot_random_effects`.\n",
        encoding="utf-8",
    )

    unadj = fit_glm(glm_df, "pep ~ indomethacin", family="binomial")
    adj = fit_glm(glm_df, f"pep ~ {glmm_rhs}", family="binomial")
    unadj_tidy = unadj.tidy(exponentiate=True)
    adj_tidy = adj.tidy(exponentiate=True)
    _write_csv(out, "glm_unadjusted", unadj_tidy)
    _write_csv(out, "glm_adjusted", adj_tidy)
    plot_forest(
        adj_tidy,
        out / "figures" / "glm_adjusted_forest.png",
        title="Adjusted binomial GLM (OR), comparison",
        xlabel="Odds ratio",
        layout="table",
    )
    (out / "glm_n.md").write_text(
        f"Fixed-effect comparison on the same n = {glm_df.height}: "
        f"unadjusted `pep ~ indomethacin`; adjusted `pep ~ {glmm_rhs}` "
        "(no site random intercept).\n",
        encoding="utf-8",
    )

    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
