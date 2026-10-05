from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Train/hold-out binomial GLM, calibration, and DCA for SUPPORT2 180-day death."""


import io
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from support import flowchart, load_parquet_dir
from statract.reporting import mermaid_flowchart
from statract.reporting import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    binary_perf,
    calibration_table,
    fit_glm,
    plot_calibration,
    plot_dca,
    plot_forest,
    threshold_tradeoff,
    write_probability_artifacts,
    write_tableone_artifacts,
)
from config import ANALYSIS_OUT, CACHE
from project import project

PRED_COVS = [
    "age",
    "sex",
    "race",
    "dzclass",
    "num_co",
    "diabetes",
    "dementia",
    "ca",
    "scoma",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "crea",
    "sod",
]
NULL_FORMULA = "death_180 ~ 1"
AGE_SEX_FORMULA = "death_180 ~ age + sex"
FULL_FORMULA = (
    "death_180 ~ age + sex + race + dzclass + num_co + diabetes + dementia + ca"
    " + scoma + meanbp + hrt + resp + temp + crea + sod"
)
VAL_FRACTION = 0.30
VAL_SEED = 2026
DECISION_THRESHOLD = 0.40


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


def _stratified_val_mask(y: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(y.size, dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if idx.size == 0:
            continue
        n_val = max(1, int(round(fraction * idx.size)))
        n_val = min(n_val, idx.size - 1) if idx.size > 1 else idx.size
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True
    return val_mask


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    target: pl.DataFrame = data["target"]
    covs = [c for c in PRED_COVS if c in target.columns]

    buf = io.StringIO()
    cohort = flowchart(target, 
        {
            "Complete prognostic covariates": pl.all_horizontal(
                [pl.col(c).is_not_null() for c in covs + ["death_180"]],
            ),
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
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Race": ("race", agg_category),
        "Disease class": ("dzclass", agg_category),
        "Comorbidities (n)": ("num_co", agg_mean_sd),
        "Diabetes": ("diabetes", agg_category),
        "Dementia": ("dementia", agg_category),
        "Cancer": ("ca", agg_category),
        "GCS coma score": ("scoma", agg_mean_sd),
        "Mean BP (mmHg)": ("meanbp", agg_mean_sd),
        "Heart rate": ("hrt", agg_mean_sd),
        "Respiratory rate": ("resp", agg_mean_sd),
        "Temperature (°C)": ("temp", agg_mean_sd),
        "Creatinine (mg/dL)": ("crea", agg_mean_sd),
        "Sodium (mEq/L)": ("sod", agg_mean_sd),
        "180-day death": ("death_180_label", agg_category),
    }
    params = {k: v for k, v in params.items() if v[0] in cohort.columns}
    # Public Table 1 API is tableone(); this helper only writes CSV/HTML/md to *_out/.
    write_tableone_artifacts(
        out,
        "table1",
        df=cohort,
        params=params,
        hue="death_180_label",
        add_all=True,
        add_pvalue=True,
    )

    y_all = cohort["death_180"].to_numpy().astype(int)
    val_mask = _stratified_val_mask(y_all, fraction=VAL_FRACTION, seed=VAL_SEED)
    train = cohort.filter(~pl.Series(val_mask))
    val = cohort.filter(pl.Series(val_mask))

    null_fit = fit_glm(train, NULL_FORMULA, family="binomial")
    age_fit = fit_glm(train, AGE_SEX_FORMULA, family="binomial")
    full_fit = fit_glm(train, FULL_FORMULA, family="binomial")
    full_tidy = full_fit.tidy(exponentiate=True)
    _write_csv(out, "glm_full_train", full_tidy)
    _write_csv(out, "glm_age_sex_train", age_fit.tidy(exponentiate=True))
    plot_forest(
        full_tidy,
        out / "figures" / "glm_full_forest.png",
        title="Multivariable GLM on training sample (OR)",
        xlabel="Odds ratio",
        layout="table",
    )

    y_train = train["death_180"].to_numpy().astype(int)
    y_val = val["death_180"].to_numpy().astype(int)
    p_null = np.asarray(null_fit.predict(val, kind="response"), dtype=float)
    p_age = np.asarray(age_fit.predict(val, kind="response"), dtype=float)
    p_full_val = np.asarray(full_fit.predict(val, kind="response"), dtype=float)
    p_full_app = np.asarray(full_fit.predict(train, kind="response"), dtype=float)

    write_probability_artifacts(
        [
            {"name": "null (validation)", "y_val": y_val, "prob_val": p_null},
            {"name": "glm_age_sex (validation)", "y_val": y_val, "prob_val": p_age},
            {"name": "glm_full (validation)", "y_val": y_val, "prob_val": p_full_val},
        ],
        out,
        positive_label="180-day death",
        dca_thresholds=np.linspace(0.05, 0.60, 12),
    )
    for name in ("fig_calibration.png", "fig_dca.png"):
        src = out / name
        if src.is_file():
            shutil.move(str(src), str(out / "figures" / name))

    # Apparent vs hold-out for the same multivariable model (optimism).
    cal_app_val = pl.concat(
        [
            calibration_table(y_train, p_full_app, model="glm_full (apparent)", split="apparent"),
            calibration_table(
                y_val,
                p_full_val,
                model="glm_full (validation)",
                split="validation",
            ),
        ],
        how="vertical_relaxed",
    )
    _write_csv(out, "table_calibration_apparent", cal_app_val)
    plot_calibration(
        cal_app_val,
        out / "figures" / "fig_calibration_apparent.png",
        title="Calibration: apparent vs validation (glm_full)",
    )
    # Direct plot_dca call (same validation table the writer already saved).
    dca = pl.read_csv(out / "table_dca.csv")
    plot_dca(dca, out / "figures" / "fig_dca.png", title="Decision curve (validation)")

    pred_cls = (p_full_val >= DECISION_THRESHOLD).astype(int)
    perf = binary_perf(y_val, pred_cls)
    perf_frame = pl.DataFrame({k: [v] for k, v in perf.items()}).with_columns(
        pl.lit(DECISION_THRESHOLD).alias("threshold"),
        pl.lit("glm_full (validation)").alias("model"),
    )
    _write_csv(out, "binary_perf_val", perf_frame)
    trade = threshold_tradeoff(
        y_val,
        p_full_val,
        thresholds=np.array([0.20, 0.30, 0.40, 0.50, 0.60]),
    )
    _write_csv(out, "threshold_tradeoff_val", trade)

    (out / "split.md").write_text(
        f"Stratified 70/30 split (seed={VAL_SEED}): train n={train.height}, "
        f"validation n={val.height}. Models are fit on train only. "
        f"`binary_perf` uses threshold {DECISION_THRESHOLD} on glm_full. "
        "SUPPORT physiology scores (`sps`, `aps`, `surv2m`, `surv6m`) are not predictors.\n",
        encoding="utf-8",
    )
    (out / "n.md").write_text(
        f"Fetched SUPPORT2 n={target.height}; complete-covariate cohort n={cohort.height}; "
        f"train n={train.height}; validation n={val.height}; "
        f"validation 180-day death rate={float(y_val.mean()):.3f}.\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
