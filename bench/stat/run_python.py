"""Time the Python estimators and write numeric payloads for the R comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import polars as pl

from statract import (
    accelerated_failure,
    breusch_godfrey_test,
    breusch_pagan_test,
    calibrate_cox,
    calibrate_logistic,
    cluster_covariance,
    conditional_tree,
    cox_ph,
    cumulative_incidence,
    durbin_watson_test,
    fine_gray_regression,
    fit_glm,
    fit_mixed,
    fit_ols,
    gam,
    gamm,
    hc_covariance,
    impute_chained,
    likelihood_ratio_test,
    log_rank,
    match_sample,
    newey_west_covariance,
    p_adjust,
    prop_test,
    ramsey_reset_test,
    roc_curve,
    roc_test,
    smooth,
    standardize_cox,
    survival_curve,
    t_test,
    validate_cox,
    validate_logistic,
    wald_test,
    wilcox_test,
)

CACHE = Path("/tmp/statract-bench")
REPEATS = 5

OLS_X = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "sex",
    "race",
    "marital3",
    "workclass4",
    "relationship3",
    "us_native",
]
GLM_X = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "sex",
    "race",
    "marital3",
    "workclass4",
    "us_native",
    "hours_per_week",
]
MATCH_X = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "race",
    "marital3",
    "workclass4",
    "relationship3",
    "us_native",
    "hours_per_week",
]
SURV_X = ["age", "sex", "num_co", "scoma", "meanbp", "hrt", "temp", "resp", "diabetes", "ca"]
# rms-style logistic validation drops temp: statract's lrm.fit port judges the
# information matrix singular when temp (about 37, nearly constant) sits next to
# the intercept (smallest singular value below 1e-7 x the largest entry), and
# stops with "did not converge". The 9 other columns fit.
LRM_X = [name for name in SURV_X if name != "temp"]
# Second, smaller logistic model for the paired ROC comparison.
ROC_X2 = ["age", "sex", "num_co", "ca"]
# Numeric columns compared between in-hospital deaths and survivors (t tests, p_adjust).
HTEST_X = ["age", "num_co", "scoma", "meanbp", "hrt", "temp", "resp"]
STD_COX_X = ["diabetes", "age", "sex", "num_co", "meanbp", "ca"]
GAMM_GRID = [50.0, 70.0, 90.0, 110.0, 130.0]
GLMM_FORMULA = {
    "poisson": "num_co ~ age + sex + meanbp + ca + (1 | dzgroup)",
    "negative_binomial": "slos ~ age + sex + num_co + meanbp + ca + (1 | dzgroup)",
}
MI_COLUMNS = ["hospdead", "age", "sex", "num_co", "meanbp", "hrt", "alb", "bili", "pafi", "wblc", "income4"]
MI_X = ["age", "sex", "num_co", "meanbp", "hrt", "alb", "bili", "pafi", "wblc", "income4"]
STAR_X = [
    "gender",
    "race",
    "freelunch",
    "birthyear",
    "grade",
    "classtype",
    "urban",
    "tyears",
    "tgen",
    "classsize",
]


def _timed(fn, repeats: int | None = None):
    fn()
    samples = []
    result = None
    for _ in range(REPEATS if repeats is None else repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result


def _fit_payload(fit, *, tests: bool) -> dict:
    tidy = fit.tidy()
    payload = {
        "terms": tidy["term"].to_list(),
        "coef": [float(v) for v in tidy["estimate"].to_list()],
        "se": [float(v) for v in tidy["std_error"].to_list()],
        "loglik": float(fit.log_likelihood),
    }
    if tests:
        payload["t"] = [float(v) for v in tidy["statistic"].to_list()]
        payload["p"] = [float(v) for v in tidy["p_value"].to_list()]
    return payload


def _cov_payload(names: list[str], cov: np.ndarray) -> dict:
    return {"terms": names, "cov": np.asarray(cov, dtype=float).tolist()}


def _test_payload(result) -> dict:
    return {"stat": float(result.statistic), "p": float(result.p_value)}


def _curve_payload(curve, times: list[float]) -> dict:
    table = curve.at(times)
    rows = []
    for record in table.iter_rows(named=True):
        rows.append(
            {
                "group": None if "group" not in record else record["group"],
                "time": float(record["time"]),
                "estimate": float(record["estimate"]),
                "low": float(record["conf_low"]),
                "high": float(record["conf_high"]),
            }
        )
    return {"rows": rows}


def _htest_payload(result) -> dict:
    payload = {"stat": float(result.statistic), "p": float(result.p_value)}
    if result.parameter is not None:
        payload["df"] = float(result.parameter)
    if result.conf_int is not None:
        payload["ci"] = [float(v) for v in result.conf_int]
    payload["estimate"] = [float(v) for v in result.estimates.values()]
    return payload


def _by_outcome(frame: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Values of ``column`` among in-hospital deaths (x) and the rest (y)."""
    died = frame["hospdead"].to_numpy() == 1
    values = frame[column].to_numpy().astype(float)
    return values[died], values[~died]


def _split_labels(node, out: list[str]) -> None:
    """Splits of a conditional tree, depth first, as text both sides can write."""
    split = node.split
    if split is None or node.left is None or node.right is None:
        return
    if split.break_at is not None:
        out.append(f"{split.column}<={split.break_at:.10g}")
    else:
        out.append(f"{split.column} in {{{','.join(sorted(split.left_levels or ()))}}}")
    _split_labels(node.left, out)
    _split_labels(node.right, out)


def _with_treat(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((pl.col("sex") == "Female").cast(pl.Float64).alias("treat"))


def run_task(task: dict, frame: pl.DataFrame, manifest: dict) -> dict:
    kind = task["kind"]
    option = task["option"]
    record = {"id": task["id"], "slice": task["slice"], "n": frame.height, "error": None, "parts": {}}
    try:
        if kind == "ols":
            seconds, fit = _timed(lambda: fit_ols(frame, "hours_per_week", OLS_X))
            record["parts"]["main"] = {**_fit_payload(fit, tests=True), "seconds": seconds}
        elif kind == "ols_w":
            seconds, fit = _timed(lambda: fit_ols(frame, "hours_per_week", OLS_X, weights="fnlwgt"))
            record["parts"]["main"] = {**_fit_payload(fit, tests=True), "seconds": seconds}
        elif kind == "glm_bin":
            seconds, fit = _timed(lambda: fit_glm(frame, "income_gt_50k", GLM_X, family="binomial"))
            record["parts"]["main"] = {**_fit_payload(fit, tests=False), "seconds": seconds}
        elif kind == "glm_gamma":
            seconds, fit = _timed(lambda: fit_glm(frame, "hours_per_week", OLS_X, family="gamma"))
            record["parts"]["main"] = {**_fit_payload(fit, tests=False), "seconds": seconds}
        elif kind == "glm_pois":
            predictors = manifest["slices"][task["slice"]]["predictors"]
            seconds, fit = _timed(lambda: fit_glm(frame, "cnt", predictors, family="poisson"))
            record["parts"]["main"] = {**_fit_payload(fit, tests=False), "seconds": seconds}
        elif kind == "hc":
            fit = fit_ols(frame, "hours_per_week", OLS_X)
            for hc_type in option["types"]:
                seconds, cov = _timed(lambda hc_type=hc_type: hc_covariance(fit, hc_type))
                record["parts"][hc_type] = {**_cov_payload(fit.names, cov), "seconds": seconds}
        elif kind == "cl1":
            fit = fit_ols(frame, "read", STAR_X)
            seconds, cov = _timed(lambda: cluster_covariance(fit, "school", data=frame))
            record["parts"]["main"] = {**_cov_payload(fit.names, cov), "seconds": seconds}
        elif kind == "cl2":
            fit = fit_ols(frame, "read", STAR_X)
            seconds, cov = _timed(lambda: cluster_covariance(fit, ["school", "grade"], data=frame))
            record["parts"]["main"] = {**_cov_payload(fit.names, cov), "seconds": seconds}
        elif kind == "nw":
            predictors = manifest["slices"][task["slice"]]["predictors"]
            fit = fit_ols(frame, "cnt", predictors)
            seconds, cov = _timed(lambda: newey_west_covariance(fit))
            record["parts"]["main"] = {**_cov_payload(fit.names, cov), "seconds": seconds}
        elif kind == "wald":
            full = fit_ols(frame, "hours_per_week", OLS_X)
            reduced = fit_ols(frame, "hours_per_week", [name for name in OLS_X if name != "us_native"])
            seconds, result = _timed(lambda: wald_test(full, reduced))
            record["parts"]["main"] = {**_test_payload(result), "seconds": seconds}
        elif kind == "lr":
            full = fit_ols(frame, "hours_per_week", OLS_X)
            reduced = fit_ols(frame, "hours_per_week", [name for name in OLS_X if name != "us_native"])
            seconds, result = _timed(lambda: likelihood_ratio_test(full, reduced))
            record["parts"]["main"] = {**_test_payload(result), "seconds": seconds}
        elif kind == "bp":
            fit = fit_ols(frame, "hours_per_week", OLS_X)
            for studentize in option["studentize"]:
                label = "student" if studentize else "raw"
                seconds, result = _timed(lambda studentize=studentize: breusch_pagan_test(fit, studentize=studentize))
                record["parts"][label] = {**_test_payload(result), "seconds": seconds}
        elif kind == "reset":
            fit = fit_ols(frame, "hours_per_week", OLS_X)
            seconds, result = _timed(lambda: ramsey_reset_test(fit, power=(2, 3)))
            record["parts"]["main"] = {**_test_payload(result), "seconds": seconds}
        elif kind == "dw":
            predictors = manifest["slices"][task["slice"]]["predictors"]
            fit = fit_ols(frame, "cnt", predictors)
            seconds, result = _timed(lambda: durbin_watson_test(fit, alternative="greater"))
            record["parts"]["main"] = {**_test_payload(result), "seconds": seconds}
        elif kind == "bg":
            predictors = manifest["slices"][task["slice"]]["predictors"]
            fit = fit_ols(frame, "cnt", predictors)
            for order in option["orders"]:
                seconds, result = _timed(
                    lambda order=order: breusch_godfrey_test(fit, order=order, distribution="chi2")
                )
                record["parts"][str(order)] = {**_test_payload(result), "seconds": seconds}
        elif kind in {"km", "km_sex", "na"}:
            times = manifest["km_times"]
            by = "sex" if kind == "km_sex" else None
            curve_kind = "nelson_aalen" if kind == "na" else "kaplan_meier"

            def _curve(by=by, curve_kind=curve_kind):
                fitted = survival_curve(frame, "d_time", "death", by, kind=curve_kind, confidence="log")
                return _curve_payload(fitted, times)

            seconds, payload = _timed(_curve)
            record["parts"]["main"] = {**payload, "seconds": seconds}
        elif kind == "lrk":
            specs = (("rho0", {"rho": 0.0}), ("rho1", {"rho": 1.0}), ("strata", {"rho": 0.0, "strata": "ca"}))
            for label, kwargs in specs:
                seconds, result = _timed(lambda kwargs=kwargs: log_rank(frame, "d_time", "death", "sex", **kwargs))
                record["parts"][label] = {**_test_payload(result), "seconds": seconds}
        elif kind == "cox":
            strata = "dzgroup" if option["strata"] else None
            ties = option["ties"]
            seconds, fit = _timed(lambda: cox_ph(frame, "d_time", "death", SURV_X, strata=strata, ties=ties))
            record["parts"]["main"] = {**_fit_payload(fit, tests=False), "seconds": seconds}
        elif kind == "aft":
            distribution = option["distribution"]
            seconds, fit = _timed(
                lambda: accelerated_failure(frame, "d_time", "death", SURV_X, distribution=distribution)
            )
            record["parts"]["main"] = {**_fit_payload(fit, tests=False), "seconds": seconds}
        elif kind == "match":
            treated = _with_treat(frame)
            distance = option["distance"]

            def _match(distance=distance):
                if distance == "exact":
                    matched = match_sample(treated, "treat", ["race"], method="exact", exact=["race"])
                elif distance == "cem":
                    matched = match_sample(treated, "treat", MATCH_X, method="cem", cutpoints="sturges")
                else:
                    matched = match_sample(
                        treated, "treat", MATCH_X, method="nearest", distance=distance, order="data"
                    )
                payload = {"weights": [float(v) for v in matched.weights]}
                # Exact and CEM compare weights. The subclass product is not part of that check.
                if distance in {"exact", "cem"}:
                    payload["pairs"] = []
                else:
                    pairs = matched.pairs()
                    payload["pairs"] = sorted(
                        (int(a), int(b))
                        for a, b in zip(pairs["treated"].to_list(), pairs["control"].to_list(), strict=True)
                    )
                if distance == "logit":
                    smd = matched.balance().filter(pl.col("term") == "age")["smd_matched"][0]
                    payload["smd_age"] = float(smd)
                return payload

            seconds, payload = _timed(_match)
            record["parts"]["main"] = {**payload, "seconds": seconds}
        elif kind == "gam":
            k = int(option["k"])
            seconds, fit = _timed(lambda: gam(frame, "cnt", [smooth("temp", k=k)]))
            record["parts"]["main"] = {
                "coef": [float(v) for v in fit.coefficients],
                "sp": float(fit.smoothing_parameter),
                "edf": float(fit.edf),
                "reml": float(fit.reml),
                "seconds": seconds,
            }
        elif kind == "lmm":
            groups = "student" if option["slopes"] else "class_id"
            slopes = ["grade"] if option["slopes"] else None
            method = option["method"]
            seconds, fit = _timed(lambda: fit_mixed(frame, "read", STAR_X, groups=groups, slopes=slopes, method=method))
            se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
            record["parts"]["main"] = {
                "terms": list(fit.names),
                "coef": [float(v) for v in fit.coefficients],
                "se": [float(v) for v in se],
                "re": np.asarray(fit.group_covariance, dtype=float).tolist(),
                "sigma2": float(fit.residual_variance),
                "loglik": float(fit.log_likelihood),
                "n_iter": int(fit.n_iter),
                "function_evals": int(fit.function_evals),
                "seconds": seconds,
            }
        elif kind == "ttest":
            x, y = _by_outcome(frame, "meanbp")
            for label, equal in (("welch", False), ("pooled", True)):
                seconds, result = _timed(lambda equal=equal: t_test(x, y, var_equal=equal))
                record["parts"][label] = {**_htest_payload(result), "seconds": seconds}
        elif kind == "wilcox":
            x, y = _by_outcome(frame, "meanbp")
            seconds, result = _timed(lambda: wilcox_test(x, y, conf_int=True))
            record["parts"]["main"] = {**_htest_payload(result), "seconds": seconds}
        elif kind == "prop":
            died = frame["hospdead"].to_numpy() == 1
            female = frame["sex"].to_numpy() == "female"
            counts = [int(np.sum(died & female)), int(np.sum(died & ~female))]
            totals = [int(np.sum(female)), int(np.sum(~female))]
            seconds, result = _timed(lambda: prop_test(counts, totals))
            record["parts"]["main"] = {**_htest_payload(result), "seconds": seconds}
        elif kind == "padj":
            raw = [t_test(*_by_outcome(frame, name)).p_value for name in HTEST_X]
            for method in option["methods"]:
                seconds, adjusted = _timed(lambda method=method: p_adjust(raw, method))
                record["parts"][method] = {"adjusted": [float(v) for v in adjusted], "seconds": seconds}
        elif kind == "roc":
            truth = frame["hospdead"]
            score1 = fit_glm(frame, "hospdead", SURV_X, family="binomial").predict(kind="link")
            score2 = fit_glm(frame, "hospdead", ROC_X2, family="binomial").predict(kind="link")

            def _roc():
                r1 = roc_curve(truth, score1)
                r2 = roc_curve(truth, score2)
                return r1, r2, r1.ci_auc(), r2.ci_auc(), roc_test(r1, r2)

            seconds, (r1, r2, ci1, ci2, test) = _timed(_roc)
            record["parts"]["main"] = {
                "auc": [float(r1.auc), float(r2.auc)],
                "var": [r1.var_auc(), r2.var_auc()],
                "ci": [ci1[0], ci1[2], ci2[0], ci2[2]],
                "stat": float(test.statistic),
                "p": float(test.p_value),
                "seconds": seconds,
            }
        elif kind in {"val_lrm", "val_cph"}:
            b = int(option["B"])
            seed = manifest["seed"]
            if kind == "val_lrm":
                seconds, table = _timed(lambda: validate_logistic(frame, "hospdead", LRM_X, B=b, seed=seed))
            else:
                seconds, table = _timed(lambda: validate_cox(frame, "d_time", "death", SURV_X, B=b, seed=seed))
            record["parts"]["main"] = {
                "terms": table["index"].to_list(),
                "index_orig": [float(v) for v in table["index_orig"].to_list()],
                # Bootstrap columns are kept for reading only; the resamples differ from R.
                "index_corrected": [float(v) for v in table["index_corrected"].to_list()],
                "seconds": seconds,
            }
        elif kind == "cal_lrm":
            b = int(option["B"])
            seconds, cal = _timed(
                lambda: calibrate_logistic(frame, "hospdead", LRM_X, B=b, seed=manifest["seed"])
            )
            record["parts"]["main"] = {
                "predy": [float(v) for v in cal.table["predy"].to_list()],
                "calibrated_orig": [float(v) for v in cal.table["calibrated_orig"].to_list()],
                "seconds": seconds,
            }
        elif kind == "cal_cph":
            b = int(option["B"])
            seconds, cal = _timed(
                lambda: calibrate_cox(
                    frame, "d_time", "death", SURV_X, u=float(option["u"]), m=int(option["m"]), B=b, seed=manifest["seed"]
                )
            )
            record["parts"]["main"] = {
                field: [float(v) for v in cal.table[field].to_list()]
                for field in ("mean_predicted", "KM", "std_err", "index_orig")
            } | {"seconds": seconds}
        elif kind == "std_cox":
            exposure = option["exposure"]
            formula = "Surv(d_time, death) ~ " + " + ".join(STD_COX_X)
            times = manifest["km_times"]
            seconds, fit = _timed(
                lambda: standardize_cox(frame, formula, values={exposure: option["values"]}, times=times)
            )
            record["parts"]["main"] = {
                "estimate": np.asarray(fit.estimates, dtype=float).ravel().tolist(),
                "cov": np.concatenate([np.asarray(c, dtype=float).ravel() for c in fit.covariances]).tolist(),
                "seconds": seconds,
            }
        elif kind == "cuminc":
            times = manifest["km_times"]
            specs = (("main", None), ("strata", "ca"))
            for label, strata in specs:
                seconds, fit = _timed(
                    lambda strata=strata: cumulative_incidence(frame, "d_time", "cause", "sex", strata=strata)
                )
                tests = fit.tests.sort("cause")
                part = {
                    "stat": [float(v) for v in tests["statistic"].to_list()],
                    "p": [float(v) for v in tests["p_value"].to_list()],
                    "seconds": seconds,
                }
                if strata is None:
                    table = fit.at(times).with_columns(
                        (pl.col("group") + " " + pl.col("cause")).alias("key")
                    ).sort(["key", "time"])
                    part["estimate"] = [float(v) for v in table["estimate"].to_list()]
                    part["variance"] = [float(v) for v in table["variance"].to_list()]
                record["parts"][label] = part
        elif kind == "crr":
            seconds, fit = _timed(
                lambda: fine_gray_regression(frame, "d_time", "cause", SURV_X, cause=option["cause"])
            )
            record["parts"]["main"] = {
                "terms": list(fit.names),
                "coef": [float(v) for v in fit.coefficients],
                "se": [float(v) for v in np.sqrt(np.diag(fit.covariance))],
                "loglik": float(fit.log_likelihood),
                "seconds": seconds,
            }
        elif kind == "gamm":
            k = int(option["k"])
            seconds, fit = _timed(
                lambda: gamm(
                    frame,
                    "hospdead",
                    [smooth("meanbp", k=k, basis="cr")],
                    random="(1 | dzgroup)",
                    predictors=["age"],
                    family="binomial",
                )
            )
            tidy = fit.tidy()
            effect = fit.partial_effect("meanbp", GAMM_GRID)
            record["parts"]["main"] = {
                "terms": tidy["term"].to_list(),
                "coef": [float(v) for v in tidy["estimate"].to_list()],
                "re": float(fit.variance_table()["variance"][0]),
                "loglik": float(fit.log_likelihood),
                "smooth": [float(v) for v in effect["fit"].to_list()],
                # edf is recorded, not compared: gamm4 0.2-6 reports it from a mis-pivoted Cholesky.
                "edf": [float(v) for v in fit.edf],
                "seconds": seconds,
            }
        elif kind == "glmm":
            family = option["family"]
            formula = GLMM_FORMULA[family]
            seconds, fit = _timed(lambda: fit_mixed(frame, formula, family=family, engine="laplace"))
            payload = {
                "terms": list(fit.names),
                "coef": [float(v) for v in fit.coefficients],
                "se": [float(v) for v in np.sqrt(np.diag(fit.covariance))],
                "loglik": float(fit.log_likelihood),
                "re": float(np.asarray(fit.group_covariance, dtype=float)[0, 0]),
                "seconds": seconds,
            }
            if family == "negative_binomial":
                payload["theta"] = float(fit.theta)
            record["parts"]["main"] = payload
        elif kind == "ctree":
            seconds, tree = _timed(lambda: conditional_tree(frame, "hospdead", SURV_X))
            splits: list[str] = []
            _split_labels(tree.root, splits)
            tests = tree.tests()
            record["parts"]["main"] = {
                "splits": splits,
                "n_terminal": [tree.n_terminal()],
                "stat": [float(v) for v in tests["statistic"].to_list()],
                "p": [float(v) for v in tests["p_value"].to_list()],
                "pred": [float(v) for v in tree.predict(frame)],
                "seconds": seconds,
            }
        elif kind == "mice":
            work = frame.select(MI_COLUMNS)
            m = int(option["m"])
            maxit = int(option["maxit"])

            def _mice():
                imputed = impute_chained(work, m=m, n_iter=maxit, seed=manifest["seed"])
                pooled = imputed.pool(lambda d: fit_glm(d, "hospdead", MI_X, family="binomial"))
                return imputed, pooled

            seconds, (imputed, pooled) = _timed(_mice, option.get("repeats"))
            record["parts"]["main"] = {
                # Only the deterministic parts are compared: which columns are
                # imputed, by which method, and how many cells. The draws differ.
                "methods": sorted(f"{name}:{method}" for name, method in imputed.methods.items()),
                "nmis": [int(imputed.missing[name]) for name in sorted(imputed.missing)],
                "pooled_terms": pooled["term"].to_list(),
                "pooled_estimate": [float(v) for v in pooled["estimate"].to_list()],
                "seconds": seconds,
            }
        else:
            raise ValueError(f"unknown kind {kind}")
    except Exception as exc:  # noqa: BLE001 - one task must not stop the suite
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["trace"] = traceback.format_exc(limit=8)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", default=None, help="run only these task IDs")
    args = parser.parse_args()
    manifest = json.loads((CACHE / "prepared" / "manifest.json").read_text())
    frames = {name: pl.read_parquet(info["parquet"]) for name, info in manifest["slices"].items()}
    out = []
    for task in manifest["tasks"]:
        if args.task is not None and task["id"] not in args.task:
            continue
        print(f"python {task['id']} {task['slice']}", flush=True)
        out.append(run_task(task, frames[task["slice"]], manifest))
    dest = CACHE / "results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "python.json").write_text(json.dumps(out))
    failed = [row["id"] + " " + row["slice"] for row in out if row["error"]]
    print("python failed", len(failed))
    for name in failed:
        print(" ", name)


if __name__ == "__main__":
    main()
