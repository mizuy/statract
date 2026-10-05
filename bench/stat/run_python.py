"""Time the Python estimators and write numeric payloads for the R comparison."""

from __future__ import annotations

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
    cluster_covariance,
    cox_ph,
    durbin_watson_test,
    fit_glm,
    fit_mixed,
    fit_ols,
    gam,
    hc_covariance,
    likelihood_ratio_test,
    log_rank,
    match_sample,
    newey_west_covariance,
    ramsey_reset_test,
    smooth,
    survival_curve,
    wald_test,
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


def _timed(fn):
    fn()
    samples = []
    result = None
    for _ in range(REPEATS):
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
        else:
            raise ValueError(f"unknown kind {kind}")
    except Exception as exc:  # noqa: BLE001 - one task must not stop the suite
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["trace"] = traceback.format_exc(limit=8)
    return record


def main() -> None:
    manifest = json.loads((CACHE / "prepared" / "manifest.json").read_text())
    frames = {name: pl.read_parquet(info["parquet"]) for name, info in manifest["slices"].items()}
    out = []
    for task in manifest["tasks"]:
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
