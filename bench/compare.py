"""Join the Python and R payloads and write the benchmark table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CACHE = Path("/tmp/statract-bench")
OUT = Path(__file__).resolve().parents[1] / "docs" / "models" / "benchmarks" / "comparison.csv"


def _max_rel(left, right) -> float:
    left_arr = np.atleast_1d(np.asarray(left, dtype=float))
    right_arr = np.atleast_1d(np.asarray(right, dtype=float))
    scale = np.maximum(np.abs(right_arr), 1e-12)
    return float(np.max(np.abs(left_arr - right_arr) / scale))


def _max_corr(left, right) -> float:
    """Covariance difference on the correlation scale, |dV_ij| / sqrt(V_ii V_jj).

    Equal to the relative error on the diagonal. Off the diagonal it does not
    divide by a near-zero covariance, which the elementwise ratio does.
    """
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    sd = np.sqrt(np.abs(np.diag(right_arr)))
    scale = np.maximum(np.outer(sd, sd), 1e-300)
    return float(np.max(np.abs(left_arr - right_arr) / scale))


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _align(py: dict, r: dict, fields: list[str]) -> tuple[dict, dict, str | None]:
    if "terms" not in py:
        return py, r, None
    if py["terms"] == r["terms"]:
        return py, r, None
    if set(py["terms"]) != set(r["terms"]):
        return py, r, "term names differ: " + ",".join(py["terms"]) + " vs " + ",".join(r["terms"])
    order = [r["terms"].index(name) for name in py["terms"]]
    aligned = dict(r)
    aligned["terms"] = list(py["terms"])
    for field in fields:
        if field not in r:
            continue
        values = r[field]
        if field == "cov":
            matrix = np.asarray(values, dtype=float)[np.ix_(order, order)]
            aligned["cov"] = matrix
        else:
            aligned[field] = [values[i] for i in order]
    return py, aligned, None


def _curve_vectors(payload: dict) -> dict[str, list[float]]:
    rows = sorted(payload["rows"], key=lambda row: (str(row.get("group")), float(row["time"])))
    return {
        "estimate": [row["estimate"] for row in rows],
        "low": [row["low"] for row in rows],
        "high": [row["high"] for row in rows],
    }


def _pairs(payload: dict) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in pair) for pair in payload.get("pairs", [])}


def _match_profiles(slice_name: str) -> list[tuple]:
    """Covariate row of each unit, so a swap between identical controls is a tie."""
    import sys

    import polars as pl

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_python import MATCH_X

    manifest = json.loads((CACHE / "prepared" / "manifest.json").read_text())
    frame = pl.read_parquet(manifest["slices"][slice_name]["parquet"])
    return frame.select(MATCH_X).rows()


def _pair_check(task_id: str, py: dict, r: dict, profiles: list[tuple] | None) -> dict:
    left = _pairs(py)
    right = _pairs(r)
    note = ""
    same = left == right
    if not same and profiles is not None:
        # Greedy matching takes the first of several equidistant controls. When
        # the controls have the same covariates, the order of equal distances is
        # a tie-break, not a different match.
        swapped = sorted(set(t for t, _ in left ^ right))
        left_map = dict(left)
        right_map = dict(right)
        if left_map.keys() == right_map.keys() and all(
            profiles[left_map[t]] == profiles[right_map[t]] for t in swapped
        ):
            same = True
            note = f"{len(swapped)} pairs take a control with identical covariates"
    if not same:
        note = f"python {len(left)} r {len(right)}, {len(left & right)} shared"
        if task_id == "m-mah":
            note += MAHALANOBIS_NOTE
    return {
        "quantity": "pairs",
        "error": 0.0 if same else 1.0,
        "criterion": "set",
        "tol": 0.0,
        "passed": same,
        "note": note,
    }


MAHALANOBIS_NOTE = (
    "; the pooled covariance of all factor levels is singular, so MatchIt's pivoted"
    " Cholesky depends on the LAPACK build. Pairs agree on x86-64 Linux reference LAPACK"
)


# Checks of the tasks added for the later functions (htest, ROC, rms, stdReg2,
# cmprsk, gamm4, glmmTMB, partykit, mice). Tolerances follow docs/models/vs-r.md.
# "equal" is exact equality of strings or integers. A field missing from both
# sides is skipped, so parts of one task may carry different fields.
LATER_CHECKS: dict[str, list[tuple[str, str, float]]] = {
    "ttest": [("stat", "rel", 1e-10), ("df", "rel", 1e-10), ("p", "abs", 1e-10), ("ci", "rel", 1e-10), ("estimate", "rel", 1e-10)],
    # The Hodges-Lehmann estimate and interval come from uniroot (tol 1e-4) on both sides.
    "wilcox": [("stat", "rel", 1e-10), ("p", "abs", 1e-10), ("estimate", "rel", 1e-6), ("ci", "rel", 1e-6)],
    "prop": [("stat", "rel", 1e-10), ("df", "rel", 1e-10), ("p", "abs", 1e-10), ("ci", "rel", 1e-10), ("estimate", "rel", 1e-10)],
    "padj": [("adjusted", "rel", 1e-8)],
    "roc": [("auc", "rel", 1e-10), ("var", "rel", 1e-8), ("ci", "rel", 1e-8), ("stat", "rel", 1e-8), ("p", "abs", 1e-8)],
    # Bootstrap resamples differ between R and numpy: only the apparent column.
    "val-lrm": [("index_orig", "abs", 1e-6)],
    "val-cph": [("index_orig", "abs", 1e-6)],
    "cal-lrm": [("predy", "abs", 1e-6), ("calibrated_orig", "abs", 1e-6)],
    "cal-cph": [("mean_predicted", "abs", 1e-6), ("KM", "abs", 1e-6), ("std_err", "abs", 1e-6), ("index_orig", "abs", 1e-6)],
    "std-cox": [("estimate", "rel", 1e-6), ("cov", "rel", 1e-6)],
    "cuminc": [("estimate", "rel", 1e-8), ("variance", "rel", 1e-8), ("stat", "rel", 1e-8), ("p", "abs", 1e-8)],
    "crr": [("coef", "rel", 1e-8), ("se", "rel", 1e-8), ("loglik", "rel", 1e-8)],
    # edf and SEs are left out: gamm4 0.2-6 reports them from a mis-pivoted Cholesky.
    "gamm": [("coef", "abs", 3e-5), ("re", "rel", 1e-4), ("loglik", "abs", 1e-5), ("smooth", "abs", 3e-5)],
    # Coefficients: rtol 1e-4 with atol 2e-5, as the glmm_count fixture (meanbp is near 0).
    "glmm-pois": [("coef", "close", 1e-4), ("se", "rel", 1e-4), ("loglik", "abs", 1e-6), ("re", "rel", 1e-3)],
    "glmm-nb": [("coef", "close", 1e-4), ("se", "rel", 1e-4), ("loglik", "abs", 1e-6), ("re", "rel", 1e-3), ("theta", "rel", 1e-4)],
    "ctree": [("splits", "equal", 0.0), ("n_terminal", "equal", 0.0), ("stat", "rel", 1e-6), ("p", "abs", 1e-6), ("pred", "rel", 1e-8)],
    # mice draws from R's generator: only the method per column and the missing counts.
    "mice": [("methods", "equal", 0.0), ("nmis", "equal", 0.0)],
}


CLOSE_ATOL = 2e-5


def _close_error(left, right, rtol: float) -> float:
    """max |dL| / (atol + rtol |R|); 1 is the edge of numpy's allclose."""
    left_arr = np.atleast_1d(np.asarray(left, dtype=float))
    right_arr = np.atleast_1d(np.asarray(right, dtype=float))
    return float(np.max(np.abs(left_arr - right_arr) / (CLOSE_ATOL + rtol * np.abs(right_arr))))


def _equal(left, right) -> bool:
    def norm(value):
        return [value] if not isinstance(value, list) else list(value)

    return norm(left) == norm(right)


def _checks(task_id: str) -> list[tuple[str, str, float]]:
    """Return (field, rule, tolerance). rule is rel, abs, corr (covariance), or equal."""
    if task_id in LATER_CHECKS:
        return LATER_CHECKS[task_id]
    if task_id.startswith("lmm"):
        # lme-python loses accuracy on STAR's uncentred birth year (cond(X)
        # near 7e6). Centring it brings theta and the log-likelihood onto
        # lmer's; uncentred, the gap is up to 1.2e-3 in the coefficients and
        # 4e-5 in the log-likelihood. Recheck when lme-python is updated.
        return [
            ("coef", "rel", 3e-3),
            ("se", "rel", 1e-4),
            ("re", "rel", 3e-3),
            ("sigma2", "rel", 1e-4),
            ("loglik", "abs", 1e-4),
        ]
    if task_id.startswith("gam"):
        return [("sp", "rel", 1e-3), ("edf", "rel", 1e-4), ("reml", "abs", 1e-6), ("coef", "rel", 1e-6)]
    if task_id in {"cl1", "cl2"}:
        # STAR's uncentred birth year leaves cond(X) near 7e6. lm/vcovCL and
        # statract both sit 1e-8 to 3e-7 from a high-precision reference.
        return [("cov", "corr", 1e-6)]
    if task_id in {"hc", "nw"} or task_id.startswith("hc"):
        return [("cov", "corr", 1e-8)]
    if task_id in {"wald", "lr", "bp", "reset", "dw", "bg", "lrk"} or task_id.startswith(("bp", "bg", "lrk")):
        return [("stat", "rel", 1e-6), ("p", "abs", 1e-6)]
    if task_id in {"km", "km-sex", "na"}:
        return [("estimate", "rel", 1e-6), ("low", "rel", 1e-6), ("high", "rel", 1e-6)]
    if task_id in {"m-logit", "m-mah"}:
        checks = [("pairs", "set", 0.0)]
        if task_id == "m-logit":
            checks.append(("smd_age", "rel", 1e-8))
        return checks
    if task_id in {"m-exact", "m-cem"}:
        return [("weights", "rel", 1e-8)]
    if task_id == "glm-gamma":
        return [("coef", "rel", 1e-6), ("se", "rel", 1e-6)]
    if task_id.startswith(("glm", "cox", "aft")):
        return [("coef", "rel", 1e-6), ("se", "rel", 1e-6), ("loglik", "rel", 1e-6)]
    if task_id.startswith("ols"):
        return [
            ("coef", "rel", 1e-6),
            ("se", "rel", 1e-6),
            ("t", "rel", 1e-6),
            ("p", "abs", 1e-6),
            ("loglik", "rel", 1e-6),
        ]
    raise KeyError(task_id)


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["id"], row["slice"]): row for row in rows}


def _quantity_rows(
    task_id: str, part_name: str, py: dict, r: dict, profiles: list[tuple] | None = None
) -> list[dict]:
    note = ""
    py, r, mismatch = _align(py, r, ["coef", "se", "t", "p", "cov", "index_orig", "index_corrected"])
    if mismatch:
        return [{"quantity": "terms", "error": "", "criterion": "names", "tol": 0, "passed": False, "note": mismatch}]
    if task_id in {"km", "km-sex", "na"}:
        py = _curve_vectors(py)
        r = _curve_vectors(r)
    rows = []
    for field, rule, tol in _checks(task_id):
        if field == "pairs":
            rows.append(_pair_check(task_id, py, r, profiles))
            continue
        if field not in py and field not in r and task_id in LATER_CHECKS:
            continue
        if field not in py or field not in r:
            rows.append(
                {
                    "quantity": field,
                    "error": "",
                    "criterion": rule,
                    "tol": tol,
                    "passed": False,
                    "note": "missing value",
                }
            )
            continue
        row_note = note
        if rule == "equal":
            passed = _equal(py[field], r[field])
            error = 0.0 if passed else 1.0
            if not passed:
                row_note = f"python {py[field]} r {r[field]}"[:300]
        elif rule == "close":
            error = _close_error(py[field], r[field], tol)
            passed = error <= 1.0
            if not passed:
                row_note = f"rtol {tol:g} + atol {CLOSE_ATOL:g}; error is the allclose ratio"
        elif rule == "abs":
            error = _max_abs(py[field], r[field])
            passed = error <= tol
        elif rule == "corr":
            error = _max_corr(py[field], r[field])
            passed = error <= tol
        else:
            error = _max_rel(py[field], r[field])
            passed = error <= tol
        label = field if part_name in {"", "main"} else f"{part_name}:{field}"
        rows.append({"quantity": label, "error": error, "criterion": rule, "tol": tol, "passed": passed, "note": row_note})
    if part_name not in {"", "main"}:
        for row in rows:
            if not str(row["quantity"]).startswith(part_name):
                row["quantity"] = f"{part_name}:{row['quantity']}"
    return rows


def main() -> None:
    python_rows = _index(json.loads((CACHE / "results" / "python.json").read_text()))
    r_rows = _index(json.loads((CACHE / "results" / "r.json").read_text()))
    keys = list(dict.fromkeys([*python_rows, *r_rows]))
    table = []
    for key in keys:
        py = python_rows.get(key)
        r = r_rows.get(key)
        task_id, slice_name = key
        n = (py or r).get("n")
        py_err = None if py is None else py.get("error")
        r_err = None if r is None else r.get("error")
        if py is None or r is None or py_err or r_err:
            table.append(
                {
                    "task": task_id,
                    "slice": slice_name,
                    "n": n,
                    "quantity": "run",
                    "error": "",
                    "criterion": "",
                    "tol": "",
                    "passed": False,
                    "python_s": "",
                    "r_s": "",
                    "ratio": "",
                    "note": " ".join(
                        bit for bit in ("" if py else "python missing", "" if r else "R missing", py_err or "", r_err or "") if bit
                    ),
                }
            )
            continue
        part_names = list(dict.fromkeys([*py["parts"], *r["parts"]]))
        for part in part_names:
            if part not in py["parts"] or part not in r["parts"]:
                table.append(
                    {
                        "task": task_id,
                        "slice": slice_name,
                        "n": n,
                        "quantity": part or "main",
                        "error": "",
                        "criterion": "",
                        "tol": "",
                        "passed": False,
                        "python_s": "",
                        "r_s": "",
                        "ratio": "",
                        "note": "part missing",
                    }
                )
                continue
            py_part = py["parts"][part]
            r_part = r["parts"][part]
            py_s = float(py_part["seconds"])
            r_s = float(r_part["seconds"])
            ratio = py_s / r_s if r_s else ""
            profiles = _match_profiles(slice_name) if task_id in {"m-logit", "m-mah"} else None
            for row in _quantity_rows(task_id, part, py_part, r_part, profiles):
                table.append(
                    {
                        "task": task_id,
                        "slice": slice_name,
                        "n": n,
                        "python_s": py_s,
                        "r_s": r_s,
                        "ratio": ratio,
                        **row,
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "slice", "n", "quantity", "error", "criterion", "tol", "passed", "python_s", "r_s", "ratio", "note"],
        )
        writer.writeheader()
        writer.writerows(table)
    failed = [row for row in table if row["passed"] is False or row["passed"] == "False"]
    print(f"rows {len(table)} failed {len(failed)} -> {OUT}")
    for row in failed:
        print(f"  {row['task']} {row['slice']} {row['quantity']} {row['note']}")


if __name__ == "__main__":
    main()
