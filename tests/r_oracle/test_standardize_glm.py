"""GLM standardization against ``marginaleffects`` and a hand-coded stacked sandwich.

Fixture: ``fixtures/standardize_glm.json`` from ``scripts/standardize_glm.R``.
``covariates="fixed"`` is checked against marginaleffects 0.18
``avg_predictions`` / ``avg_comparisons``. ``covariates="sampled"`` is checked
against an R re-implementation of the ``stdReg2::standardize_glm`` estimator
(stdReg2 itself is not installed).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_glm, standardize_glm

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "standardize_glm.json").read_text())
CASES = sorted(FIXTURE["cases"])
RUNS = [
    (name, j) for name in CASES for j in range(len(FIXTURE["cases"][name]["variances"]))
]


def _run_id(run: tuple[str, int]) -> str:
    name, j = run
    spec = FIXTURE["cases"][name]["variances"][j]
    return "-".join(
        str(part)
        for part in (name, spec["covariates"], spec["vcov"], spec["cluster"])
        if part
    )


@cache
def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv")


@cache
def _fit(name: str, j: int):
    case = FIXTURE["cases"][name]
    spec = case["variances"][j]
    return standardize_glm(
        _data(case["data"]),
        case["formula"],
        values={case["exposure"]: list(case["values"])},
        family=case["family"],
        weights=case.get("weights"),
        vcov=spec["vcov"],
        cluster=spec["cluster"],
        covariates=spec["covariates"],
    )


def test_versions() -> None:
    assert FIXTURE["r_version"].startswith("R version 4.3")
    assert FIXTURE["marginaleffects"] == "0.18.0"


@pytest.mark.parametrize("run", RUNS, ids=[_run_id(r) for r in RUNS])
def test_estimates_and_covariance(run: tuple[str, int]) -> None:
    want = FIXTURE["cases"][run[0]]["variances"][run[1]]
    got = _fit(*run)
    np.testing.assert_allclose(
        got.estimates, np.atleast_1d(want["estimates"]), rtol=1e-6
    )
    cov = np.asarray(want["covariance"], dtype=float)
    # marginaleffects differentiates numerically (Richardson), which leaves
    # about 1e-10 of absolute error; small off-diagonal terms need an atol.
    np.testing.assert_allclose(
        got.covariance, cov, rtol=1e-6, atol=1e-6 * float(np.max(np.diag(cov)))
    )


TABLES = [
    (name, j, t)
    for name, j in RUNS
    for t in range(len(FIXTURE["cases"][name]["variances"][j].get("tables", [])))
]


@pytest.mark.parametrize(
    "spec",
    TABLES,
    ids=[
        f"{_run_id((n, j))}-{FIXTURE['cases'][n]['variances'][j]['tables'][t]['contrast']}"
        f"-{FIXTURE['cases'][n]['variances'][j]['tables'][t]['ci']}"
        for n, j, t in TABLES
    ],
)
def test_tidy(spec: tuple[str, int, int]) -> None:
    name, j, t = spec
    case = FIXTURE["cases"][name]
    want = case["variances"][j]["tables"][t]
    contrast = want["contrast"]
    reference = case["values"][0] if contrast else None
    table = _fit(name, j).tidy(contrast=contrast, reference=reference, ci=want["ci"])
    if contrast is not None:
        # marginaleffects reports only the non-reference rows.
        ref_row = table.row(0, named=True)
        null = 0.0 if contrast == "difference" else 1.0
        assert ref_row["estimate"] == null and ref_row["std_error"] == 0.0
        table = table.slice(1)
    assert table[case["exposure"]].to_list() == list(np.atleast_1d(want["value"]))
    columns = ["estimate", "conf_low", "conf_high"] + (
        ["std_error"] if want["std_error"] is not None else []
    )
    for column in columns:
        np.testing.assert_allclose(
            table[column].to_numpy(), np.atleast_1d(want[column]), rtol=1e-6, atol=1e-12
        )


def test_point_estimate_is_mean_of_counterfactual_predictions() -> None:
    data = _data("stdglm_a")
    got = standardize_glm(data, "y ~ trt * age + sex", values={"trt": [1]})
    fit = fit_glm(data, "y ~ trt * age + sex", family="binomial")
    expected = fit.predict(data.with_columns(pl.lit(1).alias("trt"))).mean()
    np.testing.assert_allclose(got.estimates[0], expected, rtol=1e-12)


def test_sampled_adds_covariate_variance() -> None:
    # The stacked sandwich adds the spread of the covariates to an HC0-type
    # coefficient part, so it is larger than the fixed-covariate HC0 variance.
    data = _data("stdglm_a")
    fixed = standardize_glm(
        data, "y ~ trt + age + sex", values={"trt": [0, 1]}, vcov="HC0"
    )
    sampled = standardize_glm(
        data, "y ~ trt + age + sex", values={"trt": [0, 1]}, covariates="sampled"
    )
    assert np.all(np.diag(sampled.covariance) > np.diag(fixed.covariance))


def test_rejects_bad_options() -> None:
    data = _data("stdglm_a")
    with pytest.raises(ValueError, match="not in the formula"):
        standardize_glm(data, "y ~ age + sex", values={"trt": [0, 1]})
    with pytest.raises(ValueError, match="leave vcov unset"):
        standardize_glm(
            data,
            "y ~ trt + age",
            values={"trt": [0, 1]},
            vcov="HC3",
            covariates="sampled",
        )
    with pytest.raises(ValueError, match="cluster needs a sandwich"):
        standardize_glm(
            data, "y ~ trt + age", values={"trt": [0, 1]}, vcov="model", cluster="site"
        )
    with pytest.raises(ValueError, match="unseen levels"):
        standardize_glm(data, "y ~ arm + age", values={"arm": ["a", "z"]})
    fit = _fit("binary_main", 0)
    with pytest.raises(ValueError, match="ci='log'"):
        fit.tidy(contrast="difference", reference=0, ci="log")
    with pytest.raises(ValueError, match="reference must be one of"):
        fit.tidy(contrast="ratio", reference=2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _fit("poisson_offset", 0).tidy(contrast="odds_ratio", reference=0)
