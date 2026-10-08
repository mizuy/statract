"""Two-way cluster-robust Cox variance against sandwich::vcovCL.

Fixture: ``fixtures/cox_two_way.json`` from ``scripts/cox_two_way.R``.
vcovCL fails on a coxph with one coefficient; the R script writes the same
inclusion-exclusion out by hand for those cases and checks the hand version
against vcovCL on the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import cluster_covariance, coefficient_test, cox_ph

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "cox_two_way.json").read_text())
ARM = pl.Enum(["control", "low", "high"])


def _data(name: str) -> pl.DataFrame:
    return pl.read_csv(HERE / "data" / f"{name}.csv").with_columns(pl.col("arm").cast(ARM))


def _fit(case: dict):
    return cox_ph(_data(case["sample"]), case["formula"], ties=case["ties"], weights=case["weights"])


@pytest.mark.parametrize("name", sorted(FIXTURE["cases"]))
def test_cluster_covariance_matches_vcovcl(name: str) -> None:
    case = FIXTURE["cases"][name]
    fit = _fit(case)
    assert fit.names == list(np.atleast_1d(case["terms"]))
    np.testing.assert_allclose(fit.coefficients, np.atleast_1d(case["coefficients"]), rtol=1e-8)
    cov = cluster_covariance(
        fit, case["cluster"], data=_data(case["sample"]), kind=case["type"], adjust=case["adjust"]
    )
    want = np.atleast_2d(np.asarray(case["covariance"], dtype=float))
    np.testing.assert_allclose(cov, want, rtol=1e-6, atol=1e-12)


def test_one_cluster_without_adjustment_is_the_coxph_robust_variance() -> None:
    data = _data("cox2way_a")
    fit = cox_ph(data, "Surv(time, status) ~ x + arm")
    robust = cox_ph(data, "Surv(time, status) ~ x + arm", cluster="id_patient")
    cov = cluster_covariance(fit, "id_patient", data=data, adjust=False)
    np.testing.assert_allclose(cov, robust.covariance, rtol=1e-10)


def test_two_way_table_and_guards() -> None:
    data = _data("cox2way_a")
    fit = cox_ph(data, "Surv(time, status) ~ x")
    cov = cluster_covariance(fit, ["id_patient", "e_examiner"], data=data)
    table = coefficient_test(fit, cov)
    np.testing.assert_allclose(table["std_error"].to_numpy(), np.sqrt(np.diag(cov)))
    # Labels as an array aligned with the fitted rows give the same answer.
    labels = np.column_stack(
        [data[c].gather(fit.row_index.tolist()).to_numpy() for c in ("id_patient", "e_examiner")]
    )
    np.testing.assert_allclose(cluster_covariance(fit, labels), cov, rtol=1e-12)
    with pytest.raises(ValueError, match="HC0 or HC1"):
        cluster_covariance(fit, ["id_patient"], data=data, kind="HC3")
    gaps = data.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("e_examiner")).alias("e_examiner"))
    with pytest.raises(ValueError, match="missing"):
        cluster_covariance(cox_ph(gaps, "Surv(time, status) ~ x"), ["id_patient", "e_examiner"], data=gaps)
