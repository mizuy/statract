"""Deviance summaries and nested-model comparisons against mgcv.

Fixture: ``fixtures/gam_summary.json`` from ``scripts/gam_summary.R``.
``compare_gams`` takes the df change from the plain edf; mgcv corrects the
edf for smoothing-parameter uncertainty, so only the deviance change is
compared, and the p-value is checked against its own definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from statract import compare_gams, gam, smooth

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "gam_summary.json").read_text())["cases"]
BINARY = pl.read_csv(HERE / "data" / "gamm_binary_a.csv").with_columns(off=(pl.col("age") / 50).log())
GAUSS = pl.read_csv(HERE / "data" / "gam_a.csv")


def _check(fit, want: dict, rtol: float = 1e-4) -> None:
    row = fit.glance().row(0, named=True)
    for key in ("deviance", "null_deviance", "deviance_explained", "adj_r_squared", "edf"):
        np.testing.assert_allclose(row[key], want[key], rtol=rtol, err_msg=key)


def test_binomial_summary_and_chisq_comparison() -> None:
    case = FIXTURE["binomial"]
    small = gam(BINARY, "y", [smooth("pre_size_mm", k=10)], family="binomial")
    large = gam(BINARY, "y", [smooth("pre_size_mm", k=10), smooth("age", k=10)], family="binomial")
    _check(small, case["small"])
    # The two-smooth REML optimum is flat; mgcv stops about 1e-3 edf away.
    _check(large, case["large"], rtol=5e-4)
    table = compare_gams(small, large)
    row = table.row(1, named=True)
    np.testing.assert_allclose(row["deviance_change"], case["deviance_change"], atol=5e-3)
    np.testing.assert_allclose(row["df"], large.edf - small.edf)
    np.testing.assert_allclose(row["p_value"], stats.chi2.sf(row["deviance_change"], row["df"]))
    assert table["p_value"][0] is None


def test_weighted_gaussian_summary_and_f_comparison() -> None:
    case = FIXTURE["gaussian_weighted"]
    small = gam(GAUSS, "y", [smooth("x", k=10)], weights="w")
    large = gam(GAUSS, "y", [smooth("x", k=10), smooth("z", k=10)], weights="w")
    _check(small, case["small"])
    _check(large, case["large"])
    row = compare_gams(small, large).row(1, named=True)
    np.testing.assert_allclose(row["deviance_change"], case["deviance_change"], rtol=1e-4)
    scale = large.glance()["scale"][0]
    f_value = row["deviance_change"] / row["df"] / scale
    np.testing.assert_allclose(row["statistic"], f_value)
    np.testing.assert_allclose(row["p_value"], stats.f.sf(f_value, row["df"], row["residual_df"]))


@pytest.mark.parametrize(
    ("name", "build", "rtol"),
    [
        ("poisson_offset", lambda: gam(BINARY, "n", [smooth("age", k=8)], family="poisson", offset="off"), 1e-4),
        ("gaussian", lambda: gam(GAUSS, "y", [smooth("x", k=10)]), 1e-4),
        # The smooth is nearly linear (edf near 2), where the REML optimum is flat.
        ("gamma", lambda: gam(GAUSS, "cost", [smooth("x", k=8)], family="gamma"), 5e-4),
    ],
)
def test_single_model_summary(name: str, build, rtol: float) -> None:
    _check(build(), FIXTURE[name], rtol=rtol)


def test_compare_gams_rejects_mixed_families() -> None:
    gaussian = gam(GAUSS, "y", [smooth("x", k=10)])
    gamma = gam(GAUSS, "cost", [smooth("x", k=8)], family="gamma")
    with pytest.raises(ValueError, match="family"):
        compare_gams(gaussian, gamma)
