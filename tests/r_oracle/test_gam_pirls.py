"""Binomial GAMs whose REML optimum is a weakly penalized fit.

Fixture: ``fixtures/gam_pirls.json`` from ``scripts/gam_pirls.R`` (mgcv,
synthetic data). P-IRLS without step halving diverged at small smoothing
parameters, so the search stopped short and the edf came out too low.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import gam, smooth

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "gam_pirls.json").read_text())
CASES = sorted(FIXTURE["cases"])


@pytest.mark.parametrize("name", CASES)
def test_gam_binomial_matches_mgcv(name: str) -> None:
    case = FIXTURE["cases"][name]
    df = pl.read_csv(HERE / "data" / f"{case['data']}.csv")
    if case["by"]:
        fit = gam(df, "y", [smooth("diameter", k=5, basis="tp", by="tech")], predictors=["tech"],
                  family="binomial", method="reml")
    else:
        fit = gam(df.filter(pl.col("tech") == "EMR"), "y", [smooth("diameter", k=5, basis="tp")],
                  family="binomial", method="reml")
    table = fit.smooth_table()
    edf = np.atleast_1d(case["edf"])
    np.testing.assert_allclose(table["edf"].to_numpy(), edf, atol=2e-3)
    np.testing.assert_allclose(table["statistic"].to_numpy(), np.atleast_1d(case["chisq"]), rtol=1e-2)
    assert fit.glance()["deviance"][0] == pytest.approx(case["deviance"], rel=1e-5)
