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
    got = table["edf"].to_numpy()
    # A smooth shrunk to its linear part (edf near 1) sits on a flat REML
    # ridge: the criterion keeps falling as its smoothing parameter grows.
    # mgcv stops on a gradient tolerance (a_by: gradient -1.6e-4 at edf
    # 1.0024), so its edf there is a stopping point, not the optimum. Ours
    # may go further along the ridge, so only bound it from both sides.
    flat = edf < 1.01
    np.testing.assert_allclose(got[~flat], edf[~flat], atol=2e-3)
    assert np.all(got[flat] >= 1 - 1e-6)
    assert np.all(got[flat] <= edf[flat] + 2e-3)
    np.testing.assert_allclose(table["statistic"].to_numpy(), np.atleast_1d(case["chisq"]), rtol=1e-2)
    assert fit.glance()["deviance"][0] == pytest.approx(case["deviance"], rel=1e-5)
