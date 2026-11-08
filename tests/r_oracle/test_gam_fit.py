"""``gam`` bases, tensors, ``by``, ``select``, and GCV against mgcv.

Fixture: ``fixtures/gam_fit.json`` from ``scripts/gam_fit.R`` on
``data/gam_b.csv``. Each case compares the total edf, the edf of each smooth,
the deviance, and link-scale predictions with standard errors on a fixed grid.

Cases with one smoothing parameter agree to optimizer precision. Cases with
several are compared more loosely, because the criterion is flat near its
optimum: there statract's criterion must be within ``CRITERION_SLACK`` of
mgcv's. statract bounds each log smoothing parameter, so a factor ``by``
level that mgcv shrinks to a straight line (edf 1) keeps an edf a little
above 1 here; that is most of the slack.

A factor ``by`` smooth is one block in statract (one edf and one test over
all levels), so its edf is compared with the sum of mgcv's per-level edfs.

The criterion of a ``tp`` smooth differs from mgcv's by a constant that does
not depend on the smoothing parameter (the scaling of the unpenalized
columns enters the REML and ML scores), so it is not compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import gam, smooth
from statract.models.gam import tensor_interaction, tensor_smooth

HERE = Path(__file__).resolve().parent
_DATA = json.loads((HERE / "fixtures" / "gam_fit.json").read_text())
FIXTURE = _DATA["cases"]
DATA = pl.read_csv(HERE / "data" / "gam_b.csv")
GRID = pl.DataFrame(_DATA["grid"], strict=False).with_columns(pl.col("x", "z", "noise", "hour").cast(pl.Float64))
CRITERION_SLACK = 0.02

S = smooth
# name: (smooths, keyword arguments, mgcv edf indices summed per statract block, exact)
CASES = {
    "tp": ([S("x", basis="tp")], {}, None, True),
    "tp_ml": ([S("x", basis="tp")], {"method": "ml"}, None, False),
    "cc": ([S("hour", basis="cc")], {}, None, True),
    "ps": ([S("x", basis="ps")], {}, None, True),
    "re": ([S("x"), S("site", basis="re")], {}, None, True),
    "te": ([tensor_smooth("x", "z", k=(5, 5))], {}, None, False),
    "ti": ([S("x", k=5), S("z", k=5), tensor_interaction("x", "z", k=(5, 5))], {}, None, False),
    "by_factor": ([S("x", k=8, by="grp")], {"predictors": ["grp"]}, [[0, 1, 2]], False),
    "by_numeric": ([S("x", k=8, by="z")], {}, None, True),
    "select": ([S("x"), S("noise")], {"select": True}, None, False),
    "select_te": ([S("noise", k=8), tensor_smooth("x", "z", k=(5, 5))], {"select": True}, None, False),
    "select_by_factor": ([S("x", k=8, by="grp")], {"predictors": ["grp"], "select": True}, [[0, 1, 2]], False),
    "gcv": ([S("x")], {"method": "gcv"}, None, True),
    "gcv_gamma": ([S("x")], {"method": "gcv", "gamma": 1.4}, None, True),
    "poisson_by_cc": (
        [S("x", k=8, by="grp"), S("hour", basis="cc", k=8)],
        {"predictors": ["grp"], "family": "poisson", "outcome": "count"},
        [[0, 1, 2], [3]],
        False,
    ),
    "poisson_select": (
        [S("x", k=8), S("noise", k=8)],
        {"family": "poisson", "select": True, "outcome": "count"},
        None,
        False,
    ),
}


def _fit(name: str):
    smooths, kwargs, groups, exact = CASES[name]
    kwargs = dict(kwargs)
    outcome = kwargs.pop("outcome", "y")
    return gam(DATA, outcome, smooths, **kwargs), groups, exact


@pytest.mark.parametrize("name", list(CASES))
def test_gam_matches_mgcv(name: str) -> None:
    want = FIXTURE[name]
    fit, groups, exact = _fit(name)
    want_edf = np.asarray(want["edf"])
    if groups is not None:
        want_edf = np.array([want_edf[g].sum() for g in groups])
    edf_tol, dev_tol, pred_atol = (1e-4, 1e-6, 1e-4) if exact else (5e-3, 2e-4, 2e-2)
    np.testing.assert_allclose(fit.edf, want["total_edf"], rtol=edf_tol)
    np.testing.assert_allclose(fit.block_edf, want_edf, rtol=edf_tol, atol=edf_tol)
    np.testing.assert_allclose(fit.glance()["deviance"][0], want["deviance"], rtol=dev_tol)
    eta, se = fit.predict(GRID, kind="link", se=True)
    np.testing.assert_allclose(eta, want["predict"], atol=pred_atol)
    np.testing.assert_allclose(se, want["predict_se"], rtol=10 * edf_tol, atol=pred_atol / 10)


@pytest.mark.parametrize("name", [n for n in CASES if not n.startswith("tp")])
def test_criterion_matches_mgcv(name: str) -> None:
    want = FIXTURE[name]["criterion"]
    fit, _groups, exact = _fit(name)
    if exact:
        np.testing.assert_allclose(fit.reml, want, rtol=1e-6)
    else:
        assert fit.reml <= want + CRITERION_SLACK


def test_smooth_table_with_select_and_factor_by() -> None:
    """``select=True`` on a factor ``by`` adds one penalty per level; the table still has one row per level."""
    fit, _groups, _exact = _fit("select_by_factor")
    table = fit.smooth_table()
    assert table.height == 3
    np.testing.assert_allclose(table["edf"].to_numpy(), fit.block_edf[0])
