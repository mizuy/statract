"""Diagnostic accuracy against ``epiR::epi.tests``.

Fixture: ``fixtures/diagnostic.json`` from ``scripts/diagnostic.R`` (epiR 2.0.67):
four tables (one with a zero cell), five interval methods, two levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statract import diagnostic_accuracy
from statract.models.diagnostic import EPIR_NAMES

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures" / "diagnostic.json").read_text())
CASES = FIXTURE["cases"]
_SPECIAL = {"NaN": np.nan, "NA": np.nan, "Inf": np.inf, "-Inf": -np.inf}

# epiR's agresti puts the adjusted proportion in ``est`` and uses it in the
# likelihood-ratio and Youden point estimates; statract keeps the plain ratio.
_AGRESTI_LIMITS_ONLY = {"ap", "tp", "se", "sp", "diag.ac", "pv.pos", "pv.neg", "p.rout", "p.rin", "p.tpdn", "p.tndp", "p.dntp", "p.dptn", "youden", "nndx"}


def _num(values) -> np.ndarray:
    return np.asarray([_SPECIAL.get(v, v) if isinstance(v, str) else v for v in values], dtype=float)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['table']}-{c['method']}-{c['conf_level']}")
def test_matches_epi_tests(case):
    c = case["counts"]
    acc = diagnostic_accuracy(tp=c["tp"], fp=c["fp"], fn=c["fn"], tn=c["tn"], method=case["method"], conf_level=case["conf_level"])
    ours = {EPIR_NAMES[row["statistic"]]: row for row in acc.tidy().iter_rows(named=True)}
    est, lo, hi = _num(case["est"]), _num(case["lower"]), _num(case["upper"])
    for i, name in enumerate(case["statistic"]):
        row = ours[name]
        if case["method"] == "agresti":
            if name not in _AGRESTI_LIMITS_ONLY:
                continue
            if name in ("youden", "nndx"):
                np.testing.assert_allclose([row["conf_low"], row["conf_high"]], [lo[i], hi[i]], rtol=1e-10, err_msg=name)
                continue
        else:
            np.testing.assert_allclose(row["estimate"], est[i], rtol=1e-10, err_msg=name)
        np.testing.assert_allclose([row["conf_low"], row["conf_high"]], [lo[i], hi[i]], rtol=1e-10, atol=1e-14, err_msg=name)


def test_arrays_and_counts_agree():
    rng = np.random.default_rng(1)
    d = rng.random(400) < 0.2
    t = np.where(d, rng.random(400) < 0.8, rng.random(400) < 0.15)
    t = t.astype(float)
    t[[3, 50]] = np.nan
    acc = diagnostic_accuracy(t, d)
    keep = ~np.isnan(t)
    tt, dd = t[keep] == 1, d[keep]
    assert (acc.tp, acc.fp, acc.fn, acc.tn) == (
        int(np.sum(tt & dd)),
        int(np.sum(tt & ~dd)),
        int(np.sum(~tt & dd)),
        int(np.sum(~tt & ~dd)),
    )
    assert acc.n == 398


def test_predictive_values_at_study_prevalence():
    acc = diagnostic_accuracy(tp=670, fp=202, fn=74, tn=640)
    pv = acc.predictive_values((670 + 74) / acc.n)
    tidy = {r["statistic"]: r["estimate"] for r in acc.tidy().iter_rows(named=True)}
    np.testing.assert_allclose(pv["ppv"][0], tidy["ppv"], rtol=1e-12)
    np.testing.assert_allclose(pv["npv"][0], tidy["npv"], rtol=1e-12)


def test_bad_input():
    with pytest.raises(ValueError):
        diagnostic_accuracy([1, 0, 2], [1, 0, 1])
    with pytest.raises(ValueError):
        diagnostic_accuracy(tp=1, fp=2, fn=3)
    with pytest.raises(ValueError):
        diagnostic_accuracy(tp=1, fp=2, fn=3, tn=4, method="wald")
