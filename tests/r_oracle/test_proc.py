"""statract.models.roc against pROC 1.18.5 (fixture from tests/r_oracle/scripts/proc.R)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from statract.models.roc import roc_curve, roc_test

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "proc.json").read_text())
SAMPLES = sorted(FIXTURE["samples"])
CURVES = [("roc1", "y", "score1"), ("roc2", "y", "score2"), ("roc_other", "y_other", "score_other")]


def _num(v):
    """Decode jsonlite ``na="string"`` values."""
    if isinstance(v, list):
        return [_num(x) for x in v]
    return {"NA": None, "Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan}.get(v, v) if isinstance(v, str) else v


def _arr(v) -> np.ndarray:
    return np.asarray([np.nan if x is None else x for x in np.atleast_1d(_num(v)).tolist()], dtype=float)


def _curve(sample: str, truth: str, score: str):
    data = FIXTURE["samples"][sample]["data"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return roc_curve(_num(data[truth]), _num(data[score]))


def test_versions():
    assert FIXTURE["proc_version"] == "1.18.5"
    assert FIXTURE["r_version"].startswith("R version")


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("key,truth,score", CURVES)
def test_curve_and_auc(sample, key, truth, score):
    exp = FIXTURE["samples"][sample][key]
    r = _curve(sample, truth, score)
    assert r.direction == exp["direction"]
    assert r.controls.size == exp["n_controls"]
    assert r.cases.size == exp["n_cases"]
    np.testing.assert_allclose(r.auc, exp["auc"], rtol=1e-10)
    np.testing.assert_allclose(r.thresholds, _arr(exp["thresholds"]), rtol=1e-10)
    np.testing.assert_allclose(r.sensitivities, _arr(exp["sensitivities"]), rtol=1e-10)
    np.testing.assert_allclose(r.specificities, _arr(exp["specificities"]), rtol=1e-10)
    frame = r.frame()
    assert frame.columns == ["threshold", "sensitivity", "specificity"]
    assert frame.height == len(exp["thresholds"])


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("key,truth,score", CURVES)
def test_delong_var_ci(sample, key, truth, score):
    exp = FIXTURE["samples"][sample][key]
    r = _curve(sample, truth, score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(r.var_auc(), exp["var"], rtol=1e-8, atol=1e-15)
        np.testing.assert_allclose(r.ci_auc(), _arr(exp["ci95"]), rtol=1e-8)
        np.testing.assert_allclose(r.ci_auc(level=0.90), _arr(exp["ci90"]), rtol=1e-8)


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("key,truth,score", CURVES)
def test_coords(sample, key, truth, score):
    exp = FIXTURE["samples"][sample][key]
    data = FIXTURE["samples"][sample]["data"]
    r = _curve(sample, truth, score)
    cases = [
        (r.coords("best", best_method="youden", ret=list(exp["best_youden"])), exp["best_youden"]),
        (r.coords("best", best_method="closest.topleft", ret=list(exp["best_topleft"])), exp["best_topleft"]),
        (r.coords(_num(data["thresholds"]), ret=list(exp["coords_at"])), exp["coords_at"]),
    ]
    for got, want in cases:
        assert got.columns == list(want)
        for col, val in want.items():
            np.testing.assert_allclose(got[col].to_numpy(), _arr(val), rtol=1e-10, err_msg=col)


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize(
    "test_key,other,kwargs",
    [
        ("test_paired", "roc2", {}),
        ("test_paired_greater", "roc2", {"alternative": "greater"}),
        ("test_unpaired", "roc_other", {}),
    ],
)
def test_roc_test(sample, test_key, other, kwargs):
    exp = FIXTURE["samples"][sample][test_key]
    r1 = _curve(sample, "y", "score1")
    r2 = _curve(sample, *dict((k, (t, s)) for k, t, s in CURVES)[other])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = roc_test(r1, r2, **kwargs)
    assert res.method == exp["method"]
    assert res.paired is (exp["df"] is None)
    np.testing.assert_allclose(res.statistic, exp["statistic"], rtol=1e-8)
    np.testing.assert_allclose(res.p_value, exp["p_value"], atol=1e-8)
    np.testing.assert_allclose([res.auc1, res.auc2], [exp["auc1"], exp["auc2"]], rtol=1e-10)
    if exp["df"] is not None:
        np.testing.assert_allclose(res.df, exp["df"], rtol=1e-8)


def test_paired_flag_errors():
    r1 = _curve("balanced", "y", "score1")
    ro = _curve("balanced", "y_other", "score_other")
    with pytest.raises(ValueError):
        roc_test(r1, ro, paired=True)


def test_direction_and_levels():
    r = roc_curve([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert r.direction == ">"
    assert r.auc == 1.0
    r = roc_curve(["b", "b", "a", "a"], [0.1, 0.4, 0.35, 0.8], levels=("b", "a"))
    assert r.auc == 0.75
    r = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], direction=">")
    assert r.auc == 0.25


def test_plot_roc(tmp_path):
    from statract.models.roc import plot_roc

    r1 = _curve("balanced", "y", "score1")
    r2 = _curve("balanced", "y", "score2")
    out = plot_roc({"score1": r1, "score2": r2}, tmp_path / "roc.png")
    assert out.exists()
