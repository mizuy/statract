"""Deviance summaries and nested-model comparisons against mgcv.

Fixture: ``fixtures/gam_summary.json`` from ``scripts/gam_summary.R``.
``compare_gams`` takes the df change from the plain edf; mgcv corrects the
edf for smoothing-parameter uncertainty, so only the deviance change is
compared, and the p-value is checked against its own definition.

``smooth_table`` (Wood 2013 tests): ``ref_df`` and the statistic are compared
with ``s.table``. The p-values come from the null distribution in the paper
(sec. 2.3), computed by quadrature, and are checked against simulation;
mgcv 1.9-1 reports somewhat larger p-values, kept in the fixture for
reference.

``glance`` ``edf2`` and ``aic`` (Wood, Pya & Säfken 2016) are compared with
``sum(edf2)`` and ``AIC()``. ``edf2`` depends on the choice of the factor of
the coefficient covariance and on how flat smoothing parameters are found,
so it agrees to 1%; the AIC moves by twice that difference.
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


def _check(fit, want: dict, rtol: float = 1e-4, statistic: bool = True) -> None:
    row = fit.glance().row(0, named=True)
    for key in ("deviance", "null_deviance", "deviance_explained", "adj_r_squared", "edf"):
        np.testing.assert_allclose(row[key], want[key], rtol=rtol, err_msg=key)
    table = fit.smooth_table()
    want_table = want["s_table"]
    np.testing.assert_allclose(table["edf"].to_numpy(), want_table["edf"], rtol=max(rtol, 1e-3), atol=1e-3)
    np.testing.assert_allclose(table["ref_df"].to_numpy(), want_table["ref_df"], rtol=max(rtol, 1e-3), atol=2e-3)
    if statistic:
        np.testing.assert_allclose(table["statistic"].to_numpy(), want_table["statistic"], rtol=1e-3)


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
    # Gamma: mgcv's scale is the Pearson estimate, statract's the REML one
    # (6% apart here), and the F statistic scales with it.
    _check(build(), FIXTURE[name], rtol=rtol, statistic=name != "gamma")


_MODELS = {
    ("binomial", "small"): lambda: gam(BINARY, "y", [smooth("pre_size_mm", k=10)], family="binomial"),
    ("binomial", "large"): lambda: gam(
        BINARY, "y", [smooth("pre_size_mm", k=10), smooth("age", k=10)], family="binomial"
    ),
    ("gaussian_weighted", "small"): lambda: gam(GAUSS, "y", [smooth("x", k=10)], weights="w"),
    # The z smooth is at working infinity, so its smoothing parameter gets no variance.
    ("gaussian_weighted", "large"): lambda: gam(GAUSS, "y", [smooth("x", k=10), smooth("z", k=10)], weights="w"),
    ("poisson_offset", None): lambda: gam(BINARY, "n", [smooth("age", k=8)], family="poisson", offset="off"),
    ("gaussian", None): lambda: gam(GAUSS, "y", [smooth("x", k=10)]),
    ("gamma", None): lambda: gam(GAUSS, "cost", [smooth("x", k=8)], family="gamma"),
}


@pytest.mark.parametrize("key", list(_MODELS), ids=lambda key: "-".join(k for k in key if k))
def test_corrected_edf_and_aic(key) -> None:
    case, part = key
    want = FIXTURE[case] if part is None else FIXTURE[case][part]
    row = _MODELS[key]().glance().row(0, named=True)
    np.testing.assert_allclose(row["edf2"], want["edf2"], rtol=1e-2)
    np.testing.assert_allclose(row["log_likelihood"], want["log_likelihood"], rtol=1e-5)
    np.testing.assert_allclose(row["df"] - row["edf2"], want["df"] - want["edf2"])
    np.testing.assert_allclose(row["aic"], want["aic"], atol=2 * 1e-2 * want["edf2"])


def test_compare_gams_aic_is_corrected() -> None:
    small = _MODELS[("binomial", "small")]()
    large = _MODELS[("binomial", "large")]()
    table = compare_gams(small, large)
    np.testing.assert_allclose(table["aic"].to_numpy(), [small.glance()["aic"][0], large.glance()["aic"][0]])


def test_smooth_test_null_distribution() -> None:
    from statract.models.gam.smooth_test import null_sf, smooth_p_value, wood_statistic

    # Integer rank: chi-squared on r.
    np.testing.assert_allclose(null_sf([3.0, 12.0], [1, 1, 0], [2, 1, 1]), stats.chi2.sf([3.0, 12.0], 3), rtol=1e-8)
    # Non-integer rank: chi2(k-2) + nu1 chi2(1) + nu2 chi2(1) by simulation.
    rng = np.random.default_rng(20261022)
    n = 400_000
    for weights, dofs, x in [([1, 1.2046, 0.5598], [0, 1, 1], 2.1275), ([1, 0.85, 0.67], [2, 1, 1], 6.0)]:
        q = weights[1] * rng.chisquare(1, n) + weights[2] * rng.chisquare(1, n)
        if dofs[0]:
            q += rng.chisquare(dofs[0], n)
        np.testing.assert_allclose(null_sf(x, weights, dofs)[0], np.mean(q > x), atol=3e-3)
        # Estimated scale: P(Q > t W / kappa), W ~ chi2(kappa).
        w = rng.chisquare(40, n)
        np.testing.assert_allclose(smooth_p_value(x, weights, dofs, 40), np.mean(q > x * w / 40), atol=3e-3)
    # Integer rank with an estimated scale is the F test.
    np.testing.assert_allclose(smooth_p_value(9.0, [1, 1, 0], [2, 1, 1], 50), stats.f.sf(3.0, 3, 50), rtol=1e-4)
    # The statistic does not depend on the signs the eigen solver picks.
    basis = rng.normal(size=(60, 5))
    cov = np.cov(rng.normal(size=(5, 30)))
    beta = rng.normal(size=5)
    first = wood_statistic(beta, cov, basis, 2.4)[0]
    flipped = wood_statistic(beta, cov, -basis, 2.4)[0]
    np.testing.assert_allclose(first, flipped)


def test_compare_gams_rejects_mixed_families() -> None:
    gaussian = gam(GAUSS, "y", [smooth("x", k=10)])
    gamma = gam(GAUSS, "cost", [smooth("x", k=8)], family="gamma")
    with pytest.raises(ValueError, match="family"):
        compare_gams(gaussian, gamma)
