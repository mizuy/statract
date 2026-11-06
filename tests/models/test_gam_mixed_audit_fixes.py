"""Regression tests for mixed-model and GAM fixes from the 2026-11-05 audit."""

import numpy as np
import polars as pl
import pytest
from scipy import special

from statract.models.gam import gam, smooth
from statract.models.gam.general import _optimize, _penalty_rank
from statract.models.gamm import gamm
from statract.models.glmm_extras import glmm_random_effects, median_odds_ratio
from statract.models.mixed import fit_mixed


def _poisson_frame():
    rng = np.random.default_rng(0)
    n = 300
    g = rng.integers(0, 15, n)
    x = rng.normal(size=n)
    t = rng.uniform(1, 3, n)
    b = rng.normal(0, 0.7, 15)
    y = rng.poisson(t * np.exp(0.2 + 0.5 * x + b[g]))
    return pl.DataFrame({"y": y, "x": x, "g": g, "t": t, "h": rng.integers(0, 6, n)})


def test_formula_offset_predict_needs_only_offset_columns():
    df = _poisson_frame()
    fit = fit_mixed(df, "y ~ x + offset(log(t)) + (1|g)", family="poisson")
    new = pl.DataFrame({"x": [0.0, 1.0], "t": [1.0, 2.0]})
    expected = np.exp(fit.coefficients[0] + fit.coefficients[1] * np.array([0.0, 1.0]) + np.log([1.0, 2.0]))
    np.testing.assert_allclose(fit.predict(new), expected)
    with_null_y = new.with_columns(pl.Series("y", [None, None], dtype=pl.Int64))
    np.testing.assert_allclose(fit.predict(with_null_y), expected)
    with pytest.raises(ValueError, match="offset columns"):
        fit.predict(pl.DataFrame({"x": [0.0]}))


def test_formula_offset_predict_several_terms():
    df = _poisson_frame()
    fit = fit_mixed(df, "y ~ x + offset(log(t)) + (1|g) + (1|h)", family="poisson")
    new = pl.DataFrame({"x": [0.5], "t": [2.0]})
    expected = np.exp(fit.coefficients[0] + 0.5 * fit.coefficients[1] + np.log(2.0))
    np.testing.assert_allclose(fit.predict(new), [expected])


def _vc_frame():
    rng = np.random.default_rng(2)
    n = 500
    x = rng.uniform(0, 1, n)
    z = rng.uniform(1, 3, n)
    truth = z * (2 + np.sin(2 * np.pi * x))
    return pl.DataFrame({"x": x, "z": z, "y": truth + rng.normal(0, 0.2, n)}), truth


def test_numeric_by_is_not_centred():
    df, truth = _vc_frame()
    fit = gam(df, "y", [smooth("x", by="z")], method="reml")
    assert np.sqrt(np.mean((fit.predict() - truth) ** 2)) < 0.1
    # With z also parametric, the smooth stays centred and the fit is the same.
    both = gam(df, "y", [smooth("x", by="z")], predictors=["z"], method="reml")
    assert np.sqrt(np.mean((both.predict() - truth) ** 2)) < 0.1


def test_integer_by_is_numeric_and_string_by_is_factor():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 1, n)
    k = rng.integers(1, 4, n)
    y = k * np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
    df = pl.DataFrame({"x": x, "k": k, "y": y})
    numeric = gam(df, "y", [smooth("x", by="k")], method="reml")
    assert [block.name for block in numeric.blocks] == ["x"]
    factor = gam(df.with_columns(pl.col("k").cast(pl.String)), "y", [smooth("x", by="k")], method="reml")
    assert [block.name for block in factor.blocks] == ["x|k"]
    boolean = gam(df.with_columns(pl.col("k") > 1), "y", [smooth("x", by="k")], method="reml")
    assert [block.name for block in boolean.blocks] == ["x|k"]


def test_predict_null_factor_is_nan_and_unseen_level_raises():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(0, 1, n)
    arm = rng.choice(["a", "b"], n)
    y = np.sin(2 * np.pi * x) + (arm == "b") * 2 + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "arm": arm, "y": y})
    fit = gam(df, "y", [smooth("x")], predictors=["arm"], method="ml")
    out = fit.predict(pl.DataFrame({"x": [0.5, 0.5, 0.5], "arm": ["a", "b", None]}))
    assert np.isfinite(out[:2]).all() and np.isnan(out[2])
    assert out[1] - out[0] > 1.5
    with pytest.raises(ValueError, match="unseen levels"):
        fit.predict(pl.DataFrame({"x": [0.5], "arm": ["zzz"]}))
    by_fit = gam(df, "y", [smooth("x", by="arm")], predictors=["arm"], method="ml")
    out = by_fit.predict(pl.DataFrame({"x": [0.5, 0.5], "arm": ["a", None]}))
    assert np.isfinite(out[0]) and np.isnan(out[1])
    with pytest.raises(ValueError, match="unseen levels"):
        by_fit.predict(pl.DataFrame({"x": [0.5], "arm": ["zzz"]}))


def test_gamma_with_reml_or_ml_raises():
    df, _truth = _vc_frame()
    with pytest.raises(ValueError, match="gamma"):
        gam(df, "y", [smooth("x")], method="reml", gamma=2.0)
    with pytest.raises(ValueError, match="gamma"):
        gam(df, "y", [smooth("x")], method="ml", gamma=2.0)
    gam(df, "y", [smooth("x")], method="gcv", gamma=2.0)


def test_select_uses_unpenalized_dimension():
    rng = np.random.default_rng(1)
    n = 100
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    fit = gam(df, "y", [smooth("x"), smooth("z")], select=True)
    p = fit.x.shape[1]
    assert fit.null_space_dim == p - _penalty_rank(p, fit.penalties) == 1
    score = _optimize(fit.x, fit.y, fit.penalties, None, "gaussian", "reml", 1, 1.0)[1]
    assert fit.reml == pytest.approx(score, rel=1e-6)


def test_factor_by_small_level_warns():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.uniform(0, 1, n)
    arm = np.where(np.arange(n) < 2, "rare", rng.choice(["a", "b"], n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "arm": arm, "y": y})
    with pytest.warns(UserWarning, match="rare"):
        gam(df, "y", [smooth("x", by="arm")], predictors=["arm"], method="ml")


def test_gamm_predict_null_parametric_row_is_nan():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.uniform(0, 1, n)
    w = rng.normal(size=n)
    g = rng.integers(0, 10, n)
    b = rng.normal(0, 0.6, 10)
    y = (rng.uniform(size=n) < special.expit(np.sin(2 * np.pi * x) + 0.5 * w + b[g])).astype(int)
    df = pl.DataFrame({"y": y, "x": x, "w": w, "g": g})
    fit = gamm(df, "y", [smooth("x", basis="cr")], random="(1|g)", predictors=["w"])
    new = pl.DataFrame({"x": [0.2, 0.5, 0.7], "w": [0.0, 1.0, None]})
    out = fit.predict(new)
    np.testing.assert_allclose(out[:2], fit.predict(new.head(2)))
    assert np.isnan(out[2])


def test_glmm_random_effects_counts_with_several_terms():
    df = _poisson_frame()
    fit = fit_mixed(df, "y ~ x + (1|g) + (1|h)", family="poisson")
    out = glmm_random_effects(fit, df.select("g", "h"))
    for name in ("g", "h"):
        part = out.filter(pl.col("group") == name)
        counts = df.group_by(name).len()
        expected = dict(zip(counts[name].cast(pl.String).to_list(), counts["len"].to_list(), strict=True))
        assert dict(zip(part["level"].to_list(), part["n"].to_list(), strict=True)) == expected
    only_h = glmm_random_effects(fit, df["h"])
    assert only_h.filter(pl.col("group") == "g")["n"].null_count() == only_h.filter(pl.col("group") == "g").height
    assert only_h.filter(pl.col("group") == "h")["n"].sum() == df.height


def test_median_odds_ratio_checks_fit():
    df = _poisson_frame()
    poisson = fit_mixed(df, "y ~ x + (1|g)", family="poisson")
    with pytest.raises(ValueError, match="binomial"):
        median_odds_ratio(poisson)
    binary = df.with_columns((pl.col("y") > 2).cast(pl.Int64).alias("yb"))
    probit = fit_mixed(binary, "yb ~ x + (1|g)", family="binomial", link="probit")
    with pytest.raises(ValueError, match="logit"):
        median_odds_ratio(probit)
    slope_only = fit_mixed(binary, "yb ~ x + (0 + x|g)", family="binomial")
    with pytest.raises(ValueError, match="random intercept"):
        median_odds_ratio(slope_only)
    logit = fit_mixed(binary, "yb ~ x + (1|g)", family="binomial")
    assert median_odds_ratio(logit)["median_odds_ratio"][0] >= 1.0


def test_select_tensor_gets_one_null_space_penalty():
    from statract.models.gam import tensor_smooth

    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    z2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "z1": z1, "z2": z2, "y": y})
    plain = gam(df, "y", [smooth("x"), tensor_smooth("z1", "z2", k=(5, 5))])
    fit = gam(df, "y", [smooth("x"), tensor_smooth("z1", "z2", k=(5, 5))], select=True)
    te_cols = fit.blocks[1].columns
    te_pens = [m for sl, m in fit.penalties if np.array_equal(sl, te_cols)]
    # Two marginal penalties plus exactly one null-space penalty.
    assert len(plain.penalties) == 3
    assert len(te_pens) == 3
    assert len(fit.penalties) == len(plain.penalties) + 2
    # The extra penalty spans the null space of the summed marginal penalties.
    total = te_pens[0] / np.linalg.norm(te_pens[0]) + te_pens[1] / np.linalg.norm(te_pens[1])
    null_dim = te_cols.size - np.linalg.matrix_rank(total, hermitian=True)
    assert np.linalg.matrix_rank(te_pens[2], hermitian=True) == null_dim
    # Only the intercept is left unpenalized.
    p = fit.x.shape[1]
    assert _penalty_rank(p, fit.penalties) == p - 1
    assert fit.null_space_dim == 1
    # The pure-noise tensor term is shrunk to (near) zero edf.
    assert fit.block_edf[1] < 0.05
    assert fit.block_edf[0] > 4.0


def test_select_factor_by_gets_one_null_space_penalty_per_level():
    rng = np.random.default_rng(6)
    n = 300
    x = rng.uniform(0, 1, n)
    arm = rng.choice(["a", "b", "c"], n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "arm": arm, "y": y})
    plain = gam(df, "y", [smooth("x", by="arm")], predictors=["arm"])
    fit = gam(df, "y", [smooth("x", by="arm")], predictors=["arm"], select=True)
    extra = fit.penalties[len(plain.penalties) :]
    assert len(extra) == 3
    assert [sl.size for sl, _m in extra] == [sl.size // 3 for sl in [fit.blocks[0].columns]] * 3
    p = fit.x.shape[1]
    assert fit.null_space_dim == p - _penalty_rank(p, fit.penalties) == 3
