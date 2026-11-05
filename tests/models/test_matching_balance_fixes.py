"""Regression tests: balance table columns, missing covariates in matrix
distances, reestimate, optimal k:1 shortage, subclass and CEM bins, McNemar
levels and correction, weighted RESET."""

import warnings

import numpy as np
import polars as pl
import pytest

from statract.models.fit import fit_ols
from statract.models.htest import mcnemar_test
from statract.models.linear_tests import ramsey_reset_test
from statract.models.matching import _coarsen, _quantile_subclass, match_sample


def _frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    x2 = x + rng.normal(size=n)
    return pl.DataFrame({"t": t, "x": x, "x2": x2})


def _with_null(df, col="x2", row=3):
    values = df[col].to_list()
    values[row] = None
    return df.with_columns(pl.Series(col, values, dtype=pl.Float64))


# --- balance -----------------------------------------------------------------


def test_balance_keeps_boolean_and_factor_covariates():
    df = _frame()
    rng = np.random.default_rng(5)
    df = df.with_columns(
        pl.Series("b", rng.random(df.height) < 0.4),
        pl.Series("g", rng.choice(["a", "b", "c"], df.height)).cast(pl.Enum(["a", "b", "c"])),
        pl.Series("s", rng.choice(["no", "yes"], df.height)),
    )
    table = match_sample(df, "t", ["x", "b", "g", "s"]).balance()
    assert table["term"].to_list() == ["distance", "x", "b", "g_a", "g_b", "g_c", "s_yes"]
    t = df["t"].to_numpy() == 1
    b = df["b"].to_numpy().astype(float)
    row = table.filter(pl.col("term") == "b")
    assert row["mean_treated_all"][0] == pytest.approx(b[t].mean())
    p1, p0 = b[t].mean(), b[~t].mean()
    sd = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    assert row["smd_all"][0] == pytest.approx((p1 - p0) / sd)
    g_b = (df["g"].cast(pl.String).to_numpy() == "b").astype(float)
    row = table.filter(pl.col("term") == "g_b")
    q1, q0 = g_b[t].mean(), g_b[~t].mean()
    sd = np.sqrt((q1 * (1 - q1) + q0 * (1 - q0)) / 2)
    assert row["smd_all"][0] == pytest.approx((q1 - q0) / sd)


def test_balance_atc_columns_are_treated_and_control():
    df = _frame()
    table = match_sample(df, "t", ["x"], estimand="ATC").balance()
    t = df["t"].to_numpy() == 1
    x = df["x"].to_numpy()
    row = table.filter(pl.col("term") == "x")
    assert row["mean_treated_all"][0] == pytest.approx(x[t].mean())
    assert row["mean_control_all"][0] == pytest.approx(x[~t].mean())
    sd = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
    assert row["smd_all"][0] == pytest.approx((x[t].mean() - x[~t].mean()) / sd)
    assert row["smd_all"][0] > 0


def _rel_var(v, w):
    m = np.average(v, weights=w)
    return np.sum(w * (v - m) ** 2) / (w.sum() - np.sum(w**2) / w.sum())


def test_balance_variance_ratio_uses_weights_and_n_matched_counts_matched():
    df = _frame()
    m = match_sample(df, "t", ["x"], replace=True)
    w = m.weights
    t = df["t"].to_numpy() == 1
    x = df["x"].to_numpy()
    ft, fc = t & (w > 0), ~t & (w > 0)
    assert len(np.unique(w[fc])) > 1
    row = m.balance().filter(pl.col("term") == "x")
    assert row["variance_ratio_matched"][0] == pytest.approx(_rel_var(x[ft], w[ft]) / _rel_var(x[fc], w[fc]))
    assert row["n_all"][0] == df.height
    assert row["n_matched"][0] == int((w > 0).sum()) < df.height


# --- missing covariates in matrix distances ------------------------------------


@pytest.mark.parametrize("distance", ["scaled_euclidean", "robust_mahalanobis"])
def test_matrix_distance_missing_row_unmatched(distance):
    df = _frame()
    with_null = _with_null(df)
    m = match_sample(with_null, "t", ["x", "x2"], distance=distance)
    assert m.weights[3] == 0 and m.subclass[3] == 0
    # The other rows' coordinates come from the complete rows only.
    ref = match_sample(df.filter(pl.int_range(pl.len()) != 3), "t", ["x", "x2"], distance=distance)
    keep = np.flatnonzero(np.arange(df.height) != 3)
    assert np.allclose(m.distance[keep], ref.distance)


def test_reestimate_with_missing_covariate():
    df = _with_null(_frame())
    m = match_sample(df, "t", ["x", "x2"], discard="both", reestimate=True)
    assert m.weights[3] == 0
    assert np.isnan(m.distance[3])
    assert (m.weights > 0).sum() > 0


def test_optimal_ratio_shortage_warns():
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"t": [1] * 12 + [0] * 18, "x": rng.normal(size=30)})
    with pytest.warns(UserWarning, match="falling back to 1:1"):
        m = match_sample(df, "t", ["x"], method="optimal", ratio=2)
    assert m.pairs().height == 12
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        match_sample(df, "t", ["x"], method="optimal", ratio=1)


# --- subclass and CEM bins -----------------------------------------------------


def test_subclass_cutpoints_from_focal_group_right_closed():
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.25, 0.9])
    treat = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    discarded = np.zeros(8, dtype=bool)
    # Treated quartiles 0.1, 0.2, 0.3, 0.4, 0.5; 0.2 sits on a cut point
    # and belongs to the lower (right-closed) bin.
    labels = _quantile_subclass(dist, 4, discarded, treat)
    assert labels.tolist() == [1, 1, 2, 3, 4, 1, 2, 4]
    atc = _quantile_subclass(dist, 2, discarded, ~treat)
    assert atc.tolist() == [1, 1, 2, 2, 2, 1, 1, 2]
    ate = _quantile_subclass(dist, 2, discarded, None)
    assert ate.tolist() == [1, 1, 2, 2, 2, 1, 1, 2]


def test_cem_bins_right_closed():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    codes = _coarsen(values, np.array([0.0, 2.0, 4.0]))
    assert codes.tolist() == [1, 1, 1, 2, 2]


# --- htest / RESET ---------------------------------------------------------------


def test_mcnemar_enum_common_levels():
    enum = pl.Enum(["a", "b", "c"])
    x = pl.Series(["a", "a", "b", "b", "a", "b"], dtype=enum)
    y = pl.Series(["b", "c", "b", "c", "c", "b"], dtype=enum)
    res = mcnemar_test(x, y)
    table = np.array([[0, 1, 2], [0, 2, 1], [0, 0, 0]])
    ref = mcnemar_test(table)
    assert res.statistic == pytest.approx(ref.statistic) == pytest.approx(4.0)
    assert res.parameter == 3.0
    # Plain strings use the sorted union.
    res_str = mcnemar_test(x.cast(pl.String), y.cast(pl.String))
    assert res_str.statistic == pytest.approx(4.0)


def test_mcnemar_correction_not_clamped():
    # R: mcnemar.test(matrix(c(10, 5, 5, 10), 2)) gives 0.1.
    assert mcnemar_test([[10, 5], [5, 10]]).statistic == pytest.approx(0.1)


def _reset_reference(x, y, w):
    sw = np.sqrt(w)
    b = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
    fh = x @ b
    xf = np.column_stack([x, fh**2, fh**3])
    r1 = np.sum((sw * (y - fh)) ** 2)
    b2 = np.linalg.lstsq(xf * sw[:, None], y * sw, rcond=None)[0]
    r2 = np.sum((sw * (y - xf @ b2)) ** 2)
    return (r1 - r2) / 2 / (r2 / (len(y) - xf.shape[1]))


def test_reset_uses_weights_and_offset():
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    o = rng.normal(size=n)
    y = 1 + x + 0.3 * x**2 + o + rng.normal(size=n)
    w = rng.uniform(0.2, 3, n)
    df = pl.DataFrame({"x": x, "z": z, "y": y, "w": w, "o": o})
    design = np.column_stack([np.ones(n), x, z])
    fit = fit_ols(df, "y ~ x + z", weights="w")
    assert ramsey_reset_test(fit).statistic == pytest.approx(_reset_reference(design, y, w))
    fit_o = fit_ols(df, "y ~ x + z + offset(o)")
    sw = np.ones(n)
    b = np.linalg.lstsq(design, y - o, rcond=None)[0]
    fh = design @ b + o
    xf = np.column_stack([design, fh**2, fh**3])
    r1 = np.sum((y - o - design @ b) ** 2)
    b2 = np.linalg.lstsq(xf * sw[:, None], y - o, rcond=None)[0]
    r2 = np.sum((y - o - xf @ b2) ** 2)
    expected = (r1 - r2) / 2 / (r2 / (n - 5))
    assert ramsey_reset_test(fit_o).statistic == pytest.approx(expected)


def test_reset_rank_deficient_design():
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 1 + x + 0.3 * x**2 + rng.normal(size=n)
    df = pl.DataFrame({"x": x, "x2": 2 * x, "y": y})
    res = ramsey_reset_test(fit_ols(df, "y ~ x + x2"))
    ref = ramsey_reset_test(fit_ols(df, "y ~ x"))
    assert res.df == (2.0, 96.0)
    assert res.statistic == pytest.approx(ref.statistic)
