"""Regression tests: ATE estimand, missing treatment/covariates, k:1 weights."""

import numpy as np
import polars as pl
import pytest

from statract.models.matching import match_sample


def _frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    return pl.DataFrame({"t": t, "x": x, "g": rng.integers(0, 3, n)})


@pytest.mark.parametrize("method", ["nearest", "optimal"])
def test_ate_rejected_for_pair_matching(method):
    with pytest.raises(ValueError, match="ATE"):
        match_sample(_frame(), "t", ["x"], method=method, estimand="ATE")


def _ate_subclass_reference(t, sub):
    """Unnormalized ATE rule: n_s / n_treated_s and n_s / n_control_s."""
    w = np.zeros(len(t))
    for s in np.unique(sub[sub > 0]):
        m = sub == s
        n1, n0 = int((m & (t == 1)).sum()), int((m & (t == 0)).sum())
        w[m & (t == 1)] = (n1 + n0) / n1
        w[m & (t == 0)] = (n1 + n0) / n0
    for flag in (0, 1):
        k = (t == flag) & (w > 0)
        w[k] *= k.sum() / w[k].sum()
    return w


@pytest.mark.parametrize("method", ["exact", "subclass", "cem", "full"])
def test_ate_subclass_weights(method):
    df = _frame()
    t = df["t"].to_numpy()
    covs = ["g"] if method == "exact" else ["x"]
    ate = match_sample(df, "t", covs, method=method, estimand="ATE")
    att = match_sample(df, "t", covs, method=method, estimand="ATT")
    assert not np.allclose(ate.weights, att.weights)
    sub = np.asarray(ate.subclass)
    # Every kept subclass holds both groups.
    for s in np.unique(sub[sub > 0]):
        assert set(t[sub == s]) == {0, 1}
    np.testing.assert_allclose(ate.weights, _ate_subclass_reference(t, sub))


def test_null_treatment_raises():
    df = pl.DataFrame({"t": [1, 0, None, None, 1, 0], "g": ["a", "a", "a", "b", "b", "b"]})
    for method in ["exact", "cem", "nearest"]:
        with pytest.raises(ValueError, match="missing"):
            match_sample(df, "t", ["g"], method=method)


def test_cem_null_covariate_unmatched_and_cutpoints_ignore_nulls():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n).tolist()
    x[0] = None
    x[1] = None
    t = [1, 0, *rng.integers(0, 2, n - 2).tolist()]
    df = pl.DataFrame({"t": t, "x": x})
    m = match_sample(df, "t", ["x"], method="cem")
    assert m.weights[0] == 0 and m.weights[1] == 0
    assert m.subclass[0] == 0 and m.subclass[1] == 0
    complete = match_sample(df[2:], "t", ["x"], method="cem")
    np.testing.assert_allclose(m.weights[2:], complete.weights)
    np.testing.assert_array_equal(m.subclass[2:], complete.subclass)


def test_cem_nan_covariate_unmatched():
    df = pl.DataFrame({"t": [1, 0, 1, 0, 1, 0], "x": [0.1, 0.2, float("nan"), float("nan"), 0.3, 0.25]})
    m = match_sample(df, "t", ["x"], method="cem", cutpoints=1)
    assert m.weights[2] == 0 and m.weights[3] == 0
    assert (m.weights[[0, 1, 4, 5]] > 0).all()


def test_exact_null_not_its_own_stratum():
    df = pl.DataFrame({"t": [1, 0, 1, 0, 1, 0], "g": ["a", "a", None, None, "b", "c"]})
    m = match_sample(df, "t", ["g"], method="exact")
    np.testing.assert_array_equal(m.weights, [1, 1, 0, 0, 0, 0])
    assert m.subclass[2] == 0 and m.subclass[3] == 0


def test_nearest_exact_null_never_matched():
    df = pl.DataFrame(
        {"t": [1, 0, 1, 0, 0], "x": [0.0, 0.1, 0.5, 0.55, 2.0], "g": ["a", "a", None, None, "a"]}
    )
    m = match_sample(df, "t", ["x"], distance=df["x"].to_numpy(), exact=["g"])
    assert m.weights[2] == 0 and m.weights[3] == 0
    o = match_sample(df, "t", ["x"], method="optimal", distance=df["x"].to_numpy(), exact=["g"])
    assert o.weights[2] == 0 and o.weights[3] == 0


def test_ratio_weights_sum_of_inverse_k():
    df = _frame()
    t = df["t"].to_numpy()
    m = match_sample(df, "t", ["x"], ratio=2, caliper=0.05)
    pairs = m.pairs()
    k = pairs.group_by("treated").len()
    kt = dict(zip(k["treated"].to_list(), k["len"].to_list(), strict=True))
    assert set(kt.values()) == {1, 2}
    expected = np.zeros(len(t))
    for a, b in zip(pairs["treated"].to_list(), pairs["control"].to_list(), strict=True):
        expected[b] += 1 / kt[a]
    c = (t == 0) & (expected > 0)
    expected[c] *= c.sum() / expected[c].sum()
    np.testing.assert_allclose(m.weights[c], expected[c])
    np.testing.assert_allclose(m.weights[t == 1][m.weights[t == 1] > 0], 1.0)


def test_ratio_weights_with_replacement():
    # Treated at 0 and 1 each take two controls (sharing 0.5); the treated at 3
    # finds only one control inside the caliper.
    df = pl.DataFrame({"t": [1, 1, 1, 0, 0, 0, 0], "d": [0.0, 1.0, 3.0, -0.1, 0.5, 1.1, 3.05]})
    m = match_sample(
        df, "t", ["d"], distance=df["d"].to_numpy(), ratio=2, replace=True, order="data",
        caliper=0.55, std_caliper=False,
    )
    # Raw control weights 1/2, 1, 1/2, 1, scaled to sum to 4.
    np.testing.assert_allclose(m.weights, [1, 1, 1, 2 / 3, 4 / 3, 2 / 3, 4 / 3])
