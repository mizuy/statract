"""API contracts for the Python estimators. Numeric parity with R lives in tests/r_oracle."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import (
    accelerated_failure,
    cox_ph,
    fine_gray,
    fit_glm,
    fit_mixed,
    fit_ols,
    gam,
    match_sample,
    smooth,
    sm_summary2df,
    tensor_interaction,
    tensor_smooth,
)
from statract.design import design_matrix, formula_model_matrix


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, None, 4.0, 3.0, 2.5, 1.5],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "stage": ["II", "I", "III", "I", "III", "II", "I"],
            "event": [1, 0, 1, 1, 0, 1, 0],
        }
    )


def test_ols_drops_null_rows_and_tidy_has_the_public_columns():
    fit = fit_ols(_frame(), "y", ["x", "stage"])
    assert fit.n_obs == 6
    assert fit.names[0] == "(Intercept)"
    assert "stageII" in fit.names
    table = fit.tidy()
    assert table.columns == ["term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
    expanded = fit.tidy(exponentiate=True)
    assert "exp_estimate" in expanded.columns
    glance = fit.glance()
    assert glance["n_obs"][0] == 6
    assert glance.columns[:4] == ["n_obs", "residual_df", "log_likelihood", "aic"]


def test_string_factors_sort_and_enum_keeps_definition_order():
    strings = design_matrix(_frame().drop_nulls(), ["stage"], extra=["y"])
    assert strings.names == ["(Intercept)", "stageII", "stageIII"]
    ordered = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "stage": pl.Series("stage", ["b", "a", "b"], dtype=pl.Enum(["b", "a"])),
        }
    )
    design = design_matrix(ordered, ["stage"], extra=["y"])
    assert design.names == ["(Intercept)", "stagea"]


def test_binomial_response_is_a_probability():
    data = pl.DataFrame({"y": [0, 1, 0, 1, 1, 0], "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]})
    fit = fit_glm(data, "y", ["x"], family="binomial")
    probability = fit.predict(kind="response")
    assert probability.min() > 0
    assert probability.max() < 1
    link = fit.predict(kind="link")
    assert link.shape == probability.shape


def test_gam_accepts_the_planned_scope():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 40)
    z = rng.uniform(0, 1, 40)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 40)
    data = pl.DataFrame({"y": y, "yb": (y > 0).astype(float), "x": x, "z": z, "g": np.where(z > 0.5, "a", "b")})
    assert smooth("x", basis="tp").basis == "tp"
    assert tensor_interaction("x", "z", k=(4, 4)).basis == "ti"
    binary = gam(data, "yb", [smooth("x", k=5), smooth("z", k=5)], family="binomial")
    assert np.isfinite(binary.edf)
    tensor = gam(data, "y", [tensor_smooth("x", "z", k=(4, 4))])
    assert tensor.predict(kind="terms").height == 40
    selected = gam(data, "y", [smooth("x", k=5)], select=True, method="gcv")
    assert np.isfinite(selected.edf)


def test_mixed_module_imports_engine_lazily():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / "src/statract/mixed.py"
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "pandas" not in roots
        assert "mixedlm" not in roots
        assert "mixedlm_rs" not in roots
        assert "lme_python" not in roots


def test_mixed_model_tidy_and_slope_check():
    rng = __import__("numpy").random.default_rng(0)
    g = rng.integers(0, 6, size=40)
    x = rng.normal(size=40)
    re = rng.normal(size=6)
    y = 0.5 + 0.2 * x + re[g] + rng.normal(scale=0.5, size=40)
    data = pl.DataFrame({"y": y, "x": x, "g": g})
    fit = fit_mixed(data, "y", ["x"], groups="g")
    assert fit.names == ["(Intercept)", "x"]
    assert fit.variance_table()["group"].to_list() == ["g", "residual"]
    assert fit.tidy().columns[0] == "term"
    with pytest.raises(ValueError, match="fixed effects"):
        fit_mixed(data, "y", ["x"], groups="g", slopes=["missing"])


def test_exact_pairs_are_the_subclass_product():
    data = pl.DataFrame(
        {
            "treat": [1, 0, 1, 0, 1],
            "bin": ["H", "H", "L", "L", "H"],
            "x": [1.0, 2.0, 3.0, 5.0, 4.0],
        }
    )
    matched = match_sample(data, "treat", ["bin"], method="exact")
    got = list(zip(matched.pairs()["treated"].to_list(), matched.pairs()["control"].to_list(), strict=True))
    # H: treated 0, 4 against control 1. L: treated 2 against control 3.
    assert got == [(0, 1), (4, 1), (2, 3)]
    # Normalization keeps the within-cell size ratio: H has two treated per control, L has one.
    assert matched.weights[1] / matched.weights[3] == 2.0
    assert matched.weights[0] == 1.0


def test_scalar_nearest_matches_the_full_scan():
    rng = np.random.default_rng(7)
    n = 80
    dist = rng.normal(size=n)
    dist[rng.choice(n, size=15, replace=False)] = rng.choice(dist, size=15)
    treat = np.zeros(n, dtype=bool)
    treat[rng.choice(n, size=30, replace=False)] = True
    exact = rng.integers(0, 3, size=n)
    from statract.matching import _nearest_on_line, _nearest_scan

    control = np.flatnonzero(~treat)
    for order_name, treated in (
        ("data", np.flatnonzero(treat)),
        ("largest", np.flatnonzero(treat)[np.argsort(-dist[treat], kind="mergesort")]),
    ):
        for replace in (False, True):
            for limit in (None, 0.4):
                for labels in (None, exact):
                    got = _nearest_on_line(treated, control, dist, 2, replace, limit, labels)
                    expected = _nearest_scan(treated, control, dist, 2, replace, limit, labels)
                    assert got == expected, (order_name, replace, limit, labels is not None)


def test_logit_tie_keeps_the_adjacent_lower_score():
    """MatchIt 4.5.5 keeps the neighboring control when two scores are equal.

    Scores ``[0, 0, 1]`` with only the last row treated. Both controls sit on
    the lower-score side, and the one next to the treated row in the stable
    order is index 1, not the smaller row index.
    """
    data = pl.DataFrame({"treat": [0.0, 0.0, 1.0], "x": [0.0, 0.0, 1.0]})
    matched = match_sample(
        data,
        "treat",
        ["x"],
        method="nearest",
        distance=np.array([0.0, 0.0, 1.0]),
        order="data",
    )
    assert matched.pairs()["control"].to_list() == [1]


def test_exact_and_cem_defer_the_propensity(monkeypatch):
    calls = {"n": 0}
    import statract.matching as matching

    real = matching.fit_glm

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(matching, "fit_glm", wrapped)
    data = pl.DataFrame(
        {
            "treat": [1, 0, 1, 0, 1, 0],
            "x": [0.1, 0.2, 0.4, 0.5, 0.8, 0.9],
            "bin": ["H", "H", "L", "L", "H", "L"],
        }
    )
    exact = match_sample(data, "treat", ["bin"], method="exact")
    cem = match_sample(data, "treat", ["x"], method="cem")
    assert calls["n"] == 0
    assert np.isnan(exact.distance).all()
    exact.balance()
    cem.pairs()
    assert calls["n"] == 2
    explicit = match_sample(data, "treat", ["bin"], method="exact", distance="logit")
    assert calls["n"] == 3
    assert np.isfinite(explicit.distance).any()


def test_mahalanobis_accepts_a_factor():
    data = pl.DataFrame(
        {
            "treat": [1, 1, 0, 0, 1, 0],
            "x": [0.2, 0.4, 0.1, 0.5, 0.3, 0.6],
            "color": ["red", "blue", "red", "blue", "red", "blue"],
        }
    )
    matched = match_sample(data, "treat", ["x", "color"], method="nearest", distance="mahalanobis", order="data")
    assert matched.pairs().height >= 1
    assert np.isfinite(matched.weights).all()


def test_mahalanobis_pairs_match_matchit_455():
    """Pairs locked to one MatchIt 4.5.5 run (R 4.3.3) on this 24-row frame.

    The factor uses every level, so the pooled covariance is singular and the
    match goes through the generalized inverse.
    """
    color = (["blue", "green", "red"] * 8)[:24]
    data = pl.DataFrame(
        {
            "treat": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            "x": [
                0.2, 0.2, 1.5, 1.5, 0.2, 0.4, 1.4, 1.6, 0.3, 2.0, 0.1, 1.7,
                0.5, 0.6, 1.1, 1.2, 0.0, 2.2, 0.8, 0.9, 1.8, 0.7, 1.3, 2.4,
            ],
            "color": color,
        }
    )
    matched = match_sample(data, "treat", ["x", "color"], method="nearest", distance="mahalanobis", order="data")
    got = list(zip(matched.pairs()["treated"].to_list(), matched.pairs()["control"].to_list(), strict=True))
    assert got == [
        (0, 18),
        (1, 4),
        (2, 11),
        (3, 6),
        (8, 5),
        (9, 15),
        (12, 10),
        (14, 20),
        (16, 13),
        (19, 22),
        (21, 17),
        (23, 7),
    ]
    assert np.array_equal(matched.weights, np.ones(24))


def test_mahalanobis_scale_matches_sequential_longdouble():
    """Scale and pooled covariance stay on R's sequential long-double sum.

    ``np.sum`` adds in pairs. On this draw that sum is already a different
    long double, which is enough to move a Mahalanobis pivot.
    """
    rng = np.random.default_rng(0)
    n, width = 400, 4
    values = rng.normal(size=(n, width))
    values[:, 0] = rng.choice([0.0, 99999.0, 5013.0], size=n)
    treat = np.zeros(n, dtype=bool)
    treat[: n // 3] = True

    def sequential(column: np.ndarray) -> np.longdouble:
        total = np.longdouble(0)
        for item in column:
            total += item
        return total

    column = values[:, 1]
    assert sequential(column) != np.sum(column.astype(np.longdouble), dtype=np.longdouble)

    from statract.matching import _pooled_cov, _r_scale

    center = np.empty(width)
    for index in range(width):
        center[index] = float(sequential(values[:, index]) / n)
    centered = values - center
    denom = n - 1
    scale = np.empty(width)
    for index in range(width):
        squares = centered[:, index] * centered[:, index]
        scale[index] = np.sqrt(float(sequential(squares)) / denom)
    scale[scale == 0] = 1.0
    assert np.array_equal(_r_scale(values), centered / scale)

    shifted = np.array(centered / scale, dtype=np.float64, copy=True)
    for flag in (True, False):
        rows = np.flatnonzero(treat == flag)
        for index in range(width):
            part = shifted[rows, index]
            total = sequential(part)
            tmp = total / part.shape[0]
            if np.isfinite(float(tmp)):
                adjust = np.longdouble(0)
                for item in part:
                    adjust += np.longdouble(item) - tmp
                tmp = tmp + adjust / part.shape[0]
            shifted[rows, index] = part - float(tmp)
    means = np.empty(width)
    for index in range(width):
        part = shifted[:, index]
        total = sequential(part)
        tmp = total / n
        if np.isfinite(float(tmp)):
            adjust = np.longdouble(0)
            for item in part:
                adjust += np.longdouble(item) - tmp
            tmp = tmp + adjust / n
        means[index] = float(tmp)
    gram = np.empty((width, width))
    for left in range(width):
        for right in range(left + 1):
            total = np.longdouble(0)
            for row in range(n):
                total += (np.longdouble(shifted[row, left]) - means[left]) * (
                    np.longdouble(shifted[row, right]) - means[right]
                )
            value = float(total / (n - 1))
            gram[left, right] = value
            gram[right, left] = value
    groups = int(np.unique(treat).size)
    assert np.array_equal(_pooled_cov(centered / scale, treat), gram * (n - 1) / (n - groups))


def test_squared_euclidean_tiles_match_one_row_einsum():
    """Tiling the difference does not change the squared distance bits."""
    rng = np.random.default_rng(0)
    left = np.ascontiguousarray(rng.normal(size=(40, 21)))
    right = np.ascontiguousarray(rng.normal(size=(4000, 21)))
    from statract.matching import _squared_euclidean

    got = _squared_euclidean(left, right)
    for index in range(0, left.shape[0], 7):
        diff = left[index : index + 1, None, :] - right[None, :, :]
        ref = np.einsum("ijk,ijk->ij", diff, diff)
        assert np.array_equal(got[index : index + 1], ref)


def test_matrix_distance_rejects_subclass_matching():
    data = pl.DataFrame({"treat": [1, 1, 0, 0], "x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 0.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="subclass"):
        match_sample(data, "treat", ["x1", "x2"], method="subclass", distance="mahalanobis")


def test_sm_summary2df_warns():
    class Result:
        params = (0.1, 0.2)
        pvalues = (0.5, 0.01)

        def conf_int(self):
            return {0: [0.0, 0.1], 1: [0.2, 0.3]}

    result = Result()
    with pytest.warns(DeprecationWarning, match="Fit.tidy"):
        summary = sm_summary2df(result)
    assert summary.height == 2


def test_formula_model_matrix_uses_treatment_contrasts():
    data = pl.DataFrame(
        {
            "outcome": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "treatment": [1, 1, 0, 0, 1, 0],
            "age": [45, 50, 55, 60, 65, 70],
            "stage": ["II", "I", "III", "I", "II", "III"],
        }
    )
    y, x = formula_model_matrix("outcome ~ treatment + age", data)
    assert y.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert np.allclose(x, [[1, 1, 45], [1, 1, 50], [1, 0, 55], [1, 0, 60], [1, 1, 65], [1, 0, 70]])
    _, stage = formula_model_matrix("outcome ~ stage", data)
    assert np.allclose(
        stage,
        [[1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],
    )
    _, no_intercept = formula_model_matrix("outcome ~ age - 1", data)
    assert np.allclose(no_intercept, [[45], [50], [55], [60], [65], [70]])


def test_formula_model_matrix_drops_null_rows():
    data = pl.DataFrame({"y": [1.0, None, 0.0, 1.0], "x": [1.0, 2.0, None, 4.0]})
    y, x = formula_model_matrix("y ~ x", data)
    assert y.tolist() == [1.0, 1.0]
    assert np.allclose(x, [[1, 1], [1, 4]])
    # ``x * x`` is the same term as ``x``.
    _, squared = formula_model_matrix("y ~ x * x", data)
    assert np.allclose(squared, [[1, 1], [1, 4]])


def test_wilkinson_formula_matches_the_column_interface():
    data = pl.DataFrame(
        {
            "y": [0.2, 1.4, 0.7, 1.1, 0.4, 1.8, 0.5, 1.2],
            "x": [0.1, 0.4, 0.2, 0.8, 0.3, 0.6, 0.9, 0.15],
            "stage": ["I", "II", "I", "III", "II", "I", "III", "II"],
            "g": [1, 1, 1, 2, 2, 2, 1, 2],
        }
    )
    formula = fit_ols(data, "y ~ x + stage")
    columns = fit_ols(data, "y", ["x", "stage"])
    assert formula.names == columns.names == ["(Intercept)", "x", "stageII", "stageIII"]
    np.testing.assert_allclose(formula.coefficients, columns.coefficients)
    np.testing.assert_allclose(formula.predict(), columns.predict())
    interacted = fit_ols(data, "y ~ x * stage")
    assert interacted.names == ["(Intercept)", "x", "stageII", "stageIII", "x:stageII", "x:stageIII"]
    fresh = data.with_columns(pl.Series("stage", ["II", "I", "III", "II", "I", "III", "I", "II"]))
    np.testing.assert_allclose(interacted.predict(data), interacted.x @ interacted.coefficients)
    assert interacted.predict(fresh).shape == (fresh.height,)
    with pytest.raises(ValueError, match="unseen levels"):
        interacted.predict(data.with_columns(pl.Series("stage", ["IV"] * data.height)))
    with pytest.raises(ValueError, match="not both"):
        fit_ols(data, "y ~ x", ["x"])
    binary = data.with_columns(pl.Series("y", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]))
    binomial = fit_glm(binary, "y ~ x", family="binomial")
    direct = fit_glm(binary, "y", ["x"], family="binomial")
    np.testing.assert_allclose(binomial.coefficients, direct.coefficients)


def test_survival_accepts_a_surv_formula():
    data = pl.DataFrame(
        {
            "time": [0.4, 1.2, 0.8, 2.0, 1.5, 0.6, 1.8, 0.9],
            "event": [1, 0, 1, 1, 0, 1, 1, 0],
            "x": [0.2, -0.4, 0.1, 0.5, -0.2, 0.3, 0.0, 0.4],
            "sex": ["f", "m", "f", "m", "f", "m", "m", "f"],
            "arm": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "entry": [0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    formula = cox_ph(data, "Surv(time, event) ~ x + sex")
    columns = cox_ph(data, "time", "event", ["x", "sex"])
    assert formula.names == columns.names == ["x", "sexm"]
    np.testing.assert_allclose(formula.coefficients, columns.coefficients)
    interacted = cox_ph(data, "Surv(time, event) ~ x * sex")
    assert interacted.names == ["x", "sexm", "x:sexm"]
    stratified = cox_ph(data, "Surv(time, event) ~ x + strata(arm)")
    direct = cox_ph(data, "time", "event", ["x"], strata="arm")
    assert stratified.names == ["x"]
    np.testing.assert_allclose(stratified.coefficients, direct.coefficients)
    counting = cox_ph(data, "Surv(entry, time, event) ~ x")
    entered = cox_ph(data, "time", "event", ["x"], entry="entry")
    np.testing.assert_allclose(counting.coefficients, entered.coefficients)
    clustered = cox_ph(data, "Surv(time, event) ~ x + cluster(id)")
    labeled = cox_ph(data, "time", "event", ["x"], cluster="id")
    np.testing.assert_allclose(clustered.coefficients, labeled.coefficients)
    np.testing.assert_allclose(clustered.covariance, labeled.covariance)
    aft = accelerated_failure(data, "Surv(time, event) ~ x + sex")
    aft_columns = accelerated_failure(data, "time", "event", ["x", "sex"])
    assert aft.names == aft_columns.names
    np.testing.assert_allclose(aft.coefficients, aft_columns.coefficients)
    competing = data.with_columns(pl.Series("status", [1, 0, 2, 1, 0, 1, 2, 0]))
    expanded = fine_gray(competing, "Surv(time, status) ~ x + sex", 1)
    assert expanded.columns[:2] == ["x", "sex"]
    assert "arm" not in expanded.columns
    plain = fine_gray(competing.select(["time", "status", "x", "sex"]), "time", "status", 1)
    np.testing.assert_allclose(expanded["fgwt"].to_numpy(), plain["fgwt"].to_numpy())
    gray = cox_ph(expanded, "Surv(fgstart, fgstop, fgstatus) ~ x + sex", weights="fgwt")
    gray_columns = cox_ph(expanded, "fgstop", "fgstatus", ["x", "sex"], entry="fgstart", weights="fgwt")
    np.testing.assert_allclose(gray.coefficients, gray_columns.coefficients)
    with pytest.raises(ValueError, match="belong in cox_ph"):
        fit_ols(data, "Surv(time, event) ~ x")
    with pytest.raises(ValueError, match="already names"):
        cox_ph(data, "Surv(time, event) ~ x", ["x"])


def test_fit_mixed_accepts_a_wilkinson_formula():
    data = pl.DataFrame(
        {
            "y": [1.2, 1.5, 0.9, 1.1, 2.4, 2.1, 2.8, 2.0, 0.3, 0.6, 0.1, 0.8],
            "x": [0.1, 0.4, 0.2, 0.5, 0.2, 0.6, 0.3, 0.7, 0.1, 0.5, 0.2, 0.4],
            "g": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        }
    )
    formula = fit_mixed(data, "y ~ x + (1 | g)")
    columns = fit_mixed(data, "y", ["x"], groups="g")
    assert formula.names == columns.names
    np.testing.assert_allclose(formula.coefficients, columns.coefficients)
    np.testing.assert_allclose(formula.group_covariance, columns.group_covariance)
    slope = fit_mixed(data, "y ~ x + (1 + x | g)", method="ml")
    direct = fit_mixed(data, "y", ["x"], groups="g", slopes=["x"], method="ml")
    np.testing.assert_allclose(slope.coefficients, direct.coefficients)
    np.testing.assert_allclose(slope.group_covariance, direct.group_covariance)
    no_intercept = fit_mixed(data, "y ~ 0 + x + (0 + x | g)")
    assert no_intercept.names == ["x"]
    assert no_intercept.random_names == ["x"]
    with pytest.raises(ValueError, match="\\|\\|"):
        fit_mixed(data, "y ~ x + (1 + x || g)")
    with pytest.raises(ValueError, match="several random-effect terms"):
        fit_mixed(data, "y ~ x + (1 | g) + (1 | g)")
    binomial = fit_mixed(
        data.with_columns((pl.col("y") > 1.0).cast(pl.Int8).alias("z")),
        "z ~ x + (1 | g)",
        family="binomial",
    )
    assert binomial.family == "binomial"
    assert binomial.engine == "laplace"
    tidy = binomial.tidy(exponentiate=True)
    assert "exp_estimate" in tidy.columns


def test_ols_keeps_qr_accuracy_on_an_uncentred_birth_year():
    # A birth-year column makes the scaled design ill-conditioned. The normal
    # equations then lose about cond^2 * eps, which the sandwich bread inherits.
    rng = np.random.default_rng(20261006)
    n = 1000
    birth = 1979.0 + rng.integers(0, 4, n) + rng.uniform(0.0, 1.0, n) / 12.0
    grade = rng.integers(0, 4, n).astype(float)
    y = 500.0 - 0.8 * (birth - 1980.0) + 12.0 * grade + rng.normal(0.0, 30.0, n)
    frame = pl.DataFrame({"y": y, "birth": birth, "grade": grade})
    fit = fit_ols(frame, "y", ["birth", "grade"])
    expected, *_ = np.linalg.lstsq(fit.x, y, rcond=None)
    np.testing.assert_allclose(fit.coefficients, expected, rtol=1e-10)
    q, r = np.linalg.qr(fit.x)
    r_inv = np.linalg.inv(r)
    np.testing.assert_allclose(fit.bread() / n, r_inv @ r_inv.T, rtol=1e-10)
