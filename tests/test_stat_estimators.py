"""API contracts for the Python estimators. Numeric parity with R lives in tests/r_oracle."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from statract import (
    GLMHelper,
    accelerated_failure,
    cox_ph,
    fine_gray,
    fit_glm,
    fit_mixed,
    fit_ols,
    gam,
    match_sample,
    psmatch,
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


def test_matrix_distance_rejects_subclass_matching():
    data = pl.DataFrame({"treat": [1, 1, 0, 0], "x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 0.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="subclass"):
        match_sample(data, "treat", ["x1", "x2"], method="subclass", distance="mahalanobis")


def test_replaced_helpers_warn():
    data = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "treatment": [1, 1, 0, 0, 1, 0],
            "age": [45, 50, 55, 60, 65, 70],
            "gender": [0, 1, 0, 1, 0, 1],
            "outcome": [1, 1, 0, 0, 1, 0],
        }
    )
    with pytest.warns(DeprecationWarning, match="match_sample"):
        with pytest.raises(NotImplementedError, match="match_sample"):
            psmatch(data, "treatment", ["age", "gender"], "id")
    with pytest.warns(DeprecationWarning, match="fit_glm"):
        with pytest.raises(NotImplementedError, match="fit_glm"):
            GLMHelper("outcome ~ treatment + age", data)

    import statsmodels.api as sm

    y = [1.0, 2.0, 3.0, 4.0]
    x = sm.add_constant([0.0, 1.0, 2.0, 3.0])
    result = sm.OLS(y, x).fit()
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
    with pytest.raises(ValueError, match="exactly one"):
        fit_mixed(data, "y ~ x + (1 | g) + (1 | g)")
    binomial = fit_mixed(
        data.with_columns((pl.col("y") > 1.0).cast(pl.Int8).alias("z")),
        "z ~ x + (1 | g)",
        family="binomial",
    )
    assert binomial.family == "binomial"
    assert binomial.engine == "lme"
    tidy = binomial.tidy(exponentiate=True)
    assert "exp_estimate" in tidy.columns
