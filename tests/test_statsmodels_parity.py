"""Fits that used to call statsmodels, checked against values it produced.

The expected numbers come from statsmodels 0.15.0 on the same seeded data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from statract import fit_glm, fit_ols, proportion_ci
from statract.models.matching import _probit_probability
from statract.surv.diagnostics import _maybe_lowess


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    n = 60
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1 + 0.5 * x1 - x2 + rng.normal(size=n)
    w = rng.uniform(0.5, 2, n)
    off = rng.normal(size=n) * 0.3
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x1 - 0.5 * x2)))).astype(int)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": 2 * x1 - x2, "w": w, "off": off, "t": t})


def _se(fit) -> np.ndarray:
    return np.sqrt(np.diag(fit.covariance))


def test_src_has_no_statsmodels_import() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "statract"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import statsmodels" in text or "from statsmodels" in text:
            offenders.append(str(path.relative_to(root.parents[1])))
    assert offenders == []


def test_gaussian_glm_with_weights_and_offset() -> None:
    fit = fit_glm(_data(), "y", ["x1", "x2"], family="gaussian", weights="w", offset="off")
    np.testing.assert_allclose(
        fit.coefficients, [0.9158365722042258, 0.46480368939618055, -0.9295974578029969], rtol=1e-10
    )
    np.testing.assert_allclose(
        _se(fit), [0.1407691087229088, 0.16361824954878879, 0.16539768217971834], rtol=1e-10
    )
    assert fit.log_likelihood == pytest.approx(-90.47812467951276, rel=1e-12)
    assert fit.deviance == pytest.approx(81.70029366331164, rel=1e-12)
    assert fit.scale == pytest.approx(1.433338485321257, rel=1e-12)
    assert fit.aic == pytest.approx(186.95624935902552, rel=1e-12)
    assert fit.bic == pytest.approx(193.23928304569182, rel=1e-12)
    assert fit.residual_df == 57


def test_gaussian_glm_rank_deficient_uses_minimum_norm() -> None:
    fit = fit_glm(_data(), "y", ["x1", "x2", "x3"], family="gaussian", weights="w", offset="off")
    np.testing.assert_allclose(
        fit.coefficients,
        [0.9158365722042265, -0.15493125613560527, -0.6197299850371034, 0.3098674727658925],
        rtol=1e-10,
    )
    assert fit.log_likelihood == pytest.approx(-90.47812467951276, rel=1e-12)
    assert fit.aic == pytest.approx(186.95624935902552, rel=1e-12)
    assert fit.residual_df == 57


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (
            None,
            {
                "coef": [0.8798916511029344, -0.19263163020708973, -0.6750207749591686, 0.28975751454498955],
                "se": [0.1270582365903756, 0.07384467553715358, 0.14098089824563553, 0.05338092865439118],
                "llf": -82.39498978052873,
                "aic": 170.78997956105746,
                "bic": 177.07301324772376,
                "r2": 0.4643784239669698,
                "adj_r2": 0.4455846844570389,
                "f": 24.709208282980907,
            },
        ),
        (
            "w",
            {
                "coef": [0.8702934761260745, -0.1700336895664365, -0.6407377379498975, 0.3006703588170242],
                "se": [0.12838006815915118, 0.07341582258588272, 0.13872318983881565, 0.05396443028865167],
                "llf": -84.9505722048853,
                "aic": 175.9011444097706,
                "bic": 182.1841780964369,
                "r2": 0.46930021608380157,
                "adj_r2": 0.45067917103411037,
                "f": 25.20267873427355,
            },
        ),
    ],
)
def test_ols_rank_deficient_uses_pinv(weights, expected) -> None:
    fit = fit_ols(_data(), "y", ["x1", "x2", "x3"], weights=weights)
    np.testing.assert_allclose(fit.coefficients, expected["coef"], rtol=1e-10)
    np.testing.assert_allclose(_se(fit), expected["se"], rtol=1e-10)
    assert fit.residual_df == 57
    assert fit.log_likelihood == pytest.approx(expected["llf"], rel=1e-12)
    assert fit.aic == pytest.approx(expected["aic"], rel=1e-12)
    assert fit.bic == pytest.approx(expected["bic"], rel=1e-12)
    assert fit.r_squared == pytest.approx(expected["r2"], rel=1e-12)
    assert fit.adj_r_squared == pytest.approx(expected["adj_r2"], rel=1e-12)
    assert fit.f_statistic == pytest.approx(expected["f"], rel=1e-12)


def test_probit_propensity() -> None:
    probability, _rows = _probit_probability(_data(), "t", ["x1", "x2"])
    np.testing.assert_allclose(
        probability[:5],
        [0.6248680017597386, 0.7080020970803229, 0.7458059246254912, 0.3675889858819978, 0.7343090902779019],
        rtol=1e-7,
    )


def test_lowess_matches_statsmodels() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = np.sin(x) + rng.normal(size=200) * 0.3
    xs, fitted = _maybe_lowess(x, y)
    np.testing.assert_array_equal(xs, np.sort(x))
    np.testing.assert_allclose(
        fitted[::40],
        [-1.1423431442062395, -0.6757863933376617, -0.24780151830577682, 0.2186429693208238, 0.6462464372386598],
        rtol=1e-12,
    )
    _xt, tied = _maybe_lowess(np.round(x, 1), y)
    np.testing.assert_allclose(
        tied[::40],
        [-1.1331920524091992, -0.667819566315129, -0.2199512118302377, 0.2487165932544484, 0.6412880187024889],
        rtol=1e-12,
    )
    _xs, narrow = _maybe_lowess(x[:30], y[:30], frac=0.3)
    np.testing.assert_allclose(
        narrow[::6],
        [-0.9270186597442176, -0.5813220613353184, -0.30069223697114206, 0.2858472117357837, 0.6745480823483054],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, 0, None], (0.21899421248918433, 1.0)),
        ([0, 0, 0], (0.0, 0.0)),
        ([1, 1, 1, 1], (1.0, 1.0)),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], (0.05185365230812369, 0.49360089314642175)),
    ],
)
def test_proportion_ci_is_clipped_wald(values, expected) -> None:
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Int64})
    ci = frame.select(proportion_ci(pl.col("x"), alpha=0.1).alias("ci"))["ci"][0]
    assert (ci["lo"], ci["hi"]) == pytest.approx(expected, rel=1e-12)
