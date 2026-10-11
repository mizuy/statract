"""Properties of the cubic regression, cyclic cubic, and P-spline bases.

No R fixture covers ``ps`` or ``cc``; these tests check the defining
properties from Wood (2017, *Generalized Additive Models*, 2nd ed.,
secs. 5.3.1-5.3.3) and Eilers & Marx (1996).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from statract.models.gam import _cr_basis, gam, smooth
from statract.models.gam.general import _cc, _ps


def _integral_curvature_squared(
    f, lower: float, upper: float, n: int = 200001
) -> float:
    grid = np.linspace(lower, upper, n)
    step = grid[1] - grid[0]
    values = f(grid)
    second = np.gradient(np.gradient(values, step), step)
    return float(np.trapezoid(second[2:-2] ** 2, grid[2:-2]))


def test_cr_interpolates_knot_values_and_penalty_is_curvature() -> None:
    knots = np.array([0.0, 0.7, 1.5, 2.0, 3.2, 4.0])
    basis, penalty = _cr_basis(knots, knots)
    np.testing.assert_allclose(basis, np.eye(knots.size), atol=1e-12)
    beta = np.array([0.3, -1.0, 0.5, 2.0, -0.4, 1.1])
    rough = _integral_curvature_squared(
        lambda g: _cr_basis(g, knots, penalty=False)[0] @ beta, 0.0, 4.0
    )
    np.testing.assert_allclose(beta @ penalty @ beta, rough, rtol=1e-3)
    # Linear beyond the end knots: second differences vanish.
    outside = (
        _cr_basis(np.array([-2.0, -1.0, 0.0, 4.0, 5.0, 6.0]), knots, penalty=False)[0]
        @ beta
    )
    np.testing.assert_allclose(np.diff(outside[:3], 2), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diff(outside[3:], 2), 0.0, atol=1e-12)


def test_ps_partition_of_unity_and_uniform_values() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(2.0, 7.0, 300)
    k = 10
    basis, penalty, null_dim, rebuild = _ps("x", x, k)
    assert basis.shape == (300, k)
    assert null_dim == 2
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rebuild({"x": x}), basis)
    # Penalty rank k - 2; constant and linear coefficient sequences are unpenalized.
    assert np.linalg.matrix_rank(penalty) == k - 2
    index = np.arange(k, dtype=float)
    np.testing.assert_allclose(penalty @ np.ones(k), 0.0, atol=1e-12)
    np.testing.assert_allclose(penalty @ index, 0.0, atol=1e-12)
    # Second-difference penalty: sum of squared second differences.
    beta = rng.normal(size=k)
    np.testing.assert_allclose(beta @ penalty @ beta, np.sum(np.diff(beta, 2) ** 2))
    # Equally spaced knots over the range widened by 0.1 % per side; at an
    # interior knot a cubic B-spline basis takes the values 1/6, 2/3, 1/6.
    low, high = x.min(), x.max()
    span = high - low
    step = (span * 1.002) / (k - 3)
    knot = low - 0.001 * span + 2 * step
    row = rebuild({"x": np.array([knot])})[0]
    np.testing.assert_allclose(np.sort(row[row > 1e-12]), [1 / 6, 1 / 6, 2 / 3])
    # Linear coefficients reproduce a linear function of x.
    grid = np.linspace(low, high, 50)
    line = rebuild({"x": grid}) @ index
    np.testing.assert_allclose(np.diff(line, 2), 0.0, atol=1e-10)


def test_cc_is_periodic_and_penalty_is_curvature() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 24.0, 400)
    k = 8
    basis, penalty, null_dim, rebuild = _cc("x", x, k)
    assert basis.shape == (400, k - 1)
    assert null_dim == 1
    assert np.linalg.matrix_rank(penalty) == k - 2
    np.testing.assert_allclose(penalty @ np.ones(k - 1), 0.0, atol=1e-10)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    knots = np.quantile(np.unique(x), np.linspace(0, 1, k))
    at_knots = rebuild({"x": knots})
    np.testing.assert_allclose(at_knots[:-1], np.eye(k - 1), atol=1e-12)
    np.testing.assert_allclose(at_knots[-1], at_knots[0], atol=1e-12)

    beta = rng.normal(size=k - 1)

    def f(g):
        return rebuild({"x": np.asarray(g, dtype=float)}) @ beta

    lo, hi = knots[0], knots[-1]
    period = hi - lo
    # Value, slope and curvature match across the join (one-sided differences).
    eps = 1e-3
    left = f(hi - np.array([0.0, eps, 2 * eps]))
    right = f(lo + np.array([0.0, eps, 2 * eps]))
    np.testing.assert_allclose(left[0], right[0], atol=1e-12)
    np.testing.assert_allclose(
        (left[0] - left[1]) / eps, (right[1] - right[0]) / eps, atol=1e-2
    )
    np.testing.assert_allclose(
        (left[0] - 2 * left[1] + left[2]) / eps**2,
        (right[0] - 2 * right[1] + right[2]) / eps**2,
        atol=5e-2,
    )
    # Values outside are wrapped into the period.
    probe = np.array([lo + 0.3, lo + 5.1, hi - 0.2])
    np.testing.assert_allclose(f(probe + period), f(probe), atol=1e-12)
    np.testing.assert_allclose(f(probe - 2 * period), f(probe), atol=1e-12)
    rough = _integral_curvature_squared(f, lo, hi)
    np.testing.assert_allclose(beta @ penalty @ beta, rough, rtol=1e-3)


def test_gam_fits_ps_and_cc_smooths() -> None:
    rng = np.random.default_rng(3)
    n = 400
    hour = rng.uniform(0.0, 24.0, n)
    z = rng.uniform(-2.0, 2.0, n)
    y = np.sin(2 * np.pi * hour / 24) + z**2 + rng.normal(0.0, 0.2, n)
    data = pl.DataFrame({"hour": hour, "z": z, "y": y})
    fit = gam(
        data, "y", [smooth("hour", k=10, basis="cc"), smooth("z", k=10, basis="ps")]
    )
    fitted = np.asarray(fit.predict(data))
    truth = np.sin(2 * np.pi * hour / 24) + z**2
    assert np.sqrt(np.mean((fitted - truth) ** 2)) < 0.08
