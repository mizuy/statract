"""Log(scale) inference in ``accelerated_failure``.

The survival::rotterdam Weibull example once reported ``Log(scale)`` with a
zero standard error: a plain Newton step from the least-squares start pointed
downhill, the line search stalled at the start, and the indefinite Hessian there
gave a negative variance that the tidy table clipped to zero. The checks below
use a finite-difference Hessian of the Weibull log-likelihood, not R values.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import statract.surv.aft as aft_module
from statract import accelerated_failure

FORMULA = "Surv(time, event) ~ x + z"


def _days_scale(seed: int = 0, n: int = 400) -> pl.DataFrame:
    """Follow-up in days with about 30% events: the least-squares start sits far from the MLE."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.binomial(1, 0.5, n)
    t = np.exp(9.0 - 0.5 * x - 0.3 * z + 0.75 * np.log(rng.exponential(size=n)))
    c = rng.uniform(300.0, 6000.0, n)
    return pl.DataFrame(
        {"time": np.maximum(np.minimum(t, c), 1.0), "event": (t <= c).astype(int), "x": x, "z": z}
    )


def _weibull_negloglik(theta: np.ndarray, x: np.ndarray, y: np.ndarray, event: np.ndarray) -> float:
    # Extreme-value density on log time: log f(u) = u - exp(u), log S(u) = -exp(u).
    beta, log_scale = theta[:-1], theta[-1]
    scale = np.exp(log_scale)
    u = (y - x @ beta) / scale
    ll = np.where(event, u - np.exp(u) - log_scale - y, -np.exp(u))
    return -float(np.sum(ll))


def _numeric_hessian(fun, theta: np.ndarray, h: float = 1e-4) -> np.ndarray:
    k = theta.shape[0]
    eye = np.eye(k) * h
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            out[i, j] = (
                fun(theta + eye[i] + eye[j])
                - fun(theta + eye[i] - eye[j])
                - fun(theta - eye[i] + eye[j])
                + fun(theta - eye[i] - eye[j])
            ) / (4 * h * h)
    return out


def test_weibull_log_scale_se_matches_full_hessian() -> None:
    data = _days_scale()
    fit = accelerated_failure(data, FORMULA, distribution="weibull")
    assert fit.converged
    assert fit.names[-1] == "Log(scale)"
    event = data["event"].to_numpy().astype(bool)

    def nll(theta: np.ndarray) -> float:
        return _weibull_negloglik(theta, fit.x, fit.y, event)

    assert nll(fit.coefficients) == pytest.approx(-fit.log_likelihood, rel=1e-10)
    reference = np.sqrt(np.diag(np.linalg.inv(_numeric_hessian(nll, fit.coefficients))))

    tidy = fit.tidy(exponentiate=True)
    row = tidy.filter(pl.col("term") == "Log(scale)").row(0, named=True)
    np.testing.assert_allclose(tidy["std_error"].to_numpy(), reference, rtol=1e-4)
    assert row["std_error"] > 0
    assert row["statistic"] == pytest.approx(row["estimate"] / row["std_error"])
    assert row["conf_low"] < row["estimate"] < row["conf_high"]
    assert row["conf_high"] - row["conf_low"] == pytest.approx(2 * 1.959963984540054 * row["std_error"])
    assert row["exp_estimate"] == pytest.approx(fit.scale)
    assert row["exp_conf_low"] == pytest.approx(np.exp(row["conf_low"]))
    assert row["exp_conf_high"] == pytest.approx(np.exp(row["conf_high"]))


def test_non_converged_fit_warns_and_reports_no_zero_se(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reinstate the undamped Newton step that stalled on this kind of data.
    def plain_newton(hess: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return np.linalg.solve(-hess, grad)

    monkeypatch.setattr(aft_module, "_ascent_step", plain_newton)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = accelerated_failure(_days_scale(), FORMULA, distribution="weibull")
    assert not fit.converged
    assert np.diag(fit.covariance)[-1] <= 0
    row = fit.tidy(exponentiate=True).filter(pl.col("term") == "Log(scale)").row(0, named=True)
    assert np.isfinite(row["estimate"])
    for column in ("std_error", "statistic", "p_value", "conf_low", "conf_high", "exp_conf_low", "exp_conf_high"):
        assert np.isnan(row[column]), column


def test_converged_fit_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        accelerated_failure(_days_scale(), FORMULA, distribution="lognormal")
