"""AFT Newton steps stay uphill when most follow-up is censored."""

import numpy as np
import polars as pl

from statract import accelerated_failure


def _heavily_censored(n: int = 3000, seed: int = 15) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.binomial(1, 0.4, n)
    lp = 0.5 * x - 0.4 * z
    # Weibull shape 0.5, about 5% events by the end of follow-up at 30.
    t_event = (rng.exponential(1.0, n) / (0.009 * np.exp(lp))) ** 2
    t_loss = rng.exponential(250.0, n)
    time = np.minimum.reduce([t_event, t_loss, np.full(n, 30.0)])
    event = (t_event <= np.minimum(t_loss, 30.0)).astype(int)
    return pl.DataFrame({"time": np.maximum(time, 0.01), "event": event, "x": x, "z": z})


def test_weibull_converges_with_heavy_censoring() -> None:
    data = _heavily_censored()
    assert data["event"].mean() < 0.1
    weibull = accelerated_failure(data, "Surv(time, event) ~ x + z", distribution="weibull")
    expo = accelerated_failure(data, "Surv(time, event) ~ x + z", distribution="exponential")
    assert weibull.converged
    # The exponential model is the Weibull model with scale 1.
    assert weibull.log_likelihood >= expo.log_likelihood
    # True shape 0.5 is scale 2 on the log-time scale.
    assert 1.5 < weibull.scale < 2.7
