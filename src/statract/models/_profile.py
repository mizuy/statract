"""Profile-likelihood interval limits by exact root finding.

A limit is where the signed root of the likelihood-ratio drop,
``sign(b - b_hat) * sqrt(drop(b))``, equals a normal quantile. ``MASS``'s
``confint.profile.glm`` interpolates a spline through a few profile points;
here Brent's method solves for the crossing itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import optimize


def profile_limit(
    drop: Callable[[float], float],
    estimate: float,
    scale: float,
    quantile: float,
    *,
    max_doublings: int = 40,
) -> float:
    """Value of the coefficient on one side where ``sqrt(drop(b))`` reaches ``|quantile|``.

    ``drop(b)`` is the likelihood-ratio statistic with the coefficient fixed at
    ``b`` (zero at ``estimate``). The side is the sign of ``quantile``. ``scale``
    (a standard error) sets the first bracket at ``estimate + quantile * scale``.
    When the drop levels off before the quantile, as under separation, the
    limit is infinite.
    """
    sign = 1.0 if quantile > 0 else -1.0
    target = abs(float(quantile))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(estimate), 1.0)

    def gap(b: float) -> float:
        value = drop(b)
        if not np.isfinite(value):
            return np.inf
        return float(np.sqrt(max(value, 0.0))) - target

    lo = float(estimate)
    distance = target * scale
    previous = -target
    for _ in range(max_doublings):
        hi = float(estimate + sign * distance)
        value = gap(hi)
        if value >= 0:
            a, b = (lo, hi) if sign > 0 else (hi, lo)
            xtol = 1e-12 * max(scale, abs(hi))
            return float(optimize.brentq(gap, a, b, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200))
        if value - previous < 1e-10 * target and distance > 8 * target * scale:
            break
        previous = value
        lo = hi
        distance *= 2.0
    return sign * np.inf
