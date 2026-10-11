"""E-values for unmeasured confounding (VanderWeele and Ding 2017).

``e_value`` follows the formulas of the R package ``EValue``
(``evalues.RR``, ``evalues.OR``, ``evalues.HR``, ``evalues.MD``). The estimate
is first put on the risk-ratio scale:

- ``"RR"``: as is.
- ``"OR"`` and ``"HR"`` with ``rare=True`` (outcome below about 15%): as is.
- ``"OR"`` with ``rare=False``: ``sqrt(OR)``.
- ``"HR"`` with ``rare=False``: ``(1 - 0.5**sqrt(HR)) / (1 - 0.5**sqrt(1/HR))``.
- ``"SMD"`` (standardized mean difference ``d`` with standard error ``se``):
  ``exp(0.91 d)``, interval ``exp(0.91 d ± 1.78 se)``.

For a risk ratio ``RR`` above ``true`` the E-value is ``r + sqrt(r (r - 1))``
with ``r = RR / true``; below ``true`` it uses ``r = true / RR``. For the
interval only the limit nearer ``true`` gets an E-value; it is 1 when the
interval crosses ``true``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import polars as pl

Measure = Literal["RR", "OR", "HR", "SMD"]
_MEASURES = ("RR", "OR", "HR", "SMD")


def _to_rr(value: float, measure: str, rare: bool) -> float:
    if measure == "RR" or (measure in ("OR", "HR") and rare):
        return value
    if measure == "OR":
        return math.sqrt(value)
    if measure == "HR":
        return (1 - 0.5 ** math.sqrt(value)) / (1 - 0.5 ** math.sqrt(1 / value))
    raise ValueError(measure)


def _threshold(rr: float, true: float) -> float:
    r = rr / true if rr > true else true / rr
    return r + math.sqrt(r * (r - 1))


def e_value(
    estimate: float,
    lower: float | None = None,
    upper: float | None = None,
    *,
    measure: Measure = "RR",
    rare: bool | None = None,
    se: float | None = None,
    true: float | None = None,
) -> pl.DataFrame:
    """E-value of an estimate and of its confidence limit nearer the null.

    ``measure`` is ``"RR"``, ``"OR"``, ``"HR"`` or ``"SMD"``. ``rare`` is
    required for ``"OR"`` and ``"HR"``. ``"SMD"`` needs ``se`` and builds the
    interval itself. ``true`` is the value to shift the estimate to, on the
    scale of ``measure`` (default 1, or 0 for ``"SMD"``).

    Returns one row: ``rr``, ``rr_lower``, ``rr_upper`` (the estimate and
    limits on the risk-ratio scale), ``e_value`` and ``e_value_ci``.
    ``e_value_ci`` is null when no limit is given.
    """
    if measure not in _MEASURES:
        raise ValueError(f"measure must be one of {_MEASURES}, got {measure!r}")
    if measure in ("OR", "HR") and rare is None:
        raise ValueError(f"measure={measure!r} needs rare=True or rare=False")
    if measure == "SMD":
        if se is None:
            raise ValueError('measure="SMD" needs se')
        if lower is not None or upper is not None:
            raise ValueError('measure="SMD" builds the interval from se; do not pass lower or upper')
        if se <= 0:
            raise ValueError("se must be positive")
        d = float(estimate)
        rr = math.exp(0.91 * d)
        rr_lo = math.exp(0.91 * d - 1.78 * se)
        rr_hi = math.exp(0.91 * d + 1.78 * se)
        true_rr = math.exp(0.91 * (0.0 if true is None else float(true)))
    else:
        for name, v in (("estimate", estimate), ("lower", lower), ("upper", upper)):
            if v is not None and not v > 0:
                raise ValueError(f"{name} must be positive on the {measure} scale, got {v}")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower must not exceed upper")
        if lower is not None and lower > estimate or upper is not None and upper < estimate:
            raise ValueError("the estimate must lie inside its interval")
        flag = bool(rare)
        rr = _to_rr(float(estimate), measure, flag)
        rr_lo = None if lower is None else _to_rr(float(lower), measure, flag)
        rr_hi = None if upper is None else _to_rr(float(upper), measure, flag)
        true_rr = _to_rr(1.0 if true is None else float(true), measure, flag)
        if not true_rr > 0:
            raise ValueError("true must be positive")

    point = 1.0 if rr == true_rr else _threshold(rr, true_rr)
    ci: float | None
    if rr > true_rr:
        ci = None if rr_lo is None else (1.0 if rr_lo <= true_rr else _threshold(rr_lo, true_rr))
    elif rr < true_rr:
        ci = None if rr_hi is None else (1.0 if rr_hi >= true_rr else _threshold(rr_hi, true_rr))
    else:
        ci = None if rr_lo is None and rr_hi is None else 1.0
    return pl.DataFrame(
        {
            "rr": [rr],
            "rr_lower": [rr_lo],
            "rr_upper": [rr_hi],
            "e_value": [point],
            "e_value_ci": [ci],
        },
        schema={k: pl.Float64 for k in ("rr", "rr_lower", "rr_upper", "e_value", "e_value_ci")},
    )


def e_value_curve(rr: float, exposure_rr: np.ndarray | float) -> np.ndarray:
    """Confounder–outcome risk ratio that, with ``exposure_rr``, just explains away ``rr``.

    The bias factor of Ding and VanderWeele (2016) equals ``rr`` on this curve:
    ``RR_UY = rr (1 - RR_AU) / (rr - RR_AU)`` for ``RR_AU > rr`` (with ``rr``
    above 1; a protective ``rr`` is inverted first). Points with
    ``RR_AU <= rr`` cannot explain it away and give ``inf``. The curve meets
    the diagonal at the E-value.
    """
    if not rr > 0:
        raise ValueError("rr must be positive")
    r = rr if rr >= 1 else 1 / rr
    x = np.asarray(exposure_rr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x > r, r * (1 - x) / (r - x), np.inf)
    return out
