"""E-values against published examples (VanderWeele and Ding 2017) and the theory text."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statract import e_value, e_value_curve


def _row(frame):
    return frame.row(0, named=True)


def test_risk_ratio_paper_example():
    # Annals 2017: RR 3.9 (1.8-8.7) -> E-values 7.26 and 3.0.
    r = _row(e_value(3.9, 1.8, 8.7))
    assert r["e_value"] == pytest.approx(3.9 + math.sqrt(3.9 * 2.9))
    assert round(r["e_value"], 2) == 7.26
    assert r["e_value_ci"] == pytest.approx(3.0)


def test_protective_theory_text():
    # Chapter 6: standardized RR 0.498, upper limit 0.755 (0.76 in the text) -> 3.43 and 1.98.
    r = _row(e_value(0.498, 0.33, 0.755))
    assert round(r["e_value"], 2) == 3.43
    assert round(r["e_value_ci"], 2) == 1.98


def test_interval_crossing_null_gives_one():
    assert _row(e_value(1.4, 0.9, 2.1))["e_value_ci"] == 1.0
    assert _row(e_value(0.8, 0.6, 1.05))["e_value_ci"] == 1.0
    assert _row(e_value(1.4))["e_value_ci"] is None


def test_odds_and_hazard_ratio_conversions():
    common_or = _row(e_value(4.0, 2.25, 9.0, measure="OR", rare=False))
    assert common_or["rr"] == pytest.approx(2.0)
    assert common_or["rr_lower"] == pytest.approx(1.5)
    assert common_or["e_value"] == pytest.approx(2 + math.sqrt(2))
    rare_or = _row(e_value(4.0, 2.25, 9.0, measure="OR", rare=True))
    assert rare_or["rr"] == 4.0
    hr = 0.6
    expect = (1 - 0.5 ** math.sqrt(hr)) / (1 - 0.5 ** math.sqrt(1 / hr))
    assert _row(e_value(hr, measure="HR", rare=False))["rr"] == pytest.approx(expect)
    with pytest.raises(ValueError):
        e_value(2.0, measure="OR")


def test_smd():
    r = _row(e_value(0.5, measure="SMD", se=0.1))
    assert r["rr"] == pytest.approx(math.exp(0.455))
    assert r["rr_lower"] == pytest.approx(math.exp(0.455 - 0.178))
    rr_lo = math.exp(0.455 - 0.178)
    assert r["e_value_ci"] == pytest.approx(rr_lo + math.sqrt(rr_lo * (rr_lo - 1)))


def test_shift_to_true():
    r = _row(e_value(3.0, true=1.5))
    assert r["e_value"] == pytest.approx(2 + math.sqrt(2))


def test_curve_meets_diagonal_at_e_value():
    rr = 2.01
    ev = _row(e_value(rr))["e_value"]
    np.testing.assert_allclose(e_value_curve(rr, ev), ev, rtol=1e-12)
    assert np.isinf(e_value_curve(rr, 1.5))
    np.testing.assert_allclose(e_value_curve(1 / rr, ev), ev, rtol=1e-12)
