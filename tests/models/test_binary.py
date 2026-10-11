"""Tests for statract.models.binary."""

from __future__ import annotations

import numpy as np

from statract.models.binary import binary_perf, threshold_tradeoff


def test_binary_perf_basic() -> None:
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    perf = binary_perf(y, pred)
    assert perf["tp"] == 1
    assert perf["fn"] == 1
    assert perf["fp"] == 1
    assert perf["tn"] == 1
    assert perf["sensitivity"] == 0.5
    assert perf["specificity"] == 0.5
    assert perf["undertreatment_rate"] == 0.5
    assert perf["overtreatment_rate"] == 0.5


def test_threshold_tradeoff_shape() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    df = threshold_tradeoff(y, p, thresholds=np.array([0.5]))
    assert df.height == 1
    assert "sensitivity" in df.columns
