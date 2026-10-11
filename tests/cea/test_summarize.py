"""Tests for statract.cea.summarize (dampack-style ICER tables)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statract.cea.summarize import STATUS_D, STATUS_ED, STATUS_ND, calculate_icers, net_monetary_benefit


def test_nmb_basic() -> None:
    nmb = net_monetary_benefit([100.0, 200.0], [1.0, 1.5], wtp=100.0)
    np.testing.assert_allclose(nmb, [0.0, -50.0])


def test_two_strategies_icer() -> None:
    """Cheap/low-effect vs expensive/high-effect → both ND, ICER on second."""
    df = calculate_icers(
        cost=[10_000.0, 25_000.0],
        effect=[5.0, 6.0],
        strategies=["A", "B"],
    )
    assert df["strategy"].to_list() == ["A", "B"]
    assert df["status"].to_list() == [STATUS_ND, STATUS_ND]
    assert math.isnan(df["icer"][0])
    np.testing.assert_allclose(df["icer"][1], 15_000.0)


def test_strong_dominance() -> None:
    """B costs more and has less effect than A → D."""
    df = calculate_icers(
        cost=[10.0, 20.0, 15.0],
        effect=[2.0, 1.0, 3.0],
        strategies=["A", "B", "C"],
    )
    by = {r["strategy"]: r for r in df.iter_rows(named=True)}
    assert by["B"]["status"] == STATUS_D
    assert by["A"]["status"] == STATUS_ND
    assert by["C"]["status"] == STATUS_ND
    # Frontier A → C
    np.testing.assert_allclose(by["C"]["icer"], (15.0 - 10.0) / (3.0 - 2.0))


def test_extended_dominance() -> None:
    """Classic 3-strategy extended dominance example.

    A: cost 0, effect 0
    B: cost 100, effect 0.5  (ED: ICER A→B = 200, B→C = 50 < 200)
    C: cost 150, effect 1.5
    """
    df = calculate_icers(
        cost=[0.0, 100.0, 150.0],
        effect=[0.0, 0.5, 1.5],
        strategies=["A", "B", "C"],
    )
    by = {r["strategy"]: r for r in df.iter_rows(named=True)}
    assert by["A"]["status"] == STATUS_ND
    assert by["B"]["status"] == STATUS_ED
    assert by["C"]["status"] == STATUS_ND
    # Frontier A → C: ICER = 150 / 1.5 = 100
    np.testing.assert_allclose(by["C"]["icer"], 100.0)
    assert math.isnan(by["B"]["icer"])


def test_duplicate_strategy_names_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        calculate_icers([1.0, 2.0], [1.0, 2.0], strategies=["A", "A"])


def test_length_mismatch_raise() -> None:
    with pytest.raises(ValueError):
        calculate_icers([1.0, 2.0], [1.0], strategies=["A", "B"])
