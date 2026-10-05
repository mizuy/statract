"""Tests for statract.cea.markov (3-state textbook cohort)."""

from __future__ import annotations

import numpy as np

from statract.cea.markov import simulate_cohort_markov


def _sick_dead_P(*, p_hs: float = 0.05, p_hd: float = 0.01, p_sd: float = 0.1) -> np.ndarray:
    """Healthy / Sick / Dead transition matrix (rows from, cols to)."""
    # Healthy → H, S, D
    # Sick → S, D (no recovery)
    # Dead → Dead
    return np.array(
        [
            [1.0 - p_hs - p_hd, p_hs, p_hd],
            [0.0, 1.0 - p_sd, p_sd],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_absorbing_dead_and_trace() -> None:
    P = _sick_dead_P()
    res = simulate_cohort_markov(
        states=("Healthy", "Sick", "Dead"),
        initial={"Healthy": 1.0},
        ages=range(40, 51),
        transition=P,
        utility={"Healthy": 1.0, "Sick": 0.6, "Dead": 0.0},
        cost={"Healthy": 0.0, "Sick": 1000.0, "Dead": 0.0},
        discount_rate=0.0,
        record_trace=True,
    )
    assert res.trace is not None
    assert res.trace.height == 11
    # Mass conserved
    totals = res.trace.select(
        (res.trace["Healthy"] + res.trace["Sick"] + res.trace["Dead"]).alias("t"),
    )["t"]
    np.testing.assert_allclose(totals.to_numpy(), np.ones(11), atol=1e-12)
    # Dead increases over time
    assert res.trace["Dead"][-1] > res.trace["Dead"][0]
    assert res.ly > 0
    assert res.qaly > 0
    assert res.cost >= 0


def test_discount_reduces_totals() -> None:
    P = _sick_dead_P()
    kwargs = dict(
        states=("Healthy", "Sick", "Dead"),
        initial=[1.0, 0.0, 0.0],
        ages=range(50, 81),
        transition=P,
        utility=[1.0, 0.5, 0.0],
        cost=[10.0, 100.0, 0.0],
    )
    undisc = simulate_cohort_markov(**kwargs, discount_rate=0.0)
    disc = simulate_cohort_markov(**kwargs, discount_rate=0.03)
    assert disc.cost < undisc.cost
    assert disc.qaly < undisc.qaly
    assert disc.ly < undisc.ly


def test_on_cycle_hook() -> None:
    """Event cost at age 45 and move some Healthy → Sick."""
    P = _sick_dead_P(p_hs=0.0, p_hd=0.0, p_sd=0.0)  # no natural transitions

    def on_cycle(age: int, mass: np.ndarray) -> tuple[np.ndarray, float, float]:
        if age != 45:
            return mass, 0.0, 0.0
        m = mass.copy()
        move = 0.2 * m[0]
        m[0] -= move
        m[1] += move
        return m, 500.0, 0.0

    res = simulate_cohort_markov(
        states=("Healthy", "Sick", "Dead"),
        initial=[1.0, 0.0, 0.0],
        ages=range(40, 50),
        transition=P,
        utility=[1.0, 0.5, 0.0],
        cost=0.0,
        discount_rate=0.0,
        on_cycle=on_cycle,
        record_trace=True,
    )
    assert res.cost == 500.0
    # After age 45 hook, Sick should be 0.2 at start of subsequent cycles
    row46 = res.trace.filter(res.trace["age"] == 46).row(0, named=True)
    np.testing.assert_allclose(row46["Sick"], 0.2)
    np.testing.assert_allclose(row46["Healthy"], 0.8)


def test_age_dependent_transition_callable() -> None:
    def transition(age: int, mass: np.ndarray) -> np.ndarray:
        p_hs = 0.01 * (age - 40)
        return mass @ _sick_dead_P(p_hs=min(p_hs, 0.2))

    res = simulate_cohort_markov(
        states=("Healthy", "Sick", "Dead"),
        initial={"Healthy": 1.0},
        ages=range(40, 60),
        transition=transition,
        utility=1.0,
        living_mask=[True, True, False],
    )
    assert 0 < res.ly <= 20.0
