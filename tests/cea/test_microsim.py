"""Tests for statract.cea.microsim (3-state textbook model against the cohort kernel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from statract.cea import (
    MicrosimResult,
    distribution_table,
    simulate_cohort_markov,
    simulate_microsim,
)

AGES = range(50, 71)
UTILITY = np.array([1.0, 0.6, 0.0])
COST = np.array([0.0, 1000.0, 0.0])
DISCOUNT = 0.03


@dataclass
class State:
    p_sick: float
    st: np.ndarray
    cost: np.ndarray
    qaly: np.ndarray
    ly: np.ndarray


@dataclass(frozen=True)
class SickDead:
    """Healthy -> Sick -> Dead. An arm is the Healthy -> Sick probability."""

    n_cycles: int = len(AGES)
    n_streams: int = 1
    n_cells: int = 1
    p_healthy_dead: float = 0.01
    p_sick_dead: float = 0.1

    def init_population(self, ids: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return {"cell": ids % self.n_cells}

    def init_arm(self, population: dict[str, np.ndarray], arm: float) -> State:
        n = population["cell"].size
        return State(arm, np.zeros(n, dtype=np.int8), np.zeros(n), np.zeros(n), np.zeros(n))

    def step(self, state: State, cycle: int, uniforms: np.ndarray) -> None:
        df = (1.0 + DISCOUNT) ** -cycle
        st = state.st
        state.qaly += df * UTILITY[st]
        state.cost += df * COST[st]
        state.ly += df * (st != 2)
        u = uniforms[0]
        healthy, sick = st == 0, st == 1
        st[healthy & (u < self.p_healthy_dead)] = 2
        st[healthy & (u >= self.p_healthy_dead) & (u < self.p_healthy_dead + state.p_sick)] = 1
        st[sick & (u < self.p_sick_dead)] = 2

    def is_done(self, state: State) -> bool:
        return bool((state.st == 2).all())

    def outcomes(self, state: State) -> dict[str, np.ndarray]:
        return {"cost": state.cost, "qaly": state.qaly, "ly": state.ly}

    def tallies(self, state: State) -> dict[str, np.ndarray]:
        return {"alive": np.array([float((state.st != 2).sum())])}


def _run(model: SickDead | None = None, *, arms=(("low", 0.10), ("high", 0.05)), **kwargs) -> MicrosimResult:
    options = {"n_trials": 200_000, "comparator": "low", "intervention": "high", "wtp": 1000.0, "seed": 3}
    return simulate_microsim(model or SickDead(), list(arms), **(options | kwargs))


def _cohort(p_sick: float):
    p = np.array(
        [
            [1.0 - p_sick - 0.01, p_sick, 0.01],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
    )
    return simulate_cohort_markov(
        states=("Healthy", "Sick", "Dead"),
        initial=[1.0, 0.0, 0.0],
        ages=AGES,
        transition=p,
        utility=UTILITY,
        cost=COST,
        discount_rate=DISCOUNT,
    )


@pytest.mark.parametrize(("arm", "p_sick"), [("low", 0.10), ("high", 0.05)])
def test_means_match_cohort_kernel(arm: str, p_sick: float) -> None:
    result = _run(n_trials=1_000_000)
    expected = _cohort(p_sick)
    for metric in ("cost", "qaly", "ly"):
        z = (result.mean(arm, metric) - getattr(expected, metric)) / result.se(arm, metric)
        assert abs(z) < 4.0, metric


def test_paired_increment_matches_cohort_and_beats_independent_runs() -> None:
    result = _run(n_trials=1_000_000)
    low, high = _cohort(0.10), _cohort(0.05)
    delta = result.delta["qaly"]
    assert abs(delta.mean - (high.qaly - low.qaly)) / delta.se < 4.0
    independent = np.hypot(result.se("low", "qaly"), result.se("high", "qaly"))
    assert delta.se < independent


def test_identical_arms_have_exactly_zero_difference() -> None:
    result = _run(arms=(("low", 0.10), ("high", 0.10)))
    for metric in ("cost", "qaly", "ly", "net_benefit"):
        assert result.delta[metric].mean == 0.0
        assert result.delta[metric].se == 0.0


def test_result_does_not_depend_on_jobs() -> None:
    one = _run(n_jobs=1, chunk_size=50_000)
    two = _run(n_jobs=2, chunk_size=50_000)
    assert one.mean("low", "qaly") == two.mean("low", "qaly")
    assert one.delta["cost"].mean == two.delta["cost"].mean
    assert one.tallies["low"]["alive"] == two.tallies["low"]["alive"]
    np.testing.assert_array_equal(one.persons["high"]["qaly"], two.persons["high"]["qaly"])


def test_seed_changes_result() -> None:
    assert _run(seed=1).mean("low", "qaly") != _run(seed=2).mean("low", "qaly")


def test_strata_are_weighted_equally() -> None:
    result = _run(SickDead(n_cells=4), n_trials=100_003)
    moments = result.moments["low"]["qaly"]
    assert moments.n.sum() == 100_003
    assert moments.mean == pytest.approx(float((moments.total / moments.n).mean()))


def test_increment_table_net_benefit_and_icer() -> None:
    result = _run()
    table = result.increment_table()
    row = {r["metric"]: r for r in table.iter_rows(named=True)}
    assert row["icer"]["delta"] == pytest.approx(row["cost"]["delta"] / row["qaly"]["delta"])
    assert row["net_benefit"]["delta"] == pytest.approx(1000.0 * row["qaly"]["delta"] - row["cost"]["delta"])
    assert row["icer"]["se"] > 0.0


def test_distribution_table() -> None:
    result = _run()
    table = distribution_table(result, ["qaly"])
    assert table.height == 2
    row = table.filter(table["arm"] == "low").row(0, named=True)
    x = result.persons["low"]["qaly"]
    assert row["Median"] == pytest.approx(float(np.median(x)))
    assert row["Size (n)"] == x.size
    assert row["Mean"] == pytest.approx(result.mean("low", "qaly"))


def test_distribution_table_needs_persons() -> None:
    with pytest.raises(ValueError, match="keep_persons"):
        distribution_table(_run(keep_persons=False))


@pytest.mark.parametrize("kwargs", [{"n_trials": 0}, {"chunk_size": 0}, {"comparator": "other"}])
def test_rejects_bad_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _run(**kwargs)
