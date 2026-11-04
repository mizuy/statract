"""Individual-level discrete-time microsimulation kernel (disease-agnostic).

Disease-specific logic belongs outside this package. The caller supplies a
``MicrosimModel``. This module only does the following:

- Splits trials into chunks and runs them in processes. Results do not depend on
  ``n_jobs``.
- Draws uniforms per ``(seed, chunk, cycle)``: ``n_streams`` values for every
  person, whatever their state. A value is fixed by person, cycle and stream, and
  every arm uses the same values (common random numbers).
- Summarizes per-person outcomes into means and standard errors. Differences
  between arms are taken within the same person.
- Keeps per-person outcomes and reports their distribution.

Keep the cycle order of ``simulate_cohort_markov`` inside ``step``: state rewards,
then events, then transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl


class MicrosimModel(Protocol):
    """``step`` updates ``state`` in place. The model decides what an ``arm`` holds."""

    n_cycles: int
    n_streams: int
    n_cells: int

    def init_population(self, ids: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Attributes shared by all arms. Must include ``cell`` (stratum, all 0 if none)."""
        ...

    def init_arm(self, population: dict[str, np.ndarray], arm: Any) -> Any: ...

    def step(self, state: Any, cycle: int, uniforms: np.ndarray) -> None:
        """``uniforms`` has shape (n_streams, n_people), including people who have left."""
        ...

    def is_done(self, state: Any) -> bool: ...

    def outcomes(self, state: Any) -> dict[str, np.ndarray]:
        """Per-person outcomes: metric name -> array of length n_people."""
        ...

    def tallies(self, state: Any) -> dict[str, np.ndarray]:
        """Arrays summed over people and chunks (e.g. traces by age). Empty if none."""
        ...


@dataclass(frozen=True)
class Moments:
    """Per-stratum (n, sum, sum of squares). The mean weights strata equally."""

    n: np.ndarray
    total: np.ndarray
    squares: np.ndarray

    @staticmethod
    def of(values: np.ndarray, cell: np.ndarray, n_cells: int) -> Moments:
        return Moments(
            n=np.bincount(cell, minlength=n_cells).astype(np.float64),
            total=np.bincount(cell, weights=values, minlength=n_cells),
            squares=np.bincount(cell, weights=values * values, minlength=n_cells),
        )

    def __add__(self, other: Moments) -> Moments:
        return Moments(self.n + other.n, self.total + other.total, self.squares + other.squares)

    @property
    def mean(self) -> float:
        return float((self.total / self.n).mean())

    @property
    def se(self) -> float:
        cell_mean = self.total / self.n
        cell_var = (self.squares - self.n * cell_mean**2) / (self.n - 1.0)
        return float(np.sqrt((np.clip(cell_var, 0.0, None) / self.n).sum())) / self.n.size


@dataclass(frozen=True)
class MicrosimResult:
    arms: tuple[str, ...]
    n: int
    seed: int
    moments: dict[str, dict[str, Moments]]  # arm -> metric -> Moments
    delta: dict[str, Moments]  # metric -> paired intervention - comparator; has net_benefit
    comparator: str
    intervention: str
    wtp: float
    persons: dict[str, dict[str, np.ndarray]] | None
    tallies: dict[str, dict[str, np.ndarray]]

    def mean(self, arm: str, metric: str) -> float:
        return self.moments[arm][metric].mean

    def se(self, arm: str, metric: str) -> float:
        return self.moments[arm][metric].se

    def increment_table(self, *, cost: str = "cost", effect: str = "qaly") -> pl.DataFrame:
        """Paired increments with SEs. The ICER SE uses the delta method and is unreliable when large."""
        rows = [
            {"metric": m, "delta": d.mean, "se": d.se}
            for m, d in self.delta.items()
        ]
        dc, dq = self.delta[cost], self.delta[effect]
        nb = self.delta["net_benefit"]
        var_c, var_q, var_nb = dc.se**2, dq.se**2, nb.se**2
        cov = (self.wtp**2 * var_q + var_c - var_nb) / (2.0 * self.wtp) if self.wtp else 0.0
        icer = dc.mean / dq.mean if dq.mean != 0.0 else float("nan")
        var_icer = (var_c - 2.0 * icer * cov + icer**2 * var_q) / dq.mean**2 if dq.mean else float("nan")
        rows.append({"metric": "icer", "delta": icer, "se": float(np.sqrt(max(var_icer, 0.0)))})
        return pl.DataFrame(rows)


_WORKER: dict[str, Any] = {}


def _init_worker(model: MicrosimModel, arms: Sequence[tuple[str, Any]], options: dict[str, Any]) -> None:
    _WORKER.update(model=model, arms=arms, options=options)


def _rng(*key: int) -> np.random.Generator:
    return np.random.Generator(np.random.SFC64(list(key)))


def _run_chunk(job: tuple[int, int, int]) -> dict[str, Any]:
    chunk, start, stop = job
    model: MicrosimModel = _WORKER["model"]
    arms = _WORKER["arms"]
    opt = _WORKER["options"]
    seed = opt["seed"]

    ids = np.arange(start, stop)
    population = model.init_population(ids, _rng(seed, chunk, 0))
    states = [model.init_arm(population, spec) for _, spec in arms]
    for cycle in range(model.n_cycles):
        if all(model.is_done(s) for s in states):
            break
        uniforms = _rng(seed, chunk, cycle + 1).random((model.n_streams, ids.size))
        for s in states:
            model.step(s, cycle, uniforms)

    cell = population["cell"]
    outcomes = {name: model.outcomes(s) for (name, _), s in zip(arms, states, strict=True)}
    moments = {
        name: {m: Moments.of(v, cell, model.n_cells) for m, v in values.items()}
        for name, values in outcomes.items()
    }
    low, high = outcomes[opt["comparator"]], outcomes[opt["intervention"]]
    delta = {m: Moments.of(high[m] - low[m], cell, model.n_cells) for m in high}
    net_benefit = opt["wtp"] * (high[opt["effect"]] - low[opt["effect"]]) - (high[opt["cost"]] - low[opt["cost"]])
    delta["net_benefit"] = Moments.of(net_benefit, cell, model.n_cells)
    return {
        "moments": moments,
        "delta": delta,
        "persons": outcomes if opt["keep_persons"] else None,
        "tallies": {name: model.tallies(s) for (name, _), s in zip(arms, states, strict=True)},
    }


def simulate_microsim(
    model: MicrosimModel,
    arms: Sequence[tuple[str, Any]],
    *,
    n_trials: int,
    comparator: str,
    intervention: str,
    wtp: float,
    cost: str = "cost",
    effect: str = "qaly",
    seed: int = 1,
    chunk_size: int = 100_000,
    n_jobs: int = 1,
    keep_persons: bool = True,
) -> MicrosimResult:
    """Run all arms on the same random numbers.

    Parameters
    ----------
    model
        Disease-specific states and transitions. Must be picklable if ``n_jobs > 1``.
    arms
        ``(name, spec)`` pairs. ``spec`` is passed to ``model.init_arm``.
    n_trials
        Number of people per arm.
    comparator, intervention
        Arm names. ``delta`` is intervention minus comparator.
    wtp
        Willingness to pay for ``delta["net_benefit"]``.
    cost, effect
        Metric names in ``model.outcomes``.
    seed, chunk_size
        Together they fix the result. Keep ``chunk_size`` constant to reproduce a run.
    n_jobs
        Worker processes. Does not change the result.
    keep_persons
        Keep per-person outcomes, needed by ``distribution_table``.
    """
    if n_trials <= 0 or chunk_size <= 0:
        raise ValueError("n_trials and chunk_size must be positive")
    names = tuple(name for name, _ in arms)
    if comparator not in names or intervention not in names:
        raise ValueError("comparator and intervention must be arm names")
    options = {
        "seed": int(seed),
        "comparator": comparator,
        "intervention": intervention,
        "wtp": float(wtp),
        "cost": cost,
        "effect": effect,
        "keep_persons": keep_persons,
    }
    jobs = [
        (i, start, min(start + chunk_size, n_trials))
        for i, start in enumerate(range(0, n_trials, chunk_size))
    ]
    if n_jobs <= 1:
        _init_worker(model, arms, options)
        parts = [_run_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(model, arms, options)) as ex:
            parts = list(ex.map(_run_chunk, jobs))

    moments = parts[0]["moments"]
    delta = parts[0]["delta"]
    tallies = parts[0]["tallies"]
    for p in parts[1:]:
        for name in names:
            moments[name] = {m: moments[name][m] + p["moments"][name][m] for m in moments[name]}
            tallies[name] = {k: tallies[name][k] + p["tallies"][name][k] for k in tallies[name]}
        delta = {m: delta[m] + p["delta"][m] for m in delta}
    persons = None
    if keep_persons:
        persons = {
            name: {m: np.concatenate([p["persons"][name][m] for p in parts]) for m in parts[0]["persons"][name]}
            for name in names
        }
    return MicrosimResult(
        arms=names,
        n=int(n_trials),
        seed=int(seed),
        moments=moments,
        delta=delta,
        comparator=comparator,
        intervention=intervention,
        wtp=float(wtp),
        persons=persons,
        tallies=tallies,
    )


_PERCENTILES = (("2.5%", 2.5), ("10%", 10.0), ("Median", 50.0), ("90%", 90.0), ("97.5%", 97.5))


def distribution_table(result: MicrosimResult, metrics: Sequence[str] | None = None) -> pl.DataFrame:
    """Per-person distribution, one row per arm and metric."""
    if result.persons is None:
        raise ValueError("distribution_table needs keep_persons=True")
    rows: list[dict[str, float | str]] = []
    for arm in result.arms:
        for metric, x in result.persons[arm].items():
            if metrics is not None and metric not in metrics:
                continue
            var = float(x.var(ddof=1))
            mean = float(x.mean())
            se = float(np.sqrt(var / x.size))
            row: dict[str, float | str] = {
                "arm": arm,
                "metric": metric,
                "Mean": mean,
                "Std Deviation": float(np.sqrt(var)),
                "Minimum": float(x.min()),
            }
            values = np.percentile(x, [p for _, p in _PERCENTILES])
            row.update({label: float(v) for (label, _), v in zip(_PERCENTILES, values, strict=True)})
            row.update(
                {
                    "Maximum": float(x.max()),
                    "Sum": float(x.sum()),
                    "Size (n)": float(x.size),
                    "Variance": var,
                    "SQRT[Variance/Size]": se,
                    "95% Lower Bound": mean - 1.96 * se,
                    "95% Upper Bound": mean + 1.96 * se,
                },
            )
            rows.append(row)
    return pl.DataFrame(rows)
