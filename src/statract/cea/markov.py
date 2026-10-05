"""Cohort discrete-time Markov kernel (heemod-like, disease-agnostic).

SSL / CRC screening logic belongs in the analysis project. This module only
propagates named states, applies state rewards, and optional per-cycle hooks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

TransitionFn = Callable[[int, np.ndarray], np.ndarray]
"""``P(age, mass) -> mass_after_transition`` or return transition matrix.

Prefer returning a new mass vector. If an ``(n, n)`` matrix is returned, it is
applied as ``mass @ P``.
"""

OnCycleFn = Callable[[int, np.ndarray], tuple[np.ndarray, float, float]]
"""``on_cycle(age, mass) -> (mass, extra_cost, extra_qaly)`` before transition."""

_MASS_ATOL = 1e-10
_MASS_RTOL = 1e-8


@dataclass(frozen=True)
class CohortMarkovResult:
    """Discounted totals plus optional state-membership trace."""

    cost: float
    qaly: float
    ly: float
    trace: pl.DataFrame | None


def _as_reward_vector(
    states: Sequence[str],
    values: Mapping[str, float] | Sequence[float] | np.ndarray | float,
    *,
    name: str,
) -> np.ndarray:
    n = len(states)
    if isinstance(values, (int, float)):
        return np.full(n, float(values), dtype=np.float64)
    if isinstance(values, Mapping):
        out = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(states):
            if s in values:
                out[i] = float(values[s])
        return out
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have length {n}")
    return arr


def _check_finite(arr: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_mass(
    mass: np.ndarray,
    *,
    name: str,
    expected_total: float | None = None,
    normalize: bool = False,
) -> np.ndarray:
    arr = np.asarray(mass, dtype=np.float64)
    _check_finite(arr, name=name)
    if np.any(arr < -_MASS_ATOL):
        raise ValueError(f"{name} must be non-negative")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if normalize:
        if total <= 0:
            raise ValueError(f"{name} must have positive total mass")
        return arr / total
    if expected_total is not None and not np.isclose(
        total,
        expected_total,
        rtol=_MASS_RTOL,
        atol=_MASS_ATOL,
    ):
        raise ValueError(f"{name} must conserve cohort mass")
    return arr


def _validate_transition_matrix(P: np.ndarray, *, n: int) -> np.ndarray:
    if P.ndim != 2 or P.shape != (n, n):
        raise ValueError("transition matrix must be square (n_state, n_state)")
    _check_finite(P, name="transition matrix")
    if np.any(P < -_MASS_ATOL):
        raise ValueError("transition matrix probabilities must be non-negative")
    P = np.clip(P, 0.0, None)
    if not np.allclose(P.sum(axis=1), 1.0, rtol=_MASS_RTOL, atol=_MASS_ATOL):
        raise ValueError("transition matrix rows must sum to 1")
    return P


def _apply_transition(
    age: int,
    mass: np.ndarray,
    transition: TransitionFn | np.ndarray | Mapping[int, np.ndarray],
) -> np.ndarray:
    expected_total = float(mass.sum())
    if callable(transition):
        out = transition(age, mass)
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape != mass.shape:
                raise ValueError("transition callable must return mass vector of same length")
            return _validate_mass(arr, name="transition mass", expected_total=expected_total)
        if arr.ndim == 2 and arr.shape == (mass.shape[0], mass.shape[0]):
            P = _validate_transition_matrix(arr, n=mass.shape[0])
            return _validate_mass(mass @ P, name="transition mass", expected_total=expected_total)
        raise ValueError("transition callable must return mass vector or square matrix")
    if isinstance(transition, Mapping):
        try:
            P = np.asarray(transition[age], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"transition mapping missing age {age}") from exc
    else:
        P = np.asarray(transition, dtype=np.float64)
    P = _validate_transition_matrix(P, n=mass.shape[0])
    return _validate_mass(mass @ P, name="transition mass", expected_total=expected_total)


def simulate_cohort_markov(
    *,
    states: Sequence[str],
    initial: Mapping[str, float] | Sequence[float] | np.ndarray,
    ages: Sequence[int],
    transition: TransitionFn | np.ndarray | Mapping[int, np.ndarray],
    utility: Mapping[str, float] | Sequence[float] | np.ndarray | float,
    cost: Mapping[str, float] | Sequence[float] | np.ndarray | float = 0.0,
    discount_rate: float = 0.0,
    living_mask: Sequence[bool] | np.ndarray | None = None,
    on_cycle: OnCycleFn | None = None,
    record_trace: bool = False,
) -> CohortMarkovResult:
    """Run a cohort Markov simulation over calendar ages.

    Cycle order each age ``a`` (cycle index ``t = a - ages[0]``):

    1. Accrue discounted LY / QALY / state costs on current membership.
    2. Optional ``on_cycle`` (screening / resection events).
    3. Apply ``transition`` to obtain membership for the next age
       (skipped on the final age).

    Parameters
    ----------
    states
        Ordered state names.
    initial
        Initial distribution. Non-negative values with a positive total are
        renormalized to sum to 1. Negative mass is rejected.
    ages
        Inclusive calendar ages, one per cycle (e.g. ``range(40, 101)``).
    transition
        Constant matrix, age→matrix map, or callable returning mass or matrix.
        Entries must be finite and non-negative, and matrix rows must sum to 1.
        Mass after a transition must conserve the cohort total. Negative mass
        is rejected rather than zeroed and kept in the simulation.
    utility, cost
        Per-cycle state rewards.
    discount_rate
        Annual discount rate applied as ``1 / (1 + r)**t``. Finite and >= 0.
    living_mask
        Which states count toward LY. Default: all states whose name does not
        start with ``death`` (case-insensitive) and is not exactly ``dead``.
    on_cycle
        Optional hook before transition. Returned mass must conserve the
        cohort total.
    record_trace
        If True, return a DataFrame of state membership at cycle start.
    """
    state_list = [str(s) for s in states]
    n = len(state_list)
    if n == 0:
        raise ValueError("states must be non-empty")
    if len(set(state_list)) != n:
        raise ValueError("state names must be unique")
    age_list = [int(a) for a in ages]
    if not age_list:
        raise ValueError("ages must be non-empty")

    if isinstance(initial, Mapping):
        mass = np.asarray([float(initial.get(s, 0.0)) for s in state_list], dtype=np.float64)
    else:
        mass = np.asarray(initial, dtype=np.float64)
        if mass.shape != (n,):
            raise ValueError(f"initial must have length {n}")
    mass = _validate_mass(mass, name="initial", normalize=True)

    u_vec = _as_reward_vector(state_list, utility, name="utility")
    c_vec = _as_reward_vector(state_list, cost, name="cost")

    if living_mask is None:
        living = np.array(
            [
                not (s.lower() == "dead" or s.lower().startswith("death"))
                for s in state_list
            ],
            dtype=bool,
        )
    else:
        living = np.asarray(living_mask, dtype=bool)
        if living.shape != (n,):
            raise ValueError(f"living_mask must have length {n}")

    disc = float(discount_rate)
    if not np.isfinite(disc) or disc < 0:
        raise ValueError("discount_rate must be a finite non-negative value")
    start_age = age_list[0]
    tot_cost = tot_qaly = tot_ly = 0.0
    trace_rows: list[dict[str, float | int]] = []

    for age in age_list:
        t = age - start_age
        df = 1.0 / ((1.0 + disc) ** t) if disc != 0.0 else 1.0

        if record_trace:
            row: dict[str, float | int] = {"age": age, "cycle": t}
            for i, s in enumerate(state_list):
                row[s] = float(mass[i])
            trace_rows.append(row)

        living_mass = float(mass[living].sum())
        tot_ly += df * living_mass
        tot_qaly += df * float(mass @ u_vec)
        tot_cost += df * float(mass @ c_vec)

        if on_cycle is not None:
            expected_total = float(mass.sum())
            mass, extra_c, extra_q = on_cycle(age, mass)
            mass = np.asarray(mass, dtype=np.float64)
            if mass.shape != (n,):
                raise ValueError("on_cycle must return mass of length n_state")
            mass = _validate_mass(mass, name="on_cycle mass", expected_total=expected_total)
            tot_cost += df * float(extra_c)
            tot_qaly += df * float(extra_q)

        if age != age_list[-1]:
            mass = _apply_transition(age, mass, transition)

    trace = pl.DataFrame(trace_rows) if record_trace else None
    return CohortMarkovResult(
        cost=float(tot_cost),
        qaly=float(tot_qaly),
        ly=float(tot_ly),
        trace=trace,
    )
