"""Disease-agnostic cost-effectiveness analysis (heemod / dampack-inspired).

Modules
-------
markov
    Cohort discrete-time state-transition kernel.
microsim
    Individual-level microsimulation kernel with common random numbers.
summarize
    ICER tables with strong / extended dominance and NMB.
sensitivity
    One-way DSA, PSA runner, CE plane, CEAC, EVPI.

Disease-specific natural history and cost inputs belong outside this package.
"""

from __future__ import annotations

from statract.cea.markov import CohortMarkovResult, simulate_cohort_markov
from statract.cea.microsim import (
    MicrosimModel,
    MicrosimResult,
    Moments,
    distribution_table,
    simulate_microsim,
)
from statract.cea.sensitivity import (
    PsaResult,
    ce_plane,
    ceac,
    evpi,
    one_way_dsa,
    run_psa,
    tornado_table,
)
from statract.cea.summarize import (
    STATUS_D,
    STATUS_ED,
    STATUS_ND,
    calculate_icers,
    net_monetary_benefit,
)

__all__ = [
    "CohortMarkovResult",
    "MicrosimModel",
    "MicrosimResult",
    "Moments",
    "PsaResult",
    "STATUS_D",
    "STATUS_ED",
    "STATUS_ND",
    "calculate_icers",
    "ce_plane",
    "ceac",
    "distribution_table",
    "evpi",
    "net_monetary_benefit",
    "one_way_dsa",
    "run_psa",
    "simulate_cohort_markov",
    "simulate_microsim",
    "tornado_table",
]
