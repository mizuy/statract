from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Typed DARTH Sick-Sicker teaching parameters (no CSV in git).

Source (hypothetical disease; redistribution of table values for teaching
with citation is the point of the tutorial):

Alarid-Escudero F, Krijkamp EM, Enns EA, Yang A, Hunink MGM,
Pechlivanoglou P, Jalal H. An Introductory Tutorial on Cohort
State-Transition Models in R Using a Cost-Effectiveness Analysis Example.
Med Decis Making. 2023;43(1):3-20. doi:10.1177/0272989X221103163
Open access: https://doi.org/10.1177/0272989X221103163
Companion code: https://github.com/DARTH-git/Cohort-modeling-tutorial

Base-case scalars and PSA distribution names follow Table 1 of that paper
(time-homogeneous version: constant ``p_HD``, no tunnels / transition rewards).
"""


import math

import polars as pl

from support import ProjectPath, cache

project = ProjectPath(__file__)

# Calendar ages: 25 inclusive through 100 inclusive → 76 reward cycles,
# 75 transitions (DARTH n_t = 100-25 = 75 with n_t+1 occupancy rows).
AGE_START = 25
AGE_END = 100
DISCOUNT_RATE = 0.03
WTP = 100_000.0
N_PSA = 1000
PSA_SEED = 2026

STRATEGIES = (
    "Standard of care",
    "Strategy A",
    "Strategy B",
    "Strategy AB",
)

# Table 1 base-case (MDM 2023 introductory tutorial).
BASE_PARAMS: dict[str, float] = {
    "p_HD": 0.002,
    "p_HS1": 0.15,
    "p_S1H": 0.5,
    "p_S1S2": 0.105,
    "hr_S1": 3.0,
    "hr_S2": 10.0,
    "or_S1S2": 0.6,
    "c_H": 2000.0,
    "c_S1": 4000.0,
    "c_S2": 15000.0,
    "c_D": 0.0,
    "c_trtA": 12000.0,
    "c_trtB": 13000.0,
    "u_H": 1.0,
    "u_S1": 0.75,
    "u_S2": 0.5,
    "u_D": 0.0,
    "u_trtA": 0.95,
}

# One-way DSA teaching ranges (not a published OWSA table).
DSA_RANGES: dict[str, tuple[float, float]] = {
    "p_HS1": (0.10, 0.20),
    "p_S1S2": (0.050, 0.160),
    "hr_S2": (5.0, 15.0),
    "c_S2": (10_000.0, 20_000.0),
    "c_trtA": (8_000.0, 16_000.0),
    "c_trtB": (9_000.0, 17_000.0),
    "or_S1S2": (0.40, 0.80),
}

# PSA: R gamma() here is shape-scale (mean = shape * scale).
# lognormal is meanlog, sdlog. beta is shape1, shape2.
PSA_DISTS: dict[str, tuple[str, float, float]] = {
    "p_HS1": ("beta", 30.0, 170.0),
    "p_S1H": ("beta", 60.0, 60.0),
    "p_S1S2": ("beta", 84.0, 716.0),
    "hr_S1": ("lognormal", math.log(3.0), 0.01),
    "hr_S2": ("lognormal", math.log(10.0), 0.02),
    "or_S1S2": ("lognormal", math.log(0.6), 0.1),
    "c_H": ("gamma", 100.0, 20.0),
    "c_S1": ("gamma", 177.8, 22.5),
    "c_S2": ("gamma", 225.0, 66.7),
    "c_trtA": ("gamma", 576.0, 20.8),
    "c_trtB": ("gamma", 676.0, 19.2),
    "u_H": ("beta", 200.0, 3.0),
    "u_S1": ("beta", 130.0, 45.0),
    "u_S2": ("beta", 230.0, 230.0),
    "u_trtA": ("beta", 300.0, 15.0),
}


def _psa_label(name: str) -> str:
    spec = PSA_DISTS.get(name)
    if spec is None:
        return "fixed"
    kind, a, b = spec
    pretty_log = {"hr_S1": "3.0", "hr_S2": "10.0", "or_S1S2": "0.6"}
    if kind == "lognormal" and name in pretty_log:
        return f"lognormal(log({pretty_log[name]}), {b})"
    return f"{kind}({a}, {b})"


def _param_rows() -> pl.DataFrame:
    rows = []
    for name, value in BASE_PARAMS.items():
        rows.append(
            {
                "parameter": name,
                "base": value,
                "psa_distribution": _psa_label(name),
                "source": "Alarid-Escudero et al. MDM 2023 Table 1",
            },
        )
    rows.append(
        {
            "parameter": "discount_rate",
            "base": DISCOUNT_RATE,
            "psa_distribution": "fixed",
            "source": "Alarid-Escudero et al. MDM 2023 Table 1 (d_c = d_e)",
        },
    )
    rows.append(
        {
            "parameter": "age_start",
            "base": float(AGE_START),
            "psa_distribution": "fixed",
            "source": "n_age_init = 25",
        },
    )
    rows.append(
        {
            "parameter": "age_end",
            "base": float(AGE_END),
            "psa_distribution": "fixed",
            "source": "n_age_max = 100",
        },
    )
    return pl.DataFrame(rows)


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    params = _param_rows()
    dsa = pl.DataFrame(
        {
            "parameter": list(DSA_RANGES.keys()),
            "low": [DSA_RANGES[k][0] for k in DSA_RANGES],
            "high": [DSA_RANGES[k][1] for k in DSA_RANGES],
        },
    )
    return {"params": params, "dsa_ranges": dsa}


if __name__ == "__main__":
    build()
