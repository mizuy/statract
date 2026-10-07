"""Write the seeded synthetic CSVs that the R oracle scripts read.

Run from the repository root: ``uv run python tests/r_oracle/scripts/make_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"


def _glmm_count(seed: int, n_groups: int, size: tuple[int, int], *, slope_sd: float, theta: float | None, base: float):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(size[0], size[1] + 1, n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    x = rng.normal(size=n)
    arm = rng.choice(["control", "low", "high"], size=n)
    years = rng.uniform(0.5, 3.0, n)
    b0 = rng.normal(0, 0.6, n_groups)
    b1 = rng.normal(0, slope_sd, n_groups)
    effect = np.select([arm == "low", arm == "high"], [0.3, -0.4], 0.0)
    mu = years * np.exp(base + 0.4 * x + effect + b0[group] + b1[group] * x)
    if theta is None:
        y = rng.poisson(mu)
    else:
        y = rng.negative_binomial(theta, theta / (theta + mu))
    return pl.DataFrame(
        {
            "y": y.astype(np.int64),
            "x": np.round(x, 6),
            "arm": arm,
            "years": np.round(years, 6),
            "site": [f"s{g:03d}" for g in group],
        }
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    # (1) Poisson counts, random intercept, 30 sites.
    _glmm_count(20261007, 30, (15, 25), slope_sd=0.0, theta=None, base=0.2).write_csv(DATA / "glmm_count_a.csv")
    # (2) Overdispersed counts with a random slope.
    _glmm_count(20261008, 25, (20, 40), slope_sd=0.3, theta=1.5, base=0.5).write_csv(DATA / "glmm_count_b.csv")
    # (3) Many small sites and low counts (most outcomes are zero or one).
    _glmm_count(20261009, 120, (2, 4), slope_sd=0.0, theta=3.0, base=-1.2).write_csv(DATA / "glmm_count_c.csv")


if __name__ == "__main__":
    main()
