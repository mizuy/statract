from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Simulate the teaching data for the theory text: delayed bleeding after colonoscopic resection.

The data are made up. Every arrow of the DAG is written in ``simulate``, so
each chapter can compare an estimate with the truth:

- age -> antithrombotic, hypertension, bleed
- antithrombotic -> hypertension, clip, bleed
- size_mm -> clip, bleed
- proximal -> clip, bleed
- clip -> bleed

``hypertension`` and ``male`` have no arrow into ``bleed``. ``clip`` lowers the
risk, more so for lesions of 20 mm or more.
"""

import numpy as np
import polars as pl

from support import ProjectPath, save_data

project = ProjectPath(__file__)

N = 3000
SEED = 1

# True coefficients of the outcome model (log odds).
TRUE_BLEED = {
    "intercept": -3.6,
    "age_per10": 0.15,
    "antithrombotic": 0.9,
    "size_per10": 0.55,
    "proximal": 0.6,
    "clip_small": -0.2,  # lesions < 20 mm
    "clip_large": -1.0,  # lesions >= 20 mm
}


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bleed_logit(
    age: np.ndarray,
    antithrombotic: np.ndarray,
    size_mm: np.ndarray,
    proximal: np.ndarray,
    clip: np.ndarray,
) -> np.ndarray:
    """True log odds of delayed bleeding."""
    b = TRUE_BLEED
    clip_effect = np.where(size_mm >= 20, b["clip_large"], b["clip_small"])
    return (
        b["intercept"]
        + b["age_per10"] * (age - 70) / 10
        + b["antithrombotic"] * antithrombotic
        + b["size_per10"] * (size_mm - 15) / 10
        + b["proximal"] * proximal
        + clip_effect * clip
    )


def simulate(n: int = N, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(68, 10, n), 30, 95).round()
    male = rng.binomial(1, 0.6, n)
    antithrombotic = rng.binomial(1, _expit(-1.6 + 0.08 * (age - 68)))
    hypertension = rng.binomial(
        1, _expit(-0.4 + 0.09 * (age - 68) + 1.8 * antithrombotic)
    )
    size_mm = np.clip(np.exp(rng.normal(np.log(14), 0.55, n)), 5, 80).round()
    proximal = rng.binomial(1, 0.55, n)
    # Endoscopists clip large, proximal lesions and patients on antithrombotics.
    clip = rng.binomial(
        1,
        _expit(
            -2.4
            + 1.8 * (size_mm >= 20)
            + 0.08 * (size_mm - 15)
            + 1.6 * antithrombotic
            + 0.8 * proximal
        ),
    )
    bleed = rng.binomial(
        1, _expit(bleed_logit(age, antithrombotic, size_mm, proximal, clip))
    )
    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age.astype(int),
            "male": male,
            "antithrombotic": antithrombotic,
            "hypertension": hypertension,
            "size_mm": size_mm.astype(int),
            "proximal": proximal,
            "clip": clip,
            "bleed": bleed,
        }
    )


if __name__ == "__main__":
    df = simulate()
    save_data(project.cache / "build" / "bleeding.parquet", df)
    print(df.describe())
