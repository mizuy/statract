"""Fetch survival::heart and cache a counting-process analysis frame.

Public Rdatasets CSV of the Stanford heart transplant data. lifelines is not used.
"""

from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import polars as pl

from project import project
from support import cache, snapshot_cache

RDATASETS_HEART = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/heart.csv"


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


@snapshot_cache(project.snapshot / "heart_transplants.parquet")
def load_snapshot() -> pl.DataFrame:
    """Rdatasets `survival/heart` (Crowley and Hu 1977), not committed."""
    df = _drop_rownames(pl.read_csv(RDATASETS_HEART))
    needed = {"start", "stop", "event", "age", "year", "surgery", "transplant", "id"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"heart snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    df_raw = load_snapshot()
    target = df_raw.with_columns(
        pl.col("age").alias("age_normalized"),
        pl.col("event").cast(pl.Boolean).alias("event_bool"),
        pl.col("surgery").cast(pl.Int8),
        pl.col("transplant").cast(pl.Int8),
        (pl.col("stop") - pl.col("start")).alias("survival_time"),
    )
    return {"target": target}


if __name__ == "__main__":
    frame = build()["target"]
    print(f"heart rows={frame.height} cols={frame.columns}")
