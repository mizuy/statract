from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch Vanderbilt RHC CSV and cache an analysis-ready frame. Do not commit the CSV."""


import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RHC_CSV = "https://hbiostat.org/data/repo/rhc.csv"


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


def _yes_bin(col: str, alias: str | None = None) -> pl.Expr:
    name = alias or col
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .is_in(["yes", "true", "1"])
        .cast(pl.Int8)
        .alias(name)
    )


@snapshot_cache(project.snapshot / "rhc.parquet")
def load_snapshot() -> pl.DataFrame:
    """Teaching extract from hbiostat.org (Connors JAMA 1996). Fetch only."""
    df = _drop_rownames(pl.read_csv(RHC_CSV, infer_schema_length=5000))
    needed = {"swang1", "dth30", "age"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"rhc snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    def _num(col: str) -> pl.Expr:
        return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)

    target = raw.with_columns(
        (pl.col("swang1").cast(pl.Utf8).str.strip_chars() == "RHC").cast(pl.Int8).alias("rhc"),
        _yes_bin("dth30", "dth30_bin"),
        _yes_bin("death", "death_bin") if "death" in raw.columns else pl.lit(None).alias("death_bin"),
        _num("age").alias("age"),
        pl.col("sex").cast(pl.Utf8).str.strip_chars().alias("sex") if "sex" in raw.columns else pl.lit(None).alias("sex"),
        pl.col("race").cast(pl.Utf8).str.strip_chars().alias("race") if "race" in raw.columns else pl.lit(None).alias("race"),
        pl.col("cat1").cast(pl.Utf8).str.strip_chars().alias("cat1") if "cat1" in raw.columns else pl.lit(None).alias("cat1"),
        pl.col("ca").cast(pl.Utf8).str.strip_chars().alias("ca") if "ca" in raw.columns else pl.lit(None).alias("ca"),
        pl.col("dnr1").cast(pl.Utf8).str.strip_chars().alias("dnr1") if "dnr1" in raw.columns else pl.lit(None).alias("dnr1"),
        pl.col("ninsclas").cast(pl.Utf8).str.strip_chars().alias("ninsclas")
        if "ninsclas" in raw.columns
        else pl.lit(None).alias("ninsclas"),
        pl.col("income").cast(pl.Utf8).str.strip_chars().alias("income")
        if "income" in raw.columns
        else pl.lit(None).alias("income"),
    )
    for col in [
        "edu",
        "aps1",
        "scoma1",
        "meanbp1",
        "hrt1",
        "resp1",
        "temp1",
        "pafi1",
        "alb1",
        "hema1",
        "bili1",
        "crea1",
        "sod1",
        "cardiohx",
        "chfhx",
        "chrpulhx",
        "dementhx",
        "renalhx",
        "liverhx",
        "malighx",
    ]:
        if col in target.columns:
            target = target.with_columns(_num(col).alias(col))
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
