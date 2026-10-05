from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch survival::pbc and cache the randomized analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import cache, snapshot_cache
from project import project

RDATASETS_PBC = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbc.csv"


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


def _fetch_via_r() -> pl.DataFrame | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "pbc.csv"
        code = (
            "if (!requireNamespace('survival', quietly=TRUE)) quit(status=2)\n"
            "data(pbc, package='survival')\n"
            f"write.csv(pbc, file='{dest.as_posix()}', row.names=FALSE)\n"
        )
        proc = subprocess.run(
            [rscript, "--vanilla", "-e", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dest.exists():
            return None
        return _drop_rownames(pl.read_csv(dest))


def _fetch_via_url() -> pl.DataFrame:
    return _drop_rownames(pl.read_csv(RDATASETS_PBC))


@snapshot_cache(project.snapshot / "pbc.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `survival::pbc`, else public Rdatasets CSV (same table, not committed)."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"id", "time", "status", "trt", "age", "sex", "bili"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"pbc snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    trt_enum = pl.Enum(["D-penicillamine", "placebo"])
    target = raw.with_columns(
        pl.col("time").cast(pl.Float64),
        pl.col("status").cast(pl.Int32),
        pl.col("age").cast(pl.Float64),
        pl.col("bili").cast(pl.Float64),
        pl.col("albumin").cast(pl.Float64) if "albumin" in raw.columns else pl.lit(None).alias("albumin"),
        pl.col("edema").cast(pl.Float64) if "edema" in raw.columns else pl.lit(None).alias("edema"),
        pl.col("stage").cast(pl.Float64) if "stage" in raw.columns else pl.lit(None).alias("stage"),
        pl.col("sex").cast(pl.Utf8).str.to_lowercase().alias("sex"),
        pl.when(pl.col("trt").cast(pl.Float64) == 1.0)
        .then(pl.lit("D-penicillamine"))
        .when(pl.col("trt").cast(pl.Float64) == 2.0)
        .then(pl.lit("placebo"))
        .otherwise(pl.lit(None))
        .alias("trt_label"),
        (pl.col("trt").cast(pl.Float64) == 1.0).cast(pl.Int8).alias("dp"),
        # status: 0 censored, 1 transplant (competing), 2 death (event of interest)
        (pl.col("status") == 2).alias("death_naive"),
    ).with_columns(pl.col("trt_label").cast(trt_enum))
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
