from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch survival::retinopathy and cache an eye-level analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RDATASETS_RETINOPATHY = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/retinopathy.csv"

TRT_ENUM = pl.Enum(["control", "treated"])
TYPE_ENUM = pl.Enum(["adult", "juvenile"])


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
        dest = Path(tmp) / "retinopathy.csv"
        code = (
            "if (!requireNamespace('survival', quietly=TRUE)) quit(status=2)\n"
            "data(retinopathy, package='survival')\n"
            f"write.csv(retinopathy, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_RETINOPATHY))


def _prepare(df: pl.DataFrame) -> pl.DataFrame:
    needed = {"id", "laser", "eye", "age", "type", "trt", "futime", "status", "risk"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"retinopathy snapshot missing columns: {sorted(missing)}")
    return df.with_columns(
        pl.when(pl.col("trt") == 1)
        .then(pl.lit("treated"))
        .otherwise(pl.lit("control"))
        .cast(TRT_ENUM)
        .alias("trt_label"),
        pl.col("type").cast(TYPE_ENUM),
        (pl.col("status") == 1).alias("event"),
        pl.col("futime").cast(pl.Float64),
        pl.col("age").cast(pl.Float64),
        pl.col("risk").cast(pl.Float64),
        pl.col("id").cast(pl.Int64),
    )


@snapshot_cache(project.snapshot / "retinopathy.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `survival::retinopathy`, else public Rdatasets CSV (same table, not committed)."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    return _prepare(df)


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    eyes = load_snapshot()
    return {"eyes": eyes}


if __name__ == "__main__":
    build()
