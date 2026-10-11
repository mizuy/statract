from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch survival::pbcseq and cache patient-level plus longitudinal frames."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RDATASETS_PBCSEQ = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"


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
        dest = Path(tmp) / "pbcseq.csv"
        code = (
            "if (!requireNamespace('survival', quietly=TRUE)) quit(status=2)\n"
            "data(pbcseq, package='survival')\n"
            f"write.csv(pbcseq, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_PBCSEQ))


@snapshot_cache(project.snapshot / "pbcseq.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `survival::pbcseq`, else public Rdatasets CSV."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"id", "day", "bili", "trt"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"pbcseq snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    long = raw.with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("day").cast(pl.Float64),
        (pl.col("day").cast(pl.Float64) / 365.25).alias("day_years"),
        pl.col("bili").cast(pl.Float64),
        pl.col("age").cast(pl.Float64) if "age" in raw.columns else pl.lit(None).alias("age"),
        pl.col("sex").cast(pl.Utf8).str.to_lowercase().alias("sex") if "sex" in raw.columns else pl.lit(None).alias("sex"),
        pl.col("albumin").cast(pl.Float64) if "albumin" in raw.columns else pl.lit(None).alias("albumin"),
        (pl.col("trt").cast(pl.Float64) == 1.0).cast(pl.Int8).alias("dp"),
        pl.when(pl.col("trt").cast(pl.Float64) == 1.0)
        .then(pl.lit("D-penicillamine"))
        .when(pl.col("trt").cast(pl.Float64).is_in([0.0, 2.0]))
        .then(pl.lit("placebo"))
        .otherwise(pl.lit(None))
        .alias("trt_label"),
    ).with_columns(
        pl.when(pl.col("bili") > 0).then(pl.col("bili").log()).otherwise(None).alias("log_bili"),
    )
    patient = (
        long.filter(pl.col("trt").is_not_null())
        .sort(["id", "day"])
        .group_by("id", maintain_order=True)
        .first()
    )
    return {"raw": raw, "long": long, "patient": patient}


if __name__ == "__main__":
    build()
