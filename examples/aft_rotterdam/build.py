from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch survival::rotterdam and cache the analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RDATASETS_ROTTERDAM = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rotterdam.csv"


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
        dest = Path(tmp) / "rotterdam.csv"
        code = (
            "if (!requireNamespace('survival', quietly=TRUE)) quit(status=2)\n"
            "data(rotterdam, package='survival')\n"
            f"write.csv(rotterdam, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_ROTTERDAM))


@snapshot_cache(project.snapshot / "rotterdam.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `survival::rotterdam`, else public Rdatasets CSV."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"dtime", "death", "hormon", "age"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"rotterdam snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    size_enum = pl.Enum(["<=20", "20-50", ">50"])
    hormon_enum = pl.Enum(["no hormone", "hormone"])
    target = raw.with_columns(
        pl.col("dtime").cast(pl.Float64),
        (pl.col("death").cast(pl.Int32) == 1).alias("event"),
        pl.col("age").cast(pl.Float64),
        pl.col("size").cast(pl.Utf8).alias("size") if "size" in raw.columns else pl.lit(None).alias("size"),
        pl.col("nodes").cast(pl.Float64) if "nodes" in raw.columns else pl.lit(None).alias("nodes"),
        pl.col("grade").cast(pl.Int32) if "grade" in raw.columns else pl.lit(None).alias("grade"),
        pl.when(pl.col("hormon").cast(pl.Float64) == 1.0)
        .then(pl.lit("hormone"))
        .otherwise(pl.lit("no hormone"))
        .alias("hormon_label"),
        (pl.col("hormon").cast(pl.Float64) == 1.0).cast(pl.Int8).alias("hormon"),
        (pl.col("chemo").cast(pl.Float64) == 1.0).cast(pl.Int8).alias("chemo")
        if "chemo" in raw.columns
        else pl.lit(None).alias("chemo"),
        (pl.col("meno").cast(pl.Float64) == 1.0).cast(pl.Int8).alias("meno")
        if "meno" in raw.columns
        else pl.lit(None).alias("meno"),
    ).with_columns(
        pl.col("hormon_label").cast(hormon_enum),
        pl.col("size").cast(size_enum) if "size" in raw.columns else pl.lit(None).alias("size"),
    )
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
