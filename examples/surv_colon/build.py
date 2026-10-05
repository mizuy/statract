from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch survival::colon and cache a patient-level analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import cache, snapshot_cache
from project import project

RDATASETS_COLON = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv"


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
        dest = Path(tmp) / "colon.csv"
        code = (
            "if (!requireNamespace('survival', quietly=TRUE)) quit(status=2)\n"
            "data(colon, package='survival')\n"
            f"write.csv(colon, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_COLON))


@snapshot_cache(project.snapshot / "colon.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `survival::colon`, else public Rdatasets CSV (same table, not committed)."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"id", "rx", "sex", "age", "nodes", "status", "time", "etype"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"colon snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    rx_enum = pl.Enum(["Obs", "Lev", "Lev+5FU"])
    patient = (
        raw.filter(pl.col("etype") == 1)
        .with_columns(
            pl.col("rx").cast(rx_enum),
            pl.when(pl.col("sex") == 1).then(pl.lit("male")).otherwise(pl.lit("female")).alias("sex_label"),
            (pl.col("status") == 1).alias("event"),
            pl.col("time").cast(pl.Float64),
            pl.col("age").cast(pl.Float64),
            pl.col("nodes").cast(pl.Float64),
        )
    )
    return {"raw": raw, "patient": patient}


if __name__ == "__main__":
    build()
