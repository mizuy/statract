from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch MASS::epil (Thall & Vail 1990) and cache the analysis frames."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

# MASS is GPL-2 | GPL-3. The CSV is fetched at run time and never committed.
RDATASETS_EPIL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"


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
        dest = Path(tmp) / "epil.csv"
        code = (
            "data(epil, package='MASS')\n"
            f"write.csv(epil, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_EPIL))


@snapshot_cache(project.snapshot / "epil.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `MASS::epil`, else the public Rdatasets CSV."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"y", "trt", "base", "age", "subject", "period"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"epil snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    # Long format: one row per patient x 2-week visit (59 x 4 = 236).
    visits = raw.with_columns(
        pl.col("y").cast(pl.Int64),
        pl.col("trt").cast(pl.Utf8).cast(pl.Enum(["placebo", "progabide"])),
        (pl.col("trt").cast(pl.Utf8) == "progabide").cast(pl.Int8).alias("progabide"),
        pl.col("base").cast(pl.Float64),
        pl.col("age").cast(pl.Float64),
        # Baseline count covers 8 weeks; each visit covers 2 weeks.
        (pl.col("base") / 4).log().alias("log_base2wk"),
        pl.col("period").cast(pl.Int64),
    ).sort(pl.col("subject").cast(pl.Int64), "period").with_columns(
        pl.col("subject").cast(pl.Utf8),
    ).select(
        "subject", "period", "y", "trt", "progabide", "base", "age", "log_base2wk"
    )
    # One row per patient for Table 1 (baseline + total post-randomisation count).
    patients = (
        visits.group_by("subject", maintain_order=True)
        .agg(
            pl.col("trt").first(),
            pl.col("base").first(),
            pl.col("age").first(),
            pl.col("y").sum().alias("y_total"),
            (pl.col("y") == 0).any().alias("any_zero_visit"),
        )
    )
    return {"raw": raw, "visits": visits, "patients": patients}


if __name__ == "__main__":
    build()
