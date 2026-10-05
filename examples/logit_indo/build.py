from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch medicaldata::indo_rct and cache the analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import cache, snapshot_cache
from project import project

RDATASETS_INDO = "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/indo_rct.csv"


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


def _yes_bin(col: str, alias: str) -> pl.Expr:
    text = pl.col(col).cast(pl.Utf8).str.to_lowercase()
    return (
        text.str.contains(r"(?:^|_)yes$")
        | text.is_in(["1", "true", "yes"])
    ).cast(pl.Int8).alias(alias)


def _fetch_via_r() -> pl.DataFrame | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "indo_rct.csv"
        code = (
            "if (!requireNamespace('medicaldata', quietly=TRUE)) quit(status=2)\n"
            "data(indo_rct, package='medicaldata')\n"
            f"write.csv(indo_rct, file='{dest.as_posix()}', row.names=FALSE)\n"
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
    return _drop_rownames(pl.read_csv(RDATASETS_INDO))


@snapshot_cache(project.snapshot / "indo_rct.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `medicaldata::indo_rct`, else public Rdatasets CSV."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"outcome", "rx", "site", "age", "risk"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"indo_rct snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    rx_levels = ["0_placebo", "1_indomethacin"]
    target = raw.with_columns(
        pl.col("rx").cast(pl.Enum(rx_levels)),
        pl.col("site").cast(pl.Utf8),
        pl.col("age").cast(pl.Float64),
        pl.col("risk").cast(pl.Float64),
        _yes_bin("outcome", "pep"),
        _yes_bin("sod", "sod_yes") if "sod" in raw.columns else pl.lit(None).alias("sod_yes"),
        _yes_bin("pdstent", "pdstent_yes") if "pdstent" in raw.columns else pl.lit(None).alias("pdstent_yes"),
        (pl.col("rx").cast(pl.Utf8).str.contains("indomethacin")).cast(pl.Int8).alias("indomethacin"),
        pl.col("gender").cast(pl.Utf8).alias("gender") if "gender" in raw.columns else pl.lit(None).alias("gender"),
    )
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
