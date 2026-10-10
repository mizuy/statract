from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch medicaldata::licorice_gargle and cache the analysis frame."""


import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RDATASETS_CSV = "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv"
GITHUB_RDA = "https://raw.githubusercontent.com/higgi13425/medicaldata/master/data/licorice_gargle.rda"

# Time points with a throat pain score (0-10), in order.
TIMES = {
    "pacu30min": "PACU 30 min",
    "pacu90min": "PACU 90 min",
    "postOp4hour": "4 h",
    "pod1am": "POD1 morning",
}


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


def _rscript(code: str, dest: Path) -> pl.DataFrame | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    proc = subprocess.run([rscript, "--vanilla", "-e", code], check=False, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        return None
    return _drop_rownames(pl.read_csv(dest, null_values="NA"))


def _fetch_via_package() -> pl.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "licorice_gargle.csv"
        code = (
            "if (!requireNamespace('medicaldata', quietly=TRUE)) quit(status=2)\n"
            "data(licorice_gargle, package='medicaldata')\n"
            f"write.csv(licorice_gargle, file='{dest.as_posix()}', row.names=FALSE)\n"
        )
        return _rscript(code, dest)


def _fetch_via_url() -> pl.DataFrame | None:
    try:
        return _drop_rownames(pl.read_csv(RDATASETS_CSV, null_values="NA"))
    except Exception:
        return None


def _fetch_via_rda() -> pl.DataFrame | None:
    """The package's own .rda from GitHub, read by R."""
    with tempfile.TemporaryDirectory() as tmp:
        rda = Path(tmp) / "licorice_gargle.rda"
        try:
            urllib.request.urlretrieve(GITHUB_RDA, rda)
        except Exception:
            return None
        dest = Path(tmp) / "licorice_gargle.csv"
        code = f"load('{rda.as_posix()}')\nwrite.csv(licorice_gargle, file='{dest.as_posix()}', row.names=FALSE)\n"
        return _rscript(code, dest)


@snapshot_cache(project.snapshot / "licorice_gargle.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `medicaldata::licorice_gargle`, else Rdatasets CSV, else the package .rda."""
    for fetch in (_fetch_via_package, _fetch_via_url, _fetch_via_rda):
        df = fetch()
        if df is not None:
            break
    else:
        raise RuntimeError("could not fetch licorice_gargle (needs R or network access to Rdatasets / GitHub)")
    needed = {"treat", "preOp_age", "pacu30min_throatPain", "pod1am_throatPain"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"licorice_gargle snapshot missing columns: {sorted(missing)}")
    return df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    target = raw.with_columns(
        pl.col("treat")
        .cast(pl.Int64)
        .replace_strict({0: "Sugar", 1: "Licorice"}, return_dtype=pl.Utf8)
        .cast(pl.Enum(["Sugar", "Licorice"]))
        .alias("arm"),
        pl.col("preOp_age").cast(pl.Float64).alias("age"),
        pl.col("preOp_calcBMI").cast(pl.Float64).alias("bmi"),
        pl.col("preOp_gender")
        .cast(pl.Int64)
        .replace_strict({0: "Male", 1: "Female"}, return_dtype=pl.Utf8)
        .alias("sex"),
        pl.col("preOp_asa").cast(pl.Int64).cast(pl.Utf8).alias("asa"),
        pl.col("preOp_mallampati").cast(pl.Int64).cast(pl.Utf8).alias("mallampati"),
        pl.col("preOp_smoking")
        .cast(pl.Int64)
        .replace_strict({1: "Current", 2: "Past", 3: "Never"}, return_dtype=pl.Utf8)
        .alias("smoking"),
        pl.col("intraOp_surgerySize")
        .cast(pl.Int64)
        .replace_strict({1: "Small", 2: "Medium", 3: "Large"}, return_dtype=pl.Utf8)
        .alias("surgery_size"),
        *[pl.col(f"{t}_throatPain").cast(pl.Float64).alias(f"pain_{t}") for t in TIMES],
        *[(pl.col(f"{t}_throatPain") > 0).alias(f"sore_{t}") for t in TIMES],
    )
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
