from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch NHEFS (causaldata or public CSV) and cache the analysis frame."""


import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

RDATASETS_NHEFS = "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs.csv"
HERNAN_NHEFS = "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"


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
        dest = Path(tmp) / "nhefs.csv"
        code = (
            "if (!requireNamespace('causaldata', quietly=TRUE)) quit(status=2)\n"
            "data(nhefs, package='causaldata')\n"
            f"write.csv(nhefs, file='{dest.as_posix()}', row.names=FALSE)\n"
        )
        proc = subprocess.run(
            [rscript, "--vanilla", "-e", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dest.exists():
            return None
        return _drop_rownames(pl.read_csv(dest, infer_schema_length=5000))


def _fetch_via_url() -> pl.DataFrame:
    try:
        return _drop_rownames(pl.read_csv(RDATASETS_NHEFS, infer_schema_length=5000))
    except Exception:
        return _drop_rownames(pl.read_csv(HERNAN_NHEFS, infer_schema_length=5000))


@snapshot_cache(project.snapshot / "nhefs.parquet")
def load_snapshot() -> pl.DataFrame:
    """R `causaldata::nhefs`, else Rdatasets, else Hernán teaching CSV. Fetch only."""
    df = _fetch_via_r()
    if df is None:
        df = _fetch_via_url()
    needed = {"qsmk", "wt82_71"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"nhefs snapshot missing columns: {sorted(missing)}")
    return df


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = load_snapshot()
    cols = [
        "qsmk",
        "wt82_71",
        "age",
        "sex",
        "race",
        "education",
        "smokeintensity",
        "smokeyrs",
        "exercise",
        "active",
        "wt71",
    ]
    exprs = []
    for col in cols:
        if col in raw.columns:
            exprs.append(_num(col).alias(col))
    target = raw.with_columns(exprs)
    if "qsmk" in target.columns:
        target = target.with_columns(pl.col("qsmk").cast(pl.Int8))
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
