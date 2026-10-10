from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Fetch SUPPORT2 (hbiostat teaching CSV) and cache the analysis frame."""


import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import polars as pl

from support import ProjectPath, cache, snapshot_cache

project = ProjectPath(__file__)

HBIOSSTAT_ZIP = "https://hbiostat.org/data/repo/support2csv.zip"
UCI_CSV = "https://archive.ics.uci.edu/static/public/880/data.csv"


def _drop_rownames(df: pl.DataFrame) -> pl.DataFrame:
    first = df.columns[0]
    if first in {"", "rownames", "column_1", "id"} or first.startswith("Unnamed"):
        return df.drop(first)
    return df


def _read_support_csv(raw: bytes) -> pl.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    if first.startswith("age,") or first.startswith('"age"'):
        text = "rownames," + text
    df = pl.read_csv(io.StringIO(text), infer_schema_length=20000)
    return _drop_rownames(df)


def _fetch_hbiostat() -> pl.DataFrame:
    with urlopen(HBIOSSTAT_ZIP, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return _read_support_csv(zf.read(name))


def _fetch_uci() -> pl.DataFrame:
    with urlopen(UCI_CSV, timeout=60) as resp:
        return _read_support_csv(resp.read())


@snapshot_cache(project.snapshot / "support2.parquet")
def load_snapshot() -> pl.DataFrame:
    """hbiostat SUPPORT2 teaching CSV (zip); UCI CSV as fallback. Fetch only."""
    try:
        df = _fetch_hbiostat()
    except Exception:
        df = _fetch_uci()
    needed = {"age", "death", "sex", "hospdead"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"support2 snapshot missing columns: {sorted(missing)}")
    return df


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def _rename_dotted(df: pl.DataFrame) -> pl.DataFrame:
    mapping = {c: c.replace(".", "_") for c in df.columns if "." in c}
    return df.rename(mapping) if mapping else df


@cache(project.cache / "build")
def build() -> dict[str, pl.DataFrame]:
    raw = _rename_dotted(load_snapshot())
    d_time = "d_time" if "d_time" in raw.columns else "d.time"
    num_co = "num_co" if "num_co" in raw.columns else "num.co"
    target = raw.with_columns(
        _num("age").alias("age"),
        _num("death").alias("death"),
        _num("hospdead").alias("hospdead"),
        _num(d_time).alias("d_time") if d_time in raw.columns else pl.lit(None).alias("d_time"),
        _num(num_co).alias("num_co") if num_co in raw.columns else pl.lit(None).alias("num_co"),
        _num("scoma").alias("scoma") if "scoma" in raw.columns else pl.lit(None).alias("scoma"),
        _num("meanbp").alias("meanbp") if "meanbp" in raw.columns else pl.lit(None).alias("meanbp"),
        _num("hrt").alias("hrt") if "hrt" in raw.columns else pl.lit(None).alias("hrt"),
        _num("resp").alias("resp") if "resp" in raw.columns else pl.lit(None).alias("resp"),
        _num("temp").alias("temp") if "temp" in raw.columns else pl.lit(None).alias("temp"),
        _num("crea").alias("crea") if "crea" in raw.columns else pl.lit(None).alias("crea"),
        _num("sod").alias("sod") if "sod" in raw.columns else pl.lit(None).alias("sod"),
        _num("diabetes").cast(pl.Int8).alias("diabetes")
        if "diabetes" in raw.columns
        else pl.lit(None).alias("diabetes"),
        _num("dementia").cast(pl.Int8).alias("dementia")
        if "dementia" in raw.columns
        else pl.lit(None).alias("dementia"),
        pl.col("sex").cast(pl.Utf8).str.strip_chars().alias("sex"),
        pl.col("race").cast(pl.Utf8).str.strip_chars().alias("race")
        if "race" in raw.columns
        else pl.lit(None).alias("race"),
        pl.col("dzclass").cast(pl.Utf8).str.strip_chars().alias("dzclass")
        if "dzclass" in raw.columns
        else pl.lit(None).alias("dzclass"),
        pl.col("ca").cast(pl.Utf8).str.strip_chars().alias("ca")
        if "ca" in raw.columns
        else pl.lit(None).alias("ca"),
    ).with_columns(
        ((pl.col("death") == 1) & (pl.col("d_time") <= 180)).cast(pl.Int8).alias("death_180"),
    ).with_columns(
        pl.when(pl.col("death_180") == 1)
        .then(pl.lit("dead 180d"))
        .otherwise(pl.lit("alive 180d"))
        .alias("death_180_label"),
    )
    return {"raw": raw, "target": target}


if __name__ == "__main__":
    build()
