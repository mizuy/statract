#!/usr/bin/env python3
"""Collect example raw snapshots into one directory for a private data repo.

Each example's ``build.py`` defines ``load_snapshot()``, which fetches the raw
public file once and keeps it under ``examples/<stem>/snapshot/``. This script
runs those loaders (fetching only when no snapshot exists yet, or always with
``--refetch``), copies each parquet to ``<out>/<name>.parquet``, and writes
``<out>/MANIFEST.csv`` with the row count and SHA-256.

The examples read the directory back when ``STATRACT_DATA_DIR`` points at it
(see ``examples/support.py``). Do not commit the output to this repository.

Usage::

    uv run python scripts/snapshot_private_data.py ../statract-data
    uv run python scripts/snapshot_private_data.py ../statract-data --stem psm_rhc
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Every example with a fetched raw file. cea_sicksicker has none.
DEFAULT_STEMS = (
    "pred_support",
    "psm_rhc",
    "surv_colon",
    "cif_pbc",
    "aft_rotterdam",
    "cox_retinopathy",
    "lmm_pbcseq",
    "logit_indo",
    "iptw_nhefs",
)

_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"

SOURCES = {
    "pred_support": (
        "https://hbiostat.org/data/repo/support2csv.zip (fallback: UCI 880)",
        "Vanderbilt Biostatistics SUPPORT2 teaching data; UCI copy is CC BY 4.0",
    ),
    "psm_rhc": (
        "https://hbiostat.org/data/repo/rhc.csv",
        "Vanderbilt RHC (Connors JAMA 1996); teaching use only, do not redistribute",
    ),
    "surv_colon": (f"{_RDATASETS}/survival/colon.csv", "R survival, GPL-2/3"),
    "cif_pbc": (f"{_RDATASETS}/survival/pbc.csv", "R survival, GPL-2/3"),
    "aft_rotterdam": (f"{_RDATASETS}/survival/rotterdam.csv", "R survival, GPL-2/3"),
    "cox_retinopathy": (f"{_RDATASETS}/survival/retinopathy.csv", "R survival, GPL-2/3"),
    "lmm_pbcseq": (f"{_RDATASETS}/survival/pbcseq.csv", "R survival, GPL-2/3"),
    "logit_indo": (f"{_RDATASETS}/medicaldata/indo_rct.csv", "R medicaldata, MIT; cite Elmunzer NEJM 2012"),
    "iptw_nhefs": (f"{_RDATASETS}/causaldata/nhefs.csv", "R causaldata NHEFS; Hernan and Robins teaching data"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_build(stem: str):
    """Import ``examples/<stem>/build.py`` fresh (examples share module names)."""
    for name in ("build", "project", "config"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(EXAMPLES / stem))
    sys.path.insert(1, str(EXAMPLES))
    try:
        return importlib.import_module("build")
    finally:
        sys.path.remove(str(EXAMPLES / stem))
        sys.path.remove(str(EXAMPLES))


def snapshot_stem(stem: str, out: Path, *, refetch: bool) -> dict[str, object]:
    build = _load_build(stem)
    loader = build.load_snapshot
    frame = loader(force=refetch)
    src = loader._path()
    if not src.is_file():
        raise FileNotFoundError(f"{stem}: no snapshot written at {src}")
    name = loader.cache_basepath.name
    dst = out / name
    shutil.copyfile(src, dst)
    source, license_note = SOURCES.get(stem, ("see examples/<stem>/build.py", ""))
    return {
        "file": name,
        "stem": stem,
        "rows": frame.height if isinstance(frame, pl.DataFrame) else "",
        "columns": frame.width if isinstance(frame, pl.DataFrame) else "",
        "sha256": _sha256(dst),
        "source": source,
        "license": license_note,
        "snapshot": src.name,
        "copied_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _write_manifest(out: Path, rows: list[dict[str, object]]) -> Path:
    path = out / "MANIFEST.csv"
    kept: dict[str, dict[str, object]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                kept[row["file"]] = row
    for row in rows:
        kept[str(row["file"])] = row
    fields = ["file", "stem", "rows", "columns", "sha256", "source", "license", "snapshot", "copied_at"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in sorted(kept):
            writer.writerow({f: kept[key].get(f, "") for f in fields})
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="Output directory (the private data repo checkout)")
    parser.add_argument("--stem", nargs="*", default=list(DEFAULT_STEMS), help="Example stems to collect")
    parser.add_argument("--refetch", action="store_true", help="Fetch again even if a snapshot exists")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("write the snapshots outside this repository")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem in args.stem:
        if not (EXAMPLES / stem / "build.py").is_file():
            parser.error(f"unknown example stem: {stem}")
        row = snapshot_stem(stem, out, refetch=args.refetch)
        rows.append(row)
        print(f"{stem}: {row['file']} rows={row['rows']} sha256={str(row['sha256'])[:12]}…")
    print(f"wrote {_write_manifest(out, rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
