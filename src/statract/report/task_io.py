"""Task output helpers for an analysis pipeline.

Clears and creates an ``out/`` directory, writes Polars frames to CSV, and
prints ``Saved:`` lines.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

import polars as pl


def clear_task_output_dir(
    out_dir: Path,
    *,
    preserve_subdirs: tuple[str, ...] = (),
) -> None:
    """Delete generated files directly under ``out_dir`` so no stale outputs from renamed or removed files remain."""
    if not out_dir.is_dir():
        return
    preserve = {out_dir.resolve() / name for name in preserve_subdirs}
    for child in list(out_dir.iterdir()):
        if child.resolve() in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prepare_task_output(
    task_file: str | Path,
    out_dir: Path | None = None,
    *,
    clear: bool = True,
    preserve_subdirs: tuple[str, ...] = (),
) -> Path:
    """Return the task's ``out/`` (or ``out_dir``). By default, existing outputs are deleted before the run."""
    out = Path(out_dir) if out_dir is not None else Path(task_file).resolve().parent / "out"
    if clear:
        clear_task_output_dir(out, preserve_subdirs=preserve_subdirs)
    out.mkdir(parents=True, exist_ok=True)
    return out


def task_output_dir(
    task_file: str | Path,
    out_dir: Path | None = None,
    *,
    clear: bool = True,
    preserve_subdirs: tuple[str, ...] = (),
) -> Path:
    """Return ``tasks/<name>/out/`` from ``tasks/<name>/<file>.py``."""
    return prepare_task_output(
        task_file,
        out_dir,
        clear=clear,
        preserve_subdirs=preserve_subdirs,
    )


def save_frames(paths: Mapping[str, Path], frames: Mapping[str, pl.DataFrame]) -> None:
    """Check that ``paths`` and ``frames`` have the same keys, and save the Polars DataFrames as CSV."""
    miss_in_paths = sorted(set(frames) - set(paths))
    miss_in_frames = sorted(set(paths) - set(frames))
    if miss_in_paths or miss_in_frames:
        raise KeyError(
            f"save_frames: paths/frames の key が一致しません。 "
            f"paths にない frame={miss_in_paths!r}, frames にない path={miss_in_frames!r}"
        )
    for key, df in frames.items():
        df.write_csv(paths[key])


def print_saved(*paths: Path | str) -> None:
    """Print one ``Saved: <path>`` line for each entry in ``paths``."""
    for p in paths:
        if p is None:
            continue
        print(f"Saved: {p}")
