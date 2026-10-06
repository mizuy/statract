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
    """``out_dir`` 直下の生成物を削除し、改名・削除された出力の残骸を残さない。"""
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
    """task の ``out/``（または ``out_dir``）を返す。既定では実行前に既存成果物を消す。"""
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
    """``tasks/<name>/<file>.py`` から ``tasks/<name>/out/`` を返す。"""
    return prepare_task_output(
        task_file,
        out_dir,
        clear=clear,
        preserve_subdirs=preserve_subdirs,
    )


def save_frames(paths: Mapping[str, Path], frames: Mapping[str, pl.DataFrame]) -> None:
    """``paths`` と ``frames`` のキー集合を一致させ、Polars DataFrame を CSV 保存する。"""
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
    """``Saved: <path>`` を ``paths`` ごとに 1 行ずつ出力する。"""
    for p in paths:
        if p is None:
            continue
        print(f"Saved: {p}")
