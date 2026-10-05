"""Example-local paths, parquet cache, and flowchart.

Gallery scripts import this module instead of endolab. It is not part of the
public ``statract`` API and does not set facility defaults.
"""

from __future__ import annotations


import datetime
import functools
import shutil
import sys
import zoneinfo
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Any, ParamSpec, TextIO, TypeVar

import polars as pl

logger = getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

__all__ = [
    "ProjectPath",
    "cache",
    "flowchart",
    "load_data",
    "load_parquet_dir",
    "save_data",
    "snapshot_cache",
]


def get_timestamp(timezone: str | None = "UTC") -> str:
    tz = zoneinfo.ZoneInfo(timezone or "UTC")
    return datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")


def get_unique_path(path: Path | str, *, timezone: str | None = "UTC") -> Path:
    if isinstance(path, str):
        path = Path(path)
    timestamp = get_timestamp(timezone)
    unique = path.parent / f"{path.stem}_{timestamp}{path.suffix}"
    if unique.exists():
        i = 1
        while True:
            unique = path.parent / f"{path.stem}_{timestamp}_{i:02d}{path.suffix}"
            if not unique.exists():
                break
            i += 1
    return unique


class ProjectPath:
    def __init__(
        self,
        project_root: Path | str,
        *,
        data_dir: str = "data",
        cache_dir: str = "cache",
        snapshot_dir: str = "snapshot",
        log_dir: str = "log",
        save_dir: str = "save",
    ) -> None:
        if isinstance(project_root, str):
            project_root = Path(project_root)
        self.project_root = project_root.resolve()
        self.data = self.project_root / data_dir
        self.cache = self.project_root / cache_dir
        self.snapshot = self.project_root / snapshot_dir
        self.logdir_root = self.project_root / log_dir
        self.save_root = self.project_root / save_dir

    def get_logdir(self, name: str) -> Path:
        return get_unique_path(self.logdir_root / name)

    def get_save_dir(self, name: str) -> Path:
        return get_unique_path(self.save_root / name)


def save_data(cache_path: Path | str, value: pl.DataFrame | dict[str, Any]) -> None:
    """Save a frame as parquet, or a dict of frames as a directory of parquet files."""
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)
    if isinstance(value, pl.DataFrame):
        if cache_path.suffix not in {".parquet", ".ipc", ".feather"}:
            raise ValueError(f"DataFrame cache path needs a parquet suffix: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        value.write_parquet(cache_path)
        return
    if isinstance(value, dict):
        cache_path.mkdir(parents=True, exist_ok=True)
        for key, data in value.items():
            if isinstance(data, pl.DataFrame):
                save_data(cache_path / f"{key}.parquet", data)
            elif isinstance(data, dict):
                save_data(cache_path / key, data)
            else:
                raise TypeError(
                    f"example cache only stores DataFrames and dicts of DataFrames, not {type(data).__name__}"
                )
        return
    raise TypeError(f"example cache only stores DataFrames and dicts of DataFrames, not {type(value).__name__}")


def load_data(cache_path: Path | str) -> pl.DataFrame | dict[str, Any]:
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)
    if cache_path.suffix == ".parquet":
        return pl.read_parquet(cache_path)
    if cache_path.suffix in {".ipc", ".feather"}:
        return pl.read_ipc(cache_path, memory_map=False)
    if cache_path.is_dir():
        result: dict[str, Any] = {}
        for file in cache_path.glob("*"):
            if file.is_dir():
                result[file.stem] = load_data(file)
            elif file.suffix in {".parquet", ".ipc", ".feather"}:
                result[file.stem] = load_data(file)
        return result
    raise ValueError(f"Invalid cache_path: {cache_path}")


def load_parquet_dir(cache_path: Path | str) -> dict[str, pl.DataFrame]:
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)
    if not cache_path.is_dir():
        raise ValueError(f"cache_path must be a directory: {cache_path}")
    return {file.stem: pl.read_parquet(file) for file in cache_path.glob("*.parquet")}


class _Cached:
    def __init__(self, generator: Callable[..., T], cache_path: Path, *, snapshot: bool) -> None:
        if not callable(generator):
            raise TypeError("generator must be callable")
        self.generator = generator
        self.cache_basepath = cache_path
        self.snapshot = snapshot
        self.generated_object: T | None = None
        functools.update_wrapper(self, generator)

    def _path(self, *, force: bool = False) -> Path:
        if not self.snapshot:
            return self.cache_basepath
        parent = self.cache_basepath.parent
        if not force and parent.exists():
            found = list(parent.glob(f"{self.cache_basepath.stem}_*"))
            if found:
                return sorted(found)[-1]
        return get_unique_path(parent / self.cache_basepath.name)

    def __call__(self, *, force: bool = False) -> T:
        if force:
            self.clear_cache()
        if self.generated_object is None:
            cache_path = self._path(force=force)
            if cache_path.exists():
                logger.info("Loading from cache: %s", cache_path)
                self.generated_object = load_data(cache_path)  # type: ignore[assignment]
            else:
                start = perf_counter()
                ret = self.generator()
                save_data(cache_path, ret)
                logger.info("Saved %s in %.1fs", cache_path, perf_counter() - start)
                self.generated_object = ret
        return self.generated_object

    def clear_cache(self) -> None:
        if not self.snapshot:
            cache_path = self.cache_basepath
            if cache_path.exists():
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
        self.generated_object = None


def cache(path: Path | str) -> Callable[[Callable[P, T]], _Cached]:
    if isinstance(path, str):
        path = Path(path)

    def decorator(generator: Callable[P, T]) -> _Cached:
        return _Cached(generator, path, snapshot=False)

    return decorator


def snapshot_cache(path: Path | str) -> Callable[[Callable[P, T]], _Cached]:
    if isinstance(path, str):
        path = Path(path)

    def decorator(generator: Callable[P, T]) -> _Cached:
        return _Cached(generator, path, snapshot=True)

    return decorator


def flowchart(
    df: pl.DataFrame,
    flow_defs: dict[str, pl.Expr | dict[str, pl.Expr]],
    out: TextIO = sys.stdout,
    getstat: Callable[[pl.DataFrame], str] | None = None,
) -> pl.DataFrame:
    """Apply inclusion and exclusion steps and write a text flowchart."""
    if getstat is None:
        getstat = lambda frame: f"{frame.height} rows"

    out.write(f"✔ {getstat(df)}\n")

    def _write_step(*, is_exclusion: bool, step_name: str, excluded: pl.DataFrame, sub_lines: list[str]) -> None:
        header = f"|-> {step_name}" if is_exclusion else f"|- {step_name}"
        out.write(f"{header}\n")
        out.write(f"    excluded: {getstat(excluded)}\n")
        for line in sub_lines:
            out.write(line)
        out.write(f"✔ {getstat(df)}\n")

    for name, item_expr in flow_defs.items():
        is_exclusion = name.startswith("-")
        step_name = name[1:] if is_exclusion else name

        if isinstance(item_expr, pl.Expr):
            item_expr = item_expr.fill_null(False)
            if is_exclusion:
                excluded = df.filter(item_expr)
                df = df.filter(~item_expr)
            else:
                excluded = df.filter(~item_expr)
                df = df.filter(item_expr)
            _write_step(is_exclusion=is_exclusion, step_name=step_name, excluded=excluded, sub_lines=[])
        elif isinstance(item_expr, dict):
            df_before = df
            conds: list[pl.Expr] = []
            sub_lines: list[str] = []
            for sub_name, cond in item_expr.items():
                if not isinstance(cond, pl.Expr):
                    raise TypeError(f"Invalid type for exclusion criteria: {type(cond)}")
                cond = cond.fill_null(False)
                sub_is_exclusion = sub_name.startswith("-")
                label = sub_name[1:] if sub_is_exclusion else sub_name
                if is_exclusion:
                    conds.append(cond)
                    marker = "×"
                elif sub_is_exclusion:
                    conds.append(~cond)
                    marker = "×"
                else:
                    conds.append(cond)
                    marker = "✔"
                sub_lines.append(f"      | {marker} {label}: {getstat(df_before.filter(cond))}\n")
            cond = pl.any_horizontal(conds)
            if is_exclusion:
                excluded = df.filter(cond)
                df = df.filter(~cond)
            else:
                excluded = df.filter(~cond)
                df = df.filter(cond)
            _write_step(
                is_exclusion=is_exclusion,
                step_name=step_name,
                excluded=excluded,
                sub_lines=sub_lines,
            )
        else:
            raise TypeError(f"Invalid flowchart step {name!r}: {type(item_expr)}")
    out.write(f"✔ Total: {getstat(df)}\n")
    return df
