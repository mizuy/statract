"""Configuration constants and helper functions for the example project."""

import os
from pathlib import Path

from project import project

# Project paths using Project class
PROJECT_ROOT = project.project_root
CACHE = project.cache
SNAPSHOT_ROOT = project.snapshot
LOGDIR_ROOT = project.logdir_root


def get_logdir(name: str):
    """ログディレクトリを生成(タイムスタンプ付き)

    実行時にLOGDIR環境変数が設定されている場合、
    そのディレクトリを返す(Taskfileから実行された場合)。
    そうでない場合、新しくタイムスタンプ付きディレクトリを作成する。

    注: notebook内では通常、LOGDIR環境変数を直接参照することを推奨します
       (Python: `os.environ.get("LOGDIR")`, R: `Sys.getenv("LOGDIR")`)。
    """
    env_logdir = os.environ.get("LOGDIR")
    if env_logdir:
        logdir = Path(env_logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        return logdir
    return project.get_logdir(name)


def latest_logdir() -> str | None:
    """最新のログディレクトリのパスを返す(存在しない場合はNone)。"""
    if not LOGDIR_ROOT.exists():
        return None
    logdirs = sorted(LOGDIR_ROOT.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logdirs:
        return str(logdirs[0])
    return None
