from pathlib import Path

from project import project

PROJECT_ROOT = project.project_root
CACHE = project.cache
SNAPSHOT_ROOT = project.snapshot
LOGDIR_ROOT = project.logdir_root
ANALYSIS_OUT = PROJECT_ROOT / "aft_rotterdam_out"


def ensure_analysis_out() -> Path:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_OUT / "figures").mkdir(parents=True, exist_ok=True)
    return ANALYSIS_OUT
