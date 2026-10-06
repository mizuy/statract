"""Deprecated shim. Implementation lives in ``studyloop.task_io``."""

from __future__ import annotations

from statract._deprecate import warn_moved
from studyloop.task_io import (
    clear_task_output_dir,
    prepare_task_output,
    print_saved,
    save_frames,
    task_output_dir,
)

warn_moved("statract.task_io", "studyloop.task_io")

__all__ = [
    "clear_task_output_dir",
    "prepare_task_output",
    "print_saved",
    "save_frames",
    "task_output_dir",
]
