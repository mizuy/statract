"""Thin shim: task_io names still resolve to studyloop."""

from __future__ import annotations

import studyloop.task_io
from statract.task_io import prepare_task_output, save_frames


def test_task_io_shim_is_studyloop() -> None:
    assert prepare_task_output is studyloop.task_io.prepare_task_output
    assert save_frames is studyloop.task_io.save_frames
