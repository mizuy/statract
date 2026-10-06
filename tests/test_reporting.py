"""Thin shims: public reporting / task_io names still resolve to studyloop."""

from __future__ import annotations

import studyloop.reporting
import studyloop.task_io
import statract.reporting
import statract.task_io


def test_reporting_shim_is_studyloop() -> None:
    assert statract.reporting.write_csv_companion is studyloop.reporting.write_csv_companion
    assert statract.reporting.write_report_body is studyloop.reporting.write_report_body
    assert statract.reporting.mermaid_flowchart is studyloop.reporting.mermaid_flowchart


def test_task_io_shim_is_studyloop() -> None:
    assert statract.task_io.prepare_task_output is studyloop.task_io.prepare_task_output
    assert statract.task_io.save_frames is studyloop.task_io.save_frames
