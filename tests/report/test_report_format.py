"""Regression tests for report formatting fixes (audit 2026-11-05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from statract.report.artifacts import csv_companion_markdown, write_report_body
from statract.report.markdown import format_sigfigs


def test_csv_companion_keeps_cells_as_written(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("id,p,smd,age\n007,0.10,0.100,35.0\n008,1.00,0.200,40.0\n", encoding="utf-8")
    md = csv_companion_markdown(csv_path)
    assert "| 007 | 0.10 | 0.100 | 35.0 |" in md
    assert "| 008 | 1.00 | 0.200 | 40.0 |" in md


def test_csv_companion_head_keeps_cells_as_written(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("id,p\n" + "".join(f"00{i},0.10\n" for i in range(5)), encoding="utf-8")
    md = csv_companion_markdown(csv_path, max_rows=2, head_rows=2)
    assert "| 000 | 0.10 |" in md


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12345.0, "12345"), (365.0, "365"), (123456.78, "123457"), (9999.7, "10000"), (0.12345, "0.1235"), (-12345.6, "-12346")],
)
def test_format_sigfigs_no_e_notation_for_large(value: float, expected: str) -> None:
    assert format_sigfigs(value) == expected


def test_write_report_body_strips_only_headings(tmp_path: Path) -> None:
    lines = ["## Title", "#1 risk factor", "```python", "# comment", "```", "  ### Sub ###", "text"]
    body = write_report_body(tmp_path, lines).read_text(encoding="utf-8").splitlines()
    assert body == ["Title", "#1 risk factor", "```python", "# comment", "```", "Sub", "text"]
