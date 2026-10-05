"""Tests for generic analysis reporting helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from statract.reporting import (
    csv_companion_markdown,
    mermaid_flowchart,
    out_asset,
    parse_flowchart,
    write_csv_companion,
    write_csv_companions,
    write_report_body,
)


def test_csv_companion_markdown_embeds_small_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_csv(csv_path)

    md = csv_companion_markdown(csv_path)

    assert "| a | b |" in md
    assert "全文は CSV" not in md


def test_csv_companion_markdown_large_csv_includes_head(tmp_path: Path) -> None:
    csv_path = tmp_path / "large.csv"
    pl.DataFrame({"n": range(5), "label": [f"row-{i}" for i in range(5)]}).write_csv(csv_path)

    md = csv_companion_markdown(csv_path, max_rows=2, head_rows=3)

    assert "- 行数: 5" in md
    assert "- 列数: 2" in md
    assert "[large.csv](out/large.csv)" in md
    assert "先頭 3 行" in md
    assert "row-0" in md
    assert "row-2" in md
    assert "row-3" not in md


def test_csv_companion_markdown_clamps_negative_head_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "large.csv"
    pl.DataFrame({"n": range(3)}).write_csv(csv_path)

    md = csv_companion_markdown(csv_path, max_rows=1, head_rows=-1)

    assert "先頭 0 行" in md
    assert "| n |" in md
    assert "| 0 |" not in md


def test_write_csv_companion_and_companions(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "already_csv.md").write_text("ignored", encoding="utf-8")
    pl.DataFrame({"a": [1]}).write_csv(out / "table_a.csv")
    pl.DataFrame({"b": [2]}).write_csv(out / "table_b.csv")

    md_path = write_csv_companion(out / "table_a.csv")
    written = write_csv_companions(out, skip_stems=frozenset({"table_a"}))

    assert md_path == out / "table_a_csv.md"
    assert (out / "table_a_csv.md").exists()
    assert written == [out / "table_b_csv.md"]


def test_write_report_body_strips_headings(tmp_path: Path) -> None:
    path = write_report_body(tmp_path, ["# Heading", "plain", "  ## Nested"])

    assert path.name == "section_report.md"
    assert path.read_text(encoding="utf-8").splitlines() == ["Heading", "plain", "Nested"]


def test_out_asset_default_path() -> None:
    assert out_asset("fig.png") == "out/fig.png"


_SAMPLE_FLOW = """\
#### Integrated lesions ####
✔ 36606 lesions, 11203 patients
|- ESD or surgery
    excluded: 32626 lesions, 9830 patients
✔ 3980 lesions, 3662 patients
✔ Total: 3980 lesions, 3662 patients
✔ 3980 lesions, 3662 patients
|- pathological adenocarcinoma
    excluded: 551 lesions, 487 patients
✔ 3429 lesions, 3221 patients
✔ Total: 3429 lesions, 3221 patients
✔ 3429 lesions, 3221 patients
|-> nested exclusion
    excluded: 10 lesions, 9 patients
✔ 3419 lesions, 3212 patients
✔ Total: 3419 lesions, 3212 patients
✔ 3419 lesions, 3212 patients
|- exclude preoperative advanced depth (MP/SS/SE)
    excluded: 1484 lesions, 1458 patients
✔ 1776 lesions, 1689 patients
✔ Total: 1776 lesions, 1689 patients
"""


def test_parse_flowchart_multi_call() -> None:
    parsed = parse_flowchart(_SAMPLE_FLOW)
    assert parsed.title == "Integrated lesions"
    assert parsed.initial == "36606 lesions, 11203 patients"
    assert [s.name for s in parsed.steps] == [
        "ESD or surgery",
        "pathological adenocarcinoma",
        "nested exclusion",
        "exclude preoperative advanced depth (MP/SS/SE)",
    ]
    assert parsed.steps[0].excluded == "32626 lesions, 9830 patients"
    assert parsed.steps[0].remaining == "3980 lesions, 3662 patients"
    assert parsed.steps[2].is_exclusion is True
    assert parsed.steps[3].is_exclusion is False


def test_mermaid_flowchart_keep_path_and_reasons() -> None:
    md = mermaid_flowchart(_SAMPLE_FLOW)
    assert md.startswith("```mermaid\nflowchart TB\n")
    assert 'A["Integrated lesions<br/>36,606 lesions, 11,203 patients"]:::keep' in md
    assert 'X1["not ESD or surgery<br/>excluded: 32,626 lesions, 9,830 patients"]:::drop' in md
    assert 'X3["nested exclusion<br/>excluded: 10 lesions, 9 patients"]:::drop' in md
    assert (
        'X4["exclude preoperative advanced depth (MP/SS/SE)<br/>'
        'excluded: 1,484 lesions, 1,458 patients"]:::drop'
    ) in md
    assert 'E["Analysis cohort<br/>1,776 lesions, 1,689 patients"]:::keep' in md
    assert "A --> B" in md
    assert "B --> C" in md
    assert "C --> D" in md
    assert "D --> E" in md
    assert "A -.-> X1" in md
    assert md.endswith("```\n")
