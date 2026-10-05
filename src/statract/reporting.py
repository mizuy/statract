"""Helpers for analysis report artifacts and CSV Markdown companions."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._markdown import github_markdown_table_from_polars

REPORT_SECTION_FILENAME = "section_report.md"
MAX_EMBED_TABLE_ROWS = 100
MAX_EMBED_TABLE_COLUMNS = 50
DEFAULT_CSV_HEAD_ROWS = 10

_FLOW_HEADER_RE = re.compile(r"^####\s*(.+?)\s*####\s*$")
_FLOW_STEP_RE = re.compile(r"^\|-(>?)\s+(.+)$")
_FLOW_EXCLUDED_RE = re.compile(r"^\s+excluded:\s*(.+)$")
_FLOW_CHECK_RE = re.compile(r"^✔\s+(.+)$")
_FLOW_TOTAL_RE = re.compile(r"^✔\s+Total:\s*(.+)$")
_FLOW_NUM_RE = re.compile(r"\d+")


def report_body_path(out_dir: str | Path) -> Path:
    """Return the standard Markdown fragment path under an analysis output dir."""
    return Path(out_dir) / REPORT_SECTION_FILENAME


def out_asset(name: str, *, out_dir_name: str = "out") -> str:
    """Path from a task-level `{stem}_report.md` to an asset in `out/`."""
    return f"{out_dir_name.rstrip('/')}/{name}"


def write_report_body(out_dir: str | Path, lines: list[str]) -> Path:
    """Write the main Markdown fragment imported by `{stem}_report.md`.

    Generated fragments should not own report structure. Leading Markdown heading
    markers are stripped so the task-level report can control section hierarchy.
    """
    path = report_body_path(out_dir)
    plain_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            plain_lines.append(stripped.lstrip("#").strip())
        else:
            plain_lines.append(line)
    path.write_text("\n".join(plain_lines), encoding="utf-8")
    return path


def markdown_flowchart(flowchart: str) -> str:
    """Return flowchart text as a Markdown fenced code block."""
    body = flowchart.rstrip("\n")
    return f"```text\n{body}\n```\n"


@dataclass(frozen=True, slots=True)
class FlowchartStep:
    """One inclusion/exclusion step parsed from ``pp.flowchart`` text."""

    name: str
    excluded: str
    remaining: str
    is_exclusion: bool


@dataclass(frozen=True, slots=True)
class ParsedFlowchart:
    """Structured view of ``pp.flowchart`` stdout (possibly multi-call)."""

    title: str | None
    initial: str | None
    steps: list[FlowchartStep]


def _comma_stat(stat: str) -> str:
    return _FLOW_NUM_RE.sub(lambda m: f"{int(m.group()):,}", stat)


def _mermaid_label(text: str) -> str:
    """Escape text for Mermaid node labels inside ``[...]``."""
    return (
        text.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("\n", "<br/>")
    )


def parse_flowchart(flowchart: str) -> ParsedFlowchart:
    """Parse ``pp.flowchart`` text into title / initial / steps.

    Supports concatenated multi-call buffers (each call ends with
    ``✔ Total: ...`` and the next call repeats the residual count).
    """
    title: str | None = None
    initial: str | None = None
    steps: list[FlowchartStep] = []
    pending_name: str | None = None
    pending_is_exclusion = False
    pending_excluded: str | None = None

    for raw in flowchart.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        header = _FLOW_HEADER_RE.match(line)
        if header:
            title = header.group(1).strip()
            continue
        if _FLOW_TOTAL_RE.match(line):
            continue
        step = _FLOW_STEP_RE.match(line)
        if step:
            pending_name = step.group(2).strip()
            pending_is_exclusion = step.group(1) == ">"
            pending_excluded = None
            continue
        excluded = _FLOW_EXCLUDED_RE.match(line)
        if excluded and pending_name is not None:
            pending_excluded = excluded.group(1).strip()
            continue
        check = _FLOW_CHECK_RE.match(line)
        if not check:
            continue
        stat = check.group(1).strip()
        if pending_name is not None and pending_excluded is not None:
            steps.append(
                FlowchartStep(
                    name=pending_name,
                    excluded=pending_excluded,
                    remaining=stat,
                    is_exclusion=pending_is_exclusion,
                ),
            )
            pending_name = None
            pending_excluded = None
            continue
        if initial is None:
            initial = stat

    return ParsedFlowchart(title=title, initial=initial, steps=steps)


def _excluded_reason_label(step: FlowchartStep) -> str:
    name = step.name.strip()
    if step.is_exclusion or name.lower().startswith("exclude"):
        return name
    return f"not {name}"


def mermaid_flowchart(
    flowchart: str,
    *,
    final_label: str | None = "Analysis cohort",
) -> str:
    """Convert ``pp.flowchart`` text to a Mermaid ``flowchart TB`` fence.

    Main residual counts form a vertical keep path; each step branches to an
    excluded node that carries the criterion reason.
    """
    parsed = parse_flowchart(flowchart)
    if parsed.initial is None or not parsed.steps:
        return markdown_flowchart(flowchart)

    ids = [chr(ord("A") + i) for i in range(len(parsed.steps) + 1)]
    lines = [
        "```mermaid",
        "flowchart TB",
        "  classDef keep fill:#e8f5e9,stroke:#2e7d32,color:#111",
        "  classDef drop fill:#f5f5f5,stroke:#9e9e9e,color:#333",
    ]

    start_body = _comma_stat(parsed.initial)
    if parsed.title:
        start_body = f"{parsed.title}<br/>{start_body}"
    lines.append(f'  {ids[0]}["{_mermaid_label(start_body)}"]:::keep')

    for i, step in enumerate(parsed.steps):
        keep_id = ids[i + 1]
        drop_id = f"X{i + 1}"
        keep_stat = _comma_stat(step.remaining)
        if i + 1 == len(parsed.steps) and final_label:
            keep_body = f"{final_label}<br/>{keep_stat}"
        else:
            keep_body = keep_stat
        drop_body = f"{_excluded_reason_label(step)}<br/>excluded: {_comma_stat(step.excluded)}"
        lines.append(f'  {keep_id}["{_mermaid_label(keep_body)}"]:::keep')
        lines.append(f'  {drop_id}["{_mermaid_label(drop_body)}"]:::drop')

    for i in range(len(parsed.steps)):
        lines.append(f"  {ids[i]} --> {ids[i + 1]}")
        lines.append(f"  {ids[i]} -.-> X{i + 1}")

    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_text_artifact(
    path: str | Path,
    text: str,
    *,
    written: list[Path] | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Write a text/Markdown artifact and optionally append it to `written`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding=encoding)
    if written is not None:
        written.append(out)
    return out


def write_csv_artifact(
    df: Any,
    path: str | Path,
    *,
    written: list[Path] | None = None,
    companion: bool = True,
    **companion_kwargs: Any,
) -> list[Path]:
    """Write a CSV artifact and optional Markdown companion.

    `df` is expected to provide a Polars-like `write_csv(path)` method.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    paths = [out]
    if companion:
        paths.append(write_csv_companion(out, **companion_kwargs))
    if written is not None:
        written.extend(paths)
    return paths


def write_table(
    out_dir: str | Path,
    stem: str,
    df: Any,
    written: list[Path],
    *,
    skip_if_empty: bool = True,
    skip_empty_except: tuple[str, ...] = (),
) -> None:
    """``out_dir/{stem}.csv`` と Markdown companion を書き、``written`` に追記する。"""
    if skip_if_empty and df.is_empty() and stem not in skip_empty_except:
        return
    write_csv_artifact(df, Path(out_dir) / f"{stem}.csv", written=written)


def plotly_category_orders(df: Any, cols: list[str], *, reverse: bool = False) -> dict[str, list[Any]]:
    """Build stable Plotly category orders from Polars-like columns."""
    out: dict[str, list[Any]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        dtype = getattr(series, "dtype", None)
        if dtype is not None and dtype.__class__.__name__ in {"Enum", "Categorical"}:
            cats = series.cat.get_categories().to_list()
        else:
            cats = series.drop_nulls().unique().to_list()
            with suppress(TypeError):
                cats = sorted(cats, key=lambda x: (str(type(x)), str(x)))
        if reverse:
            cats = list(reversed(cats))
        out[col] = cats
    return out


def finalize_plotly_facet_figure(fig: Any) -> Any:
    """Apply common report styling to a Plotly facet figure."""
    fig.for_each_annotation(lambda a: a.update(text=(a.text or "").split("=")[-1]))
    return (
        fig.update_xaxes(title=None)
        .update_yaxes(title=None, showticklabels=False)
        .update_layout(
            template="none",
            font=dict(family="Helvetica", size=20),
            legend_title=None,
            legend_traceorder="reversed",
        )
    )


def write_plotly_artifact(
    fig: Any,
    out_dir: str | Path,
    stem: str,
    *,
    written: list[Path] | None = None,
    html: bool = True,
    png: bool = True,
    image_scale: int = 4,
) -> list[Path]:
    """Write a Plotly figure as HTML and, when available, PNG.

    PNG export depends on optional Plotly image backends such as kaleido. If PNG
    export fails, this helper writes a small Markdown note instead of failing the
    analysis task.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if html:
        html_path = directory / f"{stem}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        paths.append(html_path)
    if png:
        png_path = directory / f"{stem}.png"
        try:
            fig.write_image(png_path, scale=image_scale)
        except Exception as exc:  # noqa: BLE001
            note = directory / f"{stem}_png_unavailable.md"
            note.write_text(
                f"`{png_path.name}` は Plotly image export の失敗により未生成。\n\n"
                f"理由: `{type(exc).__name__}: {exc}`\n",
                encoding="utf-8",
            )
            paths.append(note)
        else:
            paths.append(png_path)
    if written is not None:
        written.extend(paths)
    return paths


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _csv_link(csv_path: Path, csv_link_prefix: str | None) -> str:
    if csv_link_prefix is None:
        return csv_path.name
    return f"{csv_link_prefix.rstrip('/')}/{csv_path.name}"


def csv_companion_markdown(
    csv_path: str | Path,
    *,
    max_rows: int | None = MAX_EMBED_TABLE_ROWS,
    max_columns: int | None = MAX_EMBED_TABLE_COLUMNS,
    head_rows: int = DEFAULT_CSV_HEAD_ROWS,
    csv_link_prefix: str | None = "out",
) -> str:
    """Return Markdown companion text for a CSV.

    Small CSVs are rendered fully as a Markdown table. Large CSVs include shape,
    column names, a CSV link, and a Markdown table of the first `head_rows` rows.
    """
    import polars as pl  # noqa: PLC0415

    path = Path(csv_path)
    try:
        scan = pl.scan_csv(path)
        columns = scan.collect_schema().names()
        height = scan.select(pl.len().alias("_n")).collect().item()
    except pl.exceptions.NoDataError:
        return f"`{path.name}` は空の CSV です。\n\nCSV: [{path.name}]({_csv_link(path, csv_link_prefix)})\n"
    width = len(columns)
    can_embed_rows = max_rows is None or height <= max_rows
    can_embed_columns = max_columns is None or width <= max_columns
    if can_embed_rows and can_embed_columns:
        df = pl.read_csv(path)
        return github_markdown_table_from_polars(df, max_rows=None)

    n_head = min(max(head_rows, 0), height)
    head = pl.read_csv(path, n_rows=head_rows)
    lines = [
        f"`{path.name}` は大きいため、全文は CSV を参照する。",
        "",
        f"- 行数: {height}",
        f"- 列数: {width}",
        f"- CSV: [{path.name}]({_csv_link(path, csv_link_prefix)})",
        "",
        "列名:",
        "",
    ]
    lines.extend(f"- `{_md_cell(column)}`" for column in columns)
    lines.extend(
        [
            "",
            f"先頭 {n_head} 行:",
            "",
            github_markdown_table_from_polars(head, max_rows=None).rstrip(),
        ],
    )
    return "\n".join(lines) + "\n"


def write_csv_companion(
    csv_path: str | Path,
    *,
    max_rows: int | None = MAX_EMBED_TABLE_ROWS,
    max_columns: int | None = MAX_EMBED_TABLE_COLUMNS,
    head_rows: int = DEFAULT_CSV_HEAD_ROWS,
    csv_link_prefix: str | None = "out",
) -> Path:
    """Write `{stem}_csv.md` next to a CSV and return the written path."""
    path = Path(csv_path)
    md_path = path.with_name(f"{path.stem}_csv.md")
    body = csv_companion_markdown(
        path,
        max_rows=max_rows,
        max_columns=max_columns,
        head_rows=head_rows,
        csv_link_prefix=csv_link_prefix,
    )
    md_path.write_text(
        "\n".join([f"<!-- generated from {path.name} -->", "", body.rstrip(), ""]),
        encoding="utf-8",
    )
    return md_path


def write_csv_companions(
    out_dir: str | Path,
    *,
    pattern: str = "*.csv",
    max_rows: int | None = MAX_EMBED_TABLE_ROWS,
    max_columns: int | None = MAX_EMBED_TABLE_COLUMNS,
    head_rows: int = DEFAULT_CSV_HEAD_ROWS,
    skip_stems: frozenset[str] | None = None,
    csv_link_prefix: str | None = "out",
) -> list[Path]:
    """Write `{stem}_csv.md` companions for CSV files in `out_dir`."""
    directory = Path(out_dir)
    skip = skip_stems or frozenset()
    written: list[Path] = []
    for csv_path in sorted(directory.glob(pattern)):
        if csv_path.stem.endswith("_csv") or csv_path.stem in skip:
            continue
        written.append(
            write_csv_companion(
                csv_path,
                max_rows=max_rows,
                max_columns=max_columns,
                head_rows=head_rows,
                csv_link_prefix=csv_link_prefix,
            ),
        )
    return written


def write_reporting_fragments(out_dir: str | Path, *, written: list[Path] | None = None) -> list[Path]:
    """Write generic report fragments for an analysis output directory.

    Currently this writes CSV Markdown companions. The `written` argument is
    accepted for compatibility with project-specific orchestrators.
    """
    fragments = write_csv_companions(out_dir)
    if written is not None:
        written.extend(fragments)
    return fragments
