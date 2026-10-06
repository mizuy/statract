"""Polars DataFrame → GitHub Flavored Markdown pipe table.

Self-contained copy of the report-table helpers (no endoschema / endolab import).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat

# Pandoc pipe tables strip leading ASCII spaces; em space survives into docx.
_EM_SPACE = "\u2003"
_DEFAULT_FLOAT_SIGFIGS = 4

if TYPE_CHECKING:
    import polars as pl


def format_sigfigs(value: SupportsFloat | None, *, digits: int = 4) -> str:
    """Format a float for report tables with at most ``digits`` significant figures.

    Integers that are exact (e.g. 365.0) are shown without a trailing ``.0``.
    Non-finite values fall back to ``str(...)``. ``None`` becomes ``""``.
    """
    if value is None:
        return ""
    x = float(value)
    if not math.isfinite(x):
        return str(x)
    if x == 0.0:
        return "0"
    text = f"{x:.{digits}g}"
    return text


def _preserve_leading_indent(s: str) -> str:
    # tableone category sub-rows use two ASCII spaces; one space is not nested indent.
    match = re.match(r"^(  +)(.*)$", s)
    if not match:
        return s
    level = len(match.group(1)) // 2
    return _EM_SPACE * level + match.group(2).lstrip()


def _md_cell(value: object, *, float_sigfigs: int | None = _DEFAULT_FLOAT_SIGFIGS) -> str:
    if value is None:
        return ""
    # bool is a subclass of int — keep True/False / 0/1 as-is via str().
    if float_sigfigs is not None and isinstance(value, float) and not isinstance(value, bool):
        s = format_sigfigs(value, digits=float_sigfigs)
    else:
        s = str(value)
    s = s.replace("\n", " ").replace("|", "\\|")
    return _preserve_leading_indent(s)


def github_markdown_table_from_polars(
    df: pl.DataFrame,
    *,
    max_rows: int | None = 200,
    float_sigfigs: int | None = _DEFAULT_FLOAT_SIGFIGS,
) -> str:
    """Polars DataFrame を GFM 風のパイプ表 (ヘッダ + 区切り行 + 本体) にする。"""
    import polars as pl  # noqa: PLC0415

    if not isinstance(df, pl.DataFrame):
        msg = f"expected polars.DataFrame, got {type(df)!r}"
        raise TypeError(msg)
    view = df if max_rows is None else df.head(max_rows)
    cols = list(view.columns)
    if not cols:
        return "_（列が空です）_\n"
    header = "| " + " | ".join(_md_cell(c, float_sigfigs=None) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in view.iter_rows(named=False):
        lines.append(
            "| " + " | ".join(_md_cell(v, float_sigfigs=float_sigfigs) for v in row) + " |"
        )
    return "\n".join(lines) + "\n"


def github_markdown_table_from_csv(
    path: str | Path,
    *,
    max_rows: int | None = 200,
    float_sigfigs: int | None = _DEFAULT_FLOAT_SIGFIGS,
) -> str:
    """CSV を読み、`github_markdown_table_from_polars` と同じ形式の Markdown を返す。"""
    import polars as pl  # noqa: PLC0415

    df = pl.read_csv(Path(path))
    return github_markdown_table_from_polars(df, max_rows=max_rows, float_sigfigs=float_sigfigs)
