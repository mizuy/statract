"""Compatibility shim. Implementation lives in ``studyloop._markdown``."""

from __future__ import annotations

from studyloop._markdown import (
    format_sigfigs,
    github_markdown_table_from_csv,
    github_markdown_table_from_polars,
)

__all__ = [
    "format_sigfigs",
    "github_markdown_table_from_csv",
    "github_markdown_table_from_polars",
]
