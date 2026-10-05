"""Compatibility shim for the old journal HR forest import path.

Canonical coefficient / OR / HR forests: ``from statract import plot_forest``.
Grouped n/event/panel layout still lives in ``statract.journal_forest`` and
is re-exported here so existing ``statract.figure.hr_forest`` imports keep working.
"""

from __future__ import annotations

from statract.journal_forest import (
    DEFAULT_FIGURE_WIDTH,
    DPI,
    DOUBLE_COLUMN_WIDTH_IN,
    FIGURE_WIDTH_PRESETS,
    FONT_PT,
    SINGLE_COLUMN_WIDTH_IN,
    WIDTH_IN,
    FigureWidthPreset,
    ForestLayout,
    ForestPanelSpec,
    ForestRow,
    ForestTextCol,
    build_forest_layout_panels,
    gap_row,
    panel_rows_from_prepared,
    pt_to_in,
    quartile_group_rows,
    render_hr_forest,
    resolve_figure_width,
    rows_from_prepared,
    rows_from_prepared_grouped,
    rows_from_prepared_named_groups,
    save_forest_panel_rows,
    save_hr_forest_figure,
    save_prepared_hr_forest,
)

__all__ = [
    "DEFAULT_FIGURE_WIDTH",
    "DPI",
    "DOUBLE_COLUMN_WIDTH_IN",
    "FIGURE_WIDTH_PRESETS",
    "FONT_PT",
    "SINGLE_COLUMN_WIDTH_IN",
    "WIDTH_IN",
    "FigureWidthPreset",
    "ForestLayout",
    "ForestPanelSpec",
    "ForestRow",
    "ForestTextCol",
    "build_forest_layout_panels",
    "gap_row",
    "panel_rows_from_prepared",
    "pt_to_in",
    "quartile_group_rows",
    "render_hr_forest",
    "resolve_figure_width",
    "rows_from_prepared",
    "rows_from_prepared_grouped",
    "rows_from_prepared_named_groups",
    "save_forest_panel_rows",
    "save_hr_forest_figure",
    "save_prepared_hr_forest",
]
