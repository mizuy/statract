"""Journal HR forest: row mapping, layout, and font restoration on save."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from statract.figure.hr_forest import (
    ForestRow,
    ForestTextCol,
    build_forest_layout_panels,
    resolve_figure_width,
    rows_from_prepared,
    rows_from_prepared_grouped,
    save_forest_panel_rows,
    save_prepared_hr_forest,
)
from statract import journal_forest as _journal


def _prepared() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Q1", "Q2"],
            "n": [100, 80],
            "event": [2, 5],
            "estimate": [1.0, 1.5],
            "conf.low": [1.0, 1.1],
            "conf.high": [1.0, 2.0],
            "desc": ["Reference", "1.500 (1.100-2.000)"],
            "pvalue": ["", "0.012"],
        }
    )


def test_rows_from_prepared_marks_reference() -> None:
    rows = rows_from_prepared(_prepared())
    assert rows[0].is_reference
    assert rows[0].label == "Q1"
    assert rows[1].hr == 1.5
    assert rows[1].conf_low == 1.1
    assert rows[1].pvalue == "0.012"


def test_grouped_rows_indent_quartiles() -> None:
    df = _prepared().with_columns(pl.lit("ADR").alias("group"))
    rows = rows_from_prepared_grouped(df, group_col="group", group_order=["ADR"])
    assert rows[0].row_kind == "group_header"
    assert rows[0].label == "ADR"
    assert rows[1].label == "  Q1"
    assert rows[2].label == "  Q2"


def test_shared_xlim_uses_one_range() -> None:
    low = ForestRow("low", 10, 1, 0.8, 0.7, 0.9, "0.800 (0.700-0.900)", "0.040")
    high = ForestRow("high", 10, 1, 1.4, 1.2, 1.8, "1.400 (1.200-1.800)", "0.010")
    layout = build_forest_layout_panels(
        [("a", [low]), ("b", [high])],
        shared_x_lim=True,
        width_in=resolve_figure_width(preset="single"),
    )
    assert layout.width_in == 3.15
    assert layout.panels[0].x_lim == layout.panels[1].x_lim
    assert layout.panels[0].x_lim[0] < 0.7
    assert layout.panels[0].x_lim[1] > 1.8


def test_save_restores_matplotlib_font(tmp_path) -> None:
    family = list(plt.rcParams["font.family"])
    sans = list(plt.rcParams["font.sans-serif"])
    written = save_prepared_hr_forest(_prepared(), tmp_path / "forest.png", dpi=72, also_pdf=True)
    assert [p.name for p in written] == ["forest.png", "forest.pdf"]
    assert written[0].stat().st_size > 0
    assert written[1].read_bytes().startswith(b"%PDF")
    assert list(plt.rcParams["font.family"]) == family
    assert list(plt.rcParams["font.sans-serif"]) == sans


def test_empty_frame_writes_nothing(tmp_path) -> None:
    assert save_prepared_hr_forest(pl.DataFrame(), tmp_path / "empty.png") == []


def test_figure_hr_forest_is_stat_shim() -> None:
    assert save_prepared_hr_forest is _journal.save_prepared_hr_forest
    assert ForestRow is _journal.ForestRow


def test_panel_rows_extra_column(tmp_path) -> None:
    row = ForestRow(
        "ADR",
        100,
        4,
        0.98,
        0.96,
        1.01,
        "0.980 (0.960-1.010)",
        "0.200",
        extras=(("cindex", "0.71"),),
    )
    written = save_forest_panel_rows(
        [("main", [row])],
        tmp_path / "panel.png",
        extra_cols=(ForestTextCol("cindex", "C-index"),),
        show_n_event=False,
        dpi=72,
        also_pdf=False,
        figure_width="double",
    )
    assert len(written) == 1
    assert written[0].stat().st_size > 0
