"""Regression tests for group handling in Table One (audit 2026-11-05, M29-M32, M38, low items)."""

from __future__ import annotations

import polars as pl

from statract import tableone_raw
from statract.tableone import agg, stat
from statract.tableone.table import tableone_gt_from_frame


def test_anova_drops_null_group_rows() -> None:
    x = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 30.0, 40.0, 50.0])
    g = pl.Series(["a", "a", "b", "b", "a", "b", None, None, None])
    keep = g.is_not_null()
    assert stat.stat_anova(x, g) == stat.stat_anova(x.filter(keep), g.filter(keep))


def test_chisq_drops_null_group_rows() -> None:
    x = pl.Series(["y", "n", "y", "n", "y", "n", "y", "y", "y"])
    g = pl.Series(["a", "a", "b", "b", "a", "b", None, None, None])
    keep = g.is_not_null()
    assert stat.stat_chisq(x, g) == stat.stat_chisq(x.filter(keep), g.filter(keep))


def test_one_group_is_not_tested() -> None:
    x = pl.Series(["y", "n", "y"])
    g = pl.Series(["a", "a", "a"])
    assert stat.stat_fisher(x, g) == "-"
    assert stat.stat_chisq(x, g) == "-"


def test_variable_missing_in_one_group_gives_dash() -> None:
    df = pl.DataFrame({"s": ["y", "n", "y", None, None], "g": ["A", "A", "A", "B", "B"]})
    out = tableone_raw(df, {"S": ("s", "category")}, hue="g", add_pvalue=True)
    assert out.filter(pl.col("name") == "S")["P value"].item() == "-"


def _group_columns(out: pl.DataFrame) -> list[str]:
    return [c for c in out.columns if c not in {"name", "All", "P value", "SMD"}]


def test_numeric_hue_columns_sort_numerically() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "g": [10, 2, 1, 10, 2]})
    assert _group_columns(tableone_raw(df, {"X": ("x", "mean_sd")}, hue="g")) == ["1", "2", "10"]


def test_enum_hue_columns_keep_enum_order() -> None:
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "g": pl.Series(["Placebo", "Drug", "Placebo", "Drug"], dtype=pl.Enum(["Placebo", "Drug"]))}
    )
    assert _group_columns(tableone_raw(df, {"X": ("x", "mean_sd")}, hue="g")) == ["Placebo", "Drug"]


def test_string_hue_columns_sort_as_text() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["b", "a", "10"]})
    assert _group_columns(tableone_raw(df, {"X": ("x", "mean_sd")}, hue="g")) == ["10", "a", "b"]


def test_null_hue_is_not_a_group_column() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 50.0], "g": ["a", "a", "b", "b", None]})
    out = tableone_raw(df, {"X": ("x", "mean_sd")}, hue="g", add_all=True)
    assert out.columns == ["name", "All", "a", "b"]
    # The All column still counts every row.
    assert out["All"].item() == agg.agg_mean_sd(pl.col("x")).pipe(lambda e: df.select(e)).item()


def test_null_hue_left_out_of_header_counts() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 50.0], "g": ["a", "a", "b", "b", None]})
    table = tableone_raw(df, {"X": ("x", "mean_sd")}, hue="g")
    gt = tableone_gt_from_frame(table, source_df=df, hue="g")
    labels = [str(c.column_label) for c in gt._boxhead]
    assert not any("None" in label for label in labels)


def test_absent_level_uses_cell_format() -> None:
    df = pl.DataFrame({"s": ["F", "M", "F", "F"], "g": ["A", "A", "B", "B"]})
    out = tableone_raw(df, {"Sex": ("s", "category")}, hue="g")
    assert out.filter(pl.col("name") == "  M")["B"].item() == "0 (0.0%)"
    out_p = tableone_raw(df, {"Sex": ("s", "category_p")}, hue="g")
    assert out_p.filter(pl.col("name") == "  M")["B"].item() == "0.0"


def test_row_groups_work_with_category_rows() -> None:
    df = pl.DataFrame({"sex": ["F", "M", "F"], "age": [40.0, 50.0, 60.0]})
    table = tableone_raw(df, {"Background | Sex": ("sex", "category"), "Background | Age": ("age", "mean_sd")})
    gt = tableone_gt_from_frame(table)
    data = gt._tbl_data
    assert "row_group" in data.columns
    assert set(data["row_group"].to_list()) == {"Background"}
    assert not any(" | " in name for name in data["name"].to_list())


def test_agg_size_counts_nulls() -> None:
    df = pl.DataFrame({"x": [1, 2, None, 4]})
    assert df.select(agg.agg_size(pl.col("x"))).item() == 4


def test_fixed_does_not_overflow() -> None:
    df = pl.DataFrame({"x": [1e18, -2.5e20, 1e300, 1.25]})
    out = df.select(agg._fixed(pl.col("x"), 1)).to_series().to_list()
    assert out[0] == "1000000000000000000.0"
    assert out[1] == "-250000000000000000000.0"
    assert out[2] is not None and out[2].endswith(".0")
    assert out[3] == "1.2"
