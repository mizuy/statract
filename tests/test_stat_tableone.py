"""Comprehensive tests for the statract.tableone.table module."""

import polars as pl
import pytest

from statract.tableone.table import (
    ColumnNotFoundError,
    InvalidParameterError,
    TableOneError,
    TableOneParam,
    tableone,
    tableone_raw,
)
from statract.tableone.agg import (
    agg_category_n,
    agg_category_np,
    agg_count,
    agg_mean_sd,
    agg_median_range,
    agg_ratio_np,
)
from great_tables import GT

from statract.tableone.stat import stat_anova, stat_fisher, stat_kruskal, standardized_difference


class TestTableOneParam:
    """Test TableOneParam type."""

    def test_tableone_param_creation(self):
        """Test creating TableOneParam."""
        param = TableOneParam(column=pl.col("age"), aggfunc=None, statfunc=None)
        assert param.column.meta.output_name() == "age"
        assert param.aggfunc is None
        assert param.statfunc is None


class TestTableOneErrors:
    """Test error handling."""

    def test_empty_dataframe(self):
        """Test error when dataframe is empty."""
        df = pl.DataFrame({"age": []})
        params = {"Age": ("age", agg_mean_sd)}
        with pytest.raises(InvalidParameterError, match="non-empty"):
            tableone_raw(df, params)

    def test_empty_params(self):
        """Test error when params is empty."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        with pytest.raises(InvalidParameterError, match="non-empty"):
            tableone_raw(df, {})

    def test_column_not_found(self):
        """Test error when column is not found."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        params = {"Age": ("missing_column", agg_mean_sd)}
        with pytest.raises(ColumnNotFoundError, match="not found"):
            tableone_raw(df, params)

    def test_invalid_param_type(self):
        """Test error when param type is invalid."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        params = {"Age": "invalid"}  # type: ignore
        with pytest.raises(InvalidParameterError, match="Invalid param type"):
            tableone_raw(df, params)

    def test_invalid_hue_type(self):
        """Test error when hue type is invalid."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        params = {"Age": ("age", agg_mean_sd)}
        with pytest.raises(InvalidParameterError, match="Invalid hue type"):
            tableone_raw(df, params, hue=123)  # type: ignore


class TestTableOneBasic:
    """Test basic tableone functionality."""

    def test_tableone_no_hue(self):
        """Test tableone without hue (overall statistics)."""
        df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert "name" in result.columns
        assert "All" in result.columns  # Should have "All" column when no hue
        assert result.height == 1
        assert result["name"][0] == "Age"

    def test_tableone_with_hue(self):
        """Test tableone with hue (grouped statistics)."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "age": [25, 30, 35, 40],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue="group")
        assert isinstance(result, pl.DataFrame)
        assert "name" in result.columns
        assert "A" in result.columns
        assert "B" in result.columns
        assert result.height == 1

    def test_tableone_with_tuple_params(self):
        """Test tableone with tuple params."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_tableone_with_tableoneparam(self):
        """Test tableone with TableOneParam objects."""
        df = pl.DataFrame({"age": [25, 30, 35]})
        params = {"Age": TableOneParam(column=pl.col("age"), aggfunc=agg_mean_sd, statfunc=None)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1


class TestTableOneCategory:
    """Test tableone with category variables."""

    def test_tableone_category_no_hue(self):
        """Test tableone with category variable without hue."""
        df = pl.DataFrame({"category": ["A", "B", "C", "A", "B"]})
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns  # Should have "All" column when no hue
        assert result.height >= 2  # Variable name + categories
        # Check that first row is variable name
        assert result["name"][0] == "Category"
        # Check that subsequent rows are indented categories
        assert any(name.startswith("  ") for name in result["name"][1:])

    def test_tableone_category_with_hue(self):
        """Test tableone with category variable with hue."""
        df = pl.DataFrame(
            {
                "group": ["X", "X", "Y", "Y"],
                "category": ["A", "B", "A", "B"],
            }
        )
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params, hue="group")
        assert isinstance(result, pl.DataFrame)
        assert "X" in result.columns
        assert "Y" in result.columns
        assert result.height >= 2  # Variable name + categories

    def test_tableone_category_example(self):
        """Test tableone with the example from docstring."""
        df = pl.DataFrame(
            {
                "hue": ["a", "a", "a", "b", "b", "b"],
                "diameter": [10, 20, 15, 5, 10, 10],
                "depth": ["Tis", "T1a", "T1b", "T2", "T1b", "T3"],
            }
        )
        params = {
            "size": ("diameter", agg_median_range),
            "depth": ("depth", agg_category_n),
        }
        result = tableone_raw(df, params, hue="hue")
        assert isinstance(result, pl.DataFrame)
        assert "name" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns
        # Check that depth has multiple rows (variable name + categories)
        depth_rows = [i for i, name in enumerate(result["name"]) if name == "depth" or name.startswith("  ")]
        assert len(depth_rows) >= 2


class TestTableOneAggregationFunctions:
    """Test different aggregation functions."""

    def test_tableone_mean_sd(self):
        """Test mean_sd aggregation."""
        df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        params = {"Age": ("age", "mean_sd")}
        result = tableone_raw(df, params)
        assert result.height == 1

    def test_tableone_median_range(self):
        """Test median_range aggregation."""
        df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        params = {"Age": ("age", "median_range")}
        result = tableone_raw(df, params)
        assert result.height == 1

    def test_tableone_category_np(self):
        """Test category_np aggregation."""
        df = pl.DataFrame({"category": ["A", "B", "C"]})
        params = {"Category": ("category", "category_np")}
        result = tableone_raw(df, params)
        assert result.height >= 2

    def test_tableone_ratio_np(self):
        """Test ratio_np aggregation."""
        df = pl.DataFrame({"flag": [True, False, True, True]})
        params = {"Flag": ("flag", "ratio_np")}
        result = tableone_raw(df, params)
        assert result.height == 1


class TestTableOneMultipleHue:
    """Test tableone with multiple hue columns."""

    def test_tableone_multiple_hue(self):
        """Test tableone with multiple hue columns."""
        df = pl.DataFrame(
            {
                "group1": ["A", "A", "B", "B"],
                "group2": ["X", "Y", "X", "Y"],
                "age": [25, 30, 35, 40],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue=["group1", "group2"])
        assert isinstance(result, pl.DataFrame)
        # Should have columns for each combination
        assert result.height == 1

    def test_tableone_hue_as_expr(self):
        """Test tableone with hue as pl.Expr."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "age": [25, 30, 35, 40],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue=pl.col("group"))
        assert isinstance(result, pl.DataFrame)
        assert "A" in result.columns
        assert "B" in result.columns


class TestTableOneEdgeCases:
    """Test edge cases."""

    def test_tableone_single_row(self):
        """Test tableone with single row dataframe."""
        df = pl.DataFrame({"age": [25]})
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_tableone_single_category(self):
        """Test tableone with single category value."""
        df = pl.DataFrame({"category": ["A", "A", "A"]})
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert result.height >= 2  # Variable name + category

    def test_tableone_missing_values(self):
        """Test tableone with missing values."""
        df = pl.DataFrame({"age": [25, None, 35, None, 45]})
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_tableone_multiple_params(self):
        """Test tableone with multiple parameters."""
        df = pl.DataFrame(
            {
                "age": [25, 30, 35],
                "height": [170, 175, 180],
                "category": ["A", "B", "C"],
            }
        )
        params = {
            "Age": ("age", agg_mean_sd),
            "Height": ("height", agg_median_range),
            "Category": ("category", agg_category_n),
        }
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns  # Should have "All" column when no hue
        assert result.height >= 3  # At least one row per parameter


class TestTableOneAddAll:
    """Test add_all parameter functionality."""

    def test_tableone_add_all_true(self):
        """Test tableone with add_all=True adds 'All' column as leftmost."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "age": [25, 30, 35, 40],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue="group", add_all=True)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns
        assert "A" in result.columns
        assert "B" in result.columns
        # Check that "All" is the first data column (after "name")
        columns = result.columns
        assert columns[0] == "name"
        assert columns[1] == "All"  # "All" should be leftmost data column
        assert result.height == 1

    def test_tableone_add_all_false(self):
        """Test tableone with add_all=False does not add 'All' column."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "age": [25, 30, 35, 40],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue="group", add_all=False)
        assert isinstance(result, pl.DataFrame)
        assert "All" not in result.columns  # Should not have "All" when add_all=False
        assert "A" in result.columns
        assert "B" in result.columns

    def test_tableone_add_all_with_category(self):
        """Test add_all with category variables."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "category": ["X", "Y", "X", "Y"],
            }
        )
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params, hue="group", add_all=True)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns
        assert "A" in result.columns
        assert "B" in result.columns
        # Check column order: name, All, then group columns
        columns = result.columns
        assert columns[0] == "name"
        assert columns[1] == "All"
        assert result.height >= 2  # Variable name + categories

    def test_tableone_add_all_no_hue(self):
        """Test that add_all has no effect when hue is None."""
        df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        params = {"Age": ("age", agg_mean_sd)}
        result = tableone_raw(df, params, hue=None, add_all=True)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns  # Should have "All" column when no hue
        # Should be same as add_all=False when hue is None
        result_no_add_all = tableone_raw(df, params, hue=None, add_all=False)
        assert result.equals(result_no_add_all)


class TestTableOneEnum:
    """Test enum type handling in tableone."""

    def test_tableone_enum_ordering_no_hue(self):
        """Test that enum categories are displayed in enum order without hue."""
        df = pl.DataFrame(
            {
                "category": pl.Series(["B", "B", "A"], dtype=pl.Enum(["A", "B", "C", "D"])),
            }
        )
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        assert "All" in result.columns
        # Check that categories are in enum order: A, B, C, D
        category_rows = [row for row in result["name"] if row.startswith("  ")]
        assert category_rows == ["  A", "  B", "  C", "  D"]
        # Check that missing categories show 0
        assert result.filter(pl.col("name") == "  C")["All"][0] == "0"
        assert result.filter(pl.col("name") == "  D")["All"][0] == "0"

    def test_tableone_enum_ordering_with_hue(self):
        """Test that enum categories are displayed in enum order with hue."""
        df = pl.DataFrame(
            {
                "group": ["X", "X", "Y"],
                "category": pl.Series(["B", "B", "A"], dtype=pl.Enum(["A", "B", "C", "D"])),
            }
        )
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params, hue="group")
        assert isinstance(result, pl.DataFrame)
        assert "X" in result.columns
        assert "Y" in result.columns
        # Check that categories are in enum order: A, B, C, D
        category_rows = [row for row in result["name"] if row.startswith("  ")]
        assert category_rows == ["  A", "  B", "  C", "  D"]
        # Check that missing categories show 0 for all groups
        c_row = result.filter(pl.col("name") == "  C")
        assert c_row["X"][0] == "0"
        assert c_row["Y"][0] == "0"
        d_row = result.filter(pl.col("name") == "  D")
        assert d_row["X"][0] == "0"
        assert d_row["Y"][0] == "0"

    def test_tableone_enum_all_categories_displayed(self):
        """Test that all enum categories are displayed even when count is 0."""
        df = pl.DataFrame(
            {
                "category": pl.Series(["A"], dtype=pl.Enum(["A", "B", "C", "D", "E"])),
            }
        )
        params = {"Category": ("category", agg_category_n)}
        result = tableone_raw(df, params)
        assert isinstance(result, pl.DataFrame)
        # Check that all enum categories are present
        category_rows = [row for row in result["name"] if row.startswith("  ")]
        assert len(category_rows) == 5  # A, B, C, D, E
        assert "  A" in category_rows
        assert "  B" in category_rows
        assert "  C" in category_rows
        assert "  D" in category_rows
        assert "  E" in category_rows
        # Check that only A has count > 0
        assert result.filter(pl.col("name") == "  A")["All"][0] == "1"
        assert result.filter(pl.col("name") == "  B")["All"][0] == "0"
        assert result.filter(pl.col("name") == "  C")["All"][0] == "0"
        assert result.filter(pl.col("name") == "  D")["All"][0] == "0"
        assert result.filter(pl.col("name") == "  E")["All"][0] == "0"


class TestTableOneGreatTables:
    """Test great_tables styling wrapper."""

    def test_tableone_gt_returns_gt_and_applies_labels_and_indent(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B"],
                "category": pl.Series(["X", "Y", "X"], dtype=pl.Enum(["X", "Y", "Z"])),
            }
        )
        params = {"Category": ("category", agg_category_n)}

        gt = tableone(
            df,
            params,
            hue="group",
            add_all=True,
            group_labels={"All": "All", "A": "Group A", "B": "Group B"},
        )
        assert isinstance(gt, GT)

        html = gt.as_raw_html()
        # group labels applied
        assert "Group A" in html
        assert "Group B" in html
        # indent converted to full-width space for category rows
        assert "　X" in html

    def test_write_tableone_artifacts(self, tmp_path):
        from statract.tableone.table import write_tableone_artifacts

        df = pl.DataFrame(
            {
                "group": ["A", "A", "B"],
                "age": [10.0, 12.0, 20.0],
            }
        )
        params = {"Age": ("age", agg_mean_sd)}
        written = write_tableone_artifacts(
            tmp_path,
            "table_baseline_demo",
            df=df,
            params=params,
            hue="group",
            add_all=True,
        )
        stems = {p.name for p in written}
        assert "table_baseline_demo.csv" in stems
        assert "table_baseline_demo_csv.md" in stems
        assert "table_baseline_demo.html" in stems
        assert "table_baseline_demo_gt.md" in stems
        assert (tmp_path / "table_baseline_demo_gt.md").read_text(encoding="utf-8")

    def test_gt_fragment_is_scoped_and_stable(self, tmp_path):
        """The _gt.md fragment uses a fixed id, scopes every CSS rule, and has no blank lines."""
        import re

        from statract.tableone.table import write_tableone_artifacts

        df = pl.DataFrame({"group": ["A", "A", "B"], "age": [10.0, 12.0, 20.0]})
        params = {"Age": ("age", agg_mean_sd)}
        texts = []
        for sub in ("a", "b"):
            write_tableone_artifacts(tmp_path / sub, "t1", df=df, params=params, hue="group")
            texts.append((tmp_path / sub / "t1_gt.md").read_text(encoding="utf-8"))
        assert texts[0] == texts[1]
        text = texts[0]
        assert '<div id="tableone-t1"' in text
        assert "\n\n" not in text
        style = re.search(r"<style>(.*)</style>", text, flags=re.S).group(1)
        for selector in re.findall(r"([^{}]+)\{", style):
            for part in selector.split(","):
                assert part.strip().startswith("#tableone-t1"), part

    def test_column_order_keeps_stub_leftmost(self):
        """column_order 適用後も stub(name) が左端、指定列順が保たれること。"""
        import re

        from statract.tableone.table import tableone_gt_from_frame

        table = pl.DataFrame(
            {
                "name": ["Age"],
                "All Post": ["1"],
                "ESD Pre": ["2"],
                "All Pre": ["3"],
                "ESD Post": ["4"],
                "Ope Pre": ["5"],
                "Ope Post": ["6"],
            }
        )
        order = ["All Pre", "All Post", "ESD Pre", "ESD Post", "Ope Pre", "Ope Post"]
        html = tableone_gt_from_frame(table, column_order=order).as_raw_html()
        ids = re.findall(r'scope="col" id="([^"]+)"', html)
        assert ids == [
            "name",
            "All-Pre",
            "All-Post",
            "ESD-Pre",
            "ESD-Post",
            "Ope-Pre",
            "Ope-Post",
        ]


class TestTableOnePValue:
    """Test p-value column option."""

    def test_tableone_add_pvalue_false_no_column(self):
        df = pl.DataFrame({"group": ["A", "B"], "age": [10, 20]})
        params = {"Age": ("age", agg_mean_sd, None)}
        result = tableone_raw(df, params, hue="group", add_pvalue=False)
        assert "P value" not in result.columns

    def test_tableone_add_pvalue_true_regular_and_category(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "age": [10, 11, 20, 21],
                "sex": ["M", "F", "M", "M"],
            }
        )
        params = {
            "Age": ("age", agg_mean_sd, None),
            "Sex": ("sex", agg_category_n, None),
        }
        result = tableone_raw(df, params, hue="group", add_pvalue=True)
        assert "P value" in result.columns

        # Regular row has a p-value (not empty/"-"/Failed)
        age_p = result.filter(pl.col("name") == "Age")["P value"][0]
        assert age_p not in ("", "-", "Failed")

        # Category header has a p-value; sub-rows are blank
        sex_p = result.filter(pl.col("name") == "Sex")["P value"][0]
        assert sex_p not in ("", "-", "Failed")
        sub = result.filter(pl.col("name").str.starts_with("  "))
        assert all(v == "" for v in sub["P value"].to_list())

    def test_tableone_add_pvalue_statfunc_none_is_auto(self):
        # statfunc=None => auto (not '-') by design
        df = pl.DataFrame({"group": ["A", "A", "B", "B"], "age": [10, 11, 20, 21]})
        params = {"Age": ("age", agg_mean_sd, None)}
        result = tableone_raw(df, params, hue="group", add_pvalue=True)
        assert result.filter(pl.col("name") == "Age")["P value"][0] != "-"

    def test_tableone_add_pvalue_multi_hue_crosses_the_groups(self):
        df = pl.DataFrame(
            {
                "g1": ["A", "A", "A", "B", "B", "B"] * 2,
                "g2": ["X", "Y"] * 6,
                "age": [10.0, 30.0, 11.0, 31.0, 12.0, 33.0, 13.0, 29.0, 10.5, 30.5, 12.5, 32.0],
            }
        )
        params = {"Age": ("age", agg_mean_sd, None)}
        result = tableone_raw(df, params, hue=["g1", "g2"], add_pvalue=True)
        assert [c for c in result.columns if "/" in c] == ["A / X", "A / Y", "B / X", "B / Y"]
        groups = df.select(pl.concat_str([pl.col("g1"), pl.col("g2")], separator=" / ")).to_series()
        assert result.filter(pl.col("name") == "Age")["P value"][0] == stat_anova(df["age"], groups)

    def test_tableone_median_summary_gets_a_rank_test(self):
        df = pl.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "los": [1, 2, 2, 3, 40, 5, 6, 7, 8, 9]})
        median = tableone_raw(df, {"LOS": ("los", agg_median_range, None)}, hue="group", add_pvalue=True)
        mean = tableone_raw(df, {"LOS": ("los", agg_mean_sd, None)}, hue="group", add_pvalue=True)
        assert median["P value"][0] == stat_kruskal(df["los"], df["group"])
        assert mean["P value"][0] == stat_anova(df["los"], df["group"])
        named = tableone_raw(df, {"LOS": ("los", agg_mean_sd, "kruskal")}, hue="group", add_pvalue=True)
        assert named["P value"][0] == median["P value"][0]

    def test_fisher_on_a_larger_table_is_reproducible(self):
        df = pl.DataFrame(
            {"g": ["A", "B", "C"] * 10, "c": ["u", "v", "w", "w", "v"] * 6},
        )
        first = stat_fisher(df["c"], df["g"])
        assert all(stat_fisher(df["c"], df["g"]) == first for _ in range(3))


class TestTableOneSMD:
    """Standardized mean differences, as R tableone reports them."""

    def test_numeric_and_categorical_smd(self):
        df = pl.DataFrame(
            {
                "group": ["A"] * 4 + ["B"] * 4,
                "age": [10.0, 12.0, 14.0, 16.0, 13.0, 15.0, 17.0, 19.0],
                "sex": ["M", "M", "F", "F", "M", "F", "F", "F"],
            }
        )
        result = tableone_raw(
            df,
            {"Age": ("age", agg_mean_sd, None), "Sex": ("sex", agg_category_n, None)},
            hue="group",
            add_pvalue=True,
            add_smd=True,
        )
        assert result.columns[-2:] == ["P value", "SMD"]
        # |13 - 16| / sqrt((sd^2 + sd^2) / 2) with sd^2 = 20 / 3 in both groups
        assert result.filter(pl.col("name") == "Age")["SMD"][0] == f"{3 / (20 / 3) ** 0.5:.3f}"
        # Binary: |0.5 - 0.25| / sqrt((0.25 + 0.1875) / 2)
        assert result.filter(pl.col("name") == "Sex")["SMD"][0] == f"{0.25 / ((0.25 + 0.1875) / 2) ** 0.5:.3f}"
        sub = result.filter(pl.col("name").str.starts_with("  "))
        assert all(v == "" for v in sub["SMD"].to_list())

    def test_smd_averages_pairs_with_three_groups(self):
        values = pl.Series([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        groups = pl.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
        pairs = [
            standardized_difference(values.filter(groups.is_in(pair)), groups.filter(groups.is_in(pair)))
            for pair in (["a", "b"], ["a", "c"], ["b", "c"])
        ]
        assert standardized_difference(values, groups) == pytest.approx(sum(pairs) / 3)
        assert pairs[0] == pytest.approx(1.0)

    def test_multinomial_smd_does_not_depend_on_the_dropped_level(self):
        x = pl.Series(["u", "v", "w", "w", "v", "u", "u", "w", "v", "v"])
        g = pl.Series(["A"] * 5 + ["B"] * 5)
        y = x.replace({"u": "z"})
        assert standardized_difference(x, g) == pytest.approx(standardized_difference(y, g))
