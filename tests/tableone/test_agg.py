"""Tests for statract.tableone.agg module."""

import numpy as np
import polars as pl

from statract.tableone.agg import (
    agg_bool_np,
    agg_bool_pnn,
    agg_category_base,
    agg_mean_sd,
    agg_median_iqr,
    agg_range,
)


class TestAggCategoryBase:
    """Test agg_category_base function."""

    def test_basic_category_aggregation(self):
        """Test basic category aggregation without enum_categories."""
        df = pl.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
        result = df.select(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 3  # A, B, C

        # Check structure
        for item in category_list:
            # Polars returns struct elements as dicts in to_list()
            assert isinstance(item, dict)
            assert "name" in item
            assert "n" in item
            assert "total" in item
            assert "p" in item

        # Check values
        category_dict = {item["name"]: item for item in category_list}
        assert category_dict["A"]["n"] == 3
        assert category_dict["B"]["n"] == 2
        assert category_dict["C"]["n"] == 1
        assert category_dict["A"]["total"] == 6
        assert category_dict["B"]["total"] == 6
        assert category_dict["C"]["total"] == 6
        assert abs(category_dict["A"]["p"] - 0.5) < 1e-10
        assert abs(category_dict["B"]["p"] - 1.0 / 3.0) < 1e-10
        assert abs(category_dict["C"]["p"] - 1.0 / 6.0) < 1e-10

    def test_with_enum_categories(self):
        """Test category aggregation with enum_categories."""
        df = pl.DataFrame({"category": ["B", "B", "A"]})
        enum_categories = ["A", "B", "C", "D"]

        result = df.select(agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 4  # All enum categories

        # Check order matches enum_categories
        assert category_list[0]["name"] == "A"
        assert category_list[1]["name"] == "B"
        assert category_list[2]["name"] == "C"
        assert category_list[3]["name"] == "D"

        # Check values
        category_dict = {item["name"]: item for item in category_list}
        assert category_dict["A"]["n"] == 1
        assert category_dict["B"]["n"] == 2
        assert category_dict["C"]["n"] == 0  # Missing category
        assert category_dict["D"]["n"] == 0  # Missing category

        assert category_dict["A"]["total"] == 3
        assert category_dict["B"]["total"] == 3
        assert category_dict["C"]["total"] == 3
        assert category_dict["D"]["total"] == 3

        assert abs(category_dict["A"]["p"] - 1.0 / 3.0) < 1e-10
        assert abs(category_dict["B"]["p"] - 2.0 / 3.0) < 1e-10
        assert category_dict["C"]["p"] == 0.0
        assert category_dict["D"]["p"] == 0.0

    def test_empty_enum_categories(self):
        """Test with empty enum_categories list."""
        df = pl.DataFrame({"category": ["A", "B", "A"]})

        result = df.select(agg_category_base(pl.col("category"), enum_categories=[]).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 0

    def test_with_null_values(self):
        """Test category aggregation with NULL values."""
        df = pl.DataFrame({"category": ["A", "B", None, "A", None, "B"]})

        result = df.select(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 2  # Only A and B (NULL excluded)

        category_dict = {item["name"]: item for item in category_list}
        assert category_dict["A"]["n"] == 2
        assert category_dict["B"]["n"] == 2
        assert category_dict["A"]["total"] == 4  # NULL values excluded
        assert category_dict["B"]["total"] == 4

    def test_with_enum_categories_and_null(self):
        """Test enum_categories with NULL values."""
        df = pl.DataFrame({"category": ["B", None, "A", None]})
        enum_categories = ["A", "B", "C"]

        result = df.select(agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert len(category_list) == 3

        category_dict = {item["name"]: item for item in category_list}
        assert category_dict["A"]["n"] == 1
        assert category_dict["B"]["n"] == 1
        assert category_dict["C"]["n"] == 0
        assert category_dict["A"]["total"] == 2  # NULL excluded
        assert category_dict["B"]["total"] == 2
        assert category_dict["C"]["total"] == 2

    def test_single_category(self):
        """Test with single category value."""
        df = pl.DataFrame({"category": ["A", "A", "A"]})

        result = df.select(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert len(category_list) == 1
        assert category_list[0]["name"] == "A"
        assert category_list[0]["n"] == 3
        assert category_list[0]["total"] == 3
        assert category_list[0]["p"] == 1.0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"category": []})

        result = df.select(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 0

    def test_empty_dataframe_with_enum(self):
        """Test empty DataFrame with enum_categories."""
        df = pl.DataFrame({"category": []})
        enum_categories = ["A", "B", "C"]

        result = df.select(agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert len(category_list) == 3

        # All should have n=0, total=0
        for item in category_list:
            assert item["n"] == 0
            assert item["total"] == 0
            # p should be 0 or NaN when total is 0
            assert item["p"] == 0.0 or (item["p"] != item["p"])  # NaN check

    def test_with_group_by(self):
        """Test agg_category_base with group_by."""
        df = pl.DataFrame(
            {
                "group": ["X", "X", "Y", "Y"],
                "category": ["A", "B", "A", "B"],
            }
        )

        result = df.group_by("group", maintain_order=True).agg(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 2

        # Check group X
        x_result = result.filter(pl.col("group") == "X")["result"].to_list()[0]
        assert len(x_result) == 2  # A and B
        x_dict = {item["name"]: item for item in x_result}
        assert x_dict["A"]["n"] == 1
        assert x_dict["B"]["n"] == 1
        assert x_dict["A"]["total"] == 2

        # Check group Y
        y_result = result.filter(pl.col("group") == "Y")["result"].to_list()[0]
        assert len(y_result) == 2  # A and B
        y_dict = {item["name"]: item for item in y_result}
        assert y_dict["A"]["n"] == 1
        assert y_dict["B"]["n"] == 1
        assert y_dict["A"]["total"] == 2

    def test_with_group_by_and_enum(self):
        """Test agg_category_base with group_by and enum_categories."""
        df = pl.DataFrame(
            {
                "group": ["X", "X", "Y", "Y"],
                "category": ["B", "B", "A", "A"],
            }
        )
        enum_categories = ["A", "B", "C"]

        result = df.group_by("group", maintain_order=True).agg(
            agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result")
        )

        assert result.height == 2

        # Check group X
        x_result = result.filter(pl.col("group") == "X")["result"].to_list()[0]
        assert len(x_result) == 3  # All enum categories
        x_dict = {item["name"]: item for item in x_result}
        assert x_dict["A"]["n"] == 0
        assert x_dict["B"]["n"] == 2
        assert x_dict["C"]["n"] == 0

        # Check group Y
        y_result = result.filter(pl.col("group") == "Y")["result"].to_list()[0]
        assert len(y_result) == 3  # All enum categories
        y_dict = {item["name"]: item for item in y_result}
        assert y_dict["A"]["n"] == 2
        assert y_dict["B"]["n"] == 0
        assert y_dict["C"]["n"] == 0

    def test_all_null_values(self):
        """Test with all NULL values."""
        df = pl.DataFrame({"category": [None, None, None]})

        result = df.select(agg_category_base(pl.col("category")).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert isinstance(category_list, list)
        assert len(category_list) == 0  # No categories when all NULL

    def test_all_null_with_enum(self):
        """Test all NULL values with enum_categories."""
        df = pl.DataFrame({"category": [None, None, None]})
        enum_categories = ["A", "B", "C"]

        result = df.select(agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]
        assert len(category_list) == 3

        # All should have n=0, total=0
        for item in category_list:
            assert item["n"] == 0
            assert item["total"] == 0

    def test_enum_order_preserved(self):
        """Test that enum_categories order is preserved."""
        df = pl.DataFrame({"category": ["Z", "Y", "X"]})
        enum_categories = ["X", "Y", "Z", "W"]

        result = df.select(agg_category_base(pl.col("category"), enum_categories=enum_categories).alias("result"))

        assert result.height == 1
        category_list = result["result"].to_list()[0]

        # Check order matches enum_categories
        assert category_list[0]["name"] == "X"
        assert category_list[1]["name"] == "Y"
        assert category_list[2]["name"] == "Z"
        assert category_list[3]["name"] == "W"


class TestNumberFormat:
    """Fixed decimal places and R type 7 quartiles in the string aggregators."""

    def test_mean_sd_keeps_trailing_zeros(self):
        df = pl.DataFrame({"x": [51.41, 73.41]})  # mean 62.41, SD 15.556...
        out = df.select(agg_mean_sd(pl.col("x"), precision=2)).item()
        assert out == "62.41 (15.56)"
        df = pl.DataFrame({"x": [1.0, 3.0]})  # SD = sqrt(2)
        assert df.select(agg_mean_sd(pl.col("x"), precision=2)).item() == "2.00 (1.41)"
        assert df.select(agg_mean_sd(pl.col("x"), precision=0)).item() == "2 (1)"

    def test_median_iqr_uses_type7_and_comma_space(self):
        # R: quantile(1:4) gives 1.75 and 3.25 (type 7, linear interpolation).
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        assert (
            df.select(agg_median_iqr(pl.col("x"), precision=2)).item()
            == "2.50 [1.75, 3.25]"
        )

    def test_median_iqr_matches_numpy_linear(self):
        rng = np.random.default_rng(0)
        x = rng.exponential(2.0, size=37)
        q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        out = (
            pl.DataFrame({"x": x})
            .select(agg_median_iqr(pl.col("x"), precision=3))
            .item()
        )
        assert out == f"{med:.3f} [{q1:.3f}, {q3:.3f}]"

    def test_group_by_and_nulls(self):
        df = pl.DataFrame({"g": ["a", "a", "b"], "x": [1.0, 2.0, None]})
        out = (
            df.group_by("g", maintain_order=True)
            .agg(agg_median_iqr(pl.col("x")))
            .to_dicts()
        )
        assert out == [{"g": "a", "x": "1.5 [1.2, 1.8]"}, {"g": "b", "x": None}]

    def test_negative_and_ties(self):
        df = pl.DataFrame({"x": [-0.25, -2.5]})
        assert df.select(agg_range(pl.col("x"))).item() == "-2.5--0.2"
        # Same as sprintf("%.2f", 0.125) in R and Python: "0.12".
        assert (
            pl.DataFrame({"x": [0.125]}).select(agg_range(pl.col("x"), 2)).item()
            == "0.12-0.12"
        )

    def test_percent_keeps_trailing_zeros(self):
        df = pl.DataFrame(
            {"b": [True, False, False, False, False, False, False, False]}
        )
        assert df.select(agg_bool_np(pl.col("b"), precision=2)).item() == "1 (12.50%)"
        assert (
            df.select(agg_bool_pnn(pl.col("b"), precision=2)).item() == "12.50% (1/8)"
        )
