"""Tests for statract.tableone.agg module."""

import polars as pl

from statract.tableone.agg import agg_category_base


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
