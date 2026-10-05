"""Aggregation functions for Polars DataFrames.

This module provides aggregation functions for descriptive statistics,
including mean, median, category counts, and various ratio calculations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import polars as pl

# Type aliases
AggFunc: TypeAlias = Callable[[pl.Expr], pl.Expr]


def agg_size(expr: pl.Expr) -> pl.Expr:
    """Count total number of rows (including nulls).

    Args:
        expr: Polars expression for the column to count

    Returns:
        Polars expression that evaluates to the count

    Examples:
        >>> import polars as pl
        >>> from statract import agg_size
        >>>
        >>> df = pl.DataFrame({"x": [1, 2, None, 4]})
        >>> df.select(agg_size(pl.col("x")).alias("count"))
        shape: (1, 1)
        ┌───────┐
        │ count │
        │ ---   │
        │ u32   │
        ╞═══════╡
        │ 3     │
        └───────┘
    """
    return expr.count()


def agg_count(expr: pl.Expr) -> pl.Expr:
    """Count non-null values in a column.

    Args:
        expr: Polars expression for the column to count

    Returns:
        Polars expression that evaluates to the count of non-null values

    Examples:
        >>> import polars as pl
        >>> from statract import agg_count
        >>>
        >>> df = pl.DataFrame({"x": [1, 2, None, 4]})
        >>> df.select(agg_count(pl.col("x")).alias("count"))
        shape: (1, 1)
        ┌───────┐
        │ count │
        │ ---   │
        │ u32   │
        ╞═══════╡
        │ 3     │
        └───────┘
    """
    return expr.count()


def agg_nunique(expr: pl.Expr) -> pl.Expr:
    """Count number of unique values in a column.

    Args:
        expr: Polars expression for the column to count unique values

    Returns:
        Polars expression that evaluates to the count of unique values

    Examples:
        >>> import polars as pl
        >>> from statract import agg_nunique
        >>>
        >>> df = pl.DataFrame({"x": ["A", "B", "A", "C", "B"]})
        >>> df.select(agg_nunique(pl.col("x")).alias("n_unique"))
        shape: (1, 1)
        ┌──────────┐
        │ n_unique │
        │ ---      │
        │ u32      │
        ╞══════════╡
        │ 3        │
        └──────────┘
    """
    return expr.n_unique()


def _format_n(n: pl.Expr, precision: int = 1) -> pl.Expr:
    """Format number with precision."""
    return n.round(precision)


def _format_p(p: pl.Expr, precision: int = 1) -> pl.Expr:
    """Format percentage with precision."""
    return (p * 100).round(precision)


def _format_np(n: pl.Expr, p: pl.Expr, precision: int = 1) -> pl.Expr:
    """Format number (percentage)."""
    return pl.format("{} ({}%)", n.round(precision), (p * 100).round(precision))


def _format_pnn(n: pl.Expr, total: pl.Expr, p: pl.Expr, precision: int = 1) -> pl.Expr:
    """Format percentage (count/total)."""
    return pl.format("{}% ({}/{})", _format_p(p, precision), n, total)


def agg_bool_np(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean as count (percentage)."""
    n = (expr > 0).sum()
    total = expr.is_not_null().sum()
    p = n / total
    return _format_np(n, p, precision)


def agg_bool_pnn(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean as percentage (count/total)."""
    n = (expr > 0).sum()
    total = expr.is_not_null().sum()
    p = n / total
    return _format_pnn(n, total, p, precision)


def agg_bool_p(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean as percentage."""
    n = (expr > 0).sum()
    total = expr.is_not_null().sum()
    p = n / total

    # Format percentage with precision
    return _format_p(p, precision)


def agg_bool_n(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean as count."""
    return (expr > 0).sum()


def agg_bool_ci(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean with confidence interval (placeholder).

    CI calculation requires additional statistical tooling; for now we return the
    plain percentage formatter.
    """
    return agg_bool_p(expr, precision)


def agg_bool_pnnci(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean as percentage (count/total) with CI (placeholder).

    CI calculation requires additional statistical tooling; for now we return the
    plain pnn formatter.
    """
    return agg_bool_pnn(expr, precision)


def agg_mean(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as mean value.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to the rounded mean

    Examples:
        >>> import polars as pl
        >>> from statract import agg_mean
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_mean(pl.col("age")).alias("mean_age"))
        shape: (1, 1)
        ┌──────────┐
        │ mean_age │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 35.0     │
        └──────────┘
    """
    return expr.mean().round(precision)


def agg_mean_sd(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as "mean (SD)" format.

    Commonly used in medical research Table One for continuous variables.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to a string in "mean (SD)" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_mean_sd
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_mean_sd(pl.col("age")).alias("age_summary"))
        shape: (1, 1)
        ┌─────────────┐
        │ age_summary │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ 35.0 (7.9)  │
        └─────────────┘
    """
    mean = expr.mean().round(precision)
    sd = expr.std().round(precision)
    return pl.format("{} ({})", mean, sd)


def agg_median(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as median value.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to the rounded median

    Examples:
        >>> import polars as pl
        >>> from statract import agg_median
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_median(pl.col("age")).alias("median_age"))
        shape: (1, 1)
        ┌────────────┐
        │ median_age │
        │ ---        │
        │ f64        │
        ╞════════════╡
        │ 35.0       │
        └────────────┘
    """
    return expr.median().round(precision)


def agg_median_range(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as "median (min-max)" format.

    Commonly used in medical research Table One for skewed continuous variables.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to a string in "median (min-max)" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_median_range
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_median_range(pl.col("age")).alias("age_summary"))
        shape: (1, 1)
        ┌─────────────────┐
        │ age_summary     │
        │ ---             │
        │ str             │
        ╞═════════════════╡
        │ 35.0 (25.0-45.0)│
        └─────────────────┘
    """
    median = expr.median().round(precision)
    min_val = expr.min().round(precision)
    max_val = expr.max().round(precision)
    return pl.format("{} ({}-{})", median, min_val, max_val)


# Convenience functions for common format types
def agg_ratio(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column as percentage.

    Calculates the percentage of True/positive values in a boolean column.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to a percentage string

    Examples:
        >>> import polars as pl
        >>> from statract import agg_ratio
        >>>
        >>> df = pl.DataFrame({"success": [True, True, False, True, False]})
        >>> df.select(agg_ratio(pl.col("success")).alias("success_rate"))
        shape: (1, 1)
        ┌──────────────┐
        │ success_rate │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ 60.0         │
        └──────────────┘
    """
    return agg_bool_p(expr, 1)


def agg_ratio_n(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column as count of True values.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to the count of True values

    Examples:
        >>> import polars as pl
        >>> from statract import agg_ratio_n
        >>>
        >>> df = pl.DataFrame({"success": [True, True, False, True, False]})
        >>> df.select(agg_ratio_n(pl.col("success")).alias("n_success"))
        shape: (1, 1)
        ┌───────────┐
        │ n_success │
        │ ---       │
        │ u32       │
        ╞═══════════╡
        │ 3         │
        └───────────┘
    """
    return agg_bool_n(expr)


def agg_ratio_np(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column as "count (percentage%)" format.

    Commonly used in medical research Table One for binary variables.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to a string in "n (p%)" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_ratio_np
        >>>
        >>> df = pl.DataFrame({"success": [True, True, False, True, False]})
        >>> df.select(agg_ratio_np(pl.col("success")).alias("success_summary"))
        shape: (1, 1)
        ┌─────────────────┐
        │ success_summary │
        │ ---             │
        │ str             │
        ╞═════════════════╡
        │ 3 (60.0%)       │
        └─────────────────┘
    """
    return agg_bool_np(expr, 1)


def agg_ratio_pnn(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column as "percentage% (count/total)" format.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to a string in "p% (n/total)" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_ratio_pnn
        >>>
        >>> df = pl.DataFrame({"success": [True, True, False, True, False]})
        >>> df.select(agg_ratio_pnn(pl.col("success")).alias("success_summary"))
        shape: (1, 1)
        ┌─────────────────┐
        │ success_summary │
        │ ---             │
        │ str             │
        ╞═════════════════╡
        │ 60.0% (3/5)     │
        └─────────────────┘
    """
    return agg_bool_pnn(expr, 1)


def agg_ratio_ci(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column with confidence interval (placeholder).

    Note: CI calculation requires additional statistical tooling; currently
    returns plain percentage format.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to a percentage string
    """
    return agg_bool_ci(expr, 1)


def agg_ratio_pnnci(expr: pl.Expr) -> pl.Expr:
    """Aggregate boolean/binary column as percentage with count and CI (placeholder).

    Note: CI calculation requires additional statistical tooling; currently
    returns plain pnn format.

    Args:
        expr: Polars expression for a boolean column

    Returns:
        Polars expression that evaluates to a string in "p% (n/total)" format
    """
    return agg_bool_pnnci(expr, 1)


def agg_range(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as "min-max" range format.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to a string in "min-max" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_range
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_range(pl.col("age")).alias("age_range"))
        shape: (1, 1)
        ┌───────────┐
        │ age_range │
        │ ---       │
        │ str       │
        ╞═══════════╡
        │ 25.0-45.0 │
        └───────────┘
    """
    min_val = expr.min().round(precision)
    max_val = expr.max().round(precision)
    return pl.format("{}-{}", min_val, max_val)


def agg_median_iqr(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate numeric column as "median [Q1, Q3]" format with interquartile range.

    Commonly used in medical research Table One for skewed continuous variables.

    Args:
        expr: Polars expression for the numeric column
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Polars expression that evaluates to a string in "median [Q1, Q3]" format

    Examples:
        >>> import polars as pl
        >>> from statract import agg_median_iqr
        >>>
        >>> df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        >>> df.select(agg_median_iqr(pl.col("age")).alias("age_summary"))
        shape: (1, 1)
        ┌───────────────────┐
        │ age_summary       │
        │ ---               │
        │ str               │
        ╞═══════════════════╡
        │ 35.0 [30.0,40.0]  │
        └───────────────────┘
    """
    median = expr.median().round(precision)
    p25 = expr.quantile(0.25).round(precision)
    p75 = expr.quantile(0.75).round(precision)
    return pl.format("{} [{},{}]", median, p25, p75)


def agg_category_base(
    expr: pl.Expr, enum_categories: list[str] | None = None
) -> pl.Expr:
    """Aggregate category as base structure with category name, n, n_total, and p.

    If enum_categories is provided, all enum categories are included
    with count 0 for missing ones, sorted by enum order.

    Args:
        expr: Column expression to aggregate
        enum_categories: If provided, ensures all enum categories are included
                        with count 0 for missing ones, sorted by enum order.

    Returns:
        pl.Expr of type list[struct[4]] where each struct contains:
        - name (String): category name
        - n (Int64): count for this category
        - total (Int64): total count (non-null)
        - p (Float64): percentage (n / total)

        Note: This expression needs to be evaluated in a DataFrame context
        using group_by().agg() to create the list[struct[4]].
    """
    # Exclude null categories from the output (common for TableOne-style summaries).
    # We still keep `n_total` as the non-null denominator for percentages.
    expr_non_null = expr.drop_nulls()
    n_total = expr.is_not_null().sum()
    vc = expr_non_null.value_counts()

    # Get the column name from the original expression
    col_name = expr.meta.output_name() or "value"

    # If enum_categories is provided, ensure all categories are present
    if enum_categories is not None:
        # Create list of all enum categories as structs
        # Create struct without alias to avoid duplicate field name errors
        enum_structs = []
        for cat in enum_categories:
            cat_count = (expr == pl.lit(cat)).sum()
            cat_percentage = cat_count / n_total
            # Create struct with category name, n, total, p
            enum_structs.append(
                pl.struct(
                    [
                        pl.lit(cat).alias("name"),  # category name
                        cat_count.cast(pl.Int64).alias("n"),  # n
                        n_total.cast(pl.Int64).alias("total"),  # total
                        cat_percentage.cast(pl.Float64).alias("p"),  # p
                    ],
                ),
            )
        # Return list of all enum structs in order using pl.concat_list()
        # Without alias, pl.concat_list works correctly
        if len(enum_structs) == 0:
            return pl.lit([]).cast(
                pl.List(
                    pl.Struct(
                        [
                            pl.Field("name", pl.String),
                            pl.Field("n", pl.Int64),
                            pl.Field("total", pl.Int64),
                            pl.Field("p", pl.Float64),
                        ],
                    ),
                ),
            )
        return pl.concat_list(enum_structs)

    # Default: create list[struct[4]] from value_counts (nulls excluded)
    # - In a plain `select`, `value_counts()` yields a struct per unique value across rows.
    #   We call `implode()` so the result becomes a single `list[struct[2]]` row.
    # - In a `group_by(...).agg`, `value_counts()` already yields `list[struct[2]]`;
    #   `implode()` is a no-op in that context (keeps it as `list[struct[2]]`).
    #
    # Then we transform each element into struct[4] with {name, n, total, p}.
    # We compute `total` inside `list.eval` to avoid referencing named columns.
    vc_list = vc.implode()

    return vc_list.list.eval(
        pl.struct(
            [
                pl.element().struct.field(col_name).alias("name"),  # category name
                pl.element().struct.field("count").cast(pl.Int64).alias("n"),  # n
                pl.element()
                .struct.field("count")
                .cast(pl.Int64)
                .sum()
                .alias("total"),  # total
                (
                    pl.element().struct.field("count").cast(pl.Float64)
                    / pl.element().struct.field("count").cast(pl.Float64).sum()
                ).alias("p"),  # p
            ],
        ),
    )


def agg_category_np(
    expr: pl.Expr, precision: int = 1, enum_categories: list[str] | None = None
) -> pl.Expr:
    base = agg_category_base(expr, enum_categories=enum_categories)
    # Transform base struct[4] to struct[2] with formatted values
    # base is list[struct[4]]: [category_name, n, n_total, p]
    # Result should be list[struct[2]]: [category_name, formatted_np]
    return base.list.eval(
        pl.struct(
            [
                pl.element().struct.field("name"),  # category name
                _format_np(
                    pl.element().struct.field("n"),  # n
                    pl.element().struct.field("p"),  # p
                    precision,
                ),
            ],
        ),
    )


def agg_category_p(
    expr: pl.Expr, precision: int = 1, enum_categories: list[str] | None = None
) -> pl.Expr:
    base = agg_category_base(expr, enum_categories=enum_categories)
    # Transform base struct[4] to struct[2] with formatted percentage
    return base.list.eval(
        pl.struct(
            [
                pl.element().struct.field("name"),  # category name
                _format_p(pl.element().struct.field("p"), precision),  # p
            ],
        ),
    )


def agg_category_n(
    expr: pl.Expr, precision: int = 1, enum_categories: list[str] | None = None
) -> pl.Expr:  # noqa: ARG001
    base = agg_category_base(expr, enum_categories=enum_categories)
    # Transform base struct[4] to struct[2] with formatted count
    return base.list.eval(
        pl.struct(
            [
                pl.element().struct.field("name"),  # category name
                pl.element().struct.field("n").cast(pl.Int64),  # n
            ],
        ),
    )


def agg_category_pnn(
    expr: pl.Expr, precision: int = 1, enum_categories: list[str] | None = None
) -> pl.Expr:
    """Aggregate category as percentage (count/total).

    Args:
        expr: Column expression to aggregate
        precision: Decimal precision for formatting
        enum_categories: If provided, ensures all enum categories are included
                        with count 0 for missing ones, sorted by enum order.

    Returns:
        pl.Expr of type list[struct[2]] where each struct contains:
        - field 0: category value (String)
        - field 1: formatted value (String) as "percentage% (count/total)"
    """
    base = agg_category_base(expr, enum_categories=enum_categories)
    # Transform base struct[4] to struct[2] with formatted percentage (count/total)
    return base.list.eval(
        pl.struct(
            [
                pl.element().struct.field("name"),  # category name
                _format_pnn(
                    pl.element().struct.field("n").cast(pl.Int64),  # n
                    pl.element().struct.field("total").cast(pl.Int64),  # total
                    pl.element().struct.field("p"),  # p
                    precision,
                ),
            ],
        ),
    )


# Default category aggregation function.
# agg_category is an alias for agg_category_np, which formats categories as "count (percentage%)".
# This is the most commonly used format for categorical variables in Table One.
#
# Example usage:
#     >>> import polars as pl
#     >>> from statract import agg_category
#     >>>
#     >>> df = pl.DataFrame({"gender": ["M", "F", "M", "F", "M"]})
#     >>> df.select(agg_category(pl.col("gender")).alias("gender_summary"))
#     # Returns list of structs with category name and "count (percentage%)" format
agg_category = agg_category_np


def agg_bool_category(expr: pl.Expr, precision: int = 1) -> pl.Expr:
    """Aggregate boolean as category (cast to categorical first)."""
    return agg_category_np(
        expr.cast(pl.Categorical(pl.Categories("boolean"))), precision
    )
