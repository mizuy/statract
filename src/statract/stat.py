"""Statistical analysis utilities for Polars DataFrames.

This module provides statistical test functions and utilities for Polars DataFrames,
including ANOVA, chi-square, Fisher's exact test, and various statistical measures.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import polars as pl
import scipy.stats


# statfunc(col0, col1) -> str
StatFunc: TypeAlias = Callable[[pl.Series, pl.Series], str]


def odds(p: float) -> float:
    """Calculate odds from probability.

    Examples:
        >>> from statract import odds
        >>>
        >>> odds(0.5)  # 50% probability
        1.0
        >>> odds(0.25)  # 25% probability
        0.333...
    """
    return p / (1 - p)


def oddsratio(p: float, q: float) -> float:
    """Calculate odds ratio between two probabilities.

    Examples:
        >>> from statract import oddsratio
        >>>
        >>> oddsratio(0.5, 0.25)  # Odds ratio between 50% and 25%
        3.0
    """
    return odds(p) / odds(q)


def notnull_mean(expr: pl.Expr) -> pl.Expr:
    """Return an expression for the mean of non-null values (as a fraction in [0, 1]).

    Examples:
        >>> import polars as pl
        >>> from statract import notnull_mean
        >>>
        >>> df = pl.DataFrame({"x": [1, 2, None, 4, 5]})
        >>> df.select(notnull_mean(pl.col("x")))
        shape: (1, 1)
        ┌─────┐
        │ x   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.8 │  # 4 out of 5 are non-null
        └─────┘
    """
    return expr.is_not_null().mean()


def weighted_mean(x: pl.Expr, w: pl.Expr) -> pl.Expr:
    """Return an expression for the weighted mean.

    Examples:
        >>> import polars as pl
        >>> from statract import weighted_mean
        >>>
        >>> df = pl.DataFrame({
        ...     "value": [10, 20, 30],
        ...     "weight": [1, 2, 1]
        ... })
        >>> df.select(weighted_mean(pl.col("value"), pl.col("weight")))
        shape: (1, 1)
        ┌──────┐
        │ value│
        │ ---  │
        │ f64  │
        ╞══════╡
        │ 20.0 │  # (10*1 + 20*2 + 30*1) / (1+2+1)
        └──────┘
    """
    return (x * w).sum() / w.sum()


def weighted_cov(x: pl.Expr, y: pl.Expr, w: pl.Expr) -> pl.Expr:
    """Return an expression for the weighted covariance.

    Examples:
        >>> import polars as pl
        >>> from statract import weighted_cov
        >>>
        >>> df = pl.DataFrame({
        ...     "x": [1, 2, 3],
        ...     "y": [2, 4, 6],
        ...     "w": [1, 1, 1]
        ... })
        >>> df.select(weighted_cov(pl.col("x"), pl.col("y"), pl.col("w")))
    """
    x_mean = weighted_mean(x, w)
    y_mean = weighted_mean(y, w)
    return (w * (x - x_mean) * (y - y_mean)).sum() / w.sum()


def weighted_corr(x: pl.Expr, y: pl.Expr, w: pl.Expr) -> pl.Expr:
    """Return an expression for the weighted correlation.

    Examples:
        >>> import polars as pl
        >>> from statract import weighted_corr
        >>>
        >>> df = pl.DataFrame({
        ...     "x": [1, 2, 3, 4, 5],
        ...     "y": [2, 4, 6, 8, 10],
        ...     "w": [1, 1, 1, 1, 1]
        ... })
        >>> df.select(weighted_corr(pl.col("x"), pl.col("y"), pl.col("w")))
        # Returns correlation close to 1.0
    """
    cov_xy = weighted_cov(x, y, w)
    cov_xx = weighted_cov(x, x, w)
    cov_yy = weighted_cov(y, y, w)
    return cov_xy / (cov_xx * cov_yy).sqrt()


def proportion_ci(expr: pl.Expr, alpha: float = 0.05) -> pl.Expr:
    """Return an expression for (lo, hi) confidence interval of a proportion as a struct.

    This expects `expr` to represent a boolean-like indicator where `expr > 0`
    counts as success. The interval is the Wald (normal approximation) interval,
    clipped to [0, 1].

    Examples:
        >>> import polars as pl
        >>> from statract import proportion_ci
        >>>
        >>> # Calculate confidence interval for a proportion
        >>> df = pl.DataFrame({
        ...     "success": [1, 1, 0, 1, 0, 1, 1, 0]
        ... })
        >>>
        >>> result = df.select(
        ...     proportion_ci(pl.col("success")).alias("ci")
        ... )
        >>> print(result.select(pl.col("ci").struct.field("lo"), pl.col("ci").struct.field("hi")))
    """
    out_dtype = pl.Struct([pl.Field("lo", pl.Float64), pl.Field("hi", pl.Float64)])
    z = float(scipy.stats.norm.isf(alpha / 2.0))

    def interval(s: pl.Series) -> pl.Series:
        count = int(s.struct.field("_count")[0])
        total = int(s.struct.field("_total")[0])
        p = count / total if total else float("nan")
        half = z * math.sqrt(p * (1.0 - p) / total) if total else float("nan")
        lo = min(max(p - half, 0.0), 1.0)
        hi = min(max(p + half, 0.0), 1.0)
        return pl.Series([{"lo": lo, "hi": hi}], dtype=out_dtype)

    return pl.struct(
        [
            (expr > 0).sum().alias("_count"),
            expr.is_not_null().sum().alias("_total"),
        ],
    ).map_batches(interval, return_dtype=out_dtype)


def weighted_qcut(values: pl.Expr, weights: pl.Expr, df: pl.DataFrame, q: int | list[float]) -> pl.Series:
    """Return weighted quantile cuts.

    Creates quantile bins based on weighted cumulative distribution.
    Implemented using polars without pandas conversion.

    Args:
        values: Expression for values to bin
        weights: Expression for weights
        df: DataFrame containing the data
        q: Number of quantiles (int) or list of quantile values (e.g., [0, 0.25, 0.5, 0.75, 1.0])

    Returns:
        Series with quantile cut labels as strings (e.g., "(0.0, 0.25]", "(0.25, 0.5]", etc.)

    Examples:
        >>> import polars as pl
        >>> from statract import weighted_qcut
        >>>
        >>> df = pl.DataFrame({
        ...     "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ... })
        >>> cuts = weighted_qcut(pl.col("value"), pl.col("weight"), df, q=4)
        >>> # Returns 4 quantile bins
    """
    # Get column names
    values_name = values.meta.output_name()
    weights_name = weights.meta.output_name()

    if values_name is None or weights_name is None:
        raise ValueError("values and weights expressions must have output names")

    # Select and sort by values
    sorted_df = (
        df.select([values, weights])
        .with_row_index("_idx")
        .sort(values_name)
        .with_columns(pl.col(weights_name).cum_sum().alias("_cumsum"))
    )

    # Calculate quantiles
    if isinstance(q, int):
        quantiles = np.linspace(0, 1, q + 1).tolist()
    else:
        quantiles = q

    # Get total cumulative sum for normalization
    total_weight = sorted_df[weights_name].sum()

    # Normalize cumulative sum to [0, 1]
    sorted_df = sorted_df.with_columns((pl.col("_cumsum") / total_weight).alias("_normalized"))

    # Assign bins based on quantiles using when/then
    # Build when/then chain for each quantile bin
    when_expr = None
    for i in range(len(quantiles) - 1):
        left = quantiles[i]
        right = quantiles[i + 1]
        bin_label = f"({left:.6g}, {right:.6g}]"

        if i == 0:
            # First bin: include left boundary
            condition = (pl.col("_normalized") >= left) & (pl.col("_normalized") <= right)
        else:
            # Other bins: exclude left boundary (already handled by previous bin)
            condition = (pl.col("_normalized") > left) & (pl.col("_normalized") <= right)

        if when_expr is None:
            when_expr = pl.when(condition).then(pl.lit(bin_label))
        else:
            when_expr = when_expr.when(condition).then(pl.lit(bin_label))

    # Default to last bin (should not happen, but for safety)
    when_expr = when_expr.otherwise(pl.lit(f"({quantiles[-2]:.6g}, {quantiles[-1]:.6g}]"))

    # Apply bin assignment
    sorted_df = sorted_df.with_columns(when_expr.alias("_bin"))

    # Restore original order and return bin labels
    return sorted_df.sort("_idx").select("_bin")["_bin"]


def cohen_d(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """Return an expression for Cohen's d effect size (pooled SD, ddof=1).

    Cohen's d measures the standardized difference between two means.

    Examples:
        >>> import polars as pl
        >>> from statract import cohen_d
        >>>
        >>> df = pl.DataFrame({
        ...     "group": ["A", "A", "A", "B", "B", "B"],
        ...     "value": [10, 11, 12, 20, 21, 22]
        ... })
        >>> result = df.group_by("group").agg(pl.col("value").alias("values")).select(
        ...     cohen_d(
        ...         pl.col("values").list.get(0),
        ...         pl.col("values").list.get(1)
        ...     )
        ... )
    """
    nx = x.count()
    ny = y.count()
    varx = x.var(ddof=1)
    vary = y.var(ddof=1)
    pooled_var = ((nx - 1) * varx + (ny - 1) * vary) / (nx + ny - 2)
    pooled_sd = pooled_var.sqrt()
    return (x.mean() - y.mean()) / pooled_sd


def format_pvalue(pvalue: float, *, omit: bool = True) -> str:
    """Format p-value according to NEJM guidelines.

    Args:
        pvalue: P-value to format
        omit: If True, use P<.001 for very small values

    Returns:
        Formatted p-value string
    """
    if pvalue > 0.01:
        return f"{pvalue:.2f}"
    elif 0.01 > pvalue > 0.001:
        return f"{pvalue:.3f}"
    elif omit:
        return "P<.001"
    else:
        return f"{pvalue:.2g}"


def stat_anova(col_interval: pl.Series, col_nominal: pl.Series) -> str:
    """Perform one-way ANOVA test.

    Tests whether means of continuous variable differ across groups defined by categorical variable.

    Args:
        col_interval: Continuous variable Series
        col_nominal: Categorical variable Series

    Returns:
        Formatted p-value string (e.g., "0.05", "P<.001", or "-" if test cannot be performed)

    Examples:
        >>> import polars as pl
        >>> from statract import stat_anova
        >>>
        >>> df = pl.DataFrame({
        ...     "age": [45, 50, 55, 60, 65, 70],
        ...     "group": ["A", "A", "B", "B", "C", "C"]
        ... })
        >>> pvalue = stat_anova(df["age"], df["group"])
        >>> print(pvalue)
    """
    # Group by nominal variable and get interval values
    df = pl.DataFrame({"_interval": col_interval, "_nominal": col_nominal})
    grouped = df.group_by("_nominal").agg(pl.col("_interval"))
    groups = []
    for row in grouped.iter_rows(named=True):
        # row["_interval"] is a python list (list of values)
        vals = [v for v in row["_interval"] if v is not None]
        if vals:
            groups.append(np.asarray(vals))

    if len(groups) < 2:
        return "-"

    _stat, pvalue = scipy.stats.f_oneway(*groups)
    return format_pvalue(pvalue)


def stat_chisq(col_nominal0: pl.Series, col_nominal1: pl.Series) -> str:
    """Perform chi-square test of independence.

    Tests whether two categorical variables are independent.

    Args:
        col_nominal0: First categorical variable Series
        col_nominal1: Second categorical variable Series

    Returns:
        Formatted p-value string (e.g., "0.05", "P<.001", or "-" if test cannot be performed)

    Examples:
        >>> import polars as pl
        >>> from statract import stat_chisq
        >>>
        >>> df = pl.DataFrame({
        ...     "gender": ["M", "M", "F", "F", "M"],
        ...     "treatment": ["A", "B", "A", "B", "A"]
        ... })
        >>> pvalue = stat_chisq(df["gender"], df["treatment"])
        >>> print(pvalue)
    """
    # Create contingency table from two Series
    ct = (
        pl.DataFrame({"col0": col_nominal0, "col1": col_nominal1})
        .with_columns(pl.lit(1).alias("_v"))
        .pivot(
            index="col0",
            on="col1",
            values="_v",
            aggregate_function="len",
        )
        .fill_null(0)
    )

    if ct.height < 2 or ct.width < 2:
        return "-"

    # Convert to numpy array for scipy
    ct_array = ct.select(pl.exclude("col0")).to_numpy()
    _chi2, p, _dof, _expected = scipy.stats.chi2_contingency(ct_array)
    return format_pvalue(p)


def stat_fisher(col_nominal0: pl.Series, col_nominal1: pl.Series) -> str:
    """Perform Fisher's exact test.

    Notes:
    - For 2x2 tables, this uses `scipy.stats.fisher_exact` with default method.
    - For RxC tables (e.g. 2x3, 3x3, ...), this uses `scipy.stats.fisher_exact`
      with `MonteCarloMethod` for p-value computation.

    Args:
        col_nominal0: First categorical variable Series
        col_nominal1: Second categorical variable Series

    Returns:
        Formatted p-value string
    """
    # Build contingency table from two Series
    ct = (
        pl.DataFrame({"col0": col_nominal0, "col1": col_nominal1})
        .drop_nulls()
        .with_columns(pl.lit(1).alias("_v"))
        .pivot(index="col0", on="col1", values="_v", aggregate_function="len")
        .fill_null(0)
    )

    if ct.height < 2 or ct.width < 2:
        return "-"

    # Convert to numpy counts matrix.
    ct_array = ct.select(pl.exclude("col0")).to_numpy()
    if ct_array.shape == (2, 2):
        # 2x2 table: use default method
        _, p = scipy.stats.fisher_exact(ct_array)
        return format_pvalue(p)

    # RxC table: use MonteCarloMethod
    rng = np.random.default_rng()
    method = scipy.stats.MonteCarloMethod(rng=rng)
    result = scipy.stats.fisher_exact(ct_array, method=method)
    p = result.pvalue.item() if isinstance(result.pvalue, np.ndarray) else result.pvalue
    return format_pvalue(p)


def stat_auto(col0: pl.Series, col1: pl.Series) -> str:
    """Automatically select and perform appropriate statistical test based on column types.

    Selects and executes test function based on data types:
    - Both numeric: ANOVA
    - Both categorical: Chi-square
    - Mixed: Returns "-" (cannot auto-select)

    Args:
        col0: First column Series
        col1: Second column Series

    Returns:
        Formatted p-value string (e.g., "0.05", "P<.001", or "-" if test cannot be performed)

    Examples:
        >>> import polars as pl
        >>> from statract import stat_auto
        >>>
        >>> df = pl.DataFrame({
        ...     "age": [45, 50, 55, 60],
        ...     "group": ["A", "A", "B", "B"]
        ... })
        >>> pvalue = stat_auto(df["age"], df["group"])
        >>> # Automatically uses ANOVA and returns p-value string
    """
    # Get dtype from Series
    dtype0 = col0.dtype
    dtype1 = col1.dtype

    def dt(dtype: pl.DataType) -> str:
        match dtype:
            case (
                pl.Float32
                | pl.Float64
                | pl.Int8
                | pl.Int16
                | pl.Int32
                | pl.Int64
                | pl.UInt8
                | pl.UInt16
                | pl.UInt32
                | pl.UInt64
            ):
                return "interval"
            case pl.Enum | pl.Categorical | pl.String | pl.Boolean:
                return "nominal"
            case _:
                return "others"

    dts = (dt(dtype0), dt(dtype1))

    match dts:
        case ("nominal", "nominal"):
            return stat_fisher(col0, col1)
        case ("interval", "nominal"):
            return stat_anova(col0, col1)
        case _:
            return "-"
