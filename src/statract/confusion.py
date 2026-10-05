"""Experimental implementations for confusion matrix and related metrics.

.. warning::
    This module contains experimental implementations that may change
    or be removed in future versions. Use with caution.

.. note::
    The API and behavior of functions in this module may change without notice.
"""

from __future__ import annotations

import warnings
from typing import Any

import polars as pl


def ratio(p: int, q: int) -> str:
    """Format ratio as percentage with fraction.

    Examples:
        >>> from statract import ratio
        >>>
        >>> ratio(3, 10)
        '30.0% 3/10'
        >>> ratio(1, 4)
        '25.0% 1/4'
    """
    return f"{p / q:0.1%} {p}/{q}"


class confusion_matrix:
    """Confusion matrix for binary classification metrics.

    .. warning::
        This is an experimental implementation and may change in future versions.

    Calculates various classification metrics including sensitivity, specificity,
    precision, recall, F1 score, and likelihood ratios.

    Examples:
        >>> from statract import confusion_matrix
        >>>
        >>> # Create from counts
        >>> cm = confusion_matrix(TP=80, TN=90, FP=10, FN=20)
        >>> print(f"Sensitivity: {cm.Se:.2f}")
        >>> print(f"Specificity: {cm.Sp:.2f}")
        >>> print(f"F1 Score: {cm.F1:.2f}")
        >>> cm.summary()  # Print formatted table
        >>>
        >>> # Create from Polars expressions
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "pred": [True, True, False, False],
        ...     "truth": [True, False, True, False]
        ... })
        >>> cm = confusion_matrix.from_expr(
        ...     pl.col("pred"),
        ...     pl.col("truth"),
        ...     df
        ... )
    """

    def __init__(self, TP: int, TN: int, FP: int, FN: int) -> None:
        self.TP = TP
        self.TN = TN
        self.FP = FP  # type I error, alpha
        self.FN = FN  # type II error, beta
        n = TP + TN + FP + FN
        self.n = n
        self.Prev = (TP + FP) / n

        # sensitivity, recall, hit rate, true positive rate
        self.TPR = TP / (TP + FN)
        self.TNR = TN / (TN + FP)
        self.PPV = TP / (TP + FP)
        self.NPV = TN / (TN + FN)

        # another names
        self.Se = self.Recall = self.TPR
        self.Sp = self.TNR
        self.Precision = self.PPV
        self.FNR = 1 - self.TPR
        self.FPR = 1 - self.TNR
        self.FDR = 1 - self.PPV
        self.FOR = 1 - self.NPV

        self.Acc = (TP + TN) / n
        self.F1 = 2 * TP / (2 * TP + FP + FN)
        self.LRp = self.Se / (1 - self.Sp)
        self.LRn = self.Sp / (1 - self.Se)
        self.DOR = self.LRp / self.LRn

    def summary(self) -> None:
        tp, tn, fp, fn = self.TP, self.TN, self.FP, self.FN
        print(
            f"""
            |Predict T|Predict F|
        -----|-------------------|
        GT T | TP{tp:>5} | FN{fn:>5} | {tp + fn:>5}
        GT F | FP{fp:>5} | TN{tn:>5} | {fp + tn:>5}
        -----|-------------------|
                {tp + fp:>5} |   {fn + tn:>5} | {tp + fp + fn + tn:>5}
        """,
        )
        print(f"PPV: {ratio(tp, tp + fp)}")
        print(f"NPV: {ratio(tn, tn + fn)}")
        print(f"Se: {ratio(tp, tp + fn)}")
        print(f"Sp: {ratio(tn, tn + fp)}")

    @classmethod
    def from_expr(cls, pred: pl.Expr, truth: pl.Expr, df: pl.DataFrame) -> confusion_matrix:
        """Create confusion matrix from polars expressions.

        Args:
            pred: Prediction expression (boolean)
            truth: Truth expression (boolean)
            df: DataFrame containing the data

        Returns:
            confusion_matrix instance
        """
        result = df.select(
            [
                (pred & truth).sum().alias("TP"),
                ((~pred) & (~truth)).sum().alias("TN"),
                (pred & (~truth)).sum().alias("FP"),
                ((~pred) & truth).sum().alias("FN"),
            ]
        ).row(0)

        TP, TN, FP, FN = result[0], result[1], result[2], result[3]
        return confusion_matrix(TP, TN, FP, FN)


def sm_summary2df(results: Any) -> pl.DataFrame:
    """Convert statsmodels regression results to Polars DataFrame.

    .. deprecated::
        Use ``Fit.tidy`` from ``statract.fit_ols`` or ``fit_glm``.
    """
    warnings.warn(
        "sm_summary2df is deprecated; use Fit.tidy",
        DeprecationWarning,
        stacklevel=2,
    )
    r = pl.DataFrame(
        {
            "pvals": results.pvalues,
            "coeff": results.params,
            "conf_l": results.conf_int()[0],
            "conf_h": results.conf_int()[1],
        }
    )

    hr = (
        pl.col("coeff").exp().round(1).cast(pl.Utf8)
        + pl.lit(" (")
        + pl.col("conf_l").exp().round(1).cast(pl.Utf8)
        + pl.lit(", ")
        + pl.col("conf_h").exp().round(1).cast(pl.Utf8)
        + pl.lit(")")
    )
    return r.with_columns(
        hr.alias("HR (95% CI)"),
        pl.col("pvals").map_elements(lambda x: f"{x:.2g}", return_dtype=pl.String).alias("p value"),
    ).select(["HR (95% CI)", "p value"])
