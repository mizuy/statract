"""Tests for statract.tableone.stat module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from statract.tableone.stat import (
    proportion_ci,
    cohen_d,
    format_pvalue,
    notnull_mean,
    odds,
    oddsratio,
    stat_anova,
    stat_auto,
    stat_chisq,
    stat_fisher,
    weighted_corr,
    weighted_cov,
    weighted_mean,
)


def _scalar(df: pl.DataFrame, expr: pl.Expr) -> float:
    return float(df.select(expr.alias("_v")).item())


class TestBasicMath:
    def test_odds_oddsratio(self):
        assert odds(0.5) == 1.0
        assert oddsratio(0.5, 0.25) == odds(0.5) / odds(0.25)

    def test_format_pvalue(self):
        assert format_pvalue(0.5) == "0.50"
        assert format_pvalue(0.005) == "0.005"
        assert format_pvalue(1e-10) == "P<.001"


class TestExprAggregates:
    def test_notnull_mean_expr(self):
        df = pl.DataFrame({"x": [1, None, 3, None]})
        v = _scalar(df, notnull_mean(pl.col("x")))
        assert v == 0.5

    def test_weighted_mean_cov_corr_expr(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 4.0],
                "y": [10.0, 20.0, 40.0],
                "w": [1.0, 2.0, 1.0],
            }
        )

        wm = _scalar(df, weighted_mean(pl.col("x"), pl.col("w")))
        assert abs(wm - 2.25) < 1e-12  # (1*1 + 2*2 + 4*1) / 4

        cov = _scalar(df, weighted_cov(pl.col("x"), pl.col("y"), pl.col("w")))
        corr = _scalar(df, weighted_corr(pl.col("x"), pl.col("y"), pl.col("w")))

        # Compare against numpy, using weights.
        x = np.asarray(df["x"])
        y = np.asarray(df["y"])
        w = np.asarray(df["w"])
        mx = np.sum(w * x) / np.sum(w)
        my = np.sum(w * y) / np.sum(w)
        cov_np = np.sum(w * (x - mx) * (y - my)) / np.sum(w)
        varx = np.sum(w * (x - mx) ** 2) / np.sum(w)
        vary = np.sum(w * (y - my) ** 2) / np.sum(w)
        corr_np = cov_np / math.sqrt(varx * vary)

        assert abs(cov - cov_np) < 1e-10
        assert abs(corr - corr_np) < 1e-10

    def test_ci_lo_hi_expr(self):
        # successes: 2, total: 4
        df = pl.DataFrame({"x": [1, 1, 0, None]})
        out = df.select(proportion_ci(pl.col("x"), alpha=0.05).alias("ci"))["ci"][0]
        lo = float(out["lo"])
        hi = float(out["hi"])
        assert 0.0 <= lo <= hi <= 1.0

    def test_cohen_d_expr(self):
        # Two groups: x mean 2, y mean 4, identical variance -> d should be negative.
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]})
        d = _scalar(df, cohen_d(pl.col("x"), pl.col("y")))
        assert d < 0
        assert abs(d + 2.0) < 1e-12  # for these symmetric lists, pooled sd == 1


class TestStatFuncs:
    def test_stat_auto(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["A", "A", "B", "B"]})
        p = stat_auto(df["x"], df["g"])
        assert isinstance(p, str)
        assert p != "-"
        # Should use ANOVA for interval vs nominal

        df2 = pl.DataFrame({"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]})
        p2 = stat_auto(df2["a"], df2["b"])
        assert isinstance(p2, str)
        assert p2 != "-"
        # Should use categorical test for nominal vs nominal

    def test_stat_anova_string_refs(self):
        df = pl.DataFrame({"g": ["A", "A", "B", "B"], "x": [1.0, 2.0, 10.0, 11.0]})
        p = stat_anova(df["x"], df["g"])
        assert isinstance(p, str)
        assert p != "-"

    def test_stat_chisq_string_refs(self):
        df = pl.DataFrame({"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]})
        p = stat_chisq(df["a"], df["b"])
        assert isinstance(p, str)
        assert p != "-"

    def test_stat_fisher_2x2_no_r(self):
        # 2x2 should use scipy and not require R.
        df = pl.DataFrame({"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]})
        p = stat_fisher(df["a"], df["b"])
        assert isinstance(p, str)
        assert p != "-"

    def test_stat_fisher_rxc_no_r(self):
        # RxC table should use scipy with MonteCarloMethod and not require R.
        df = pl.DataFrame({
            "a": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "b": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y"]
        })
        p = stat_fisher(df["a"], df["b"])
        assert isinstance(p, str)
        assert p != "-"


