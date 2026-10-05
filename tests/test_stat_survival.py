"""Tests for statract.survival module."""

from __future__ import annotations

import polars as pl
import pytest

from statract.survival import (
    cumulative_survival_ci,
    log_rank_pvalue,
    timedelta_to_years,
)


def test_timedelta_to_years_expr():
    df = pl.DataFrame(
        {
            "d": pl.Series(
                [365 * 24 * 60 * 60 * 1_000_000_000, -10 * 24 * 60 * 60 * 1_000_000_000],
                dtype=pl.Duration("ns"),
            )
        }
    )
    out = df.select(timedelta_to_years(pl.col("d")).alias("y"))["y"].to_list()
    assert out[0] == pytest.approx(365 / 365.242)
    assert out[1] == 0.0  # clipped


def test_cumulative_survival_ci_expr_struct():
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0],
            "e": [True, True, False, False],
        }
    )
    out = df.select(cumulative_survival_ci("t", "e", survive_at=2.0).alias("ci"))["ci"][0]
    assert set(out.keys()) == {"rate", "lo", "hi", "summary"}
    assert 0.0 <= float(out["rate"]) <= 1.0
    assert 0.0 <= float(out["lo"]) <= float(out["hi"]) <= 1.0
    assert isinstance(out["summary"], str)


def test_cumulative_survival_python_wrapper():
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0],
            "e": [True, True, False, False],
        }
    )
    result = df.select(cumulative_survival_ci("t", "e", survive_at=2.0).alias("ci"))["ci"][0]
    assert 0.0 <= float(result["rate"]) <= 1.0
    assert 0.0 <= float(result["lo"]) <= float(result["hi"]) <= 1.0
    assert isinstance(result["summary"], str)


def test_cumulative_survival_ci_matches_survival_curve():
    from statract.surv.curve import survival_curve

    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0],
            "e": [True, True, False, False],
        }
    )
    out = df.select(cumulative_survival_ci("t", "e", survive_at=2.0).alias("ci"))["ci"][0]
    at = survival_curve(df, "t", "e").at(2.0)
    assert float(out["rate"]) == pytest.approx(float(at["estimate"][0]))
    assert float(out["lo"]) == pytest.approx(float(at["conf_low"][0]))
    assert float(out["hi"]) == pytest.approx(float(at["conf_high"][0]))


def test_log_rank_pvalue_two_groups():
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 1.5, 2.5],
            "e": [True, False, True, False],
            "g": ["A", "A", "B", "B"],
        }
    )
    p = float(df.select(log_rank_pvalue("t", "e", "g")).item())
    assert 0.0 <= p <= 1.0


def test_log_rank_pvalue_requires_two_groups():
    df = pl.DataFrame(
        {
            "t": [1.0, 2.0, 1.5],
            "e": [True, False, True],
            "g": ["A", "B", "C"],
        }
    )
    with pytest.raises(ValueError):
        df.select(log_rank_pvalue("t", "e", "g")).item()


