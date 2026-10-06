"""Left-truncated Aalen–Johansen."""

from __future__ import annotations

import polars as pl

from statract import survival_curve


def test_delayed_entry_risk_set_excludes_people_who_have_not_entered():
    data = pl.DataFrame(
        {
            "entry": [0.0, 0.5, 0.0],
            "time": [1.0, 2.0, 2.0],
            "event": [1, 0, 2],
        }
    )
    curve = survival_curve(data, "time", "event", kind="aalen_johansen", entry="entry")
    relapse = curve.frame().filter(pl.col("state") == "1").sort("time")
    # t=1: all three are at risk, one relapse. t=2: the relapse has left, one competing event.
    assert relapse["time"].to_list() == [1.0, 2.0]
    assert relapse["n_risk"].to_list() == [3.0, 2.0]
    assert relapse["estimate"].to_list()[0] == 1 / 3
    assert relapse["estimate"].to_list()[1] == 1 / 3


def test_entry_before_every_time_matches_the_ordinary_curve_at_event_times():
    data = pl.DataFrame(
        {
            "entry": [-1.0, -1.0, -1.0, -1.0],
            "time": [1.0, 1.0, 2.0, 3.0],
            "event": [1, 0, 2, 0],
        }
    )
    ordinary = survival_curve(data, "time", "event", kind="aalen_johansen")
    truncated = survival_curve(data, "time", "event", kind="aalen_johansen", entry="entry")
    left = ordinary.frame().filter((pl.col("state") == "1") & (pl.col("n_event") > 0))
    right = truncated.frame().filter((pl.col("state") == "1") & (pl.col("n_event") > 0))
    assert right["time"].to_list() == left["time"].to_list()
    assert right["estimate"].to_list() == left["estimate"].to_list()
