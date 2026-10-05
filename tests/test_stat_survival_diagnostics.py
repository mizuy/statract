"""Smoke tests for KM number-at-risk and Cox PH diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

from statract import (
    cox_ph,
    default_at_risk_xticks,
    plot_cox_residuals,
    plot_forest,
    plot_loglog,
    plot_survival,
    proportional_hazards_test,
    survival_curve,
    write_cox_diagnostic_suite,
)
from statract.survival import KmPlotCurve, add_at_risk_counts, _km_from_group


def _surv_frame(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    time = rng.uniform(10, 2000, size=n)
    event = rng.integers(0, 2, size=n).astype(bool)
    group = np.where(rng.random(n) < 0.5, "A", "B")
    age = rng.normal(60, 10, size=n)
    treat = rng.integers(0, 2, size=n)
    return pl.DataFrame(
        {
            "time": time,
            "event": event,
            "group": group,
            "age": age,
            "treat": treat,
        }
    )


def test_src_has_no_lifelines_import() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "statract"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "lifelines" in text:
            offenders.append(str(path.relative_to(root.parents[1])))
    assert offenders == []


def test_default_at_risk_xticks_day_scale_is_readable() -> None:
    ticks = default_at_risk_xticks(2500.0)
    assert ticks[0] == pytest.approx(0.0)
    assert 6 <= ticks.size <= 10
    diffs = np.diff(ticks)
    assert np.allclose(diffs, diffs[0])
    # colon-scale: 500-day steps, not 1000 plus a ragged 3329.
    colon = default_at_risk_xticks(3329.0)
    assert 6 <= colon.size <= 10
    assert 3329.0 not in set(colon.tolist())
    np.testing.assert_allclose(np.diff(colon), np.diff(colon)[0])
    assert colon[1] == pytest.approx(500.0)
    # month-scale (~retinopathy): 10-unit steps, not 0/20/40/60/80.
    months = default_at_risk_xticks(75.0)
    assert 6 <= months.size <= 10
    assert months[1] == pytest.approx(10.0)


def test_plot_survival_includes_number_at_risk_table_by_default() -> None:
    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group")
    nar_ax = getattr(ax, "_statract_at_risk_ax", None)
    assert nar_ax is not None, "expected shared-x number-at-risk table"
    ylabels = [t.get_text() for t in nar_ax.get_yticklabels()]
    assert "A" in ylabels and "B" in ylabels
    assert "at risk" in (nar_ax.get_ylabel() or "").lower()
    texts = getattr(ax, "_statract_at_risk_texts", [])
    assert texts, "expected count text artists at data x of each tick"
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_plot_survival_nar_shares_xaxis_with_curve() -> None:
    """NAR columns must use the same data x ticks/limits as the KM curve."""
    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group")
    nar_ax = getattr(ax, "_statract_at_risk_ax", None)
    assert nar_ax is not None
    np.testing.assert_allclose(nar_ax.get_xlim(), ax.get_xlim())
    expected = getattr(ax, "_statract_at_risk_xticks")
    np.testing.assert_allclose(nar_ax.get_xticks(), expected)
    # Main axis ticks include the same evaluation times.
    main = np.asarray(ax.get_xticks(), dtype=float)
    for t in expected:
        assert np.min(np.abs(main - t)) < 1e-9
    ax.figure.canvas.draw()
    km_labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text().strip()]
    assert km_labels, "KM time tick labels must stay visible (sharex must not steal them)"
    pos_km = ax.get_position()
    pos_nar = nar_ax.get_position()
    assert abs(pos_km.x0 - pos_nar.x0) < 1e-3
    assert abs(pos_km.x1 - pos_nar.x1) < 1e-3
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def _display_x(ax, x: float) -> float:
    return float(ax.transData.transform((x, 0.0))[0])


def test_plot_survival_nar_counts_align_with_curve_ticks_in_display_space() -> None:
    """Counts sit under KM ticks in display pixels, not via margin padding guesses."""
    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group")
    fig = ax.figure
    fig.canvas.draw()
    nar_ax = getattr(ax, "_statract_at_risk_ax")
    ticks = np.asarray(getattr(ax, "_statract_at_risk_xticks"), dtype=float)
    texts = list(getattr(ax, "_statract_at_risk_texts"))
    assert ticks.size and texts
    for t in ticks:
        assert abs(_display_x(ax, t) - _display_x(nar_ax, t)) < 1.5
    for artist in texts:
        x_data, _y = artist.get_position()
        assert np.min(np.abs(ticks - x_data)) < 1e-9
        x_text = float(nar_ax.transData.transform(artist.get_position())[0])
        x_tick = _display_x(ax, x_data)
        assert abs(x_text - x_tick) < 1.5
    import matplotlib.pyplot as plt
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    fig.canvas.draw()
    for t in ticks:
        assert abs(_display_x(ax, t) - _display_x(nar_ax, t)) < 1.5
    plt.close(fig)


def test_plot_survival_nar_counts_match_durations() -> None:
    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group")
    ticks = np.asarray(getattr(ax, "_statract_at_risk_xticks"), dtype=float)
    texts = list(getattr(ax, "_statract_at_risk_texts"))
    nar_ax = getattr(ax, "_statract_at_risk_ax")
    groups = [t.get_text() for t in nar_ax.get_yticklabels()]
    expected: dict[tuple[str, float], int] = {}
    for name in groups:
        durs = df.filter(pl.col("group") == name)["time"].to_numpy()
        for t in ticks:
            expected[(name, float(t))] = int(np.sum(durs >= t))
    by_group_tick: dict[tuple[str, float], str] = {}
    for artist in texts:
        x_data, y_data = artist.get_position()
        name = groups[int(round(y_data))]
        by_group_tick[(name, float(x_data))] = artist.get_text()
    for key, count in expected.items():
        assert by_group_tick[key] == str(count)
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_plot_survival_matches_survival_curve() -> None:
    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group", at_risk_counts=False)
    km_lines = {
        line.get_label(): line
        for line in ax.get_lines()
        if line.get_linestyle() not in {"None", "none", ""} and line.get_label() in {"A", "B"}
    }
    assert set(km_lines) == {"A", "B"}
    for name, line in km_lines.items():
        fitted = survival_curve(df.filter(pl.col("group") == name), "time", "event")
        t = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        for row in fitted.frame().iter_rows(named=True):
            hits = np.flatnonzero(np.isclose(t, row["time"]))
            assert hits.size, f"missing KM vertex at t={row['time']}"
            assert y[hits[0]] == pytest.approx(row["estimate"], abs=1e-9)
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_plot_survival_bw_uses_linetypes_and_grayscale() -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    df = _surv_frame()
    ax = plot_survival(df, time="time", status="event", hue="group", style="bw")
    km_lines = [
        line
        for line in ax.get_lines()
        if line.get_linestyle() not in {"None", "none", ""} and line.get_label() in {"A", "B"}
    ]
    assert len(km_lines) >= 2
    styles = {line.get_linestyle() for line in km_lines}
    assert len(styles) >= 2
    for line in km_lines:
        rgb = mcolors.to_rgb(line.get_color())
        assert max(rgb) - min(rgb) < 0.08
    with pytest.raises(ValueError, match="style"):
        plot_survival(df, time="time", status="event", hue="group", style="rainbow")
    plt.close(ax.figure)


def test_add_at_risk_counts_smoke() -> None:
    df = _surv_frame(20)
    curve = _km_from_group(df, "time", "event", "All")
    assert isinstance(curve, KmPlotCurve)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.step(np.concatenate([[0.0], curve.time]), np.concatenate([[1.0], curve.survival]), where="post")
    ticks = default_at_risk_xticks(float(df["time"].max()))
    ax.set_xticks(ticks)
    ax.set_xlim(left=0.0, right=float(ticks[-1]))
    add_at_risk_counts(curve, ax=ax, xticks=ticks)
    nar_ax = getattr(ax, "_statract_at_risk_ax", None)
    assert nar_ax is not None
    np.testing.assert_allclose(nar_ax.get_xlim(), ax.get_xlim())
    np.testing.assert_allclose(nar_ax.get_xticks(), ticks)
    plt.close(fig)


def test_proportional_hazards_test_and_diagnostic_suite(tmp_path: Path) -> None:
    df = _surv_frame(60)
    fit = cox_ph(df, "Surv(time, event) ~ treat + age")
    zph = proportional_hazards_test(fit)
    assert "global" in zph["term"].to_list()
    assert {"term", "statistic", "df", "p_value"} <= set(zph.columns)

    written = write_cox_diagnostic_suite(
        fit,
        tmp_path,
        data=df,
        time="time",
        status="event",
        by="group",
        stem="cox",
    )
    names = {p.name for p in written}
    assert "cox_schoenfeld.png" in names
    assert "cox_loglog.png" in names
    assert "cox_dfbeta.png" in names
    assert "cox_martingale.png" in names
    assert "cox_deviance.png" in names
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_plot_cox_residuals_and_loglog_return_paths(tmp_path: Path) -> None:
    df = _surv_frame(50)
    fit = cox_ph(df, "Surv(time, event) ~ treat + age")
    out = plot_cox_residuals(fit, "scaled_schoenfeld", path=tmp_path / "sch.png")
    assert Path(out).exists()
    out2 = plot_loglog(df, "time", "event", by="group", path=tmp_path / "ll.png")
    assert Path(out2).exists()


def test_plot_forest_table_layout_default(tmp_path: Path) -> None:
    df = _surv_frame(50)
    tidy = cox_ph(df, "Surv(time, event) ~ treat + age").tidy(exponentiate=True)
    out = plot_forest(tidy, tmp_path / "forest.png", title="HR", xlabel="Hazard ratio")
    assert Path(out).exists() and Path(out).stat().st_size > 0
    fig = plot_forest(tidy, layout="points")
    assert fig is not None
    assert len(fig.axes) == 1
    import matplotlib.pyplot as plt

    plt.close(fig)
