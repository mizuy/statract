"""Smoke tests for plot_forest / glmm_forestplot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from statract import fit_glm, fit_ols, glmm_forestplot, plot_forest
from statract.surv import cox_ph


def _binomial_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "x": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "z": [1.0, 2.0, 1.5, 2.5, 3.0, 0.5, 2.0, 1.0, 2.2, 3.1, 0.8, 2.8],
        }
    )


def test_plot_forest_or_returns_path(tmp_path: Path) -> None:
    fit = fit_glm(_binomial_frame(), "y ~ x + z", family="binomial")
    tidy = fit.tidy(exponentiate=True)
    out = plot_forest(tidy, tmp_path / "or_forest.png", title="OR", xlabel="Odds ratio")
    assert out == tmp_path / "or_forest.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_forest_hr_returns_figure() -> None:
    df = pl.DataFrame(
        {
            "time": [10.0, 12.0, 8.0, 20.0, 15.0, 9.0, 30.0, 25.0, 18.0, 22.0],
            "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            "treat": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [50.0, 60.0, 55.0, 70.0, 45.0, 65.0, 52.0, 58.0, 48.0, 62.0],
        }
    )
    tidy = cox_ph(df, "Surv(time, event) ~ treat + age").tidy(exponentiate=True)
    fig = plot_forest(tidy, title="HR", xlabel="Hazard ratio")
    assert fig is not None
    assert hasattr(fig, "savefig")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_forest_ols_coef(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "z": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        }
    )
    tidy = fit_ols(df, "y ~ x + z").tidy()
    out = plot_forest(tidy, tmp_path / "ols_forest.png", xlabel="Coefficient")
    assert Path(out).exists()


def test_glmm_forestplot_from_oddsratio_table(tmp_path: Path) -> None:
    summary = pl.DataFrame(
        {
            "variable": ["Covariate_1", "Covariate_2"],
            "oddsratio": [0.5, 1.2],
            "lo": [0.2, 0.9],
            "hi": [0.9, 1.6],
        }
    )
    out = glmm_forestplot(summary, path=tmp_path / "glmm_forest.png", lim=5)
    assert Path(out).exists()


def test_plot_forest_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_forest(pl.DataFrame({"term": [], "estimate": [], "conf_low": [], "conf_high": []}))


def test_plot_forest_default_is_table_layout() -> None:
    tidy = fit_glm(_binomial_frame(), "y ~ x + z", family="binomial").tidy(exponentiate=True)
    fig = plot_forest(tidy, title="OR", xlabel="Odds ratio")
    assert len(fig.axes) == 2
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "Term" in texts
    assert "OR (95% CI)" in texts
    assert "P" in texts
    tab_w = fig.axes[0].get_position().width
    forest_w = fig.axes[-1].get_position().width
    assert tab_w < forest_w * 1.25
    xs = {t.get_text(): t.get_position()[0] for t in fig.axes[0].texts}
    assert xs["OR (95% CI)"] < 0.70
    forest = fig.axes[-1]
    vlines = [line for line in forest.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
    assert any(abs(float(line.get_xdata()[0]) - 1.0) < 1e-9 for line in vlines)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_forest_bw_is_grayscale() -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    tidy = fit_glm(_binomial_frame(), "y ~ x + z", family="binomial").tidy(exponentiate=True)
    fig = plot_forest(tidy, xlabel="Odds ratio", style="bw")
    forest = fig.axes[-1]
    for line in forest.lines:
        rgb = mcolors.to_rgb(line.get_color())
        assert max(rgb) - min(rgb) < 0.08
    with pytest.raises(ValueError, match="style"):
        plot_forest(tidy, style="rainbow")
    plt.close(fig)


def test_plot_forest_formats_small_p_as_lt_001() -> None:
    import matplotlib.pyplot as plt

    tidy = pl.DataFrame(
        {
            "term": ["treat"],
            "estimate": [0.4],
            "conf_low": [0.2],
            "conf_high": [0.7],
            "exp_estimate": [1.5],
            "exp_conf_low": [1.2],
            "exp_conf_high": [1.9],
            "p_value": [1e-8],
        }
    )
    fig = plot_forest(tidy, xlabel="Hazard ratio")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "<0.001" in texts
    assert "HR (95% CI)" in texts
    plt.close(fig)


def test_plot_forest_accepts_prepared_hr_columns() -> None:
    import matplotlib.pyplot as plt

    prepared = pl.DataFrame(
        {
            "name": ["treat"],
            "n": [200],
            "event": [40],
            "estimate": [0.62],
            "conf.low": [0.41],
            "conf.high": [0.94],
            "pvalue": ["0.024"],
        }
    )
    fig = plot_forest(
        prepared,
        xlabel="Hazard ratio",
        drop_intercept=False,
        extra_cols=("n", "event"),
        layout="table",
        style="bw",
    )
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "treat" in texts
    assert "n" in texts
    assert "event" in texts
    plt.close(fig)


def test_plot_forest_also_pdf(tmp_path: Path) -> None:
    tidy = fit_glm(_binomial_frame(), "y ~ x + z", family="binomial").tidy(exponentiate=True)
    png = plot_forest(tidy, tmp_path / "or.png", xlabel="Odds ratio", also_pdf=True)
    assert Path(png).exists()
    assert Path(png).with_suffix(".pdf").exists()
