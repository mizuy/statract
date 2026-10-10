"""Tests for mixed-model MOR / RE helpers and the optional gpboost extra."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
import scipy.stats

from statract import (
    fit_mixed,
    glmm_random_effects,
    median_odds_ratio,
    plot_random_effects,
)


def _mor_unit_variance() -> float:
    return float(np.exp(np.sqrt(2.0) * scipy.stats.norm.ppf(0.75)))


class TestMedianOddsRatio:
    def test_unit_variance(self) -> None:
        frame = median_odds_ratio(1.0)
        assert frame.height == 1
        got = float(frame["median_odds_ratio"][0])
        assert got == pytest.approx(_mor_unit_variance(), rel=1e-10)

    def test_zero_variance(self) -> None:
        frame = median_odds_ratio(0.0)
        assert float(frame["median_odds_ratio"][0]) == 1.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            median_odds_ratio(-0.1)

    def test_wald_interval_present(self) -> None:
        frame = median_odds_ratio(0.25, std_error=0.05)
        assert "conf_low" in frame.columns
        assert float(frame["conf_low"][0]) < float(frame["median_odds_ratio"][0])
        assert float(frame["conf_high"][0]) > float(frame["median_odds_ratio"][0])


class TestPlotRandomEffects:
    def test_writes_png(self, tmp_path: Path) -> None:
        effects = pl.DataFrame(
            {
                "group": ["A", "B", "C"],
                "blup": [-0.2, 0.0, 0.3],
                "conf_low": [-0.5, -0.2, 0.05],
                "conf_high": [0.1, 0.2, 0.55],
            }
        )
        out = plot_random_effects(effects, tmp_path / "re.png", title="Site BLUPs")
        assert out == tmp_path / "re.png"
        assert out.exists()
        assert out.stat().st_size > 0


def _cluster_frame(n_per: int = 40, n_group: int = 4, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    site = np.repeat([f"s{i}" for i in range(n_group)], n_per)
    u = rng.normal(0, 0.6, n_group)
    u_i = np.repeat(u, n_per)
    treat = rng.integers(0, 2, size=n_per * n_group)
    eta = -0.8 + -0.7 * treat + u_i
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pl.DataFrame({"outcome": y, "treatment": treat, "hospital": site})


class TestFitMixedBinomial:
    def test_glmer_formula_and_helpers(self) -> None:
        df = _cluster_frame()
        fit = fit_mixed(df, "outcome ~ treatment + (1 | hospital)", family="binomial")
        assert fit.family == "binomial"
        assert fit.engine == "laplace"
        assert fit.converged
        tidy = fit.tidy(exponentiate=True)
        assert "treatment" in tidy["term"].to_list()
        re = glmm_random_effects(fit, df.select("hospital"))
        assert re.height == 4
        assert "n" in re.columns
        mor = median_odds_ratio(fit)
        assert float(mor["median_odds_ratio"][0]) >= 1.0
        assert float(mor["variance"][0]) == pytest.approx(float(fit.group_covariance[0, 0]))


class TestGlmmGpboostOptional:
    def test_skipped_without_extra(self) -> None:
        pytest.importorskip("gpboost")
        from statract import glmm_gpboost

        df = _cluster_frame()
        result = glmm_gpboost(
            formula="outcome ~ treatment",
            cols_group=["hospital"],
            data=df,
            return_dict=True,
        )
        if result is None or result.get("model") is None:
            pytest.skip("gpboost fit returned None")
        assert isinstance(result["model"], pl.DataFrame)
