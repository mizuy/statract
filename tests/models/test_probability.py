"""Tests for statract.models.probability."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from statract.models.probability import (
    brier_score,
    calibration_table,
    decision_curve_table,
    write_probability_artifacts,
)


def test_brier_and_calibration_shapes() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.3, 0.9, 0.4])
    assert 0.0 <= brier_score(y, p) <= 1.0
    cal = calibration_table(y, p, n_bins=4, model="m")
    assert cal.height >= 1
    assert set(cal.columns) >= {"model", "pred_mean", "obs_rate"}
    dca = decision_curve_table(y, p, model="m")
    assert dca.height >= 1
    assert "net_benefit" in dca.columns


def test_write_probability_artifacts(tmp_path: Path) -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    models = [
        {"name": "model_logistic", "y_val": y, "prob_val": np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.1])},
        {"name": "model_rf", "y_val": y, "prob_val": np.array([0.25, 0.75, 0.35, 0.65, 0.55, 0.15])},
    ]
    written = write_probability_artifacts(models, tmp_path)
    names = {p.name for p in written}
    assert "table_calibration.csv" in names
    assert "table_brier.csv" in names
    assert "table_dca.csv" in names
    assert "fig_calibration.png" in names
    assert "fig_dca.png" in names
