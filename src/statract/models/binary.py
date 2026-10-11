"""Binary classification metrics for 0/1 labels (positive class = 1)."""

from __future__ import annotations

import numpy as np
import polars as pl


def binary_perf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """2×2 metrics with positive class = 1.

    Also reports ``overtreatment_rate`` (= FPR) and ``undertreatment_rate`` (= FNR)
    for decision-oriented reporting.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = int(y_true.size)
    if n == 0:
        return {
            "n": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "n_pos": 0,
            "n_neg": 0,
            "sensitivity": None,
            "specificity": None,
            "ppv": None,
            "npv": None,
            "accuracy": None,
            "overtreatment_rate": None,
            "undertreatment_rate": None,
        }
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n_pos = tp + fn
    n_neg = tn + fp
    n_pred_pos = tp + fp
    n_pred_neg = tn + fn
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sensitivity": tp / n_pos if n_pos else None,
        "specificity": tn / n_neg if n_neg else None,
        "ppv": tp / n_pred_pos if n_pred_pos else None,
        "npv": tn / n_pred_neg if n_pred_neg else None,
        "accuracy": (tp + tn) / n,
        "overtreatment_rate": fp / n_neg if n_neg else None,
        "undertreatment_rate": fn / n_pos if n_pos else None,
    }


def decision_rates(perf: dict[str, float | int | None]) -> dict[str, float | int | None]:
    """Extract decision-oriented fields from :func:`binary_perf`."""
    return {
        "overtreatment_rate": perf.get("overtreatment_rate"),
        "undertreatment_rate": perf.get("undertreatment_rate"),
        "missed_high_risk_n": perf.get("fn"),
        "unnecessary_high_risk_n": perf.get("fp"),
    }


def threshold_tradeoff(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pl.DataFrame:
    """Sens/Spec (and related) for ``prob >= t`` as positive prediction."""
    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0, 1.0], np.linspace(0.05, 0.95, 37)]))
    rows: list[dict[str, object]] = []
    for t in thresholds:
        pred = (prob >= float(t)).astype(int)
        rows.append({"threshold": float(t), **binary_perf(y_true, pred)})
    return pl.DataFrame(rows)
