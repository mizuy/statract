"""The REML penalty log determinant keeps its structural rank."""

from __future__ import annotations

import numpy as np

from statract.models.gam.general import _logdet_penalty, _penalty_rank, _total_penalty


def test_logdet_keeps_tiny_eigenvalues_when_smoothing_parameters_differ() -> None:
    a = np.diag([1.0, 2.0, 0.0])
    b = np.diag([0.0, 1.0, 3.0])
    penalties = [(np.arange(3), a), (np.arange(3), b)]
    assert _penalty_rank(3, penalties) == 3
    lam = np.exp([-12.0, 18.0])
    penalty = _total_penalty(3, penalties, lam)
    expected = np.log(lam[0] * 1.0) + np.log(lam[0] * 2.0 + lam[1] * 1.0) + np.log(lam[1] * 3.0)
    assert np.isclose(_logdet_penalty(penalty, 3), expected)


def test_penalty_rank_excludes_the_null_space() -> None:
    penalties = [(np.arange(3), np.diag([1.0, 1.0, 0.0]))]
    assert _penalty_rank(3, penalties) == 2
    assert _penalty_rank(3, []) == 0
