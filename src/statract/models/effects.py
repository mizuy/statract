"""Tests and effect curves for spline terms (``rms::anova`` and ``rms::contrast``).

Both work on any fit that carries ``coefficients``, ``covariance``, ``names``,
and a formula ``design``: ``fit_glm``, ``fit_ols``, and ``cox_ph``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl
from scipy import stats

from .design import build_design
from .formula import Basis


def _term_columns(fit: Any, term: str) -> tuple[Basis, list[int]]:
    design = fit.design
    computed = design.computed or {}
    basis = computed.get(term)
    if not isinstance(basis, Basis):
        spline_terms = [name for name, expr in computed.items() if isinstance(expr, Basis)]
        raise ValueError(f"{term!r} is not a spline term of this fit; spline terms are {spline_terms}")
    if design.recipes is None:
        raise ValueError("the fit was not built from a formula")
    columns = [
        j
        for j, recipe in enumerate(design.recipes)
        if len(recipe) == 1 and recipe[0][0] == term and design.names[j] in fit.names
    ]
    return basis, [list(fit.names).index(design.names[j]) for j in columns]


def _wald(beta: np.ndarray, cov: np.ndarray) -> tuple[float, int, float]:
    chi2 = float(beta @ np.linalg.solve(cov, beta))
    df = beta.size
    return chi2, df, float(stats.chi2.sf(chi2, df))


def spline_test(fit: Any, term: str) -> pl.DataFrame:
    """Wald chi-square for a spline term and, for ``rcs``, its nonlinear part.

    Matches the ``term`` and ``Nonlinear`` rows of ``anova()`` on an ``rms``
    fit when the term has no interactions. ``term`` is the label as written in
    the formula, for example ``"rcs(age, 4)"``. Interaction columns are not
    pooled into the term.
    """
    basis, index = _term_columns(fit, term)
    beta = np.asarray(fit.coefficients, dtype=float)
    cov = np.asarray(fit.covariance, dtype=float)
    rows = []
    chi2, df, p = _wald(beta[index], cov[np.ix_(index, index)])
    rows.append({"term": term, "part": "overall", "statistic": chi2, "df": df, "p_value": p})
    if basis.fn == "rcs":
        rest = index[1:]
        chi2, df, p = _wald(beta[rest], cov[np.ix_(rest, rest)])
        rows.append({"term": term, "part": "nonlinear", "statistic": chi2, "df": df, "p_value": p})
    return pl.DataFrame(rows)


def _adjust_values(data: pl.DataFrame, names: Sequence[str]) -> dict[str, object]:
    """``rms::datadist`` adjust-to values: the median, or the most frequent level."""
    values: dict[str, object] = {}
    for name in names:
        series = data.get_column(name).drop_nulls()
        if series.len() == 0:
            raise ValueError(f"column {name!r} has no values")
        if series.dtype.is_numeric() and series.dtype != pl.Boolean:
            values[name] = float(np.median(series.to_numpy().astype(float)))
        else:
            counts = series.value_counts(sort=False)
            top = counts.get_column("count").max()
            candidates = counts.filter(pl.col("count") == top).get_column(name).to_list()
            values[name] = sorted(candidates, key=str)[0] if series.dtype != pl.Boolean else candidates[0]
    return values


def spline_effect(
    fit: Any,
    data: pl.DataFrame,
    variable: str,
    *,
    at: Sequence[float],
    reference: float,
    adjust: dict[str, object] | None = None,
    level: float = 0.95,
    exponentiate: bool = False,
) -> pl.DataFrame:
    """Contrast of the linear predictor at ``at`` against ``reference``.

    The same numbers as ``rms::contrast(fit, list(x = at), list(x = reference))``:
    every other variable is held at ``adjust`` (default: the median of a
    numeric column, the most frequent level of a factor, as ``datadist``).
    With ``exponentiate=True`` the estimate and limits are odds or hazard
    ratios. Intervals use the normal quantile.
    """
    design = fit.design
    if design.recipes is None:
        raise ValueError("the fit was not built from a formula")
    if variable not in data.columns:
        raise KeyError(f"column {variable!r} is not in the frame")
    computed = design.computed or {}
    from .formula import _columns_in

    needed: list[str] = []
    for recipe in design.recipes:
        for symbol, _level in recipe:
            sources = _columns_in(computed[symbol]) if symbol in computed else [symbol]
            for source in sources:
                if source not in needed:
                    needed.append(source)
    if variable not in needed:
        raise ValueError(f"{variable!r} is not in the model")
    others = [name for name in needed if name != variable]
    fixed = _adjust_values(data, others)
    fixed.update(adjust or {})
    at_values = np.asarray(at, dtype=float)
    points = np.concatenate([at_values, [float(reference)]])
    frame = pl.DataFrame({variable: points})
    for name in others:
        dtype = data.get_column(name).dtype
        frame = frame.with_columns(pl.Series(name, [fixed[name]] * points.size).cast(dtype))
    new = build_design(frame, design)
    if new.x.shape[0] != points.size:
        raise ValueError("the adjust-to values produced missing rows")
    columns = [design.names.index(name) for name in fit.names]
    x = new.x[:, columns]
    diff = x[:-1] - x[-1]
    beta = np.asarray(fit.coefficients, dtype=float)
    cov = np.asarray(fit.covariance, dtype=float)
    estimate = diff @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov, diff))
    z = float(stats.norm.ppf(0.5 + level / 2))
    low = estimate - z * se
    high = estimate + z * se
    if exponentiate:
        estimate, low, high = np.exp(estimate), np.exp(low), np.exp(high)
    return pl.DataFrame(
        {
            variable: at_values,
            "estimate": estimate,
            "std_error": se,
            "conf_low": low,
            "conf_high": high,
        }
    )
