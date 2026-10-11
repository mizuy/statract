"""Categorical responses for the ordinal and multinomial models.

The right-hand side goes through ``model_matrix``. The response is coded
0..K-1 in level order before the design is built, so rows with a missing
response drop with the other incomplete rows.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from .design import ColumnRef, Design, _level_key, column_series
from .formula import model_matrix, reject_survival_syntax

_CODE = "__statract_response_code__"


@dataclass
class CategoricalData:
    design: Design
    codes: np.ndarray
    levels: list[str]
    weights: np.ndarray
    offset: np.ndarray
    response: str


def response_levels(series: pl.Series, levels: Sequence[object] | None) -> list[str]:
    """Level order: ``levels=``, then Enum / Categorical order, then sorted values."""
    if levels is not None:
        return [_level_key(v) for v in levels]
    dtype = series.dtype
    if isinstance(dtype, pl.Enum):
        return [_level_key(v) for v in dtype.categories.to_list()]
    if dtype == pl.Boolean:
        return ["FALSE", "TRUE"]
    values = series.drop_nulls().unique().to_list()
    if dtype.is_numeric():
        return [_level_key(v) for v in sorted(values)]
    return sorted(_level_key(v) for v in values)


def _keys(series: pl.Series) -> list[str | None]:
    if series.dtype == pl.Boolean:
        return [None if v is None else ("TRUE" if v else "FALSE") for v in series.to_list()]
    return [None if v is None else _level_key(v) for v in series.to_list()]


def categorical_data(
    data: pl.DataFrame,
    formula: str,
    *,
    weights: ColumnRef | None,
    levels: Sequence[object] | None,
    drop_intercept: bool,
    min_levels: int,
) -> CategoricalData:
    """Design, response codes, level labels, weights, and offset for a categorical response."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("formula must be a Wilkinson formula such as 'y ~ x + stage'")
    lhs, rhs = formula.split("~", 1)
    response = lhs.strip()
    if response.startswith("`") and response.endswith("`"):
        response = response[1:-1]
    if response not in data.columns:
        raise ValueError(f"the response must be a column of the frame, got {lhs.strip()!r}")
    series = data.get_column(response)
    order = response_levels(series, levels)
    index = {level: i for i, level in enumerate(order)}
    keys = _keys(series)
    unknown = sorted({k for k in keys if k is not None and k not in index})
    if unknown:
        raise ValueError(f"response values {unknown} are not in levels")
    codes = pl.Series(_CODE, [None if k is None else index[k] for k in keys], dtype=pl.Int64)
    frame = data.drop(response).with_columns(codes)
    built = model_matrix(f"{_CODE} ~ {rhs}", frame)
    if built.random_effects:
        raise ValueError("random effects are not supported here")
    reject_survival_syntax(built)
    design = built.design
    y = np.asarray(built.y).astype(np.int64)
    offset = np.zeros(design.n_obs) if built.offset is None else np.asarray(built.offset, dtype=float)
    w = np.ones(design.n_obs)
    if weights is not None:
        values = column_series(data, weights).gather(design.row_index.tolist())
        keep = values.is_not_null().to_numpy()
        w = np.asarray(values.to_numpy(), dtype=float)
        if not bool(keep.all()):
            design = Design(
                x=design.x[keep],
                names=list(design.names),
                row_index=design.row_index[keep],
                predictors=list(design.predictors),
                intercept=design.intercept,
                recipes=design.recipes,
                factor_levels=design.factor_levels,
                computed=design.computed,
            )
            y = y[keep]
            offset = offset[keep]
            w = w[keep]
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
    if drop_intercept and "(Intercept)" in design.names:
        j = design.names.index("(Intercept)")
        keep_cols = [i for i in range(len(design.names)) if i != j]
        recipes = None if design.recipes is None else tuple(design.recipes[i] for i in keep_cols)
        design = Design(
            x=design.x[:, keep_cols],
            names=[design.names[i] for i in keep_cols],
            row_index=design.row_index,
            predictors=list(design.predictors),
            intercept=False,
            recipes=recipes,
            factor_levels=design.factor_levels,
            computed=design.computed,
        )
    counts = np.bincount(y[w > 0], minlength=len(order))
    if np.any(counts == 0):
        empty = [order[i] for i in np.flatnonzero(counts == 0)]
        warnings.warn(f"response levels {empty} are empty and were dropped", UserWarning, stacklevel=3)
        used = np.flatnonzero(counts > 0)
        remap = np.full(len(order), -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        y = remap[y]
        order = [order[i] for i in used]
        if np.any(y < 0):
            keep = y >= 0
            design = Design(
                x=design.x[keep],
                names=list(design.names),
                row_index=design.row_index[keep],
                predictors=list(design.predictors),
                intercept=design.intercept,
                recipes=design.recipes,
                factor_levels=design.factor_levels,
                computed=design.computed,
            )
            y, offset, w = y[keep], offset[keep], w[keep]
    if len(order) < min_levels:
        raise ValueError(f"the response needs at least {min_levels} observed levels")
    return CategoricalData(design=design, codes=y, levels=order, weights=w, offset=offset, response=response)
