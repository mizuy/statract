"""Design matrices for column-wise models.

Column names follow R treatment contrasts: the intercept is ``(Intercept)`` and a
factor column ``stage`` with levels ``I``, ``II`` becomes ``stageII`` for the
non-reference level. ``pl.Enum`` keeps its definition order. Strings are sorted.
Booleans and integers stay numeric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

ColumnRef = str | pl.Expr


@dataclass
class Predictor:
    """One column in a design, after the reference level has been dropped."""

    name: str
    kind: str  # "numeric" or "factor"
    levels: tuple[str, ...] = ()
    reference: str | None = None


@dataclass
class Design:
    """Complete-case design matrix aligned to ``row_index`` of the input frame."""

    x: np.ndarray
    names: list[str]
    row_index: np.ndarray
    predictors: list[Predictor] = field(default_factory=list)
    intercept: bool = True
    # Wilkinson formulas store one recipe per output column. ``None`` keeps the
    # main-effect path used by ``design_matrix``.
    recipes: tuple[tuple[tuple[str, str | None], ...], ...] | None = None
    factor_levels: dict[str, tuple[str, ...]] | None = None
    computed: dict[str, Any] | None = None

    @property
    def n_obs(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.x.shape[1])


def column_series(data: pl.DataFrame, ref: ColumnRef) -> pl.Series:
    """Resolve a column name or expression to a series aligned with ``data``."""
    if isinstance(ref, str):
        if ref not in data.columns:
            raise KeyError(f"column {ref!r} is not in the frame")
        return data.get_column(ref)
    selected = data.select(ref)
    if selected.width != 1:
        raise ValueError("a column expression must produce exactly one column")
    return selected.to_series()


def _level_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _factor_levels(series: pl.Series, override: Sequence[object] | None) -> list[str]:
    if override is not None:
        return [_level_key(v) for v in override]
    dtype = series.dtype
    if isinstance(dtype, pl.Enum):
        return [_level_key(v) for v in dtype.categories.to_list()]
    if isinstance(dtype, pl.Categorical):
        cats = dtype.categories
        if cats is not None:
            return [_level_key(v) for v in cats.to_list()]
    values = series.drop_nulls().unique().to_list()
    return sorted(_level_key(v) for v in values)


def _is_factor(series: pl.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, (pl.Enum, pl.Categorical, pl.Utf8, pl.String)):
        return True
    if dtype == pl.Object:
        return True
    return False


def _factor_codes(series: pl.Series, levels: Sequence[str]) -> np.ndarray:
    """Map factor values to positions in ``levels``. Unknown values are -1."""
    n = series.len()
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if series.dtype == pl.Object:
        index = {level: i for i, level in enumerate(levels)}
        return np.fromiter(
            (index.get(_level_key(value), -1) for value in series.to_list()),
            dtype=np.int32,
            count=n,
        )
    text = series if series.dtype == pl.String else series.cast(pl.Utf8)
    # A Python map is cheaper than a columnar replace on a short column.
    # The replace pays off once the column is long enough to hide that call.
    if n < 2000:
        index = {level: i for i, level in enumerate(levels)}
        values = text.to_numpy()
        return np.fromiter((index.get(value, -1) for value in values), dtype=np.int32, count=n)
    return text.replace_strict(
        list(levels), list(range(len(levels))), default=-1, return_dtype=pl.Int32
    ).to_numpy()


def _unknown_labels(series: pl.Series, codes: np.ndarray) -> list[str]:
    if series.dtype == pl.Object:
        labels = [_level_key(value) for value, code in zip(series.to_list(), codes, strict=True) if code < 0]
    else:
        text = series if series.dtype == pl.String else series.cast(pl.Utf8)
        labels = text.filter(pl.Series(codes < 0)).to_list()
    return sorted(set(labels))


def _scatter_dummies(codes: np.ndarray, n_levels: int, out: np.ndarray) -> None:
    """Write treatment-contrast columns for every level after the reference."""
    out.fill(0.0)
    positive = codes > 0
    if np.any(positive):
        out[np.flatnonzero(positive), codes[positive] - 1] = 1.0
    if out.shape[1] != n_levels - 1:
        raise ValueError("dummy width does not match the number of non-reference levels")


def design_matrix(
    data: pl.DataFrame,
    predictors: Sequence[ColumnRef],
    *,
    intercept: bool = True,
    levels: Mapping[str, Sequence[object]] | None = None,
    extra: Sequence[ColumnRef] | None = None,
) -> Design:
    """Build a complete-case design matrix.

    Rows with a null in any predictor or ``extra`` column are dropped. The
    returned ``row_index`` holds the positions of the kept rows in ``data``.
    """
    if len(data) == 0:
        raise ValueError("data has no rows")
    if _plain_names(predictors, extra):
        return _design_columns(data, predictors, extra=extra, intercept=intercept, levels=levels)
    level_map = dict(levels or {})
    series_list: list[tuple[str, pl.Series]] = []
    for ref in predictors:
        series = column_series(data, ref)
        series_list.append((series.name, series))
    extra_series = [column_series(data, ref) for ref in (extra or ())]

    n = data.height
    keep = np.ones(n, dtype=bool)
    for _, series in series_list:
        keep &= series.is_not_null().to_numpy()
    for series in extra_series:
        keep &= series.is_not_null().to_numpy()

    # Resolve factor levels before allocating, so a rebuilt level list can
    # change the width. Codes stay as integers; the dummy block is one write.
    planned: list[tuple] = []
    n_cols = 1 if intercept else 0
    for name, series in series_list:
        kept = _take(series, keep)
        if not _is_factor(series):
            planned.append((name, np.asarray(kept.to_numpy(), dtype=float), None))
            n_cols += 1
            continue
        override = level_map.get(name)
        fac_levels = _factor_levels(series, override)
        if len(fac_levels) < 2:
            raise ValueError(f"factor {name!r} needs at least two levels after dropping nulls")
        codes = _factor_codes(kept, fac_levels)
        if np.any(codes < 0):
            if override is not None:
                unknown = _unknown_labels(kept, codes)
                raise ValueError(f"factor {name!r} has levels outside levels=: {unknown[:5]}")
            fac_levels = _factor_levels(series, None)
            if len(fac_levels) < 2:
                raise ValueError(f"factor {name!r} needs at least two levels after dropping nulls")
            codes = _factor_codes(kept, fac_levels)
        planned.append((name, codes, tuple(fac_levels)))
        n_cols += len(fac_levels) - 1

    if n_cols == 0:
        raise ValueError("the design has no columns")
    n_keep = int(keep.sum())
    x = np.empty((n_keep, n_cols), dtype=float)
    names: list[str] = []
    specs: list[Predictor] = []
    col = 0
    if intercept:
        x[:, 0] = 1.0
        names.append("(Intercept)")
        col = 1
    for name, values, fac_levels in planned:
        if fac_levels is None:
            specs.append(Predictor(name=name, kind="numeric"))
            x[:, col] = values
            names.append(name)
            col += 1
            continue
        reference = fac_levels[0]
        specs.append(Predictor(name=name, kind="factor", levels=fac_levels, reference=reference))
        width = len(fac_levels) - 1
        _scatter_dummies(values, len(fac_levels), x[:, col : col + width])
        for level in fac_levels[1:]:
            names.append(f"{name}{level}")
        col += width
    row_index = np.flatnonzero(keep).astype(np.int64)
    return Design(x=x, names=names, row_index=row_index, predictors=specs, intercept=intercept)


def build_design(
    data: pl.DataFrame,
    design: Design,
    *,
    extra: Sequence[ColumnRef] | None = None,
) -> Design:
    """Apply a fitted design's levels to new rows.

    Rows that are null in a used column are dropped. A factor level that was
    not present at fit time raises ``ValueError``.
    """
    if design.recipes is not None:
        from .formula import rebuild_formula_design

        return rebuild_formula_design(data, design, extra=extra)
    n = data.height
    keep = np.ones(n, dtype=bool)
    kept_series: list[pl.Series] = []
    for spec in design.predictors:
        series = column_series(data, spec.name)
        keep &= series.is_not_null().to_numpy()
        kept_series.append(series)
    for ref in extra or ():
        keep &= column_series(data, ref).is_not_null().to_numpy()

    n_keep = int(keep.sum())
    width = len(design.names)
    x = np.empty((n_keep, width), dtype=float) if width else np.zeros((n_keep, 0))
    col = 0
    if design.intercept:
        x[:, 0] = 1.0
        col = 1
    for spec, series in zip(design.predictors, kept_series, strict=True):
        kept = _take(series, keep)
        if spec.kind == "factor":
            codes = _factor_codes(kept, spec.levels)
            if np.any(codes < 0):
                unknown = _unknown_labels(kept, codes)
                raise ValueError(f"factor {spec.name!r} has unseen levels {unknown}")
            n_levels = len(spec.levels)
            block = n_levels - 1
            _scatter_dummies(codes, n_levels, x[:, col : col + block])
            col += block
        else:
            x[:, col] = np.asarray(kept.to_numpy(), dtype=float)
            col += 1
    return Design(
        x=x,
        names=list(design.names),
        row_index=np.flatnonzero(keep).astype(np.int64),
        predictors=list(design.predictors),
        intercept=design.intercept,
    )


def _plain_names(predictors: Sequence[ColumnRef], extra: Sequence[ColumnRef] | None) -> bool:
    return all(isinstance(ref, str) for ref in predictors) and all(isinstance(ref, str) for ref in (extra or ()))


def _is_factor_dtype(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Enum, pl.Categorical, pl.Utf8, pl.String)):
        return True
    return dtype == pl.Object


def _design_complete(
    data: pl.DataFrame,
    predictors: Sequence[str],
    *,
    num_names: list[str],
    fac_names: list[str],
    intercept: bool,
    level_map: dict[str, Sequence[object]],
) -> Design | None:
    """Design matrix when every used value is present. Null rows use the general path."""
    fac_levels: dict[str, list[str]] = {}
    for name in fac_names:
        fac_levels[name] = _factor_levels(data.get_column(name), level_map.get(name))
        if len(fac_levels[name]) < 2:
            return None
    exprs: list[pl.Expr] = [pl.col(name).cast(pl.Float64).alias(name) for name in num_names]
    exprs.extend(
        pl.col(name)
        .cast(pl.Utf8)
        .replace_strict(fac_levels[name], list(range(len(fac_levels[name]))), default=-1)
        .alias(name)
        for name in fac_names
    )
    block = data.select(exprs) if exprs else None
    numeric = None
    if num_names:
        columns = [block[name].to_numpy() for name in num_names]
        numeric = np.column_stack(columns) if len(columns) > 1 else columns[0].reshape(-1, 1)
    coded = None
    if fac_names:
        columns = [np.asarray(block[name].to_numpy(), dtype=np.int32) for name in fac_names]
        coded = np.column_stack(columns) if len(columns) > 1 else columns[0].reshape(-1, 1)
        if np.any(coded < 0):
            return None
    n_cols = (1 if intercept else 0) + len(num_names)
    for name in fac_names:
        n_cols += len(fac_levels[name]) - 1
    if n_cols == 0:
        return None
    n = data.height
    x = np.empty((n, n_cols), dtype=float)
    names: list[str] = []
    specs: list[Predictor] = []
    col = 0
    if intercept:
        x[:, 0] = 1.0
        names.append("(Intercept)")
        col = 1
    num_pos = {name: i for i, name in enumerate(num_names)}
    fac_pos = {name: i for i, name in enumerate(fac_names)}
    for name in predictors:
        if name in fac_levels:
            levels_now = fac_levels[name]
            codes = coded[:, fac_pos[name]]
            reference = levels_now[0]
            specs.append(Predictor(name=name, kind="factor", levels=tuple(levels_now), reference=reference))
            width = len(levels_now) - 1
            _scatter_dummies(codes, len(levels_now), x[:, col : col + width])
            for level in levels_now[1:]:
                names.append(f"{name}{level}")
            col += width
            continue
        specs.append(Predictor(name=name, kind="numeric"))
        x[:, col] = numeric[:, num_pos[name]]
        names.append(name)
        col += 1
    return Design(
        x=x,
        names=names,
        row_index=np.arange(n, dtype=np.int64),
        predictors=specs,
        intercept=intercept,
    )


def _design_columns(
    data: pl.DataFrame,
    predictors: Sequence[str],
    *,
    extra: Sequence[str] | None,
    intercept: bool,
    levels: Mapping[str, Sequence[object]] | None,
) -> Design:
    """One frame read for every named column, then dummy coding in NumPy."""
    level_map = dict(levels or {})
    needed = list(dict.fromkeys([*predictors, *(extra or ())]))
    if not needed:
        if not intercept:
            raise ValueError("the design has no columns")
        n = data.height
        return Design(
            x=np.ones((n, 1)),
            names=["(Intercept)"],
            row_index=np.arange(n, dtype=np.int64),
            predictors=[],
            intercept=True,
        )
    missing = [name for name in needed if name not in data.columns]
    if missing:
        raise KeyError(f"column {missing[0]!r} is not in the frame")
    schema = data.schema
    num_names = [name for name in predictors if not _is_factor_dtype(schema[name])]
    fac_names = [name for name in predictors if _is_factor_dtype(schema[name])]
    if all(data.get_column(name).null_count() == 0 for name in needed):
        complete = _design_complete(
            data,
            predictors,
            num_names=num_names,
            fac_names=fac_names,
            intercept=intercept,
            level_map=level_map,
        )
        if complete is not None:
            return complete
    fac_levels: dict[str, list[str]] = {}
    for name in fac_names:
        fac_levels[name] = _factor_levels(data.get_column(name), level_map.get(name))
        if len(fac_levels[name]) < 2:
            raise ValueError(f"factor {name!r} needs at least two levels after dropping nulls")
    # One collect: null flags, numeric values, and treatment codes.
    exprs: list[pl.Expr] = [pl.col(name).is_not_null().alias(f"__k_{name}") for name in needed]
    exprs.extend(pl.col(name).cast(pl.Float64).alias(f"__n_{name}") for name in num_names)
    exprs.extend(
        pl.col(name)
        .cast(pl.Utf8)
        .replace_strict(fac_levels[name], list(range(len(fac_levels[name]))), default=-1)
        .fill_null(-1)
        .alias(f"__c_{name}")
        for name in fac_names
    )
    block = data.select(exprs)
    flag_cols = [block[f"__k_{name}"].to_numpy() for name in needed]
    flags = np.column_stack(flag_cols) if len(flag_cols) > 1 else flag_cols[0].reshape(-1, 1)
    keep = np.ones(block.height, dtype=bool)
    for column in flags.T:
        keep &= column.astype(bool, copy=False)
    num_kept = None
    if num_names:
        numeric_cols = [block[f"__n_{name}"].to_numpy() for name in num_names]
        numeric = np.column_stack(numeric_cols) if len(numeric_cols) > 1 else numeric_cols[0].reshape(-1, 1)
        num_kept = numeric if bool(keep.all()) else numeric[keep]
        num_pos = {name: i for i, name in enumerate(num_names)}
    fac_kept = None
    if fac_names:
        coded_cols = [np.asarray(block[f"__c_{name}"].to_numpy(), dtype=np.int32) for name in fac_names]
        coded = np.column_stack(coded_cols) if len(coded_cols) > 1 else coded_cols[0].reshape(-1, 1)
        fac_kept = coded if bool(keep.all()) else coded[keep]
    n_cols = (1 if intercept else 0) + len(num_names)
    for name in fac_names:
        n_cols += len(fac_levels[name]) - 1
    if n_cols == 0:
        raise ValueError("the design has no columns")
    n_keep = int(keep.sum())
    x = np.empty((n_keep, n_cols), dtype=float)
    names: list[str] = []
    specs: list[Predictor] = []
    col = 0
    if intercept:
        x[:, 0] = 1.0
        names.append("(Intercept)")
        col = 1
    for name in predictors:
        if name in fac_levels:
            levels_now = fac_levels[name]
            j = fac_names.index(name)
            codes = np.asarray(fac_kept[:, j], dtype=np.int32)
            if np.any(codes < 0):
                kept = data.get_column(name).filter(pl.Series(keep))
                if name in level_map:
                    unknown = _unknown_labels(kept, codes)
                    raise ValueError(f"factor {name!r} has levels outside levels=: {unknown[:5]}")
                levels_now = _factor_levels(data.get_column(name), None)
                fac_levels[name] = levels_now
                if len(levels_now) < 2:
                    raise ValueError(f"factor {name!r} needs at least two levels after dropping nulls")
                codes = _factor_codes(kept, levels_now)
            reference = levels_now[0]
            specs.append(Predictor(name=name, kind="factor", levels=tuple(levels_now), reference=reference))
            width = len(levels_now) - 1
            _scatter_dummies(codes, len(levels_now), x[:, col : col + width])
            for level in levels_now[1:]:
                names.append(f"{name}{level}")
            col += width
            continue
        specs.append(Predictor(name=name, kind="numeric"))
        x[:, col] = num_kept[:, num_pos[name]]
        names.append(name)
        col += 1
    return Design(
        x=x,
        names=names,
        row_index=np.flatnonzero(keep).astype(np.int64),
        predictors=specs,
        intercept=intercept,
    )


def _take(series: pl.Series, keep: np.ndarray) -> pl.Series:
    return series.filter(pl.Series(keep))


def formula_model_matrix(formula: str, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-effects matrix for a Wilkinson formula. See ``statract.models.formula``."""
    from .formula import formula_model_matrix as _expand

    return _expand(formula, data)
