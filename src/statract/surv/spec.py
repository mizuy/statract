"""Resolve a ``Surv()`` formula into the arrays the survival estimators use."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..models.design import ColumnRef, Design, column_series
from ..models.formula import _columns_in, model_matrix


@dataclass
class SurvivalFormula:
    """A survival formula aligned to ``design.row_index``."""

    design: Design
    time: np.ndarray
    event: np.ndarray
    entry: np.ndarray
    strata: np.ndarray
    cluster: np.ndarray | None
    offset: np.ndarray
    time_name: str
    event_name: str
    entry_name: str | None
    strata_names: tuple[str, ...]
    cluster_name: str | None
    covariate_names: list[str]


def parse_survival_formula(
    data: pl.DataFrame,
    formula: str,
    *,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    strata: ColumnRef | None = None,
    cluster: ColumnRef | None = None,
    entry: ColumnRef | None = None,
    drop_intercept: bool,
    allow_counting: bool,
    allow_strata: bool,
    allow_cluster: bool,
) -> SurvivalFormula:
    """Expand ``Surv(time, status) ~ ...`` and align the extra columns."""
    built = model_matrix(formula, data)
    if built.surv is None:
        raise ValueError("the formula needs a Surv() response, for example Surv(time, status) ~ age + sex")
    if built.random_effects:
        raise ValueError("random effects belong in fit_mixed, for example (1 | group)")
    if built.surv.entry is not None and not allow_counting:
        raise ValueError("Surv(start, stop, status) belongs in cox_ph or fine_gray")
    if built.surv.entry is not None and entry is not None:
        raise ValueError("Surv(start, stop, status) already names the entry time")
    if built.strata and not allow_strata:
        raise ValueError("strata() belongs in cox_ph")
    if built.cluster is not None and not allow_cluster:
        raise ValueError("cluster() belongs in cox_ph")
    if built.strata and strata is not None:
        raise ValueError("pass strata() in the formula or strata=, not both")
    if built.cluster is not None and cluster is not None:
        raise ValueError("pass cluster() in the formula or cluster=, not both")

    design = built.design
    off = np.zeros(design.n_obs) if built.offset is None else np.asarray(built.offset, dtype=float)
    keep = np.ones(design.n_obs, dtype=bool)
    extra_rows: dict[str, np.ndarray] = {}
    for key, ref in (("weights", weights), ("offset", offset), ("strata", strata), ("cluster", cluster), ("entry", entry)):
        if ref is None:
            continue
        values = column_series(data, ref).gather(design.row_index.tolist())
        keep &= values.is_not_null().to_numpy()
        extra_rows[key] = values.to_numpy()
    if not bool(keep.all()):
        design = _subset_design(design, keep)
        off = off[keep]
        extra_rows = {key: values[keep] for key, values in extra_rows.items()}
    if offset is not None:
        off = off + np.asarray(extra_rows["offset"], dtype=float)
    if drop_intercept:
        design = _drop_intercept(design)

    index = design.row_index
    strata_names = built.strata
    if strata is not None:
        strata_labels = np.asarray(extra_rows["strata"])
        strata_names = (strata if isinstance(strata, str) else column_series(data, strata).name,)
    elif strata_names:
        strata_labels = _combine_strata(data, strata_names, index)
    else:
        strata_labels = np.array(["_"] * design.n_obs)
    cluster_labels = None
    cluster_name = built.cluster
    if cluster is not None:
        cluster_labels = np.asarray(extra_rows["cluster"])
        cluster_name = cluster if isinstance(cluster, str) else column_series(data, cluster).name
    elif cluster_name is not None:
        cluster_labels = np.asarray(column_series(data, cluster_name).gather(index.tolist()).to_numpy())
    entry_name = built.surv.entry
    if entry is not None:
        entry_values = np.asarray(extra_rows["entry"], dtype=float)
        entry_name = entry if isinstance(entry, str) else column_series(data, entry).name
    elif entry_name is not None:
        entry_values = np.asarray(column_series(data, entry_name).gather(index.tolist()).to_numpy(), dtype=float)
    else:
        entry_values = np.zeros(design.n_obs)
    return SurvivalFormula(
        design=design,
        time=np.asarray(column_series(data, built.surv.time).gather(index.tolist()).to_numpy(), dtype=float),
        event=np.asarray(column_series(data, built.surv.event).gather(index.tolist()).to_numpy()),
        entry=entry_values,
        strata=strata_labels,
        cluster=cluster_labels,
        offset=off,
        time_name=built.surv.time,
        event_name=built.surv.event,
        entry_name=entry_name,
        strata_names=tuple(strata_names),
        cluster_name=cluster_name,
        covariate_names=_covariate_names(design),
    )


def combine_strata(data: pl.DataFrame, names: tuple[str, ...], row_index: np.ndarray) -> np.ndarray:
    """One label per row. Several strata columns are joined in formula order."""
    return _combine_strata(data, names, row_index)


def _combine_strata(data: pl.DataFrame, names: tuple[str, ...] | list[str], row_index: np.ndarray) -> np.ndarray:
    columns = [column_series(data, name).gather(row_index.tolist()).to_list() for name in names]
    if len(columns) == 1:
        return np.asarray(columns[0])
    return np.asarray([", ".join(str(part) for part in parts) for parts in zip(*columns, strict=True)])


def _covariate_names(design: Design) -> list[str]:
    computed = design.computed or {}
    names: list[str] = []
    for recipe in design.recipes or ():
        for symbol, _level in recipe:
            sources = _columns_in(computed[symbol]) if symbol in computed else [symbol]
            for source in sources:
                if source not in names:
                    names.append(source)
    return names


def _subset_design(design: Design, keep: np.ndarray) -> Design:
    return Design(
        x=design.x[keep],
        names=list(design.names),
        row_index=design.row_index[keep],
        predictors=list(design.predictors),
        intercept=design.intercept,
        recipes=design.recipes,
        factor_levels=design.factor_levels,
        computed=design.computed,
    )


def _drop_intercept(design: Design) -> Design:
    if not design.intercept or "(Intercept)" not in design.names:
        return design
    index = [i for i, name in enumerate(design.names) if name != "(Intercept)"]
    x = design.x[:, index] if index else np.zeros((design.n_obs, 0))
    recipes = None if design.recipes is None else tuple(design.recipes[i] for i in index)
    return Design(
        x=x,
        names=[design.names[i] for i in index],
        row_index=design.row_index,
        predictors=list(design.predictors),
        intercept=False,
        recipes=recipes,
        factor_levels=design.factor_levels,
        computed=design.computed,
    )
