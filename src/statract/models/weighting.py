"""Inverse probability of treatment weighting for a binary treatment.

``propensity_weights`` follows ``WeightIt::weightit(method = "glm")``. The
propensity score is a logistic regression fitted with ``fit_glm``. The weights
for each estimand, ``stabilize=TRUE``, and ``WeightIt::trim`` follow WeightIt
1.x. The balance table and the effective sample sizes follow ``cobalt::bal.tab``
4.x on the weighted sample. The figure is ``statract.viz.balance.plot_love``.

Outcome models are not here. Pass the ``weights`` column of ``frame()`` to
``fit_glm(..., weights=)`` with ``hc_covariance(fit, "HC0")``, or to
``cox_ph(..., weights=)``, which uses the robust variance for such weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .design import ColumnRef, Design, column_series
from .fit import Fit, fit_glm
from .formula import model_matrix

_ESTIMANDS = {"ATE", "ATT", "ATC", "ATO"}
_BINARY = {"raw", "std"}
_SD_DENOMS = {"pooled", "treated", "control", "all", "weighted"}


@dataclass
class PropensityWeights:
    """Propensity scores and weights. Arrays are aligned with the input rows.

    Rows dropped for a missing value in the formula (or the sampling weight)
    have ``nan`` in ``ps`` and ``weights``. ``fit`` is the propensity model.
    """

    data: pl.DataFrame
    formula: str
    treatment: str
    estimand: str
    ps: np.ndarray
    weights: np.ndarray
    sampling_weights: np.ndarray
    stabilized: bool
    trim: Any
    fit: Fit = field(repr=False)

    @property
    def row_index(self) -> np.ndarray:
        """Positions of the rows that have a weight."""
        return np.asarray(self.fit.row_index, dtype=int)

    def frame(self, *, ps: str = "ps", weights: str = "weights") -> pl.DataFrame:
        """Input rows with the propensity score and the weight. Dropped rows are null."""
        return self.data.with_columns(
            pl.Series(ps, self.ps).fill_nan(None),
            pl.Series(weights, self.weights).fill_nan(None),
        )

    def summary(self) -> pl.DataFrame:
        """Per arm: units, weight range and mean, and the effective sample size."""
        idx = self.row_index
        treat = self.fit.y > 0
        w = self.weights[idx] * self.sampling_weights[idx]
        rows = []
        for label, mask in (("control", ~treat), ("treated", treat)):
            part = w[mask]
            rows.append(
                {
                    "group": label,
                    "n": int(mask.sum()),
                    "weight_min": float(part.min()),
                    "weight_mean": float(part.mean()),
                    "weight_max": float(part.max()),
                    "ess": _ess(part),
                }
            )
        return pl.DataFrame(rows)

    def effective_sample_size(self) -> pl.DataFrame:
        """Kish effective sample size, ``(sum w)^2 / sum w^2``, as ``bal.tab`` prints it."""
        idx = self.row_index
        return _ess_table(self.fit.y > 0, self.sampling_weights[idx], self.weights[idx] * self.sampling_weights[idx])

    def balance(
        self,
        *,
        binary: str = "raw",
        continuous: str = "std",
        sd_denominator: str | None = None,
        threshold: float | None = None,
        distance: bool = True,
    ) -> pl.DataFrame:
        """Balance before and after weighting, as ``cobalt::bal.tab``.

        See ``balance_table`` for the columns. The first row is ``prop.score``.
        """
        idx = self.row_index
        return _balance_from_design(
            self.fit.design,
            self.fit.y > 0,
            self.weights[idx],
            self.sampling_weights[idx],
            estimand=self.estimand,
            binary=binary,
            continuous=continuous,
            sd_denominator=sd_denominator,
            threshold=threshold,
            distance=self.ps[idx] if distance else None,
        )

    def love_plot(self, path: Path | str | None = None, **kwargs: Any):
        """``plot_love`` of ``balance()``. Keyword arguments go to ``plot_love``."""
        from ..viz.balance import plot_love

        return plot_love(self.balance(), path, **kwargs)


def propensity_weights(
    data: pl.DataFrame,
    formula: str,
    *,
    estimand: str = "ATE",
    stabilize: bool = False,
    trim: float | tuple[float, float] | None = None,
    trim_lower: bool = False,
    sampling_weights: ColumnRef | None = None,
) -> PropensityWeights:
    """Propensity score weights for a binary treatment, as ``WeightIt::weightit(method="glm")``.

    The formula is ``"treat ~ age + sex + ..."``. The treatment is 0/1 or
    boolean, and 1 is the treated arm. With ``e`` the fitted propensity score:

    - ``ATE``: treated ``1/e``, control ``1/(1-e)``
    - ``ATT``: treated 1, control ``e/(1-e)``
    - ``ATC``: treated ``(1-e)/e``, control 1
    - ``ATO``: treated ``1-e``, control ``e`` (overlap weights)

    ``stabilize=True`` multiplies each weight by the share of the unit's own
    arm, weighted by ``sampling_weights``. As in WeightIt it is for ``ATE`` only.

    ``trim`` follows ``WeightIt::trim``. A number below 1 is a quantile: weights
    above it are set to it (``0.9`` and ``0.1`` mean the same). A whole number
    of 1 or more is a count: that many largest weights are set to the next
    largest. ``trim_lower=True`` also raises the lowest weights the same way. A
    pair ``(low, high)`` sets fixed bounds; this one is not in WeightIt. ``ATT``
    leaves the treated weights at 1 and ``ATC`` the control weights. Trimming is
    applied after stabilization.

    ``sampling_weights`` enter the propensity model as prior weights (WeightIt's
    ``s.weights``). They are kept apart from ``weights``, so an outcome model
    needs their product.
    """
    if estimand not in _ESTIMANDS:
        raise ValueError(f"estimand must be one of {sorted(_ESTIMANDS)}")
    if stabilize and estimand != "ATE":
        raise ValueError("stabilize is available for estimand='ATE' only, as in WeightIt")
    name = _response_name(formula)
    if name in data.columns:
        observed = data[name].drop_nulls()
        if observed.dtype != pl.Boolean and not observed.is_in([0, 1]).all():
            raise ValueError("the treatment must be 0/1 or boolean")
    fit = fit_glm(data, formula, family="binomial", weights=sampling_weights)
    y = np.asarray(fit.y, dtype=float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("the treatment must be 0/1 or boolean")
    treat = y > 0
    if treat.all() or not treat.any():
        raise ValueError("the treatment needs both arms")
    s = np.ones(len(y)) if sampling_weights is None else np.asarray(fit.weights, dtype=float)
    e = fit.predict(kind="response")
    w = _weights_from_ps(e, treat, estimand)
    if stabilize:
        p1 = float(np.sum(s * treat) / np.sum(s))
        w = w * np.where(treat, p1, 1.0 - p1)
    if trim is not None:
        w = _trim(w, treat, estimand, trim, lower=trim_lower)
    n = data.height
    idx = np.asarray(fit.row_index, dtype=int)
    ps_full = np.full(n, np.nan)
    ps_full[idx] = e
    w_full = np.full(n, np.nan)
    w_full[idx] = w
    s_full = np.full(n, np.nan)
    s_full[idx] = s
    return PropensityWeights(
        data=data,
        formula=formula,
        treatment=_response_name(formula),
        estimand=estimand,
        ps=ps_full,
        weights=w_full,
        sampling_weights=s_full,
        stabilized=stabilize,
        trim=trim,
        fit=fit,
    )


def balance_table(
    data: pl.DataFrame,
    formula: str,
    *,
    weights: ColumnRef | np.ndarray,
    estimand: str = "ATE",
    sampling_weights: ColumnRef | np.ndarray | None = None,
    binary: str = "raw",
    continuous: str = "std",
    sd_denominator: str | None = None,
    threshold: float | None = None,
    distance: ColumnRef | np.ndarray | None = None,
) -> pl.DataFrame:
    """Covariate balance under any weights, as ``cobalt::bal.tab``.

    The covariates are the right-hand side of ``formula``. A factor with three
    or more levels shows every level as ``name_level``. A two-level factor shows
    its second level. A column with two distinct values is binary.

    ``diff_*`` is the treated mean minus the control mean. For a continuous
    covariate it is divided by the standard deviation of the sample without
    the balancing weights (sampling weights kept). ``sd_denominator`` defaults
    from the estimand: ``ATT`` treated, ``ATC`` control, ``ATO`` weighted (the
    whole sample under the balancing weights, as cobalt), otherwise ``pooled``,
    ``sqrt((s1^2 + s0^2) / 2)``. A binary covariate shows the raw difference in
    proportions unless ``binary="std"``; its variance is then ``p(1-p)``.
    ``variance_ratio_*`` is treated over control with cobalt's weighted
    variance, for continuous covariates only. ``distance`` adds a first row,
    ``prop.score``, that is always standardized.

    ``threshold`` adds ``balanced``: ``|diff_adjusted| < threshold``.
    """
    if estimand not in _ESTIMANDS:
        raise ValueError(f"estimand must be one of {sorted(_ESTIMANDS)}")
    built = model_matrix(formula, data)
    design = built.design
    idx = np.asarray(design.row_index, dtype=int)
    y = np.asarray(built.y, dtype=float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("the treatment must be 0/1 or boolean")
    w = _aligned(data, weights, idx)
    s = np.ones(len(idx)) if sampling_weights is None else _aligned(data, sampling_weights, idx)
    dist = None if distance is None else _aligned(data, distance, idx)
    keep = np.isfinite(w) & np.isfinite(s)
    if dist is not None:
        keep &= np.isfinite(dist)
    if not keep.all():
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
        y, w, s = y[keep], w[keep], s[keep]
        dist = None if dist is None else dist[keep]
    return _balance_from_design(
        design,
        y > 0,
        w,
        s,
        estimand=estimand,
        binary=binary,
        continuous=continuous,
        sd_denominator=sd_denominator,
        threshold=threshold,
        distance=dist,
    )


# --- weights ---------------------------------------------------------------


def _weights_from_ps(e: np.ndarray, treat: np.ndarray, estimand: str) -> np.ndarray:
    if estimand == "ATE":
        return np.where(treat, 1.0 / e, 1.0 / (1.0 - e))
    if estimand == "ATT":
        return np.where(treat, 1.0, e / (1.0 - e))
    if estimand == "ATC":
        return np.where(treat, (1.0 - e) / e, 1.0)
    return np.where(treat, 1.0 - e, e)


def _trim(w: np.ndarray, treat: np.ndarray, estimand: str, at, *, lower: bool) -> np.ndarray:
    """``WeightIt::trim``. ATT and ATC leave the focal arm alone."""
    out = np.array(w, dtype=float, copy=True)
    if estimand == "ATT":
        mask = ~treat
    elif estimand == "ATC":
        mask = treat
    else:
        mask = np.ones(len(w), dtype=bool)
    values = out[mask]
    if isinstance(at, tuple | list):
        if len(at) != 2:
            raise ValueError("trim bounds must be a pair (low, high)")
        lo, hi = float(at[0]), float(at[1])
        if not lo <= hi:
            raise ValueError("trim bounds need low <= high")
        out[mask] = np.clip(values, lo, hi)
        return out
    at = float(at)
    if not np.isfinite(at) or at <= 0:
        raise ValueError("trim must be a positive number or a (low, high) pair")
    if at < 1:
        q = at if at >= 0.5 else 1.0 - at
        # WeightIt uses quantile(type = 3), numpy's "closest_observation".
        top = float(np.quantile(values, q, method="closest_observation"))
        bottom = float(np.quantile(values, 1.0 - q, method="closest_observation")) if lower else -np.inf
    else:
        if not float(at).is_integer():
            raise ValueError("a trim count must be a whole number")
        k = int(at)
        if k >= values.size:
            raise ValueError("trim count must be smaller than the number of weights it trims")
        ordered = np.sort(values)
        top = float(ordered[::-1][k])
        bottom = float(ordered[k]) if lower else -np.inf
    out[mask] = np.clip(values, bottom, top)
    return out


def _ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    den = float(np.sum(w * w))
    return float(np.sum(w) ** 2 / den) if den > 0 else float("nan")


def _ess_table(treat: np.ndarray, s: np.ndarray, w: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample": ["unadjusted", "adjusted"],
            "control": [_ess(s[~treat]), _ess(w[~treat])],
            "treated": [_ess(s[treat]), _ess(w[treat])],
        }
    )


# --- balance ---------------------------------------------------------------


def _cobalt_covariates(design: Design) -> tuple[list[str], np.ndarray]:
    """Covariates as cobalt splits a formula: every level of a factor with 3+ levels."""
    x = design.x
    names = list(design.names)
    recipes = design.recipes
    if recipes is None:
        keep = [j for j, name in enumerate(names) if name != "(Intercept)"]
        return [names[j] for j in keep], x[:, keep]
    levels = design.factor_levels or {}
    out_names: list[str] = []
    cols: list[np.ndarray] = []
    done: set[str] = set()
    for j, recipe in enumerate(recipes):
        if len(recipe) == 0:
            continue
        if len(recipe) == 1 and recipe[0][1] is not None:
            symbol = recipe[0][0]
            if symbol in done:
                continue
            done.add(symbol)
            columns = {
                rec[0][1]: k
                for k, rec in enumerate(recipes)
                if len(rec) == 1 and rec[0][0] == symbol and rec[0][1] is not None
            }
            all_levels = list(levels.get(symbol, tuple(columns)))
            if len(all_levels) <= 2:
                for level in all_levels:
                    if level in columns:
                        out_names.append(f"{symbol}_{level}")
                        cols.append(x[:, columns[level]])
                continue
            present = np.column_stack([x[:, k] for k in columns.values()])
            for level in all_levels:
                if level in columns:
                    col = x[:, columns[level]]
                else:
                    col = 1.0 - present.sum(axis=1)
                out_names.append(f"{symbol}_{level}")
                cols.append(col)
            continue
        out_names.append(names[j])
        cols.append(x[:, j])
    if not cols:
        raise ValueError("the formula has no covariates")
    return out_names, np.column_stack(cols)


def _binarize(col: np.ndarray) -> np.ndarray:
    """cobalt's ``binarize``: 0 stays 0, otherwise the larger value is 1."""
    values = np.unique(col)
    if 0 in values:
        return (col != 0).astype(float)
    return (col == values.max()).astype(float)


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _wvar(x: np.ndarray, w: np.ndarray, binary: bool) -> float:
    """``cobalt::col_w_sd`` squared: reliability weights, or ``p(1-p)`` for binary."""
    w = w / np.sum(w)
    m = float(np.sum(w * x))
    if binary:
        return m * (1.0 - m)
    den = 1.0 - float(np.sum(w * w))
    if den <= 0:
        return float("nan")
    return float(np.sum(w * (x - m) ** 2) / den)


def _sd_denominator(x, treat, s, binary, how, w) -> float:
    if how == "weighted":
        var = _wvar(x, w * s, binary)
    elif how == "treated":
        var = _wvar(x[treat], s[treat], binary)
    elif how == "control":
        var = _wvar(x[~treat], s[~treat], binary)
    elif how == "all":
        var = _wvar(x, s, binary)
    else:
        var = (_wvar(x[treat], s[treat], binary) + _wvar(x[~treat], s[~treat], binary)) / 2.0
    return float(np.sqrt(var))


def _balance_from_design(
    design: Design,
    treat: np.ndarray,
    w: np.ndarray,
    s: np.ndarray,
    *,
    estimand: str,
    binary: str,
    continuous: str,
    sd_denominator: str | None,
    threshold: float | None,
    distance: np.ndarray | None,
) -> pl.DataFrame:
    if binary not in _BINARY:
        raise ValueError("binary must be 'raw' or 'std'")
    if continuous not in _BINARY:
        raise ValueError("continuous must be 'raw' or 'std'")
    if sd_denominator is None:
        sd_denominator = {"ATT": "treated", "ATC": "control", "ATO": "weighted"}.get(estimand, "pooled")
    if sd_denominator not in _SD_DENOMS:
        raise ValueError(f"sd_denominator must be one of {sorted(_SD_DENOMS)}")
    treat = np.asarray(treat, dtype=bool)
    w = np.asarray(w, dtype=float)
    s = np.asarray(s, dtype=float)
    names, x = _cobalt_covariates(design)
    entries: list[tuple[str, str, np.ndarray]] = []
    if distance is not None:
        entries.append(("prop.score", "Distance", np.asarray(distance, dtype=float)))
    for name, col in zip(names, x.T, strict=True):
        is_binary = np.unique(col).size == 2
        if is_binary:
            entries.append((name, "Binary", _binarize(col)))
        else:
            entries.append((name, "Contin.", col))
    adj = w * s
    rows = []
    for name, kind, col in entries:
        is_binary = kind == "Binary"
        standardize = kind == "Distance" or (binary == "std" if is_binary else continuous == "std")
        denom = _sd_denominator(col, treat, s, is_binary, sd_denominator, w) if standardize else 1.0
        row: dict[str, Any] = {"term": name, "type": kind}
        for label, weights in (("unadjusted", s), ("adjusted", adj)):
            mt = _wmean(col[treat], weights[treat])
            mc = _wmean(col[~treat], weights[~treat])
            diff = mt - mc
            if standardize:
                diff = diff / denom if denom > 0 else float("nan")
            if is_binary:
                ratio = None
            else:
                vt = _wvar(col[treat], weights[treat], False)
                vc = _wvar(col[~treat], weights[~treat], False)
                ratio = vt / vc if vc > 0 else float("nan")
            row[f"mean_treated_{label}"] = mt
            row[f"mean_control_{label}"] = mc
            row[f"diff_{label}"] = diff
            row[f"variance_ratio_{label}"] = ratio
        rows.append(row)
    schema = {"term": pl.String, "type": pl.String}
    for label in ("unadjusted", "adjusted"):
        for key in ("mean_treated", "mean_control", "diff", "variance_ratio"):
            schema[f"{key}_{label}"] = pl.Float64
    table = pl.DataFrame(rows, schema=schema)
    if threshold is not None:
        table = table.with_columns((pl.col("diff_adjusted").abs() < float(threshold)).alias("balanced"))
    return table


# --- helpers ---------------------------------------------------------------


def _aligned(data: pl.DataFrame, ref: ColumnRef | np.ndarray, idx: np.ndarray) -> np.ndarray:
    if isinstance(ref, np.ndarray | pl.Series | list):
        values = np.asarray(ref, dtype=float)
        if values.shape[0] != data.height:
            raise ValueError("a weight vector must have one value per row of data")
        return values[idx]
    series = column_series(data, ref).gather(idx.tolist())
    return np.asarray(series.cast(pl.Float64).fill_null(np.nan).to_numpy(), dtype=float)


def _response_name(formula: str) -> str:
    return formula.split("~", 1)[0].strip()


__all__ = ["PropensityWeights", "balance_table", "propensity_weights"]
