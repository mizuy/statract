"""Regression standardization (g-computation) after a GLM.

``standardize_glm`` fits a GLM, sets the exposure to each value for every row,
and averages the predicted means. The averages are the standardized risks (or
rates, or means). ``tidy`` turns them into differences, ratios, or odds ratios.

Two variances are available.

``covariates="fixed"`` (default) treats the covariate rows as fixed and uses
the delta method with the coefficient covariance. This is
``marginaleffects::avg_predictions`` and ``avg_comparisons`` with
``type = "response"``. ``vcov`` picks the coefficient covariance: the model
one, ``HC0``–``HC3``, or a cluster sandwich (``cluster=``).

``covariates="sampled"`` also counts the sampling of the covariates. It stacks
the GLM score and the averaging equations and uses one sandwich, as
``stdReg2::standardize_glm`` does (Sjölander 2016).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

from .covariance import cluster_covariance, hc_covariance
from .design import build_design
from .fit import Fit, _link_from_eta, fit_glm
from .formula import _columns_in

_CONTRASTS = {"difference", "ratio", "odds_ratio"}
_VCOVS = {"model", "HC0", "HC1", "HC2", "HC3"}
_COVARIATES = {"fixed", "sampled"}


@dataclass
class StandardizedGLM:
    """Standardized means, one per exposure value.

    ``estimates`` holds the averaged predictions in the order of ``values``.
    ``covariance`` is their covariance.
    """

    exposure: str
    values: list[object]
    estimates: np.ndarray
    covariance: np.ndarray
    n_obs: int
    family: str
    covariates: str
    vcov: str
    fit: Fit = field(repr=False)

    def tidy(
        self,
        *,
        contrast: str | None = None,
        reference: object = None,
        ci: str = "plain",
        level: float = 0.95,
    ) -> pl.DataFrame:
        """Table of the means, or of contrasts against ``reference``.

        ``contrast`` is ``"difference"``, ``"ratio"``, or ``"odds_ratio"``.
        The reference row has a difference of 0 or a ratio of 1 and no error.
        ``ci="log"`` builds the interval on the log scale (means and ratios).
        """
        if contrast is not None and contrast not in _CONTRASTS:
            raise ValueError(
                "contrast must be None, 'difference', 'ratio', or 'odds_ratio'"
            )
        if ci not in {"plain", "log"}:
            raise ValueError("ci must be 'plain' or 'log'")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        if contrast is not None and reference is None:
            raise ValueError("a contrast needs a reference value")
        if contrast is None and reference is not None:
            raise ValueError("reference is used only with a contrast")
        if contrast == "difference" and ci == "log":
            raise ValueError(
                "ci='log' does not fit a difference: the reference row is 0"
            )
        est = self.estimates.astype(float).copy()
        cov = self.covariance.astype(float).copy()
        k = len(self.values)
        if contrast is not None:
            matches = [i for i, value in enumerate(self.values) if value == reference]
            if not matches:
                raise ValueError("reference must be one of the values")
            ref = matches[0]
            if contrast == "odds_ratio":
                if not bool(np.all((est > 0) & (est < 1))):
                    raise ValueError("an odds ratio needs means between 0 and 1")
                cov = np.diag(1 / (1 - est) ** 2) @ cov @ np.diag(1 / (1 - est) ** 2)
                est = est / (1 - est)
            if contrast == "difference":
                jac = np.eye(k)
                jac[:, ref] -= 1
                est = est - est[ref]
            else:
                jac = np.diag(np.full(k, 1 / est[ref]))
                jac[:, ref] -= est / est[ref] ** 2
                est = est / est[ref]
            cov = jac @ cov @ jac.T
            cov[ref, :] = 0
            cov[:, ref] = 0
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        z = float(stats.norm.ppf(0.5 + level / 2))
        if ci == "plain":
            low, high = est - z * se, est + z * se
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                low, high = est * np.exp(-z * se / est), est * np.exp(z * se / est)
        return pl.DataFrame(
            {
                self.exposure: self.values,
                "estimate": est,
                "std_error": se,
                "conf_low": low,
                "conf_high": high,
            },
            strict=False,
        )


def standardize_glm(
    data: pl.DataFrame,
    formula: str,
    *,
    values: dict[str, Sequence[object]],
    family: str = "binomial",
    weights: str | None = None,
    vcov: str | None = None,
    cluster: str | None = None,
    covariates: str = "fixed",
) -> StandardizedGLM:
    """Standardize a GLM over the sample (g-computation).

    ``values`` names one exposure column and the values to set it to, for
    example ``{"trt": [0, 1]}``. Each row's mean is predicted with the exposure
    set to each value, and the predictions are averaged. Terms that use the
    exposure, such as ``trt:age``, are rebuilt. ``weights`` are prior weights of
    the GLM and also weight the average.

    ``covariates="fixed"`` gives the delta-method variance given the covariate
    rows, as ``marginaleffects``. ``vcov`` is ``"model"`` (default), ``"HC0"``,
    ``"HC1"``, ``"HC2"``, or ``"HC3"``. With ``cluster`` the coefficient
    covariance is ``vcovCL`` (default ``HC0``). ``covariates="sampled"`` uses
    the stacked sandwich of ``stdReg2::standardize_glm``; ``vcov`` must stay
    unset there, and ``cluster`` sums the rows of each cluster.
    """
    if covariates not in _COVARIATES:
        raise ValueError("covariates must be 'fixed' or 'sampled'")
    if vcov is not None and vcov not in _VCOVS:
        raise ValueError("vcov must be 'model', 'HC0', 'HC1', 'HC2', or 'HC3'")
    if covariates == "sampled" and vcov is not None:
        raise ValueError("covariates='sampled' uses its own sandwich; leave vcov unset")
    if cluster is not None and vcov == "model":
        raise ValueError("cluster needs a sandwich; use vcov='HC0' or leave vcov unset")
    if not isinstance(values, dict) or len(values) != 1:
        raise ValueError(
            "values must name exactly one exposure, for example {'trt': [0, 1]}"
        )
    exposure, levels = next(iter(values.items()))
    if exposure not in data.columns:
        raise KeyError(f"column {exposure!r} is not in the frame")
    levels = list(levels)
    if not levels:
        raise ValueError("values needs at least one value")
    fit = fit_glm(data, formula, family=family, weights=weights)
    if not _uses(fit, exposure):
        raise ValueError(f"the exposure {exposure!r} is not in the formula")
    used = data[fit.row_index.tolist()]
    n = fit.n_obs
    w = np.asarray(fit.weights, dtype=float)
    labels = None
    if cluster is not None:
        series = used.get_column(cluster)
        if series.null_count():
            raise ValueError(
                f"cluster column {cluster!r} has missing values in the rows the model uses"
            )
        labels = series.to_numpy()

    beta = fit.coefficients
    k = len(levels)
    means = np.zeros((n, k))
    # Derivative of each row's mean with respect to beta, per value.
    slopes = []
    for j, level in enumerate(levels):
        frame = used.with_columns(_constant(used, exposure, level))
        x = build_design(frame, fit.design).x
        if x.shape[0] != n:
            raise ValueError("the counterfactual rows do not match the fitted rows")
        eta = x @ beta + fit.offset
        mu = _link_from_eta(fit.family, eta)
        means[:, j] = mu
        slopes.append(_mu_eta(fit.family, eta, mu)[:, None] * x)
    total = float(np.sum(w))
    estimates = w @ means / total
    grad = np.vstack([w @ s / total for s in slopes])

    if covariates == "fixed":
        if labels is not None:
            kind = "HC0" if vcov is None else vcov
            beta_cov = cluster_covariance(fit, labels, kind=kind)
        elif vcov is None or vcov == "model":
            beta_cov = fit.covariance
        else:
            beta_cov = hc_covariance(fit, vcov)
        covariance = grad @ beta_cov @ grad.T
        used_vcov = (
            ("HC0" if vcov is None else vcov)
            if labels is not None
            else (vcov or "model")
        )
    else:
        covariance = _stacked(fit, w, means, estimates, grad * total / n, labels)
        used_vcov = "stacked"
    return StandardizedGLM(
        exposure=exposure,
        values=levels,
        estimates=estimates,
        covariance=covariance,
        n_obs=n,
        family=fit.family,
        covariates=covariates,
        vcov=used_vcov,
        fit=fit,
    )


def _stacked(
    fit: Fit,
    w: np.ndarray,
    means: np.ndarray,
    estimates: np.ndarray,
    dtheta: np.ndarray,
    labels: np.ndarray | None,
) -> np.ndarray:
    """Sandwich of the stacked equations ``w (mu_k - theta_k)`` and the GLM score.

    The score is ``w (y - mu) mu' / V(mu) x`` without the dispersion, which
    cancels in the coefficient block.
    """
    n, k = means.shape
    p = fit.x.shape[1]
    score = (fit.working_residuals * fit.working_weights)[:, None] * fit.x
    res = np.column_stack([w[:, None] * (means - estimates), score])
    if labels is not None:
        _, inverse = np.unique(labels, return_inverse=True)
        summed = np.zeros((inverse.max() + 1, res.shape[1]))
        np.add.at(summed, inverse, res)
        res = summed
    meat = np.atleast_2d(np.cov(res, rowvar=False, ddof=1))
    bread = np.zeros((k + p, k + p))
    bread[:k, :k] = -np.mean(w) * np.eye(k)
    bread[:k, k:] = dtheta
    bread[k:, k:] = -(fit.x * fit.working_weights[:, None]).T @ fit.x / n
    inv = np.linalg.inv(bread)
    scale = 1 / n if labels is None else res.shape[0] / n**2
    return (inv @ meat @ inv.T * scale)[:k, :k]


def _mu_eta(family: str, eta: np.ndarray, mu: np.ndarray) -> np.ndarray:
    if family == "binomial":
        return mu * (1 - mu)
    if family == "poisson":
        return mu
    if family == "gaussian":
        return np.ones_like(eta)
    if family == "gamma":
        return -(mu**2)
    raise ValueError(family)


def _uses(fit: Fit, exposure: str) -> bool:
    design = fit.design
    computed = design.computed or {}
    for recipe in design.recipes or ():
        for symbol, _level in recipe:
            sources = _columns_in(computed[symbol]) if symbol in computed else [symbol]
            if exposure in sources:
                return True
    return False


def _constant(frame: pl.DataFrame, name: str, value: object) -> pl.Series:
    dtype = frame.schema[name]
    series = pl.Series(name, [value] * frame.height)
    try:
        return series.cast(dtype, strict=True)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        if dtype.is_numeric():
            return series.cast(pl.Float64)
        raise ValueError(
            f"value {value!r} does not fit column {name!r} ({dtype})"
        ) from error
