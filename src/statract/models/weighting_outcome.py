"""Outcome models under IPTW weights, with the propensity score's estimation in the variance.

``weighted_glm`` follows ``WeightIt::glm_weightit`` (WeightIt 1.x). The outcome
GLM is fitted with prior weights ``weights * sampling_weights``. Its covariance:

- ``vcov="asympt"`` (default): M-estimation. The outcome score is stacked on the
  propensity score's logistic score (and, with ``stabilize=True``, on the
  estimating equation of the treated share), and the covariance is the
  sandwich ``A^-1 B A^-T`` of the whole stack. The cross block of ``A`` is the
  derivative of the weights in the propensity coefficients, ``dw/de * e(1-e) z``
  (Lunceford and Davidian 2004 for ATE; the same chain rule for ATT, ATC, and
  ATO). This is WeightIt's default for a ``weightit`` object.
- ``vcov="HC0"``: the outcome block alone, as if the weights were known. It is
  ``sandwich::vcovHC(type="HC0")`` on the weighted ``glm``.
- ``vcov="bootstrap"``: rows (or clusters) are resampled, and each replicate
  re-estimates the propensity score, the weights (with stabilization and
  trimming), and the outcome model. Intervals are percentiles.

``cluster`` sums the stacked estimating functions within clusters with the
``G / (G - 1)`` factor of ``sandwich::vcovCL``. With the bootstrap it resamples
whole clusters.

``PropensityWeights.effect`` (``weighted_effect``) gives the marginal contrast
of the weighted arm means: the difference, the ratio, or the odds ratio. It is
the saturated model ``y ~ treat`` (the weighted risk difference of a weighted
linear model, for a binary outcome), with the same three variance options.

Trimmed weights are not smooth in the propensity coefficients, so
``vcov="asympt"`` refuses them. ``"HC0"`` and ``"bootstrap"`` accept them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats

from .design import ColumnRef, column_series
from .fit import Fit, _fit_glm_arrays, _link_from_eta, fit_glm

if TYPE_CHECKING:
    from .weighting import PropensityWeights

_VCOV = ("asympt", "HC0", "bootstrap")
_MEASURES = ("difference", "ratio", "odds_ratio")
_GLM_FAMILIES = ("gaussian", "binomial", "poisson", "gamma")
_WEIGHT_COLUMN = "__statract_iptw_weight"


@dataclass
class WeightedGLM:
    """Outcome GLM under propensity score weights.

    ``fit`` is the weighted GLM with ``covariance`` replaced by the chosen
    ``vcov``. ``replicates`` holds the bootstrap coefficients (one row per
    replicate) when ``vcov="bootstrap"``.
    """

    fit: Fit
    weights: PropensityWeights = field(repr=False)
    vcov: str
    cluster: np.ndarray | None = field(default=None, repr=False)
    replicates: np.ndarray | None = field(default=None, repr=False)

    @property
    def coefficients(self) -> np.ndarray:
        return self.fit.coefficients

    @property
    def covariance(self) -> np.ndarray:
        return self.fit.covariance

    @property
    def names(self) -> list[str]:
        return list(self.fit.names)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Coefficient table with z tests.

        With ``vcov="bootstrap"`` the standard error is the replicates' standard
        deviation and the interval is their percentiles.
        """
        frame = self.fit.tidy(level=level)
        if self.replicates is not None:
            alpha = (1.0 - level) / 2.0
            low, high = np.quantile(self.replicates, [alpha, 1.0 - alpha], axis=0)
            frame = frame.with_columns(pl.Series("conf_low", low), pl.Series("conf_high", high))
        if exponentiate:
            frame = frame.with_columns(pl.col("estimate", "conf_low", "conf_high").exp())
        return frame


def weighted_glm(
    weights: PropensityWeights,
    formula: str,
    *,
    family: str = "gaussian",
    vcov: str = "asympt",
    cluster: ColumnRef | np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int | None = None,
) -> WeightedGLM:
    """GLM of the outcome under the weights, as ``WeightIt::glm_weightit``.

    The data are ``weights.data`` on the rows that have a weight; the outcome
    formula must not drop any of them. ``vcov`` is ``"asympt"`` (the default;
    accounts for estimating the propensity score), ``"HC0"`` (treats the
    weights as known), or ``"bootstrap"`` (``n_boot`` replicates with
    ``numpy.random.default_rng(seed)``). ``cluster`` is a column of
    ``weights.data`` or an array with one value per row of it.
    """
    _check_vcov(weights, vcov)
    if family not in _GLM_FAMILIES:
        raise ValueError(f"family must be one of {_GLM_FAMILIES}")
    idx = weights.row_index
    ps = _PsParts.from_weights(weights)
    frame = weights.data[idx].with_columns(pl.Series(_WEIGHT_COLUMN, ps.w * ps.s))
    fit = fit_glm(frame, formula, family=family, weights=_WEIGHT_COLUMN)
    if fit.n_obs != idx.size:
        raise ValueError(
            "the outcome formula drops rows that have a weight; drop those rows before propensity_weights"
        )
    clusters = _clusters(weights, cluster)
    replicates = None
    if vcov == "bootstrap":
        x, y, off = fit.x, np.asarray(fit.y, dtype=float), fit.offset

        def stat(rows: np.ndarray, w: np.ndarray) -> np.ndarray:
            prior = w * ps.s[rows]
            refit = _fit_glm_arrays(
                y[rows], x[rows], family, prior, off[rows], list(fit.names), fit.row_index[rows], fit.design
            )
            return refit.coefficients

        replicates = _bootstrap(weights, stat, n_boot=n_boot, seed=seed, cluster=clusters)
        cov = np.atleast_2d(np.cov(replicates, rowvar=False, ddof=1))
    else:
        mu = _link_from_eta(family, fit.x @ fit.coefficients + fit.offset)
        x = fit.x
        y = np.asarray(fit.y, dtype=float)
        ratio, d_mu = _glm_derivatives(family, mu)
        u = (ps.s * (y - mu) * ratio)[:, None] * x
        j_bb = -(x * (ps.w * ps.s * d_mu * ratio)[:, None]).T @ x
        cov = ps.sandwich(u, j_bb, clusters, stacked=vcov == "asympt")
    out = replace(fit, covariance=cov)
    out._refit = getattr(fit, "_refit", None)
    return WeightedGLM(fit=out, weights=weights, vcov=vcov, cluster=clusters, replicates=replicates)


def weighted_effect(
    weights: PropensityWeights,
    outcome: ColumnRef,
    *,
    measure: str | list[str] | tuple[str, ...] = "difference",
    vcov: str = "asympt",
    cluster: ColumnRef | np.ndarray | None = None,
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int | None = None,
) -> pl.DataFrame:
    """Marginal effect of the treatment: weighted arm means and their contrast.

    ``mean_treated`` and ``mean_control`` are the means of ``outcome`` in each
    arm under ``weights * sampling_weights``; the estimand is that of the
    weights. ``measure`` is ``"difference"`` (``m1 - m0``), ``"ratio"``
    (``m1 / m0``), ``"odds_ratio"``, or a list of them. One row per measure.

    The ratio and the odds ratio are tested and given intervals on the log
    scale; their ``std_error`` is that of the log. With ``vcov="bootstrap"``
    the standard error is the replicates' standard deviation, the interval
    is their percentiles, and the p value is the normal one from that
    standard error.
    """
    _check_vcov(weights, vcov)
    measures = [measure] if isinstance(measure, str) else list(measure)
    for m in measures:
        if m not in _MEASURES:
            raise ValueError(f"measure must be one of {_MEASURES}")
    idx = weights.row_index
    series = column_series(weights.data, outcome).gather(idx.tolist())
    if series.null_count():
        raise ValueError("the outcome has missing values on rows that have a weight")
    y = np.asarray(series.cast(pl.Float64).to_numpy(), dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("the outcome has missing values on rows that have a weight")
    if "odds_ratio" in measures and not np.all((y >= 0) & (y <= 1)):
        raise ValueError("measure='odds_ratio' needs an outcome between 0 and 1")
    ps = _PsParts.from_weights(weights)
    a = ps.a
    m1, m0 = _arm_means(y, a, ps.w * ps.s)
    if ("ratio" in measures or "odds_ratio" in measures) and not (m1 > 0 and m0 > 0):
        raise ValueError("a ratio needs positive arm means")
    if "odds_ratio" in measures and not (m1 < 1 and m0 < 1):
        raise ValueError("an odds ratio needs arm means below 1")
    clusters = _clusters(weights, cluster)
    crit = float(stats.norm.ppf(0.5 + level / 2))
    rows = []
    if vcov == "bootstrap":

        def stat(rows_: np.ndarray, w: np.ndarray) -> np.ndarray:
            r1, r0 = _arm_means(y[rows_], a[rows_], w * ps.s[rows_])
            return np.array([_contrast(m, r1, r0) for m in measures])

        reps = _bootstrap(weights, stat, n_boot=n_boot, seed=seed, cluster=clusters)
        alpha = (1.0 - level) / 2.0
        for k, m in enumerate(measures):
            est = _contrast(m, m1, m0)
            se = float(np.std(reps[:, k], ddof=1))
            lo, hi = np.quantile(reps[:, k], [alpha, 1.0 - alpha])
            if m != "difference":
                lo, hi = np.exp(lo), np.exp(hi)
            rows.append(_effect_row(m, weights.estimand, vcov, m1, m0, est, se, float(lo), float(hi)))
    else:
        u = np.column_stack([ps.s * a * (y - m1), ps.s * (1.0 - a) * (y - m0)])
        j_bb = np.diag([-np.sum(ps.w * ps.s * a), -np.sum(ps.w * ps.s * (1.0 - a))])
        cov = ps.sandwich(u, j_bb, clusters, stacked=vcov == "asympt")
        for m in measures:
            grad = _contrast_gradient(m, m1, m0)
            est = _contrast(m, m1, m0)
            se = float(np.sqrt(grad @ cov @ grad))
            lo, hi = est - crit * se, est + crit * se
            if m != "difference":
                lo, hi = np.exp(lo), np.exp(hi)
            rows.append(_effect_row(m, weights.estimand, vcov, m1, m0, est, se, float(lo), float(hi)))
    out = pl.DataFrame(rows)
    return out.with_columns(
        pl.Series("p_value", 2 * stats.norm.sf(np.abs(out["statistic"].to_numpy())))
    ).select(
        "measure",
        "estimand",
        "vcov",
        "mean_treated",
        "mean_control",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    )


# --- stacked estimating equations ---------------------------------------------


@dataclass
class _PsParts:
    """Propensity block of the stack, on the rows that have a weight."""

    z: np.ndarray
    a: np.ndarray
    s: np.ndarray
    e: np.ndarray
    w: np.ndarray
    dw_dgamma: np.ndarray
    dw_dp1: np.ndarray | None
    p1: float

    @classmethod
    def from_weights(cls, pw: PropensityWeights) -> _PsParts:
        from .weighting import _weights_from_ps

        idx = pw.row_index
        a = (np.asarray(pw.fit.y, dtype=float) > 0).astype(float)
        treat = a > 0
        s = pw.sampling_weights[idx]
        e = pw.ps[idx]
        w0 = _weights_from_ps(e, treat, pw.estimand)
        p1 = float(np.sum(s * a) / np.sum(s))
        if pw.stabilized:
            c = np.where(treat, p1, 1.0 - p1)
            dw_dp1: np.ndarray | None = np.where(treat, w0, -w0)
        else:
            c = np.ones_like(e)
            dw_dp1 = None
        dw_de = _dw_de(e, treat, pw.estimand)
        z = pw.fit.x
        return cls(
            z=z,
            a=a,
            s=s,
            e=e,
            w=pw.weights[idx],
            dw_dgamma=(c * dw_de * e * (1.0 - e))[:, None] * z,
            dw_dp1=dw_dp1,
            p1=p1,
        )

    def psi(self) -> np.ndarray:
        cols = [(self.s * (self.a - self.e))[:, None] * self.z]
        if self.dw_dp1 is not None:
            cols.append((self.s * (self.a - self.p1))[:, None])
        return np.column_stack(cols)

    def jacobian(self) -> np.ndarray:
        j = -(self.z * (self.s * self.e * (1.0 - self.e))[:, None]).T @ self.z
        if self.dw_dp1 is None:
            return j
        q = j.shape[0]
        out = np.zeros((q + 1, q + 1))
        out[:q, :q] = j
        out[q, q] = -np.sum(self.s)
        return out

    def sandwich(
        self, u: np.ndarray, j_bb: np.ndarray, clusters: np.ndarray | None, *, stacked: bool
    ) -> np.ndarray:
        """Covariance of the outcome parameters.

        The outcome estimating function is ``w_i * u_i`` and ``j_bb`` its
        summed derivative in the outcome parameters.
        """
        psi_b = self.w[:, None] * u
        p = u.shape[1]
        if not stacked:
            return _sandwich(j_bb, psi_b, clusters)
        cross = [u.T @ self.dw_dgamma]
        if self.dw_dp1 is not None:
            cross.append((u.T @ self.dw_dp1)[:, None])
        j_ps = self.jacobian()
        q = j_ps.shape[0]
        j = np.zeros((p + q, p + q))
        j[:p, :p] = j_bb
        j[:p, p:] = np.hstack(cross)
        j[p:, p:] = j_ps
        psi = np.hstack([psi_b, self.psi()])
        return _sandwich(j, psi, clusters)[:p, :p]


def _sandwich(j: np.ndarray, psi: np.ndarray, clusters: np.ndarray | None) -> np.ndarray:
    if clusters is None:
        meat = psi.T @ psi
    else:
        codes, inverse = np.unique(clusters, return_inverse=True)
        g = codes.size
        if g < 2:
            raise ValueError("cluster needs at least two clusters")
        summed = np.zeros((g, psi.shape[1]))
        np.add.at(summed, inverse, psi)
        meat = g / (g - 1) * (summed.T @ summed)
    inv = np.linalg.inv(j)
    cov = inv @ meat @ inv.T
    return (cov + cov.T) / 2.0


def _dw_de(e: np.ndarray, treat: np.ndarray, estimand: str) -> np.ndarray:
    if estimand == "ATE":
        return np.where(treat, -1.0 / e**2, 1.0 / (1.0 - e) ** 2)
    if estimand == "ATT":
        return np.where(treat, 0.0, 1.0 / (1.0 - e) ** 2)
    if estimand == "ATC":
        return np.where(treat, -1.0 / e**2, 0.0)
    return np.where(treat, -1.0, 1.0)


def _glm_derivatives(family: str, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``mu.eta / V(mu)`` and ``mu.eta`` for the canonical links of ``fit_glm``."""
    if family == "gaussian":
        return np.ones_like(mu), np.ones_like(mu)
    if family == "binomial":
        return np.ones_like(mu), mu * (1.0 - mu)
    if family == "poisson":
        return np.ones_like(mu), mu
    # gamma, inverse link: mu.eta = -mu^2, V = mu^2
    return -np.ones_like(mu), -(mu**2)


# --- helpers -------------------------------------------------------------------


def _check_vcov(weights: PropensityWeights, vcov: str) -> None:
    if vcov not in _VCOV:
        raise ValueError(f"vcov must be one of {_VCOV}")
    if vcov == "asympt" and weights.trim is not None:
        raise ValueError(
            "vcov='asympt' needs untrimmed weights: trimmed weights are not smooth in the "
            "propensity coefficients. Use vcov='HC0' or vcov='bootstrap'."
        )


def _clusters(weights: PropensityWeights, cluster: ColumnRef | np.ndarray | None) -> np.ndarray | None:
    if cluster is None:
        return None
    idx = weights.row_index
    if isinstance(cluster, np.ndarray | pl.Series | list):
        values = pl.Series(list(cluster)) if isinstance(cluster, list) else pl.Series(cluster)
        if values.len() != weights.data.height:
            raise ValueError("a cluster vector must have one value per row of data")
        values = values.gather(idx.tolist())
    else:
        values = column_series(weights.data, cluster).gather(idx.tolist())
    if values.null_count():
        raise ValueError("cluster has missing values on rows that have a weight")
    return values.cast(pl.String).to_numpy()


def _arm_means(y: np.ndarray, a: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m1 = float(np.sum(w * a * y) / np.sum(w * a))
    m0 = float(np.sum(w * (1.0 - a) * y) / np.sum(w * (1.0 - a)))
    return m1, m0


def _contrast(measure: str, m1: float, m0: float) -> float:
    """The difference, or the log of the ratio or the odds ratio."""
    if measure == "difference":
        return m1 - m0
    if measure == "ratio":
        return float(np.log(m1) - np.log(m0))
    return float(np.log(m1 / (1.0 - m1)) - np.log(m0 / (1.0 - m0)))


def _contrast_gradient(measure: str, m1: float, m0: float) -> np.ndarray:
    if measure == "difference":
        return np.array([1.0, -1.0])
    if measure == "ratio":
        return np.array([1.0 / m1, -1.0 / m0])
    return np.array([1.0 / (m1 * (1.0 - m1)), -1.0 / (m0 * (1.0 - m0))])


def _effect_row(measure, estimand, vcov, m1, m0, log_est, se, lo, hi) -> dict[str, Any]:
    est = log_est if measure == "difference" else float(np.exp(log_est))
    return {
        "measure": measure,
        "estimand": estimand,
        "vcov": vcov,
        "mean_treated": m1,
        "mean_control": m0,
        "estimate": est,
        "std_error": se,
        "statistic": log_est / se if se > 0 else float("nan"),
        "conf_low": lo,
        "conf_high": hi,
    }


def _bootstrap(
    pw: PropensityWeights,
    stat,
    *,
    n_boot: int,
    seed: int | None,
    cluster: np.ndarray | None,
) -> np.ndarray:
    """Replicates of ``stat(rows, weights)``; each re-estimates the propensity score.

    ``rows`` index the rows that have a weight. A draw without both arms is
    drawn again.
    """
    from .weighting import _trim, _weights_from_ps

    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")
    rng = np.random.default_rng(seed)
    treat = np.asarray(pw.fit.y, dtype=float) > 0
    s = pw.sampling_weights[pw.row_index]
    n = treat.size
    if cluster is not None:
        _, inverse = np.unique(cluster, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(int(inverse.max()) + 1)]
    out = []
    while len(out) < int(n_boot):
        if cluster is None:
            rows = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            rows = np.concatenate([members[g] for g in pick])
        t = treat[rows]
        if t.all() or not t.any():
            continue
        ps_fit = pw.fit.refit_rows(rows)
        e = ps_fit.predict(kind="response")
        w = _weights_from_ps(e, t, pw.estimand)
        if pw.stabilized:
            sr = s[rows]
            p1 = float(np.sum(sr * t) / np.sum(sr))
            w = w * np.where(t, p1, 1.0 - p1)
        if pw.trim is not None:
            w = _trim(w, t, pw.estimand, pw.trim, lower=pw.trim_lower)
        out.append(np.atleast_1d(np.asarray(stat(rows, w), dtype=float)))
    return np.vstack(out)


__all__ = ["WeightedGLM", "weighted_effect", "weighted_glm"]
