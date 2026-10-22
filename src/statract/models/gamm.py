"""Additive mixed models fitted as GLMMs, matching ``gamm4``.

``gamm`` writes each penalized smooth as a fixed null-space part plus an iid
random effect, as ``mgcv::smooth2random`` does, and fits the whole model by
the Laplace approximation (``lme4::glmer`` in ``gamm4``). The other random
effects, such as ``(1 | examiner)``, enter the same GLMM. The smooth's
effective degrees of freedom and the covariance of the coefficients follow
``gamm4``: the other random effects are integrated into the working
covariance ``V``, and ``Vb = (X' V^-1 X + S)^-1``.

The mixed-model form of a smooth depends on the coordinates of its basis,
not only on the function space and the penalty. The thin-plate basis here
reproduces mgcv's coordinates (covariate centred, the eigen basis scaled by
the eigenvalues, columns scaled to root-mean-square one), so the fit equals
``gamm4``'s and not only ``gam(..., bs = "re")``'s, which uses REML on a
different approximation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import special, stats

from .._core.laplace import RandomTerm, fit_laplace_terms
from .design import ColumnRef, column_series, design_matrix
from .gam import Smooth

_FAMILIES = ("binomial", "poisson")
_TP_MAX_KNOTS = 2000


@dataclass
class _SmoothPart:
    name: str
    x_c: np.ndarray  # constrained basis (gam coordinates)
    s_c: np.ndarray  # its penalty
    null: np.ndarray  # eigenvectors of s_c with zero eigenvalue
    range: np.ndarray  # eigenvectors with positive eigenvalue
    range_scale: np.ndarray  # 1 / sqrt(eigenvalue)
    rebuild: Callable[[np.ndarray], np.ndarray]  # x -> constrained basis


@dataclass
class GammFit:
    """A ``gamm4``-style fit.

    ``coefficients`` and ``covariance`` cover the parametric terms and every
    smooth's coefficients in its constrained basis, like ``g$gam``'s
    ``coefficients`` and ``Vp``. ``edf`` is per smooth. ``random_terms``
    lists the other random effects with their covariance and BLUPs.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_parametric: int
    smooth_names: list[str]
    edf: np.ndarray
    smooth_variances: np.ndarray
    log_likelihood: float
    family: str
    n_obs: int
    converged: bool
    random_terms: list[dict]
    n_parameters: int
    random_effects_frame: pl.DataFrame
    _smooths: list[_SmoothPart] = field(repr=False)
    _slices: list[slice] = field(repr=False)
    _parametric_design: object = field(repr=False)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Parametric coefficients with SEs from ``Vp``, as ``summary(g$gam)$p.table``."""
        k = self.n_parametric
        est = self.coefficients[:k]
        se = np.sqrt(np.clip(np.diag(self.covariance)[:k], 0, None))
        stat = est / se
        crit = float(stats.norm.ppf(0.5 + level / 2))
        frame = pl.DataFrame(
            {
                "term": self.names[:k],
                "estimate": est,
                "std_error": se,
                "statistic": stat,
                "p_value": 2 * stats.norm.sf(np.abs(stat)),
                "conf_low": est - crit * se,
                "conf_high": est + crit * se,
            }
        )
        if exponentiate:
            frame = frame.with_columns(
                pl.col("estimate").exp().alias("exp_estimate"),
                pl.col("conf_low").exp().alias("exp_conf_low"),
                pl.col("conf_high").exp().alias("exp_conf_high"),
            )
        return frame

    def smooth_table(self) -> pl.DataFrame:
        """edf and the random-effect variance of each smooth (``1 / sp``)."""
        return pl.DataFrame(
            {"term": self.smooth_names, "edf": self.edf, "variance": self.smooth_variances}
        )

    def variance_table(self, *, level: float = 0.95) -> pl.DataFrame:
        """Variances of the other random effects, as ``VarCorr(g$mer)`` lists them.

        ``std_error`` and ``conf_low`` / ``conf_high`` are for the variance.
        The interval is a Wald interval for the log SD, from the inverse
        Hessian of the Laplace log-likelihood, transformed to the variance.
        It is not lme4's profile interval and is less accurate near zero.
        A median odds ratio interval is ``median_odds_ratio`` of each end.
        """
        crit = float(stats.norm.ppf(0.5 + level / 2))
        rows = []
        for term in self.random_terms:
            cov = np.asarray(term["covariance"])
            for i, name in enumerate(term["names"]):
                variance = float(cov[i, i])
                log_sd, se_log = term["log_sd"][i], term["log_sd_std_error"][i]
                rows.append(
                    {
                        "group": term["group"],
                        "term": name,
                        "variance": variance,
                        "std_dev": float(np.sqrt(max(variance, 0.0))),
                        "std_error": 2 * variance * se_log,
                        "conf_low": float(np.exp(2 * (log_sd - crit * se_log))),
                        "conf_high": float(np.exp(2 * (log_sd + crit * se_log))),
                    }
                )
        schema = {
            "group": pl.String,
            "term": pl.String,
            "variance": pl.Float64,
            "std_dev": pl.Float64,
            "std_error": pl.Float64,
            "conf_low": pl.Float64,
            "conf_high": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema)

    def glance(self) -> pl.DataFrame:
        """Laplace log-likelihood, AIC, and BIC, as ``logLik`` / ``AIC`` of ``g$mer``.

        ``n_parameters`` counts the unpenalized coefficients (parametric
        terms and each smooth's null space) and the variance parameters
        (one per smooth, plus each random term's covariance parameters).
        """
        ll = self.log_likelihood
        k = self.n_parameters
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "n_parameters": [k],
                "log_likelihood": [ll],
                "aic": [-2 * ll + 2 * k],
                "bic": [-2 * ll + np.log(self.n_obs) * k],
            }
        )

    def random_effects(self) -> pl.DataFrame:
        """BLUPs of the other random effects: ``group``, ``level``, ``term``, ``blup``."""
        return self.random_effects_frame

    def partial_effect(self, term: str, values: Sequence[float] | np.ndarray) -> pl.DataFrame:
        """A smooth's contribution and pointwise SE at ``values``, as ``predict(type = "terms")``."""
        if term not in self.smooth_names:
            raise KeyError(f"smooth {term!r} is not in the fit")
        i = self.smooth_names.index(term)
        sl = self._slices[i]
        x = np.asarray(values, dtype=float)
        basis = self._smooths[i].rebuild(x)
        fit = basis @ self.coefficients[sl]
        cov = self.covariance[sl, sl]
        se = np.sqrt(np.clip(np.sum((basis @ cov) * basis, axis=1), 0, None))
        return pl.DataFrame({term: x, "fit": fit, "std_error": se})

    def predict(
        self, data: pl.DataFrame, *, kind: str = "link", se: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Population prediction: parametric terms plus smooths, random effects at zero.

        ``se=True`` also returns the pointwise standard error from ``covariance``
        (``Vp``), as ``predict(g$gam, se.fit = TRUE)``. For ``kind="response"``
        it is the delta-method SE; a confidence interval is better built on the
        link scale and transformed.
        """
        if kind not in {"link", "response"}:
            raise ValueError("kind must be 'link' or 'response'")
        from .design import build_design

        pieces = [build_design(data, self._parametric_design).x]
        for part in self._smooths:
            pieces.append(part.rebuild(np.asarray(column_series(data, part.name).to_numpy(), dtype=float)))
        design = np.column_stack(pieces)
        eta = design @ self.coefficients
        mu = special.expit(eta) if self.family == "binomial" else np.exp(eta)
        point = eta if kind == "link" else mu
        if not se:
            return point
        var = np.sum((design @ self.covariance) * design, axis=1)
        se_link = np.sqrt(np.clip(var, 0, None))
        if kind == "link":
            return point, se_link
        return point, se_link * (mu * (1 - mu) if self.family == "binomial" else mu)


def gamm(
    data: pl.DataFrame,
    outcome: ColumnRef,
    smooths: Sequence[Smooth],
    *,
    random: str | None = None,
    predictors: Sequence[ColumnRef] | None = None,
    family: str = "binomial",
) -> GammFit:
    """Fit an additive mixed model, as ``gamm4(y ~ x + s(z), random = ~(1 | g))``.

    ``smooths`` are :func:`statract.smooth` terms with ``basis="tp"``
    (mgcv's default, up to 2000 unique values) or ``"cr"``, one penalty
    each and no ``by``. ``random`` is the lme4 random part, for example
    ``"(1 | examiner)"``. ``random=None`` fits the smooths alone by the same
    Laplace likelihood, so its log-likelihood can be compared with a fit
    that has random effects (:func:`compare_gamms`). ``family`` is ``"binomial"`` (logit) or
    ``"poisson"`` (log). The fit is Laplace maximum likelihood, as glmer.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}")
    if not smooths:
        raise ValueError("gamm needs at least one smooth")
    for sm in smooths:
        if sm.basis not in {"tp", "cr"} or sm.by is not None or sm.extra:
            raise ValueError("gamm takes univariate tp or cr smooths without by")
    from .formula import model_matrix

    extra: list[ColumnRef] = [outcome, *[sm.column for sm in smooths]]
    probed = []
    if random is not None:
        probed = model_matrix(f"{_name(data, outcome)} ~ 1 + {random}", data).random_effects
        if not probed:
            raise ValueError("random needs at least one term, for example '(1 | group)'")
    for effect in probed:
        extra.append(effect.group)
        extra.extend(effect.slopes)
    design = design_matrix(data, list(predictors or []), extra=extra)
    used = data.gather(design.row_index.tolist())
    y = np.asarray(column_series(used, outcome).to_numpy(), dtype=float)

    parts = []
    for sm in smooths:
        x = np.asarray(column_series(used, sm.column).to_numpy(), dtype=float)
        basis, penalty, rebuild = (_tp_mgcv if sm.basis == "tp" else _cr_mgcv)(x, sm.k)
        parts.append(_mixed_form(sm.column, basis, penalty, rebuild))

    from .mixed import _effect_columns

    term_cols = []
    if random is not None:
        built = model_matrix(f"{_name(data, outcome)} ~ 1 + {random}", used)
        term_cols = [_effect_columns(used, built.design, effect) for effect in built.random_effects]
    terms = [RandomTerm(t.z, t.codes, len(t.levels), t.structure) for t in term_cols]
    fixed = [design.x] + [p.x_c @ p.null for p in parts]
    x_fixed = np.column_stack(fixed)
    for part in parts:
        z = part.x_c @ part.range * part.range_scale
        terms.append(RandomTerm(z, np.zeros(len(y), dtype=np.int64), 1, "iid"))
    result = fit_laplace_terms(y, x_fixed, terms, None, family=family)

    n_other = len(term_cols)
    p_par = design.x.shape[1]
    # Coefficients in the constrained (gam) basis of each smooth.
    coefficients = [result.coefficients[:p_par]]
    at = p_par
    slices = []
    pos = p_par
    for i, part in enumerate(parts):
        n_null = part.null.shape[1]
        beta_null = result.coefficients[at : at + n_null]
        at += n_null
        b = result.modes[n_other + i][0]
        coefficients.append(part.null @ beta_null + part.range @ (part.range_scale * b))
        slices.append(slice(pos, pos + part.x_c.shape[1]))
        pos += part.x_c.shape[1]
    coefficients = np.concatenate(coefficients)
    smooth_var = np.array([float(result.group_covariances[n_other + i][0, 0]) for i in range(len(parts))])

    # gamm4's Vb and edf: other random effects integrated into V.
    x_gam = np.column_stack([design.x] + [p.x_c for p in parts])
    eta = x_gam @ coefficients
    for t, modes in zip(term_cols, result.modes[:n_other], strict=True):
        eta = eta + np.einsum("ij,ij->i", t.z, modes[t.codes])
    mu = special.expit(eta) if family == "binomial" else np.exp(eta)
    w = mu * (1 - mu) if family == "binomial" else mu
    xvx = _xt_vinv_x(x_gam, w, term_cols, result.group_covariances[:n_other])
    penalty = np.zeros_like(xvx)
    for part, sl, var in zip(parts, slices, smooth_var, strict=True):
        penalty[sl, sl] = part.s_c / var
    vb = np.linalg.inv(xvx + penalty)
    vb = 0.5 * (vb + vb.T)
    influence = vb @ xvx
    edf = np.array([float(np.trace(influence[sl, sl])) for sl in slices])

    random_terms = []
    frames = []
    for t, cov, modes in zip(term_cols, result.group_covariances[:n_other], result.modes[:n_other], strict=True):
        random_terms.append({"group": t.group, "names": list(t.names), "covariance": cov})
        for k, name in enumerate(t.names):
            frames.append(
                pl.DataFrame(
                    {"group": t.group, "level": list(t.levels), "term": name, "blup": modes[:, k]},
                    schema={"group": pl.String, "level": pl.String, "term": pl.String, "blup": pl.Float64},
                )
            )
    # Wald intervals for the SDs on the log scale, by the delta method from
    # the covariance of the Laplace fit's covariance-factor parameters.
    at = 0
    for term, info in zip(terms, random_terms + [None] * len(parts), strict=True):
        width = term.n_par
        if info is not None:
            par = result.term_parameters[at : at + width]
            cov_par = result.term_parameter_covariance[at : at + width, at : at + width]
            info["log_sd"], info["log_sd_std_error"] = _log_sd_wald(term, par, cov_par)
        at += width
    names = list(design.names)
    for part in parts:
        names.extend(f"s({part.name}).{j + 1}" for j in range(part.x_c.shape[1]))
    return GammFit(
        coefficients=coefficients,
        covariance=vb,
        names=names,
        n_parametric=p_par,
        smooth_names=[p.name for p in parts],
        edf=edf,
        smooth_variances=smooth_var,
        log_likelihood=result.log_likelihood,
        family=family,
        n_obs=len(y),
        converged=result.converged,
        random_terms=random_terms,
        random_effects_frame=pl.concat(frames) if frames else pl.DataFrame(
            schema={"group": pl.String, "level": pl.String, "term": pl.String, "blup": pl.Float64}
        ),
        n_parameters=int(x_fixed.shape[1] + sum(term.n_par for term in terms)),
        _smooths=parts,
        _slices=slices,
        _parametric_design=design,
    )


def compare_gamms(*fits: GammFit) -> pl.DataFrame:
    """Likelihood-ratio tests between additive mixed models on the same rows.

    Each row after the first is tested against the row before it, as
    ``anova(m0$mer, m1$mer)``. A fit with ``random=None`` is the model
    without the random effects, on the same Laplace likelihood. ``df`` is
    the change in ``n_parameters`` and ``p_value`` uses chi-squared on ``df``.

    When the larger model adds one random term, its variance is tested at
    the boundary zero, where chi-squared on ``df`` is conservative.
    ``p_value_boundary`` is the 50:50 mixture of chi-squared on ``df - 1``
    and ``df`` (Self & Liang 1987; Stram & Lee 1994), which is the usual
    reference for one added variance (``df = 1``: half the naive p-value).
    """
    if not fits:
        raise ValueError("compare_gamms needs at least one fit")
    if len({fit.n_obs for fit in fits}) > 1:
        raise ValueError("all fits must use the same rows")
    if len({fit.family for fit in fits}) > 1:
        raise ValueError("all fits must have the same family")
    rows = []
    for index, fit in enumerate(fits):
        row = fit.glance().row(0, named=True)
        row = {"model": index + 1, **row, "df": None, "statistic": None, "p_value": None, "p_value_boundary": None}
        if index > 0:
            before = fits[index - 1]
            df = fit.n_parameters - before.n_parameters
            stat = 2 * (fit.log_likelihood - before.log_likelihood) * np.sign(df)
            row["df"] = df
            row["statistic"] = stat
            if df != 0:
                width = abs(df)
                tail = float(stats.chi2.sf(max(stat, 0.0), width))
                row["p_value"] = tail
                lower = float(stats.chi2.sf(max(stat, 0.0), width - 1)) if width > 1 else float(stat <= 0)
                row["p_value_boundary"] = 0.5 * tail + 0.5 * lower
        rows.append(row)
    schema = {
        "model": pl.Int64,
        "n_obs": pl.Int64,
        "n_parameters": pl.Int64,
        "log_likelihood": pl.Float64,
        "aic": pl.Float64,
        "bic": pl.Float64,
        "df": pl.Int64,
        "statistic": pl.Float64,
        "p_value": pl.Float64,
        "p_value_boundary": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _log_sd_wald(term: RandomTerm, par: np.ndarray, cov_par: np.ndarray) -> tuple[list[float], list[float]]:
    """Log SD of each column of a random term and its delta-method SE."""

    def log_sd(values: np.ndarray) -> np.ndarray:
        factor = term.factor(values)
        return 0.5 * np.log(np.sum(factor * factor, axis=1))

    centre = log_sd(par)
    jac = np.zeros((centre.size, par.size))
    for j in range(par.size):
        step = 1e-6 * max(1.0, abs(par[j]))
        up, down = par.copy(), par.copy()
        up[j] += step
        down[j] -= step
        jac[:, j] = (log_sd(up) - log_sd(down)) / (2 * step)
    var = np.einsum("ij,jk,ik->i", jac, cov_par, jac)
    return [float(v) for v in centre], [float(np.sqrt(max(v, 0.0))) for v in var]


def _name(data: pl.DataFrame, ref: ColumnRef) -> str:
    return ref if isinstance(ref, str) else column_series(data, ref).name


def _xt_vinv_x(x, w, term_cols, covariances) -> np.ndarray:
    """X' V^-1 X with V = W^-1 + sum Z_k Sigma_k Z_k', by Woodbury."""
    xtwx = (x * w[:, None]).T @ x
    if not term_cols:
        return xtwx
    blocks = []
    for t, cov in zip(term_cols, covariances, strict=True):
        n_levels, q = len(t.levels), t.z.shape[1]
        a = np.zeros((len(w), n_levels * q))
        rows = np.repeat(np.arange(len(w)), q)
        cols = (t.codes[:, None] * q + np.arange(q)[None, :]).ravel()
        a[rows, cols] = t.z.ravel()
        blocks.append((a, np.kron(np.eye(n_levels), cov)))
    a = np.hstack([b[0] for b in blocks])
    g = np.zeros((a.shape[1], a.shape[1]))
    at = 0
    for b, cov in blocks:
        size = b.shape[1]
        g[at : at + size, at : at + size] = cov
        at += size
    aw = a * w[:, None]
    # V^-1 = W - W A (G^-1 + A'WA)^-1 A'W, written with G instead of G^-1
    # so that a zero variance stays finite: (I + G A'WA)^-1 G.
    inner = np.linalg.solve(np.eye(len(g)) + g @ (aw.T @ a), g)
    awx = aw.T @ x
    return xtwx - awx.T @ inner @ awx


def _mixed_form(name, basis, penalty, rebuild) -> _SmoothPart:
    """Sum-to-zero constraint, then the eigen split of mgcv's smooth2random."""
    center = basis.mean(axis=0)
    q, _r = np.linalg.qr(center.reshape(-1, 1), mode="complete")
    q2 = q[:, 1:]
    x_c = basis @ q2
    s_c = q2.T @ penalty @ q2
    s_c = 0.5 * (s_c + s_c.T)
    ev, vec = np.linalg.eigh(s_c)
    positive = ev > max(float(ev.max()), 0.0) * 1e-10
    return _SmoothPart(
        name=name,
        x_c=x_c,
        s_c=s_c,
        null=vec[:, ~positive],
        range=vec[:, positive],
        range_scale=1.0 / np.sqrt(ev[positive]),
        rebuild=lambda x, _r=rebuild, _q=q2: _r(np.asarray(x, dtype=float)) @ _q,
    )


def _tp_mgcv(x: np.ndarray, k: int):
    """1D thin-plate regression spline in mgcv's coordinates.

    mgcv centres the covariate, keeps the k - 2 largest eigenvalues of the
    radial matrix, parameterises the wiggly part by gamma = D delta (basis
    U Z, penalty Z' D^-1 Z), and scales every column to root-mean-square one.
    """
    shift = float(np.mean(x))
    knots = np.unique(x - shift)
    if len(knots) > _TP_MAX_KNOTS:
        raise ValueError("tp in gamm needs at most 2000 unique values (mgcv samples knots beyond that); use basis='cr'")
    k = min(k, len(knots))
    energy = np.abs(knots[:, None] - knots[None, :]) ** 3 / 12
    poly = np.column_stack([np.ones(len(knots)), knots])
    ev, vec = np.linalg.eigh(energy)
    order = np.argsort(-np.abs(ev))[:k]
    vec, ev = vec[:, order], ev[order]
    q, _r = np.linalg.qr(((poly.T @ vec) / ev).T, mode="complete")
    z = q[:, 2:]
    coef_map = (vec / ev) @ z  # knots -> wiggly columns

    def raw(xx):
        xs = np.asarray(xx, dtype=float) - shift
        radial = np.abs(xs[:, None] - knots[None, :]) ** 3 / 12
        return np.column_stack([radial @ coef_map, np.ones(len(xs)), xs])

    basis = raw(x)
    scales = np.sqrt(np.mean(basis**2, axis=0))
    penalty = np.zeros((k, k))
    penalty[: k - 2, : k - 2] = z.T @ np.diag(1.0 / ev) @ z
    penalty = penalty / np.outer(scales, scales)
    return basis / scales, penalty, lambda xx: raw(xx) / scales


def _cr_mgcv(x: np.ndarray, k: int):
    """Cubic regression spline with knots at quantiles of the unique values (mgcv ``cr``)."""
    from .gam import _cr_basis

    unique = np.unique(x)
    k = min(k, len(unique))
    knots = np.quantile(unique, np.linspace(0, 1, k))
    basis, penalty = _cr_basis(x, knots)
    return basis, penalty, lambda xx: _cr_basis(np.asarray(xx, dtype=float), knots)[0]
