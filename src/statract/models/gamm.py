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

    def variance_table(self) -> pl.DataFrame:
        """Variances of the other random effects, as ``VarCorr(g$mer)`` lists them."""
        rows = []
        for term in self.random_terms:
            cov = np.asarray(term["covariance"])
            for i, name in enumerate(term["names"]):
                rows.append(
                    {
                        "group": term["group"],
                        "term": name,
                        "variance": float(cov[i, i]),
                        "std_dev": float(np.sqrt(max(cov[i, i], 0.0))),
                    }
                )
        return pl.DataFrame(rows)

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

    def predict(self, data: pl.DataFrame, *, kind: str = "link") -> np.ndarray:
        """Population prediction: parametric terms plus smooths, random effects at zero."""
        if kind not in {"link", "response"}:
            raise ValueError("kind must be 'link' or 'response'")
        from .design import build_design

        eta = build_design(data, self._parametric_design).x @ self.coefficients[: self.n_parametric]
        for part, sl in zip(self._smooths, self._slices, strict=True):
            eta = eta + part.rebuild(np.asarray(column_series(data, part.name).to_numpy(), dtype=float)) @ self.coefficients[sl]
        if kind == "link":
            return eta
        return special.expit(eta) if self.family == "binomial" else np.exp(eta)


def gamm(
    data: pl.DataFrame,
    outcome: ColumnRef,
    smooths: Sequence[Smooth],
    *,
    random: str,
    predictors: Sequence[ColumnRef] | None = None,
    family: str = "binomial",
) -> GammFit:
    """Fit an additive mixed model, as ``gamm4(y ~ x + s(z), random = ~(1 | g))``.

    ``smooths`` are :func:`statract.smooth` terms with ``basis="tp"``
    (mgcv's default, up to 2000 unique values) or ``"cr"``, one penalty
    each and no ``by``. ``random`` is the lme4 random part, for example
    ``"(1 | examiner)"``. ``family`` is ``"binomial"`` (logit) or
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

    effects_probe = model_matrix(f"{_name(data, outcome)} ~ 1 + {random}", data)
    if not effects_probe.random_effects:
        raise ValueError("random needs at least one term, for example '(1 | group)'")
    extra: list[ColumnRef] = [outcome, *[sm.column for sm in smooths]]
    for effect in effects_probe.random_effects:
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
        random_effects_frame=pl.concat(frames),
        _smooths=parts,
        _slices=slices,
        _parametric_design=design,
    )


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
