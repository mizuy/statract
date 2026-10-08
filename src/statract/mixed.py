"""Linear and generalized linear mixed models.

``fit_mixed`` is the public mixed-model entry point, parallel to ``fit_ols`` /
``fit_glm``. Gaussian LMMs call ``lme_python.lmer`` (lme-rs). Gamma GLMMs
and non-logit binomial links call ``lme_python.glmer``. Binomial (logit), Poisson, and
negative binomial (NB2) GLMMs, with optional zero-inflation or hurdle parts,
use the Laplace / adaptive Gauss–Hermite fit in ``_laplace``, which matches
``glmmTMB`` and ``lme4::glmer``. Wilkinson formulas such as
``y ~ x + offset(log(t)) + (1 | g)`` work for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl
from scipy import stats

from .design import ColumnRef, Design, _is_factor, column_series, design_matrix
from ._laplace import COUNT_FAMILIES, LAPLACE_FAMILIES, fit_laplace_glmm
from .formula import is_formula, model_matrix, reject_survival_syntax

_FAMILIES = ("gaussian", "binomial", "poisson", "gamma", "negative_binomial")
_LME_ENGINES = {"lme", "lme-rs", "lme-python", "lme_python"}
_LAPLACE_ENGINES = {"laplace"}
_LMER_TOLERANCE = 1e-8


@dataclass
class MixedFit:
    """Fitted linear or generalized linear mixed model.

    ``coefficients`` and ``covariance`` are the fixed effects. ``tidy`` uses a
    normal reference (Wald). ``group_covariance`` is the random-effect
    covariance for ``(Intercept)`` and any random slopes of ``group_name``.
    ``theta`` is the negative binomial size (variance ``mu + mu^2 / theta``),
    with ``theta_std_error`` from the delta method on ``log(theta)``.
    A zero-inflated or hurdle fit (``zero_part``) keeps the logit-scale
    coefficients of the zero probability in ``zero_coefficients``; see
    :meth:`zero_table`.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    n_groups: int
    log_likelihood: float
    residual_variance: float | None
    group_name: str
    random_names: list[str]
    group_covariance: np.ndarray
    method: str
    converged: bool
    x: np.ndarray
    row_index: np.ndarray
    design: Design
    n_iter: int = 0
    function_evals: int = 0
    family: str = "gaussian"
    engine: str = "lme"
    n_agq: int = 1
    formula: str | None = None
    offset: np.ndarray | None = None
    theta: float | None = None
    theta_std_error: float | None = None
    zero_part: str | None = None
    zero_coefficients: np.ndarray | None = None
    zero_covariance: np.ndarray | None = None
    zero_names: list[str] | None = None
    zero_design: Design | None = field(default=None, repr=False)
    offset_ref: ColumnRef | None = field(default=None, repr=False)
    ranef_frame: pl.DataFrame | None = field(default=None, repr=False)
    random_terms: list[dict[str, Any]] | None = field(default=None, repr=False)
    raw: Any = field(default=None, repr=False)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Fixed-effect table. Intervals use a normal (Wald) reference."""
        return _wald_table(self.names, self.coefficients, self.covariance, level, exponentiate)

    def zero_table(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Coefficients of the zero part (logit of the zero probability).

        For ``zero_part="inflated"`` this is the probability of a structural
        zero; for ``"hurdle"`` it is the probability of any zero. The columns
        match :meth:`tidy`; ``exponentiate=True`` gives odds ratios.
        """
        if self.zero_coefficients is None or self.zero_covariance is None or self.zero_names is None:
            raise TypeError("this fit has no zero-inflation or hurdle part")
        return _wald_table(self.zero_names, self.zero_coefficients, self.zero_covariance, level, exponentiate)

    def variance_table(self) -> pl.DataFrame:
        """Random-effect variances (and residual variance for Gaussian LMMs).

        ``correlation`` is the correlation with the intercept. The residual
        row leaves it empty.
        """
        if self.random_terms is not None:
            return _terms_variance_table(self.random_terms)
        cov = self.group_covariance
        intercept_at = self.random_names.index("(Intercept)") if "(Intercept)" in self.random_names else None
        intercept_sd = 0.0 if intercept_at is None else float(np.sqrt(max(cov[intercept_at, intercept_at], 0.0)))
        rows: list[dict[str, object]] = []
        for i, name in enumerate(self.random_names):
            variance = float(cov[i, i])
            std_dev = float(np.sqrt(max(variance, 0.0)))
            if intercept_at is None or i == intercept_at or intercept_sd == 0 or std_dev == 0:
                correlation = None
            else:
                correlation = float(cov[i, intercept_at] / (intercept_sd * std_dev))
            rows.append(
                {
                    "group": self.group_name,
                    "term": name,
                    "variance": variance,
                    "std_dev": std_dev,
                    "correlation": correlation,
                }
            )
        if self.family == "gaussian" and self.residual_variance is not None:
            residual = float(self.residual_variance)
            rows.append(
                {
                    "group": "residual",
                    "term": "",
                    "variance": residual,
                    "std_dev": float(np.sqrt(max(residual, 0.0))),
                    "correlation": None,
                }
            )
        return pl.DataFrame(rows)

    def random_effects(self) -> pl.DataFrame:
        """Conditional modes (BLUPs) of the random effects.

        One term: a row per group with ``blup`` (the intercept) and a column
        per slope, sorted by ``blup``. Several terms or ``ar1()``: a long
        frame with ``group`` (the grouping column), ``level``, ``term``, and
        ``blup``, as ``ranef()`` lists them.
        """
        if self.ranef_frame is not None:
            return self.ranef_frame
        if self.raw is not None and getattr(self.raw, "ranef", None) is not None:
            return _ranef_frame(self.raw.ranef, self.group_name)
        raise TypeError("random effects are not available on this fit")

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        """Population mean (fixed effects only).

        ``kind`` is ``"link"`` or ``"response"``. Gaussian LMMs ignore ``kind``.
        A fit with an offset adds it: the stored one without ``data``, and the
        offset column or the formula's ``offset()`` evaluated on ``data``.
        With a zero part, ``"response"`` is the mean of the whole outcome:
        ``(1 - pi) mu`` for zero inflation and ``(1 - pi) mu / (1 - f(0))``
        for a hurdle. ``"link"`` stays the count part's linear predictor.
        """
        if kind not in {"link", "response"}:
            raise ValueError("kind must be 'link' or 'response'")
        if data is not None and self.raw is not None:
            if kind == "response" and self.family != "gaussian" and hasattr(self.raw, "predict_response"):
                return np.asarray(self.raw.predict_response(data), dtype=float)
            return np.asarray(self.raw.predict(data), dtype=float)
        if data is None:
            eta = self.x @ self.coefficients
            if self.offset is not None:
                eta = eta + self.offset
        else:
            from .design import build_design

            design = build_design(data, self.design)
            if design.x.shape[1] != len(self.coefficients):
                raise ValueError("new data produced a different number of columns")
            eta = design.x @ self.coefficients
            if self.offset is not None:
                eta = eta + self._new_offset(data, design)
        if kind == "link" or self.family == "gaussian":
            return eta
        mean = _inv_link(self.family, eta)
        if self.zero_part is None:
            return mean
        if data is None:
            zx = self.zero_design.x if self.zero_design is not None else np.ones((len(eta), 1))
        else:
            zx = self._new_zero_x(data, len(eta))
        pi = 1.0 / (1.0 + np.exp(-(zx @ self.zero_coefficients)))
        if self.zero_part == "inflated":
            return (1.0 - pi) * mean
        if self.family == "poisson":
            f0 = np.exp(-mean)
        else:
            f0 = (self.theta / (self.theta + mean)) ** self.theta
        return (1.0 - pi) * mean / (1.0 - f0)

    def _new_zero_x(self, data: pl.DataFrame, n_rows: int) -> np.ndarray:
        if self.zero_design is None or not self.zero_design.predictors:
            return np.ones((n_rows, 1))
        from .design import build_design

        built = build_design(data, self.zero_design)
        if built.n_obs != n_rows:
            raise ValueError("new data has missing values in the zero-part columns")
        return built.x


    def _new_offset(self, data: pl.DataFrame, design: Design) -> np.ndarray:
        if self.offset_ref is not None:
            series = column_series(data, self.offset_ref).gather(design.row_index.tolist())
            return np.asarray(series.to_numpy(), dtype=float)
        if self.formula is not None:
            built = model_matrix(self.formula, data)
            if built.offset is not None and len(built.offset) == design.n_obs:
                return np.asarray(built.offset, dtype=float)
        raise ValueError("predict on new data needs the offset; refit with offset= or offset() in the formula")


def fit_mixed(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None = None,
    *,
    groups: ColumnRef | None = None,
    slopes: Sequence[str] | None = None,
    method: str = "reml",
    family: str = "gaussian",
    n_agq: int = 1,
    engine: str | None = None,
    link: str | None = None,
    offset: ColumnRef | None = None,
    zero_inflation: bool | Sequence[ColumnRef] | None = None,
    hurdle: bool = False,
) -> MixedFit:
    """Fit a linear or generalized linear mixed model.

    ``outcome`` may be a Wilkinson formula such as ``"y ~ x + (1 | g)"`` or a
    response column with ``predictors`` and ``groups``. ``family`` is
    ``"gaussian"`` (default, ``lmer`` / REML or ML), ``"binomial"``,
    ``"poisson"``, ``"gamma"``, or ``"negative_binomial"`` (NB2, variance
    ``mu + mu^2 / theta``). ``method`` is ``"reml"`` or ``"ml"`` and applies to
    Gaussian LMMs only.

    ``offset`` is a column added to the linear predictor, for example the log
    of person-time in a rate model. A formula takes ``offset(log(t))`` instead.
    Offsets need the Poisson or negative binomial family.

    ``zero_inflation`` adds a zero part to a Poisson or negative binomial
    model: ``True`` for a constant zero probability (glmmTMB ``ziformula=~1``),
    or a list of columns for its logistic model. ``hurdle=True`` makes it a
    hurdle model instead (glmmTMB ``truncated_poisson`` /
    ``truncated_nbinom2``): zeros come only from the zero part and positive
    counts from the zero-truncated distribution. The zero part has fixed
    effects only. See :meth:`MixedFit.zero_table`.

    ``engine`` defaults to ``"laplace"`` for binomial (logit link), Poisson,
    and negative binomial, and to ``"lme"`` (``lme-python`` / lme-rs) for
    Gaussian, Gamma, and other binomial links. ``"laplace"`` maximises the
    Laplace approximation, or adaptive Gauss–Hermite quadrature with
    ``n_agq > 1`` and a single random intercept. The covariance inverts the
    Hessian over all parameters, as ``glmmTMB`` and ``glmer`` do.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
    if method not in {"reml", "ml"}:
        raise ValueError("method must be 'reml' or 'ml'")
    engine_key = _normalize_engine(engine, family, link)
    if hurdle and zero_inflation is None:
        zero_inflation = True
    zero_part = None if zero_inflation is None or zero_inflation is False else ("hurdle" if hurdle else "inflated")
    if zero_part is not None and engine_key != "laplace":
        raise ValueError("zero-inflated and hurdle models need the Poisson or negative binomial family")
    if family != "gaussian" and method == "reml":
        method = "ml"

    formula, design, y, labels, group_name, random_names, extra, formula_offset = _prepare(
        data, outcome, predictors, groups, slopes, offset
    )
    terms = extra.pop("__terms__", None)
    if terms is not None:
        if engine_key != "laplace":
            raise ValueError(
                "several random-effect terms and ar1() need engine='laplace' (binomial, Poisson, or negative binomial)"
            )
        if n_agq > 1:
            raise ValueError("n_agq > 1 needs a single random effect, for example (1 | g)")
    if engine_key == "laplace":
        offset_values = formula_offset
        if offset is not None:
            offset_values = np.asarray(
                column_series(data, offset).gather(design.row_index.tolist()).to_numpy(), dtype=float
            )
        zero_design = None
        if zero_part is not None:
            zero_design = _zero_design(data, design, zero_inflation)
        if terms is not None:
            return _fit_laplace_terms(
                formula,
                design,
                y,
                terms,
                offset_values,
                offset_ref=offset,
                family=family,
                zero_part=zero_part,
                zero_design=zero_design,
            )
        return _fit_laplace(
            formula,
            design,
            y,
            labels,
            group_name,
            random_names,
            extra,
            offset_values,
            offset_ref=offset,
            family=family,
            n_agq=n_agq,
            link=link,
            zero_part=zero_part,
            zero_design=zero_design,
        )
    if offset is not None or formula_offset is not None:
        raise ValueError("offsets need engine='laplace' (binomial, Poisson, or negative binomial)")
    return _fit_lme(
        data,
        formula,
        design,
        group_name,
        random_names,
        family=family,
        method=method,
        n_agq=n_agq,
        link=link,
    )


def _normalize_engine(engine: str | None, family: str, link: str | None = None) -> str:
    laplace_ok = family in COUNT_FAMILIES and link in (None, "log") or family == "binomial" and link in (None, "logit")
    if engine is None:
        return "laplace" if laplace_ok else "lme"
    key = engine.strip().lower()
    if key in _LAPLACE_ENGINES:
        if not laplace_ok:
            raise ValueError("engine='laplace' fits binomial (logit), Poisson, and negative binomial (log) models")
        return "laplace"
    if key in _LME_ENGINES:
        if family == "negative_binomial":
            raise ValueError("the negative binomial family needs engine='laplace'")
        return "lme"
    raise ValueError("engine must be 'lme' or 'laplace'")


def _prepare(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None,
    groups: ColumnRef | None,
    slopes: Sequence[str] | None,
    offset: ColumnRef | None = None,
) -> tuple[str, Design, np.ndarray, list[object], str, list[str], dict[str, np.ndarray], np.ndarray | None]:
    if is_formula(outcome):
        if predictors is not None or groups is not None or slopes is not None:
            raise ValueError("a formula already names the predictors, groups, and slopes")
        if offset is not None:
            raise ValueError("a formula takes offset() instead of offset=")
        formula = str(outcome)
        design, y, labels, group_name, random_names, extra, formula_offset = _from_formula(data, formula)
        return formula, design, y, labels, group_name, random_names, extra, formula_offset
    if predictors is None or groups is None:
        raise ValueError("predictors and groups are required when outcome is a column")
    slope_names = list(slopes or [])
    predictor_names = [_column_name(data, col) for col in predictors]
    missing = [name for name in slope_names if name not in predictor_names]
    if missing:
        raise ValueError(f"random slopes must also be fixed effects: {missing}")
    extra_refs: list[ColumnRef] = [outcome, groups]
    if offset is not None:
        extra_refs.append(offset)
    design = design_matrix(data, predictors, extra=extra_refs)
    idx = design.row_index
    y = np.asarray(column_series(data, outcome).gather(idx.tolist()).to_numpy(), dtype=float)
    labels = column_series(data, groups).gather(idx.tolist()).to_list()
    group_name = column_series(data, groups).name
    random_names = _random_names(design, slope_names)
    extra: dict[str, np.ndarray] = {}
    outcome_name = _column_name(data, outcome)
    re_rhs = " + ".join(random_names).replace("(Intercept)", "1")
    fe = " + ".join(predictor_names) if predictor_names else "1"
    formula = f"{outcome_name} ~ {fe} + ({re_rhs} | {group_name})"
    return formula, design, y, labels, group_name, random_names, extra, None


def _fit_lme(
    data: pl.DataFrame,
    formula: str,
    design: Design,
    group_name: str,
    random_names: list[str],
    *,
    family: str,
    method: str,
    n_agq: int,
    link: str | None,
) -> MixedFit:
    import lme_python

    used = data.gather(design.row_index.tolist())
    if family == "gaussian":
        # lme-python's default tolerance of 1e-6 stops about 1e-4 short of the
        # log-likelihood optimum, which moves the coefficients by 1e-4 against
        # lmer. At 1e-8 it reaches the optimum, and lmer's on a well-conditioned
        # design.
        raw = lme_python.lmer(
            formula,
            data=used,
            reml=(method == "reml"),
            control=lme_python.FitControl(tolerance=_LMER_TOLERANCE),
        )
    else:
        kwargs: dict[str, Any] = {"family_name": family, "n_agq": int(n_agq)}
        if link is not None:
            kwargs["link_name"] = link
        raw = lme_python.glmer(formula, data=used, **kwargs)
        method = "ml"
    names = list(raw.fixed_names)
    coefficients = np.asarray(raw.coefficients, dtype=float)
    se = np.asarray(raw.std_errors, dtype=float)
    v_unscaled = np.asarray(raw.v_beta_unscaled, dtype=float)
    residual = None if raw.sigma2 is None else float(raw.sigma2)
    if v_unscaled.ndim == 2 and v_unscaled.shape == (len(names), len(names)):
        covariance = v_unscaled * (1.0 if residual is None else residual)
    else:
        covariance = np.diag(np.square(se))
    group_covariance = _group_covariance_from_var_corr(raw.var_corr, random_names, group_name)
    n_groups = int(len({row[1] for row in raw.ranef if row[0] == group_name}))
    n_obs = int(raw.num_obs)
    x = design.x
    if x.shape[1] != len(coefficients) or list(design.names) != names:
        try:
            x = np.asarray(raw.design_matrix(used), dtype=float)
        except Exception:  # noqa: BLE001 — keep our design if lme cannot rebuild X
            x = design.x
    diagnostics = getattr(raw, "diagnostics", None) or {}
    n_iter = int(raw.iterations)
    inner = diagnostics.get("inner_iterations") if isinstance(diagnostics, dict) else None
    return MixedFit(
        coefficients=coefficients,
        covariance=np.asarray(covariance, dtype=float),
        names=names,
        n_obs=n_obs,
        n_groups=n_groups,
        log_likelihood=float(raw.log_likelihood),
        residual_variance=residual,
        group_name=group_name,
        random_names=list(random_names),
        group_covariance=group_covariance,
        method=method,
        converged=bool(raw.converged),
        n_iter=n_iter,
        function_evals=int(inner or 0),
        x=x,
        row_index=design.row_index,
        design=design,
        family=family,
        engine="lme",
        n_agq=1 if family == "gaussian" else int(n_agq),
        formula=str(raw.formula) if getattr(raw, "formula", None) else formula,
        raw=raw,
    )



def _fit_laplace(
    formula: str,
    design: Design,
    y: np.ndarray,
    labels: list[object],
    group_name: str,
    random_names: list[str],
    extra: dict[str, np.ndarray],
    offset: np.ndarray | None,
    *,
    offset_ref: ColumnRef | None,
    family: str,
    n_agq: int,
    link: str | None,
    zero_part: str | None = None,
    zero_design: Design | None = None,
) -> MixedFit:
    columns = []
    for name in random_names:
        if name == "(Intercept)":
            columns.append(np.ones(design.n_obs))
        elif name in design.names:
            columns.append(design.x[:, design.names.index(name)])
        else:
            columns.append(np.asarray(extra[name], dtype=float))
    z = np.column_stack(columns)
    level_values, group = np.unique(np.asarray([str(v) for v in labels]), return_inverse=True)
    result = fit_laplace_glmm(
        y,
        design.x,
        z,
        group,
        offset,
        family=family,
        n_agq=n_agq,
        zero=zero_part,
        zero_x=None if zero_design is None else zero_design.x,
    )
    modes = result.modes
    ranef: dict[str, Any] = {"group": list(level_values), "blup": modes[:, 0]}
    for k, name in enumerate(random_names[1:], start=1):
        ranef[name] = modes[:, k]
    return MixedFit(
        coefficients=result.coefficients,
        covariance=result.covariance,
        names=list(design.names),
        n_obs=design.n_obs,
        n_groups=len(level_values),
        log_likelihood=result.log_likelihood,
        residual_variance=None,
        group_name=group_name,
        random_names=list(random_names),
        group_covariance=result.group_covariance,
        method="ml",
        converged=result.converged,
        n_iter=result.n_iter,
        function_evals=result.function_evals,
        x=design.x,
        row_index=design.row_index,
        design=design,
        family=family,
        engine="laplace",
        n_agq=int(n_agq),
        formula=formula,
        offset=offset,
        theta=result.theta,
        theta_std_error=result.theta_std_error,
        offset_ref=offset_ref,
        zero_part=zero_part,
        zero_coefficients=result.zero_coefficients,
        zero_covariance=result.zero_covariance,
        zero_names=None if zero_design is None else list(zero_design.names),
        zero_design=zero_design,
        ranef_frame=pl.DataFrame(ranef).sort("blup"),
    )


@dataclass
class _TermColumns:
    group: str
    structure: str
    names: list[str]
    z: np.ndarray
    codes: np.ndarray
    levels: list[str]


def _effect_columns(data: pl.DataFrame, design: Design, effect: Any) -> _TermColumns:
    """Columns of one random-effect term on the rows the model uses."""
    idx = design.row_index
    labels = np.asarray([str(v) for v in column_series(data, effect.group).gather(idx.tolist()).to_list()])
    levels, codes = np.unique(labels, return_inverse=True)
    if effect.structure == "ar1":
        if effect.intercept or len(effect.slopes) != 1 or effect.slope_exprs:
            raise ValueError("ar1() takes one time factor without an intercept, for example ar1(year + 0 | g)")
        name = effect.slopes[0]
        series = column_series(data, name)
        if series.dtype.is_numeric():
            order = sorted(series.drop_nulls().unique().to_list())
            keys = [str(v) for v in order]
            values = [str(v) for v in series.gather(idx.tolist()).to_list()]
        else:
            from .design import _factor_levels, _level_key

            keys = _factor_levels(series, None)
            values = [_level_key(v) for v in series.gather(idx.tolist()).to_list()]
        position = {key: k for k, key in enumerate(keys)}
        z = np.zeros((len(idx), len(keys)))
        z[np.arange(len(idx)), [position[v] for v in values]] = 1.0
        return _TermColumns(effect.group, "ar1", [f"{name}{key}" for key in keys], z, codes, list(levels))
    columns: list[np.ndarray] = []
    names: list[str] = []
    if effect.intercept:
        columns.append(np.ones(len(idx)))
        names.append("(Intercept)")
    expressed = dict(effect.slope_exprs)
    for name in effect.slopes:
        if name in design.names:
            columns.append(design.x[:, design.names.index(name)])
        elif name in expressed:
            from .formula import _columns_in, _eval_arith

            keep = np.zeros(data.height, dtype=bool)
            keep[idx] = True
            cache = {col: column_series(data, col).to_numpy() for col in _columns_in(expressed[name])}
            columns.append(_eval_arith(expressed[name], cache, keep))
        else:
            series = column_series(data, name)
            if series.dtype == pl.Boolean or _is_factor(series):
                raise ValueError(f"random slope {name!r} must be numeric")
            columns.append(np.asarray(series.gather(idx.tolist()).to_numpy(), dtype=float))
        names.append(name)
    if not names:
        raise ValueError("random effect has no intercept and no slopes")
    return _TermColumns(effect.group, "us", names, np.column_stack(columns), codes, list(levels))


def _fit_laplace_terms(
    formula: str,
    design: Design,
    y: np.ndarray,
    terms: list[_TermColumns],
    offset: np.ndarray | None,
    *,
    offset_ref: ColumnRef | None,
    family: str,
    zero_part: str | None,
    zero_design: Design | None,
) -> MixedFit:
    from ._laplace import RandomTerm, fit_laplace_terms

    result = fit_laplace_terms(
        y,
        design.x,
        [RandomTerm(t.z, t.codes, len(t.levels), t.structure) for t in terms],
        offset,
        family=family,
        zero=zero_part,
        zero_x=None if zero_design is None else zero_design.x,
    )
    random_terms = []
    blups = []
    for t, cov, rho, modes in zip(terms, result.group_covariances, result.correlations, result.modes, strict=True):
        random_terms.append(
            {"group": t.group, "structure": t.structure, "names": list(t.names), "covariance": cov, "correlation": rho}
        )
        for k, name in enumerate(t.names):
            blups.append(
                pl.DataFrame(
                    {"group": t.group, "level": list(t.levels), "term": name, "blup": modes[:, k]},
                    schema={"group": pl.String, "level": pl.String, "term": pl.String, "blup": pl.Float64},
                )
            )
    first = terms[0]
    return MixedFit(
        coefficients=result.coefficients,
        covariance=result.covariance,
        names=list(design.names),
        n_obs=design.n_obs,
        n_groups=len(first.levels),
        log_likelihood=result.log_likelihood,
        residual_variance=None,
        group_name=first.group,
        random_names=list(first.names),
        group_covariance=result.group_covariances[0],
        method="ml",
        converged=result.converged,
        n_iter=result.n_iter,
        function_evals=result.function_evals,
        x=design.x,
        row_index=design.row_index,
        design=design,
        family=family,
        engine="laplace",
        n_agq=1,
        formula=formula,
        offset=offset,
        theta=result.theta,
        theta_std_error=result.theta_std_error,
        offset_ref=offset_ref,
        zero_part=zero_part,
        zero_coefficients=result.zero_coefficients,
        zero_covariance=result.zero_covariance,
        zero_names=None if zero_design is None else list(zero_design.names),
        zero_design=zero_design,
        ranef_frame=pl.concat(blups),
        random_terms=random_terms,
    )


def _zero_design(data: pl.DataFrame, design: Design, spec: Any) -> Design:
    """Design of the zero part on the rows the count model uses."""
    used = data.gather(design.row_index.tolist())
    columns = [] if spec is True else list(spec)
    zero = design_matrix(used, columns)
    if zero.n_obs != design.n_obs:
        raise ValueError("the zero-inflation columns have missing values in rows the model uses")
    return zero


def _terms_variance_table(terms: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for term in terms:
        cov = np.asarray(term["covariance"], dtype=float)
        names = term["names"]
        intercept_at = names.index("(Intercept)") if "(Intercept)" in names else None
        for i, name in enumerate(names):
            variance = float(cov[i, i])
            std_dev = float(np.sqrt(max(variance, 0.0)))
            if term["structure"] == "ar1":
                correlation = term["correlation"]
            elif intercept_at is None or i == intercept_at:
                correlation = None
            else:
                denom = np.sqrt(max(cov[intercept_at, intercept_at], 0.0)) * std_dev
                correlation = None if denom == 0 else float(cov[i, intercept_at] / denom)
            rows.append(
                {
                    "group": term["group"],
                    "term": name,
                    "structure": term["structure"],
                    "variance": variance,
                    "std_dev": std_dev,
                    "correlation": correlation,
                }
            )
    return pl.DataFrame(rows)


def _wald_table(names: list[str], estimate: np.ndarray, covariance: np.ndarray, level: float, exponentiate: bool) -> pl.DataFrame:
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = estimate / se
    crit = float(stats.norm.ppf(0.5 + level / 2))
    frame = pl.DataFrame(
        {
            "term": names,
            "estimate": estimate,
            "std_error": se,
            "statistic": stat,
            "p_value": 2 * stats.norm.sf(np.abs(stat)),
            "conf_low": estimate - crit * se,
            "conf_high": estimate + crit * se,
        }
    )
    if exponentiate:
        frame = frame.with_columns(
            pl.col("estimate").exp().alias("exp_estimate"),
            pl.col("conf_low").exp().alias("exp_conf_low"),
            pl.col("conf_high").exp().alias("exp_conf_high"),
        )
    return frame


def _group_covariance_from_var_corr(
    var_corr: Sequence[tuple[Any, ...]],
    random_names: list[str],
    group_name: str,
) -> np.ndarray:
    k = len(random_names)
    cov = np.zeros((k, k), dtype=float)
    index = {name: i for i, name in enumerate(random_names)}
    for row in var_corr:
        grp, t1, t2, value, _sd = row[0], row[1], row[2], row[3], row[4] if len(row) > 4 else None
        if str(grp) in {"Residual", "residual"}:
            continue
        if str(grp) != str(group_name):
            continue
        if t1 not in index or t2 not in index:
            continue
        i, j = index[t1], index[t2]
        cov[i, j] = float(value)
        cov[j, i] = float(value)
    return cov


def _ranef_frame(ranef: Sequence[tuple[Any, ...]], group_name: str) -> pl.DataFrame:
    by_group: dict[str, dict[str, float]] = {}
    terms: list[str] = []
    for row in ranef:
        grp, label, term, value = row[0], row[1], row[2], row[3]
        if str(grp) != str(group_name):
            continue
        key = str(label)
        if term not in terms:
            terms.append(str(term))
        by_group.setdefault(key, {})[str(term)] = float(value)
    if not by_group:
        return pl.DataFrame({"group": [], "blup": []})
    records = []
    for label, values in by_group.items():
        rec: dict[str, object] = {"group": label, "blup": values.get("(Intercept)", values[next(iter(values))])}
        for term in terms:
            if term == "(Intercept)":
                continue
            rec[term] = values.get(term)
        records.append(rec)
    frame = pl.DataFrame(records)
    return frame.sort("blup")


def _inv_link(family: str, eta: np.ndarray) -> np.ndarray:
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    if family in COUNT_FAMILIES:
        return np.exp(eta)
    if family == "gamma":
        return 1.0 / eta
    return eta


def _column_name(data: pl.DataFrame, col: ColumnRef) -> str:
    return col if isinstance(col, str) else column_series(data, col).name


def _random_names(design: Design, slope_names: list[str]) -> list[str]:
    names = ["(Intercept)"]
    lookup = {spec.name: spec for spec in design.predictors}
    for name in slope_names:
        spec = lookup.get(name)
        if spec is None or spec.kind != "numeric":
            raise ValueError(f"random slope {name!r} must be a numeric predictor")
        names.append(name)
    return names


def _from_formula(
    data: pl.DataFrame, formula: str
) -> tuple[Design, np.ndarray, list[object], str, list[str], dict[str, np.ndarray], np.ndarray | None]:
    built = model_matrix(formula, data)
    reject_survival_syntax(built)
    if not built.random_effects:
        raise ValueError("fit_mixed needs a random effect, for example (1 | group)")
    for one in built.random_effects:
        if not one.correlated:
            raise ValueError("uncorrelated random slopes (||) are not supported")
    effect = built.random_effects[0]
    design = built.design
    idx = design.row_index
    offset = None if built.offset is None else np.asarray(built.offset, dtype=float)
    if len(built.random_effects) > 1 or effect.structure != "us":
        terms = [_effect_columns(data, design, one) for one in built.random_effects]
        return design, np.asarray(built.y, dtype=float), [], effect.group, [], {"__terms__": terms}, offset
    group = column_series(data, effect.group)
    labels = group.gather(idx.tolist()).to_list()
    random_names = ["(Intercept)"] if effect.intercept else []
    extra: dict[str, np.ndarray] = {}
    expressed = dict(effect.slope_exprs)
    for name in effect.slopes:
        spec = next((item for item in design.predictors if item.name == name), None)
        if spec is not None and spec.kind != "numeric":
            raise ValueError(f"random slope {name!r} must be numeric")
        if name in design.names:
            random_names.append(name)
            continue
        if name in expressed:
            from .formula import _columns_in, _eval_arith

            keep = np.zeros(data.height, dtype=bool)
            keep[idx] = True
            cache = {col: column_series(data, col).to_numpy() for col in _columns_in(expressed[name])}
            extra[name] = _eval_arith(expressed[name], cache, keep)
            random_names.append(name)
            continue
        series = column_series(data, name)
        if series.dtype == pl.Boolean or _is_factor(series):
            raise ValueError(f"random slope {name!r} must be numeric")
        extra[name] = np.asarray(series.gather(idx.tolist()).to_numpy(), dtype=float)
        random_names.append(name)
    if not random_names:
        raise ValueError("random effect has no intercept and no slopes")
    return design, np.asarray(built.y, dtype=float), labels, effect.group, random_names, extra, offset

