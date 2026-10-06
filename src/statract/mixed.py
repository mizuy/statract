"""Linear and generalized linear mixed models.

``fit_mixed`` is the public mixed-model entry point, parallel to ``fit_ols`` /
``fit_glm``. Gaussian LMMs call ``lme_python.lmer`` (lme-rs). Binomial,
Poisson, and Gamma GLMMs call ``lme_python.glmer``. Wilkinson formulas such as
``y ~ x + (1 | g)`` work for both.

``engine="mixedlm_rs"`` is an optional Gaussian-only fallback (extra
``statract[mixedlm]``). It is not the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl
from scipy import stats

from .design import ColumnRef, Design, _is_factor, column_series, design_matrix
from .formula import is_formula, model_matrix, reject_survival_syntax

_FAMILIES = ("gaussian", "binomial", "poisson", "gamma")
_LME_ENGINES = {"lme", "lme-rs", "lme-python", "lme_python"}
_MIXEDLM_ENGINES = {"mixedlm_rs", "mixedlm-rs", "mixedlm"}
_LMER_TOLERANCE = 1e-8


@dataclass
class MixedFit:
    """Fitted linear or generalized linear mixed model.

    ``coefficients`` and ``covariance`` are the fixed effects. ``tidy`` uses a
    normal reference (Wald). ``group_covariance`` is the random-effect
    covariance for ``(Intercept)`` and any random slopes of ``group_name``.
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
    raw: Any = field(default=None, repr=False)

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Fixed-effect table. Intervals use a normal (Wald) reference."""
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        estimate = self.coefficients
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = estimate / se
        crit = float(stats.norm.ppf(0.5 + level / 2))
        frame = pl.DataFrame(
            {
                "term": self.names,
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

    def variance_table(self) -> pl.DataFrame:
        """Random-effect variances (and residual variance for Gaussian LMMs).

        ``correlation`` is the correlation with the intercept. The residual
        row leaves it empty.
        """
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
        """Unique-group BLUPs for the grouping factor (intercept, then slopes)."""
        if self.raw is not None and getattr(self.raw, "ranef", None) is not None:
            return _ranef_frame(self.raw.ranef, self.group_name)
        raise TypeError("random effects are not available on this fit")

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "response") -> np.ndarray:
        """Population mean (fixed effects only).

        ``kind`` is ``"link"`` or ``"response"``. Gaussian LMMs ignore ``kind``.
        """
        if kind not in {"link", "response"}:
            raise ValueError("kind must be 'link' or 'response'")
        if data is not None and self.raw is not None:
            if kind == "response" and self.family != "gaussian" and hasattr(self.raw, "predict_response"):
                return np.asarray(self.raw.predict_response(data), dtype=float)
            return np.asarray(self.raw.predict(data), dtype=float)
        if data is None:
            eta = self.x @ self.coefficients
        else:
            from .design import build_design

            design = build_design(data, self.design)
            if design.x.shape[1] != len(self.coefficients):
                raise ValueError("new data produced a different number of columns")
            eta = design.x @ self.coefficients
        if kind == "link" or self.family == "gaussian":
            return eta
        return _inv_link(self.family, eta)


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
    engine: str = "lme",
    link: str | None = None,
) -> MixedFit:
    """Fit a linear or generalized linear mixed model.

    ``outcome`` may be a Wilkinson formula such as ``"y ~ x + (1 | g)"`` or a
    response column with ``predictors`` and ``groups``. ``family`` is
    ``"gaussian"`` (default, ``lmer`` / REML or ML), ``"binomial"``,
    ``"poisson"``, or ``"gamma"`` (``glmer``). ``method`` is ``"reml"`` or
    ``"ml"`` and applies to Gaussian LMMs only.

    The default ``engine`` is ``"lme"`` (``lme-python`` / lme-rs). Gaussian
    fits may use ``engine="mixedlm_rs"`` when the optional extra is installed.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
    if method not in {"reml", "ml"}:
        raise ValueError("method must be 'reml' or 'ml'")
    engine_key = _normalize_engine(engine)
    if engine_key == "mixedlm_rs" and family != "gaussian":
        raise ValueError("engine='mixedlm_rs' is Gaussian-only; use engine='lme' for GLMMs")
    if family != "gaussian" and method == "reml":
        method = "ml"

    formula, design, y, labels, group_name, random_names, extra_columns = _prepare(
        data, outcome, predictors, groups, slopes
    )
    if engine_key == "mixedlm_rs":
        return _fit_mixedlm_rs(
            design,
            y,
            labels,
            group_name,
            random_names,
            extra_columns,
            method=method,
        )
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


def _normalize_engine(engine: str) -> str:
    key = engine.strip().lower()
    if key in _LME_ENGINES:
        return "lme"
    if key in _MIXEDLM_ENGINES:
        return "mixedlm_rs"
    raise ValueError("engine must be 'lme' or 'mixedlm_rs'")


def _prepare(
    data: pl.DataFrame,
    outcome: ColumnRef,
    predictors: Sequence[ColumnRef] | None,
    groups: ColumnRef | None,
    slopes: Sequence[str] | None,
) -> tuple[str, Design, np.ndarray, list[object], str, list[str], dict[str, np.ndarray]]:
    if is_formula(outcome):
        if predictors is not None or groups is not None or slopes is not None:
            raise ValueError("a formula already names the predictors, groups, and slopes")
        formula = str(outcome)
        design, y, labels, group_name, random_names, extra = _from_formula(data, formula)
        return formula, design, y, labels, group_name, random_names, extra
    if predictors is None or groups is None:
        raise ValueError("predictors and groups are required when outcome is a column")
    slope_names = list(slopes or [])
    predictor_names = [_column_name(data, col) for col in predictors]
    missing = [name for name in slope_names if name not in predictor_names]
    if missing:
        raise ValueError(f"random slopes must also be fixed effects: {missing}")
    extra_refs: list[ColumnRef] = [outcome, groups]
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
    return formula, design, y, labels, group_name, random_names, extra


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


def _fit_mixedlm_rs(
    design: Design,
    y: np.ndarray,
    labels: list[object],
    group_name: str,
    random_names: list[str],
    extra_columns: dict[str, np.ndarray],
    *,
    method: str,
) -> MixedFit:
    try:
        import mixedlm_rs
    except ImportError as exc:
        raise ImportError(
            "engine='mixedlm_rs' requires the optional extra: pip install 'statract[mixedlm]'"
        ) from exc
    exog_re = _random_exog(design, random_names, extra_columns)
    result = mixedlm_rs.MixedLM(
        y,
        design.x,
        np.asarray(labels),
        exog_re=exog_re,
        exog_names=list(design.names),
        exog_re_names=list(random_names),
    ).fit(reml=(method == "reml"))
    coefficients = np.asarray(result.fe_params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)[: coefficients.size, : coefficients.size]
    group_covariance = np.atleast_2d(np.asarray(result.cov_re, dtype=float))
    if group_covariance.shape != (len(random_names), len(random_names)):
        raise RuntimeError("random-effect covariance has an unexpected shape")
    diagnostics = result.diagnostics
    return MixedFit(
        coefficients=coefficients,
        covariance=covariance,
        names=list(design.names),
        n_obs=design.n_obs,
        n_groups=int(len(set(labels))),
        log_likelihood=float(result.llf),
        residual_variance=float(result.scale),
        group_name=group_name,
        random_names=random_names,
        group_covariance=group_covariance,
        method=method,
        converged=bool(result.converged),
        n_iter=int(diagnostics["n_iterations"]),
        function_evals=int(diagnostics["n_objective_evaluations"]),
        x=design.x,
        row_index=design.row_index,
        design=design,
        family="gaussian",
        engine="mixedlm_rs",
        n_agq=1,
        formula=None,
        raw=result,
    )


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
    if family == "poisson":
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
) -> tuple[Design, np.ndarray, list[object], str, list[str], dict[str, np.ndarray]]:
    built = model_matrix(formula, data)
    reject_survival_syntax(built)
    if len(built.random_effects) != 1:
        raise ValueError("fit_mixed needs exactly one grouping factor, for example (1 | group)")
    effect = built.random_effects[0]
    if not effect.correlated:
        raise ValueError("uncorrelated random slopes (||) are not supported")
    if built.offset is not None:
        raise ValueError("offset() is not supported in fit_mixed")
    design = built.design
    idx = design.row_index
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
    return design, np.asarray(built.y, dtype=float), labels, effect.group, random_names, extra


def _random_exog(design: Design, random_names: list[str], extra: dict[str, np.ndarray]) -> np.ndarray:
    """Columns of the random-effect design, in ``random_names`` order."""
    columns = []
    for name in random_names:
        if name == "(Intercept)":
            columns.append(np.ones(design.n_obs, dtype=float))
        elif name in design.names:
            columns.append(np.asarray(design.x[:, design.names.index(name)], dtype=float))
        else:
            columns.append(np.asarray(extra[name], dtype=float))
    return np.column_stack(columns)
