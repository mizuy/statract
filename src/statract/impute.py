"""Multiple imputation by chained equations, and Rubin's rules.

``impute_chained`` follows the defaults of R's ``mice``: predictive mean
matching for numbers, Bayesian logistic regression for two-level columns,
and multinomial logistic regression for factors with more levels. Each
column with missing values is imputed in turn from all the other columns,
for ``n_iter`` sweeps, in ``m`` independent chains.

``pool`` combines the estimates of a model fitted to each completed data
set by Rubin's rules, with the Barnard–Rubin degrees of freedom. It matches
``mice::pool`` and ``summary(pool(...), conf.int = TRUE)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize, special, stats

_N_DONORS = 5
_RIDGE = 1e-5


@dataclass
class MultipleImputation:
    """``m`` completed copies of a data frame.

    ``datasets`` holds the completed frames. ``methods`` names the method
    used for each imputed column. ``missing`` counts the missing values that
    were filled in each column.
    """

    datasets: list[pl.DataFrame]
    methods: dict[str, str]
    missing: dict[str, int]
    n_iter: int
    seed: int | None = None
    chain_means: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def m(self) -> int:
        return len(self.datasets)

    def complete(self, i: int) -> pl.DataFrame:
        """The ``i``-th completed data set (0-based)."""
        return self.datasets[i]

    def long(self) -> pl.DataFrame:
        """All completed data sets stacked, with an ``imputation`` column (1..m)."""
        return pl.concat(
            [frame.with_columns(pl.lit(i + 1).alias("imputation")) for i, frame in enumerate(self.datasets)]
        )

    def fit(self, fit_fn: Callable[[pl.DataFrame], Any]) -> list[Any]:
        """Apply ``fit_fn`` to every completed data set."""
        return [fit_fn(frame) for frame in self.datasets]

    def pool(self, fit_fn: Callable[[pl.DataFrame], Any], **kwargs: Any) -> pl.DataFrame:
        """Fit a model to every completed data set and pool it with :func:`pool`."""
        return pool(self.fit(fit_fn), **kwargs)


def impute_chained(
    data: pl.DataFrame,
    *,
    m: int = 5,
    n_iter: int = 5,
    columns: Sequence[str] | None = None,
    methods: dict[str, str] | None = None,
    seed: int | None = None,
) -> MultipleImputation:
    """Impute missing values ``m`` times by chained equations (MICE).

    ``columns`` limits the imputation model to those columns (default: all).
    Other columns are carried along unchanged and do not predict. Every
    column in the model with missing values is imputed; the rest only
    predict. ``methods`` overrides the method per column: ``"pmm"``
    (numbers), ``"logreg"`` (two levels), ``"polyreg"`` (factors), or
    ``"none"`` (leave missing and do not use as a predictor). Booleans come
    back as booleans and factors keep their dtype.

    The draws differ from R's ``mice`` (another random number generator), so
    results agree with mice in distribution, not digit by digit.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    names = list(data.columns if columns is None else columns)
    missing_cols = [c for c in names if c not in data.columns]
    if missing_cols:
        raise ValueError(f"columns not in data: {missing_cols}")
    specs = {name: _ColumnSpec.from_series(data[name]) for name in names}
    chosen = dict(methods or {})
    for name, method in chosen.items():
        if name not in specs:
            raise ValueError(f"methods names a column outside the model: {name}")
        if method not in {"pmm", "logreg", "polyreg", "none"}:
            raise ValueError(f"unknown method {method!r} for {name}")
    method_of: dict[str, str] = {}
    for name, spec in specs.items():
        if spec.n_missing == 0:
            continue
        method = chosen.get(name, spec.default_method)
        if method == "logreg" and spec.kind == "numeric" and not spec.binary_numeric:
            raise ValueError(f"logreg needs a two-level column: {name}")
        if method == "pmm" and spec.kind != "numeric":
            raise ValueError(f"pmm needs a numeric column: {name}")
        method_of[name] = method
    active = [n for n in names if method_of.get(n, "pmm") != "none"]
    targets = [n for n in active if n in method_of]
    rng = np.random.default_rng(seed)
    datasets: list[pl.DataFrame] = []
    chain_means: dict[str, np.ndarray] = {name: np.zeros((m, n_iter)) for name in targets if specs[name].kind == "numeric"}
    for chain in range(m):
        values = {name: specs[name].codes.copy() for name in active}
        # Start from random draws of the observed values, as mice does.
        for name in targets:
            spec = specs[name]
            observed = spec.codes[~spec.mask]
            if observed.size == 0:
                raise ValueError(f"column {name} has no observed values")
            values[name][spec.mask] = rng.choice(observed, size=int(spec.mask.sum()))
        for sweep in range(n_iter):
            for name in targets:
                spec = specs[name]
                x = _predictor_matrix(values, specs, [c for c in active if c != name])
                y = values[name]
                obs = ~spec.mask
                method = method_of[name]
                if method == "pmm":
                    y[spec.mask] = _impute_pmm(y[obs], x[obs], x[spec.mask], rng)
                elif method == "logreg":
                    y[spec.mask] = _impute_logreg(y[obs], x[obs], x[spec.mask], rng)
                else:
                    y[spec.mask] = _impute_polyreg(y[obs].astype(int), len(spec.levels), x[obs], x[spec.mask], rng)
                if name in chain_means:
                    chain_means[name][chain, sweep] = float(np.mean(y[spec.mask]))
        frame = data
        for name in targets:
            frame = frame.with_columns(specs[name].to_series(values[name]))
        datasets.append(frame)
    return MultipleImputation(
        datasets=datasets,
        methods={name: method_of[name] for name in targets},
        missing={name: specs[name].n_missing for name in targets},
        n_iter=n_iter,
        seed=seed,
        chain_means=chain_means,
    )


@dataclass
class _ColumnSpec:
    name: str
    kind: str  # numeric, boolean, factor
    codes: np.ndarray
    mask: np.ndarray
    levels: list[Any]
    dtype: pl.DataType
    integer: bool = False
    binary_numeric: bool = False

    @property
    def n_missing(self) -> int:
        return int(self.mask.sum())

    @property
    def default_method(self) -> str:
        if self.kind == "numeric":
            return "pmm"
        return "logreg" if len(self.levels) == 2 else "polyreg"

    @classmethod
    def from_series(cls, series: pl.Series) -> _ColumnSpec:
        mask = series.is_null().to_numpy()
        dtype = series.dtype
        if dtype.is_numeric():
            values = series.cast(pl.Float64).to_numpy().astype(float)
            values = np.where(mask, np.nan, values)
            if np.any(np.isnan(values[~mask])):
                raise ValueError(f"column {series.name} has NaN; use null for missing")
            observed = values[~mask]
            binary = observed.size > 0 and bool(np.all(np.isin(observed, (0.0, 1.0))))
            return cls(series.name, "numeric", values, mask, [], dtype, integer=dtype.is_integer(), binary_numeric=binary)
        if dtype == pl.Boolean:
            codes = np.where(mask, np.nan, series.fill_null(False).cast(pl.Float64).to_numpy())
            return cls(series.name, "boolean", codes, mask, [False, True], dtype)
        if isinstance(dtype, pl.Enum):
            levels = dtype.categories.to_list()
        else:
            levels = sorted({v for v in series.drop_nulls().cast(pl.String).to_list()})
        lookup = {level: k for k, level in enumerate(levels)}
        raw = series.cast(pl.String).to_list()
        codes = np.array([np.nan if v is None else lookup[v] for v in raw], dtype=float)
        return cls(series.name, "factor", codes, mask, levels, dtype)

    def design(self, values: np.ndarray) -> np.ndarray:
        """Predictor columns: the value itself, or dummies without the first level."""
        if self.kind != "factor" or len(self.levels) <= 2:
            return values[:, None]
        idx = values.astype(int)
        out = np.zeros((len(values), len(self.levels) - 1))
        rows = np.flatnonzero(idx > 0)
        out[rows, idx[rows] - 1] = 1.0
        return out

    def to_series(self, values: np.ndarray) -> pl.Series:
        if self.kind == "numeric":
            series = pl.Series(self.name, values)
            if self.integer:
                return series.round(0).cast(self.dtype)
            return series.cast(self.dtype)
        if self.kind == "boolean":
            return pl.Series(self.name, values > 0.5)
        labels = [self.levels[int(v)] for v in values]
        return pl.Series(self.name, labels).cast(self.dtype)


def _predictor_matrix(values: dict[str, np.ndarray], specs: dict[str, _ColumnSpec], names: list[str]) -> np.ndarray:
    blocks = [np.ones((len(next(iter(values.values()))), 1))]
    for name in names:
        blocks.append(specs[name].design(values[name]))
    return np.column_stack(blocks)


def _impute_pmm(y_obs: np.ndarray, x_obs: np.ndarray, x_mis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Predictive mean matching (mice ``pmm``, type-1 matching, 5 donors)."""
    beta_hat, beta_star = _norm_draw(y_obs, x_obs, rng)
    yhat_obs = x_obs @ beta_hat
    yhat_mis = x_mis @ beta_star
    donors = min(_N_DONORS, len(y_obs))
    out = np.empty(len(x_mis))
    for i, target in enumerate(yhat_mis):
        # Break ties at random, as mice's matcher does.
        dist = np.abs(yhat_obs - target) + rng.uniform(0, 1e-12, len(yhat_obs))
        nearest = np.argpartition(dist, donors - 1)[:donors]
        out[i] = y_obs[rng.choice(nearest)]
    return out


def _norm_draw(y: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit and one draw from its posterior (mice ``.norm.draw``)."""
    xtx = x.T @ x
    xtx = xtx + np.diag(np.diag(xtx)) * _RIDGE
    v = np.linalg.pinv(xtx)
    beta_hat = v @ x.T @ y
    resid = y - x @ beta_hat
    df = max(len(y) - x.shape[1], 1)
    sigma_star = np.sqrt(float(resid @ resid) / rng.chisquare(df))
    chol = np.linalg.cholesky(v + np.eye(len(v)) * 1e-12)
    beta_star = beta_hat + sigma_star * chol @ rng.standard_normal(len(beta_hat))
    return beta_hat, beta_star


def _impute_logreg(y_obs: np.ndarray, x_obs: np.ndarray, x_mis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bayesian logistic regression (mice ``logreg``): draw beta, then the outcome."""
    beta, cov = _logistic(y_obs, x_obs)
    chol = np.linalg.cholesky(cov + np.eye(len(cov)) * 1e-10)
    beta_star = beta + chol @ rng.standard_normal(len(beta))
    p = special.expit(x_mis @ beta_star)
    return (rng.uniform(size=len(p)) <= p).astype(float)


def _logistic(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = special.expit(x @ beta)
        w = np.clip(p * (1 - p), 1e-10, None)
        info = (x * w[:, None]).T @ x
        info += np.eye(len(info)) * _RIDGE * (1.0 + np.trace(info) / len(info))
        step = np.linalg.solve(info, x.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = special.expit(x @ beta)
    info = (x * np.clip(p * (1 - p), 1e-10, None)[:, None]).T @ x
    info += np.eye(len(info)) * _RIDGE * (1.0 + np.trace(info) / len(info))
    return beta, np.linalg.inv(info)


def _impute_polyreg(y_obs: np.ndarray, k: int, x_obs: np.ndarray, x_mis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial logistic regression (mice ``polyreg``): draw from the fitted probabilities."""
    p = x_obs.shape[1]
    onehot = np.zeros((len(y_obs), k))
    onehot[np.arange(len(y_obs)), y_obs] = 1.0

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = np.column_stack([np.zeros(p), flat.reshape(p, k - 1)])
        logits = x_obs @ coef
        log_prob = logits - special.logsumexp(logits, axis=1, keepdims=True)
        prob = np.exp(log_prob)
        penalty = 0.5 * _RIDGE * float(flat @ flat)
        value = -float(np.sum(onehot * log_prob)) + penalty
        grad = -(x_obs.T @ (onehot - prob))[:, 1:].ravel() + _RIDGE * flat
        return value, grad

    result = optimize.minimize(objective, np.zeros(p * (k - 1)), jac=True, method="L-BFGS-B")
    coef = np.column_stack([np.zeros(p), result.x.reshape(p, k - 1)])
    logits = x_mis @ coef
    prob = np.exp(logits - special.logsumexp(logits, axis=1, keepdims=True))
    cumulative = np.cumsum(prob, axis=1)
    draws = rng.uniform(size=(len(prob), 1))
    return np.minimum((draws > cumulative).sum(axis=1), k - 1).astype(float)


def pool(
    fits: Sequence[Any],
    *,
    dfcom: float | None = None,
    level: float = 0.95,
    exponentiate: bool = False,
) -> pl.DataFrame:
    """Pool estimates across imputations by Rubin's rules, as ``mice::pool``.

    Each fit needs ``names``, ``coefficients``, and ``covariance`` (any
    statract model). ``dfcom`` is the complete-data degrees of freedom;
    by default the residual df of a linear model or GLM, the number of
    events minus parameters for a Cox model, and ``n - p`` otherwise.

    Columns: ``term``, ``m``, ``estimate``, ``ubar`` (within variance), ``b``
    (between variance), ``t`` (total variance), ``dfcom``, ``df``
    (Barnard–Rubin), ``riv``, ``lambda``, ``fmi``, ``std_error``,
    ``statistic``, ``p_value``, ``conf_low``, ``conf_high``. With
    ``exponentiate=True`` it adds ``exp_estimate``, ``exp_conf_low``, and
    ``exp_conf_high``.
    """
    fits = list(fits)
    m = len(fits)
    if m < 2:
        raise ValueError("pool needs at least two fits")
    names = list(fits[0].names)
    for fit in fits[1:]:
        if list(fit.names) != names:
            raise ValueError("every fit must have the same terms in the same order")
    estimates = np.vstack([np.asarray(f.coefficients, dtype=float) for f in fits])
    variances = np.vstack([np.diag(np.asarray(f.covariance, dtype=float)) for f in fits])
    if dfcom is None:
        dfcom = _complete_df(fits[0])
    qbar = estimates.mean(axis=0)
    ubar = variances.mean(axis=0)
    b = estimates.var(axis=0, ddof=1)
    t = ubar + (1 + 1 / m) * b
    df = _barnard_rubin(m, b, t, dfcom)
    with np.errstate(divide="ignore", invalid="ignore"):
        riv = (1 + 1 / m) * b / ubar
        lam = (1 + 1 / m) * b / t
        fmi = (riv + 2 / (df + 3)) / (riv + 1)
    se = np.sqrt(t)
    statistic = qbar / se
    p_value = 2 * stats.t.sf(np.abs(statistic), df)
    crit = stats.t.ppf(0.5 + level / 2, df)
    frame = pl.DataFrame(
        {
            "term": names,
            "m": [m] * len(names),
            "estimate": qbar,
            "ubar": ubar,
            "b": b,
            "t": t,
            "dfcom": [float(dfcom)] * len(names),
            "df": df,
            "riv": riv,
            "lambda": lam,
            "fmi": fmi,
            "std_error": se,
            "statistic": statistic,
            "p_value": p_value,
            "conf_low": qbar - crit * se,
            "conf_high": qbar + crit * se,
        }
    )
    if exponentiate:
        frame = frame.with_columns(
            pl.col("estimate").exp().alias("exp_estimate"),
            pl.col("conf_low").exp().alias("exp_conf_low"),
            pl.col("conf_high").exp().alias("exp_conf_high"),
        )
    return frame


def _complete_df(fit: Any) -> float:
    """``mice:::get.dfcom``: residual df, Cox events minus p, else n minus p."""
    residual = getattr(fit, "residual_df", None)
    if residual is not None:
        return float(max(residual, 1))
    p = len(fit.coefficients)
    if getattr(fit, "family", None) == "cox" and getattr(fit, "event", None) is not None:
        return float(max(int(np.sum(np.asarray(fit.event) > 0)) - p, 1))
    n_obs = getattr(fit, "n_obs", None)
    if n_obs is not None:
        return float(max(int(n_obs) - p, 1))
    return float("inf")


def _barnard_rubin(m: int, b: np.ndarray, t: np.ndarray, dfcom: float) -> np.ndarray:
    """Barnard and Rubin (1999) degrees of freedom, as ``mice:::barnard.rubin``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (1 + 1 / m) * b / t
    lam = np.where(lam < 1e-4, 1e-4, lam)
    dfold = (m - 1) / lam**2
    if np.isinf(dfcom):
        return dfold
    dfobs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
    return dfold * dfobs / (dfold + dfobs)
