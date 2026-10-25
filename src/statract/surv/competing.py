"""Competing risks: cumulative incidence with Gray's test, and Fine–Gray regression.

The code is written from the papers:

- Gray, R. J. (1988). A class of K-sample tests for comparing the cumulative
  incidence of a competing risk. *Annals of Statistics* 16(3), 1141–1154.
  The curve is the estimate (2.3). The test is the score (2.5)–(2.6) with the
  variance (2.10), and the pooled curve (2.11) for the weight and the plug-ins.
- Fine, J. P. and Gray, R. J. (1999). A proportional hazards model for the
  subdistribution of a competing risk. *Journal of the American Statistical
  Association* 94(446), 496–509. The weighted score (6), the information (7),
  the sandwich with the censoring term (8), the Breslow baseline, and the
  predicted cumulative incidence of Section 5.
- Zhang, X., Zhang, M.-J. and Fine, J. (2011). A proportional hazards
  regression model for the subdistribution with right-censored and
  left-truncated competing risks data. *Statistics in Medicine* 30,
  1933–1951. It restates the same estimating equation and variance.

R's ``cmprsk`` (``cuminc``, ``timepoints``, ``crr``, ``predict.crr``) was used
only as a black box, to check the output. Its source was not read. Where the
papers leave a detail open (ties, small risk sets, censoring groups, when
Newton–Raphson stops), the choice was made by comparing variants with
``cmprsk`` output. The comments say which details were settled this way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ..models.design import ColumnRef, Design, build_design, column_series, design_matrix
from ..models.formula import is_formula
from .spec import parse_survival_formula


@dataclass
class CumulativeIncidence:
    """Cumulative incidence curves by group and cause, as ``cuminc`` returns them.

    ``table`` holds the step-function corners (``time``, ``estimate``,
    ``variance``) for each ``group`` and ``cause``. ``tests`` holds Gray's
    test for each cause when there are two or more groups.
    """

    table: pl.DataFrame
    tests: pl.DataFrame | None
    groups: list[str]
    causes: list[str]
    rho: float

    def frame(self) -> pl.DataFrame:
        """Corners of every curve, with ``std_error`` added."""
        return self.table.with_columns(pl.col("variance").clip(lower_bound=0).sqrt().alias("std_error"))

    def at(self, times: float | list[float] | np.ndarray) -> pl.DataFrame:
        """Estimates and variances at ``times``, matching ``timepoints``.

        Times past the last follow-up of a group give null, as in R.
        """
        grid = np.unique(np.atleast_1d(np.asarray(times, dtype=float)))
        rows: dict[str, list] = {"group": [], "cause": [], "time": [], "estimate": [], "variance": []}
        for (group, cause), part in self.table.group_by(["group", "cause"], maintain_order=True):
            x = part["time"].to_numpy()
            f = part["estimate"].to_numpy()
            v = part["variance"].to_numpy()
            index = _timepoint_index(x, grid)
            for t, k in zip(grid, index, strict=True):
                rows["group"].append(group)
                rows["cause"].append(cause)
                rows["time"].append(float(t))
                rows["estimate"].append(None if k < 0 else float(f[k]))
                rows["variance"].append(None if k < 0 else float(v[k]))
        return pl.DataFrame(
            rows,
            schema={
                "group": pl.String,
                "cause": pl.String,
                "time": pl.Float64,
                "estimate": pl.Float64,
                "variance": pl.Float64,
            },
        ).with_columns(pl.col("variance").clip(lower_bound=0).sqrt().alias("std_error"))


def cumulative_incidence(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef,
    by: ColumnRef | None = None,
    *,
    strata: ColumnRef | None = None,
    rho: float = 0.0,
    censor: object = 0,
) -> CumulativeIncidence:
    """Cumulative incidence for each cause, matching ``cmprsk::cuminc``.

    ``event`` holds a code per cause, with ``censor`` (default 0) meaning
    censored. ``by`` splits the curves and adds Gray's test for each cause.
    ``strata`` stratifies the test only. ``rho`` is the power of the weight
    ``(1 - F(t-))^rho`` in the test. Rows with a missing value are dropped.
    """
    columns = {"time": column_series(data, time), "event": column_series(data, event)}
    if by is not None:
        columns["group"] = column_series(data, by)
    if strata is not None:
        columns["strata"] = column_series(data, strata)
    frame = pl.DataFrame({key: series.alias(key) for key, series in columns.items()})
    frame = frame.filter(~pl.any_horizontal(pl.all().is_null()))
    if frame.get_column("time").dtype.is_float():
        frame = frame.filter(pl.col("time").is_not_nan())
    if frame.height == 0:
        raise ValueError("no complete rows")
    order = np.argsort(frame["time"].to_numpy(), kind="stable")
    frame = frame[order.tolist()]
    t = np.ascontiguousarray(frame["time"].to_numpy(), dtype=np.float64)
    status = [_code(value) for value in frame["event"].to_list()]
    censor_key = _code(censor)
    failed = np.array([value != censor_key for value in status])
    if by is None:
        group_labels, group_codes = ["1"], np.zeros(t.size, dtype=np.int64)
    else:
        group_labels, group_codes = _levels(frame["group"], columns["group"])
        present = np.bincount(group_codes, minlength=len(group_labels)) > 0
        keep_levels = [i for i, flag in enumerate(present) if flag]
        remap = -np.ones(len(group_labels), dtype=np.int64)
        remap[keep_levels] = np.arange(len(keep_levels))
        group_labels = [group_labels[i] for i in keep_levels]
        group_codes = remap[group_codes]
    if strata is None:
        strata_codes = np.zeros(t.size, dtype=np.int64)
        n_strata = 1
    else:
        strata_labels, strata_codes = _levels(frame["strata"], columns["strata"])
        n_strata = len(strata_labels)
    causes = sorted({value for value, flag in zip(status, failed, strict=True) if flag}, key=_sort_key)
    if not causes:
        raise ValueError("no failures")
    ng = len(group_labels)
    pieces = []
    tests = []
    for cause in causes:
        of_cause = np.array([value == cause for value in status])
        for index, group in enumerate(group_labels):
            sel = group_codes == index
            x, f, v = _cif_corners(t[sel], failed[sel], of_cause[sel])
            pieces.append(
                pl.DataFrame(
                    {
                        "group": [group] * x.size,
                        "cause": [str(cause)] * x.size,
                        "time": x,
                        "estimate": f,
                        "variance": v,
                    }
                )
            )
        if ng > 1:
            score = np.zeros(ng)
            var = np.zeros((ng, ng))
            for stratum in range(n_strata):
                sel = strata_codes == stratum
                z, v = _gray_stratum(t[sel], group_codes[sel], of_cause[sel], failed[sel], ng, float(rho))
                score += z
                var += v
            statistic = _quadratic_form(score[: ng - 1], var[: ng - 1, : ng - 1])
            tests.append(
                {
                    "cause": str(cause),
                    "statistic": statistic,
                    "p_value": float(stats.chi2.sf(statistic, ng - 1)) if statistic >= 0 else None,
                    "df": ng - 1,
                }
            )
    table = pl.concat(pieces)
    test_frame = pl.DataFrame(tests) if tests else None
    return CumulativeIncidence(
        table=table,
        tests=test_frame,
        groups=group_labels,
        causes=[str(cause) for cause in causes],
        rho=float(rho),
    )


@dataclass
class CompetingRisksFit:
    """Fine–Gray subdistribution hazards model, as ``cmprsk::crr`` returns it.

    ``covariance`` is Fine and Gray's sandwich, which accounts for the
    estimated censoring distribution. ``information`` is the negative Hessian
    of the pseudo log likelihood.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    information: np.ndarray
    score: np.ndarray
    names: list[str]
    n_obs: int
    n_missing: int
    log_likelihood: float
    null_log_likelihood: float
    converged: bool
    event_times: np.ndarray
    baseline_jumps: np.ndarray
    score_residuals: np.ndarray
    design: Design
    cause: object
    row_index: np.ndarray

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame:
        """Coefficient table, matching ``summary.crr``."""
        se = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        estimate = self.coefficients
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = estimate / se
        p_value = 2 * stats.norm.sf(np.abs(stat))
        crit = float(stats.norm.ppf(0.5 + level / 2))
        frame = pl.DataFrame(
            {
                "term": self.names,
                "estimate": estimate,
                "std_error": se,
                "statistic": stat,
                "p_value": p_value,
                "conf_low": estimate - crit * se,
                "conf_high": estimate + crit * se,
            }
        )
        if exponentiate:
            frame = frame.with_columns(
                pl.col("estimate", "conf_low", "conf_high").exp(),
            )
        return frame

    def glance(self) -> pl.DataFrame:
        """Pseudo log likelihoods and the pseudo likelihood ratio test."""
        statistic = -2 * (self.null_log_likelihood - self.log_likelihood)
        df = len(self.coefficients)
        return pl.DataFrame(
            {
                "n_obs": [self.n_obs],
                "n_missing": [self.n_missing],
                "log_likelihood": [self.log_likelihood],
                "null_log_likelihood": [self.null_log_likelihood],
                "statistic": [statistic],
                "df": [df],
                "p_value": [float(stats.chi2.sf(statistic, df))],
                "converged": [self.converged],
            }
        )

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """Cumulative incidence of the cause at each event time, matching ``predict.crr``.

        One curve per row of ``data``. The result is long, with ``row``
        (position in ``data``), ``time``, and ``estimate``.
        """
        design = build_design(data, self.design)
        if design.x.shape[1] != len(self.coefficients):
            raise ValueError("new data produced a different number of columns")
        lp = design.x @ self.coefficients
        hazard = np.cumsum(self.baseline_jumps)
        nt = self.event_times.size
        cif = 1.0 - np.exp(-np.exp(lp)[:, None] * hazard[None, :])
        return pl.DataFrame(
            {
                "row": np.repeat(design.row_index, nt),
                "time": np.tile(self.event_times, lp.size),
                "estimate": cif.ravel(),
            }
        )


def fine_gray_regression(
    data: pl.DataFrame,
    time: ColumnRef,
    event: ColumnRef | None = None,
    predictors: list[ColumnRef] | None = None,
    *,
    cause: object = 1,
    censor: object = 0,
    censor_group: ColumnRef | None = None,
    max_iter: int = 10,
    gtol: float = 1e-6,
) -> CompetingRisksFit:
    """Fine–Gray regression, matching ``cmprsk::crr``.

    ``time`` may be a formula such as ``"Surv(time, status) ~ age + sex"``.
    ``event`` holds a code per cause, ``cause`` is the event of interest, and
    ``censor`` (default 0) means censored. ``censor_group`` estimates the
    censoring distribution separately within each group, as ``cengroup``.
    Factors expand to treatment contrasts, as ``model.matrix`` would before
    ``crr``. Time-varying terms (``cov2``, ``tf``) are not supported.
    """
    n_in = data.height
    if is_formula(time):
        if event is not None or predictors is not None:
            raise ValueError("a Surv formula already names the time, the event, and the predictors")
        parsed = parse_survival_formula(
            data,
            str(time),
            drop_intercept=True,
            allow_counting=False,
            allow_strata=False,
            allow_cluster=False,
        )
        design = parsed.design
        t = parsed.time
        status = parsed.event
        group = None
        if censor_group is not None:
            group_series = column_series(data, censor_group).gather(design.row_index.tolist())
            if group_series.null_count():
                raise ValueError("censor_group has missing values")
            group = np.asarray(group_series.to_list())
    else:
        if event is None or predictors is None:
            raise ValueError("time, event, and predictors are required when time is a column")
        design = design_matrix(data, predictors, extra=[time, event] if censor_group is None else [time, event, censor_group], intercept=False)
        index = design.row_index.tolist()
        t = np.asarray(column_series(data, time).gather(index).to_numpy(), dtype=float)
        status = np.asarray(column_series(data, event).gather(index).to_list(), dtype=object)
        group = None if censor_group is None else np.asarray(column_series(data, censor_group).gather(index).to_list())
    if design.x.shape[1] == 0:
        raise ValueError("crr needs at least one covariate")
    order = np.argsort(t, kind="stable")
    t = np.ascontiguousarray(t[order], dtype=np.float64)
    x = np.ascontiguousarray(design.x[order], dtype=np.float64)
    codes = [_code(value) for value in np.asarray(status, dtype=object)[order].tolist()]
    censor_key = _code(censor)
    cause_key = _code(cause)
    cens = np.array([value == censor_key for value in codes])
    status_code = np.where(np.array([value == cause_key for value in codes]), 1, np.where(cens, 0, 2)).astype(np.int64)
    if group is None:
        censor_group_index = np.zeros(t.size, dtype=np.int64)
        n_censor_groups = 1
    else:
        labels = np.asarray(group)[order]
        uniq = sorted(set(labels.tolist()), key=_sort_key)
        lookup = {value: i for i, value in enumerate(uniq)}
        censor_group_index = np.array([lookup[value] for value in labels.tolist()], dtype=np.int64)
        n_censor_groups = len(uniq)
    if not np.any(status_code == 1):
        raise ValueError("no failures of the cause of interest")
    model = _FineGrayModel(t, status_code, x, censor_group_index, n_censor_groups)
    b = np.zeros(x.shape[1])
    lik, score, info = model.evaluate(b)
    converged = False
    for step in range(max_iter + 1):
        # The stopping rule is not in the paper. It was found by running
        # ``crr`` with many ``gtol`` and ``maxiter`` values: R stops at the
        # first iterate where max_j |U_j| * max(|b_j|, 1) < gtol * |loglik|.
        if np.max(np.abs(score) * np.maximum(np.abs(b), 1.0)) < gtol * max(abs(lik), 1.0):
            converged = True
            break
        if step == max_iter:
            break
        b, lik, score, info = _newton_step(model, b, lik, score, info)
    covariance = model.sandwich(b, info)
    null_lik = model.evaluate(np.zeros_like(b))[0]
    return CompetingRisksFit(
        coefficients=b,
        covariance=covariance,
        information=info,
        score=score,
        names=list(design.names),
        n_obs=int(t.size),
        n_missing=int(n_in - t.size),
        log_likelihood=float(lik),
        null_log_likelihood=float(null_lik),
        converged=converged,
        event_times=model.event_times,
        baseline_jumps=model.baseline_jumps(b),
        score_residuals=model.score_residuals(b),
        design=design,
        cause=cause,
        row_index=design.row_index[order],
    )


def _code(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def _sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _levels(series: pl.Series, original: pl.Series) -> tuple[list[str], np.ndarray]:
    """Factor levels in R order: Enum order, else sorted numbers or strings."""
    dtype = original.dtype
    values = series.to_list()
    if isinstance(dtype, (pl.Enum, pl.Categorical)):
        levels = [str(v) for v in dtype.categories.to_list()] if isinstance(dtype, pl.Enum) else sorted({str(v) for v in values})
        lookup = {level: i for i, level in enumerate(levels)}
        return levels, np.array([lookup[str(v)] for v in values], dtype=np.int64)
    keys = [_code(v) for v in values]
    uniq = sorted(set(keys), key=_sort_key)
    lookup = {value: i for i, value in enumerate(uniq)}
    return [str(v) for v in uniq], np.array([lookup[v] for v in keys], dtype=np.int64)


def _quadratic_form(score: np.ndarray, var: np.ndarray) -> float:
    rank = np.linalg.matrix_rank(var)
    if rank < var.shape[0]:
        return -1.0
    return float(score @ np.linalg.solve(var, score))


def _timepoint_index(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Row of the corner table that holds the curve's value at each time.

    The curve is a right-continuous step function, so the value at ``t`` is
    the last corner with time at or before ``t``. Times before the first
    corner or after the last one give -1 (null), as ``timepoints`` does.
    """
    index = np.searchsorted(x, grid, side="right") - 1
    index[grid > x[-1]] = -1
    return index.astype(np.int64)


def _safe_divide(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    """``num / den`` with 0 where ``den`` is 0."""
    num, den = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    out = np.zeros(num.shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


# Cumulative incidence curve: Gray (1988), estimate (2.3).


def _cif_corners(t: np.ndarray, failed: np.ndarray, of_cause: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corners of one cumulative incidence curve and its variance.

    ``t`` is sorted. The estimate is (2.3) of Gray (1988):
    F(t) = sum over failure times u <= t of S(u-) d1(u) / Y(u), where S is the
    Kaplan–Meier estimate for failure from any cause.

    The variance is the Aalen-type estimate. With a = F(t) - F(u),

        sum over u <= t of  a^2 (D Y - d1^2 - d2^2) / ((Y-1) (Y-D)^2)
                          + S(u-)^2 d1 (Y-d1) / (Y^2 (Y-1))
                          - 2 a S(u-) d1 (Y-d1) / (Y (Y-D) (Y-1)),

    where D = d1 + d2 counts failures from any cause at u and d2 those from
    the other causes. The paper does not print this form. It was fitted to
    ``cuminc`` output on small data sets: without ties between causes the
    first term is a^2 D / ((Y-1)(Y-D)), and the d1^2 + d2^2 part is what
    ``cuminc`` does when causes tie. Terms whose denominator is zero are
    dropped. When Y = 1, ``cuminc`` keeps the middle term as
    S(u-)^2 d1 / Y^2, so the factor (Y-d1)/(Y-1) is taken as 1 there.

    The corners follow the layout of ``cuminc``: time 0 with value 0, then
    two corners at each failure time of this cause (before and after the
    jump), then the last follow-up time of the group. The variance is only
    updated at failure times of this cause.
    """
    if t.size == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    u, first = np.unique(t, return_index=True)
    at_risk = t.size - first
    d_all = np.bincount(np.searchsorted(u, t[failed]), minlength=u.size)
    d_one = np.bincount(np.searchsorted(u, t[of_cause]), minlength=u.size)
    keep = d_all > 0
    u, y, d, d1 = u[keep], at_risk[keep].astype(float), d_all[keep].astype(float), d_one[keep].astype(float)
    surv = np.cumprod(1.0 - d / y)
    s_before = np.concatenate([[1.0], surv])[:-1]
    f_after = np.cumsum(s_before * d1 / y)
    tie = np.where(y > 1, _safe_divide(y - d1, y - 1), 1.0)
    a_coef = _safe_divide(d * y - d1**2 - (d - d1) ** 2, (y - 1) * (y - d) ** 2)
    b_term = s_before**2 * d1 / y**2 * tie
    c_coef = _safe_divide(s_before * d1 * (y - d1), y * (y - d) * (y - 1))
    # Expand a^2 = F(t)^2 - 2 F(t) F(u) + F(u)^2 and sum each piece cumulatively.
    sum_a = np.cumsum(a_coef)
    sum_af = np.cumsum(a_coef * f_after)
    sum_aff = np.cumsum(a_coef * f_after**2)
    sum_b = np.cumsum(b_term)
    sum_c = np.cumsum(c_coef)
    sum_cf = np.cumsum(c_coef * f_after)
    var = f_after**2 * sum_a - 2 * f_after * sum_af + sum_aff + sum_b - 2 * (f_after * sum_c - sum_cf)
    own = d1 > 0
    after = f_after[own]
    var_after = var[own]
    before = np.concatenate([[0.0], after])[:-1]
    var_before = np.concatenate([[0.0], var_after])[:-1]
    times = np.concatenate([[0.0], np.repeat(u[own], 2), [t[-1]]])
    last_f = after[-1] if after.size else 0.0
    last_v = var_after[-1] if after.size else 0.0
    estimate = np.concatenate([[0.0], np.column_stack([before, after]).ravel(), [last_f]])
    variance = np.concatenate([[0.0], np.column_stack([var_before, var_after]).ravel(), [last_v]])
    return times, estimate, variance


# Gray's K-sample test: Gray (1988), (2.5), (2.6), (2.10), (2.11).


def _gray_stratum(
    t: np.ndarray, group: np.ndarray, of_cause: np.ndarray, failed: np.ndarray, ng: int, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Score and variance of Gray's test within one stratum.

    The score is (2.6) with the weight K_k(t) = L(t) R_k(t) of (2.5):

        z_k = sum over failure times of L (d1_k - R_k D1 / R.),

    with R_k = Y_k (1 - F_k(t-)) / S_k(t-) and L = (1 - F0(t-))^rho, where
    F0 is the pooled curve (2.11): dF0 = D1 / h. with h_k = Y_k / S_k(t-).

    The variance is (2.10). For group r the integrands are

        a_kr(t) = d_kr(t) + (1 - G0(t)/S_r(t)) c_kr(t),
        b_kr(t) = -(G0(t)/S_r(t)) c_kr(t),

    with d_kr = L h_k (1{k=r} - h_r / h.), G0 = 1 - F0, and c_kr(t) the sum
    of d_kr dF0 / G0(u-) over failure times u after t. The a terms are
    integrated against dF0 / h_r (the plug-in for the cause of interest) and
    the b terms against S_r(t-) d2_r / Y_r / h_r (the other causes).

    Details settled from ``cuminc`` output, not stated in the paper:

    - (2.10) is written with K_k^0 / G^0; this uses h_k in its place.
    - G0 / S_r is taken just after the jump at t, and c_kr excludes t.
    - Ties: the a term is multiplied by (H_r - D1)/(H_r - 1) with
      H_r = h. S_r(t-), and the b term by (Y_r - d2_r)/(Y_r - 1). Each factor
      is 1 when its denominator would be 0 or below.
    """
    fail_times = np.unique(t[failed])
    nt = fail_times.size
    y = np.zeros((ng, nt))
    d1 = np.zeros((ng, nt))
    d2 = np.zeros((ng, nt))
    for k in range(ng):
        sel = group == k
        own = np.sort(t[sel])
        y[k] = own.size - np.searchsorted(own, fail_times, side="left")
        d1[k] = np.bincount(np.searchsorted(fail_times, t[sel & of_cause]), minlength=nt)
        d2[k] = np.bincount(np.searchsorted(fail_times, t[sel & failed & ~of_cause]), minlength=nt)
    live = y > 0
    surv_after = np.cumprod(np.where(live, 1.0 - _safe_divide(d1 + d2, y), 1.0), axis=1)
    surv_before = np.concatenate([np.ones((ng, 1)), surv_after[:, :-1]], axis=1)
    cif_after = np.cumsum(surv_before * _safe_divide(d1, y), axis=1)
    cif_before = np.concatenate([np.zeros((ng, 1)), cif_after[:, :-1]], axis=1)
    h = np.where(live, _safe_divide(y, surv_before), 0.0)
    r_k = h * (1.0 - cif_before)
    h_dot = h.sum(axis=0)
    r_dot = r_k.sum(axis=0)
    d1_dot = d1.sum(axis=0)
    df0 = _safe_divide(d1_dot, h_dot)
    g0 = 1.0 - np.cumsum(df0)
    g0_before = np.concatenate([[1.0], g0[:-1]])
    dgamma = _safe_divide(df0, g0_before)
    weight = g0_before**rho
    score = (weight * (d1 - _safe_divide(r_k, r_dot) * d1_dot)).sum(axis=1)
    share = _safe_divide(h, h_dot)
    df2 = surv_before * _safe_divide(d2, y)
    var = np.zeros((ng, ng))
    for r in range(ng):
        ratio = _safe_divide(g0, surv_after[r])
        d_kr = weight * h * ((np.arange(ng) == r)[:, None] - share[r])
        tail = np.cumsum((d_kr * dgamma)[:, ::-1], axis=1)[:, ::-1]
        rest = np.concatenate([tail[:, 1:], np.zeros((ng, 1))], axis=1)
        a = d_kr + (1.0 - ratio) * rest
        b = -ratio * rest
        h_r = h_dot * surv_before[r]
        tie1 = np.where(h_r > 1, _safe_divide(h_r - d1_dot, h_r - 1), 1.0)
        tie2 = np.where(y[r] > 1, _safe_divide(y[r] - d2[r], y[r] - 1), 1.0)
        w1 = np.where(live[r], _safe_divide(df0, h[r]), 0.0) * tie1
        w2 = np.where(live[r], _safe_divide(df2[r], h[r]), 0.0) * tie2
        var += (a * w1) @ a.T + (b * w2) @ b.T
    return score, var


# Fine–Gray regression: Fine and Gray (1999), (6), (7), (8) and Section 5.


def _censoring_before(times: np.ndarray, censored: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Kaplan–Meier estimate of the censoring survival G at ``points-``.

    Censoring is the event. Failures at the same time stay in the risk set.
    """
    knots, first = np.unique(times, return_index=True)
    at_risk = times.size - first
    n_cens = np.bincount(np.searchsorted(knots, times[censored]), minlength=knots.size)
    surv = np.cumprod(1.0 - n_cens / at_risk)
    index = np.searchsorted(knots, points, side="left") - 1
    return np.where(index >= 0, surv[np.maximum(index, 0)], 1.0)


class _FineGrayModel:
    """Weighted risk sets of Fine and Gray (1999), Section 4.

    Rows are sorted by time. ``code`` is 1 for the cause of interest, 2 for
    other causes, and 0 for censored. ``group`` holds the censoring group.

    A row with an other-cause failure at X stays in the risk set after X with
    weight w(u) = G_k(u-) / G_k(X-), where G_k is the censoring Kaplan–Meier
    estimate of its group k. This is r_i(t) G(t) / G(X_i ^ t) of Section 4;
    the left limits were checked against ``crr``. The paper suggests separate
    weights for each level of a discrete covariate, which is ``cengroup``.
    ``crr`` sets G_k(u-) to 0 once u is past the last time seen in group k
    (found from its output), and this does the same.

    Sums over rows use cumulative sums, so one evaluation costs O(n p^2).
    """

    def __init__(self, t: np.ndarray, code: np.ndarray, x: np.ndarray, group: np.ndarray, n_groups: int) -> None:
        self.t = t
        self.code = code
        self.x = x
        self.group = group
        self.n_groups = n_groups
        is_event = code == 1
        self.event_times = np.unique(t[is_event])
        nt = self.event_times.size
        self.event_index = np.searchsorted(self.event_times, t)
        # Rows with time >= u_j are at risk at event time j for j < upto.
        self.upto = np.searchsorted(self.event_times, t, side="right")
        self.events = np.bincount(self.event_index[is_event], minlength=nt).astype(float)
        self.event_x = np.zeros((nt, x.shape[1]))
        np.add.at(self.event_x, self.event_index[is_event], x[is_event])
        censored = code == 0
        self.other = code == 2
        self.g_event = np.zeros((n_groups, nt))
        self.g_own = np.ones(t.size)
        for k in range(n_groups):
            sel = group == k
            g_at = _censoring_before(t[sel], censored[sel], self.event_times)
            g_at[self.event_times > t[sel].max()] = 0.0
            self.g_event[k] = g_at
            self.g_own[sel] = _censoring_before(t[sel], censored[sel], t[sel])

    def _risk_sums(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """exp(Z b), and S0, S1, S2 of the weighted risk set at each event time."""
        nt = self.event_times.size
        p = self.x.shape[1]
        r = np.exp(self.x @ b)
        rx = r[:, None] * self.x
        rxx = rx[:, :, None] * self.x[:, None, :]
        s0 = np.zeros(nt + 1)
        s1 = np.zeros((nt + 1, p))
        s2 = np.zeros((nt + 1, p, p))
        last = self.upto - 1
        has = last >= 0
        np.add.at(s0, last[has], r[has])
        np.add.at(s1, last[has], rx[has])
        np.add.at(s2, last[has], rxx[has])
        s0 = np.cumsum(s0[::-1])[::-1][:nt]
        s1 = np.cumsum(s1[::-1], axis=0)[::-1][:nt]
        s2 = np.cumsum(s2[::-1], axis=0)[::-1][:nt]
        for k in range(self.n_groups):
            sel = self.other & (self.group == k)
            if not np.any(sel):
                continue
            scale = 1.0 / self.g_own[sel]
            start = self.upto[sel]
            a0 = np.zeros(nt + 1)
            a1 = np.zeros((nt + 1, p))
            a2 = np.zeros((nt + 1, p, p))
            np.add.at(a0, start, r[sel] * scale)
            np.add.at(a1, start, rx[sel] * scale[:, None])
            np.add.at(a2, start, rxx[sel] * scale[:, None, None])
            g = self.g_event[k]
            s0 = s0 + g * np.cumsum(a0)[:nt]
            s1 = s1 + g[:, None] * np.cumsum(a1, axis=0)[:nt]
            s2 = s2 + g[:, None, None] * np.cumsum(a2, axis=0)[:nt]
        return r, s0, s1, s2

    def evaluate(self, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Pseudo log likelihood, score (6), and information (7) at ``b``.

        Tied events use the Breslow form.
        """
        _, s0, s1, s2 = self._risk_sums(b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mean = s1 / s0[:, None]
            lik = float(self.event_x.sum(axis=0) @ b - np.sum(self.events * np.log(s0)))
            score = (self.event_x - self.events[:, None] * mean).sum(axis=0)
            info = np.einsum("j,jpq->pq", self.events / s0, s2) - np.einsum("j,jp,jq->pq", self.events, mean, mean)
        return lik, score, info

    def baseline_jumps(self, b: np.ndarray) -> np.ndarray:
        """Breslow jumps of the baseline subdistribution hazard (Section 5)."""
        _, s0, _, _ = self._risk_sums(b)
        return self.events / s0

    def score_residuals(self, b: np.ndarray) -> np.ndarray:
        """Score contribution at each event time: the sum of Z_i - Zbar(u) over events at u."""
        _, s0, s1, _ = self._risk_sums(b)
        return self.event_x - self.events[:, None] * (s1 / s0[:, None])

    def sandwich(self, b: np.ndarray, info: np.ndarray) -> np.ndarray:
        """Omega^-1 Sigma Omega^-1 with Sigma = sum of (eta_i + psi_i)^2, from (8)."""
        r, s0, s1, _ = self._risk_sums(b)
        t, x = self.t, self.x
        p = x.shape[1]
        mean = s1 / s0[:, None]
        dlam = self.events / s0
        is_event = self.code == 1
        # eta_i = int (Z_i - Zbar) w_i dM_i, with dM_i = dN_i - w_i r_i dLambda while at risk.
        eta = np.zeros((t.size, p))
        eta[is_event] = x[is_event] - mean[self.event_index[is_event]]
        cum0 = np.concatenate([[0.0], np.cumsum(dlam)])
        cum1 = np.concatenate([np.zeros((1, p)), np.cumsum(mean * dlam[:, None], axis=0)])
        part0 = cum0[self.upto]
        part1 = cum1[self.upto]
        for k in range(self.n_groups):
            sel = self.other & (self.group == k)
            if not np.any(sel):
                continue
            g = self.g_event[k]
            tail0 = np.concatenate([np.cumsum((g * dlam)[::-1])[::-1], [0.0]])
            tail1 = np.concatenate([np.cumsum((g[:, None] * mean * dlam[:, None])[::-1], axis=0)[::-1], np.zeros((1, p))])
            scale = 1.0 / self.g_own[sel]
            part0[sel] += tail0[self.upto[sel]] * scale
            part1[sel] += tail1[self.upto[sel]] * scale[:, None]
        eta -= r[:, None] * (x * part0[:, None] - part1)
        total = eta + self._censoring_term(r, s0, mean)
        sigma = total.T @ total
        inverse = np.linalg.inv(info)
        return inverse @ sigma @ inverse.T

    def _censoring_term(self, r: np.ndarray, s0: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """psi_i of (8): the integral of q(u) / pi(u) dM^c_i(u).

        q(u) sums, over other-cause rows with X < u, the sum over event times
        s >= u of (Z - Zbar(s)) w(s) r dLambda(s). pi(u) counts rows at risk
        at u, and M^c uses the Nelson–Aalen hazard of censoring.

        With censoring groups the paper gives no formula. Each group uses its
        own rows for q, pi and M^c. Inside q, ``crr`` also counts only the
        events of the group in dLambda, and this does the same (found from
        its output). With one group this is the formula of the paper.
        """
        t, x, code = self.t, self.x, self.code
        p = x.shape[1]
        psi = np.zeros((t.size, p))
        for k in range(self.n_groups):
            sel = self.group == k
            censored = sel & (code == 0)
            cens_times = np.unique(t[censored])
            if cens_times.size == 0:
                continue
            ev = sel & (code == 1)
            events_k = np.bincount(self.event_index[ev], minlength=self.event_times.size).astype(float)
            inc0 = self.g_event[k] * events_k / s0
            inc1 = inc0[:, None] * mean
            first = np.searchsorted(self.event_times, cens_times, side="left")
            tail0 = np.concatenate([np.cumsum(inc0[::-1])[::-1], [0.0]])[first]
            tail1 = np.concatenate([np.cumsum(inc1[::-1], axis=0)[::-1], np.zeros((1, p))])[first]
            oth = sel & self.other
            scale = r[oth] / self.g_own[oth]
            # An other-cause row with time X counts for censoring times u > X.
            pos = np.searchsorted(cens_times, t[oth], side="right")
            b0 = np.zeros(cens_times.size + 1)
            b1 = np.zeros((cens_times.size + 1, p))
            np.add.at(b0, pos, scale)
            np.add.at(b1, pos, scale[:, None] * x[oth])
            b0 = np.cumsum(b0)[: cens_times.size]
            b1 = np.cumsum(b1, axis=0)[: cens_times.size]
            q = b1 * tail0[:, None] - b0[:, None] * tail1
            times_k = np.sort(t[sel])
            pi = (times_k.size - np.searchsorted(times_k, cens_times, side="left")).astype(float)
            n_cens = np.bincount(np.searchsorted(cens_times, t[censored]), minlength=cens_times.size)
            q_pi = q / pi[:, None]
            comp = np.concatenate([np.zeros((1, p)), np.cumsum(q_pi * (n_cens / pi)[:, None], axis=0)])
            rows = np.flatnonzero(sel)
            value = -comp[np.searchsorted(cens_times, t[rows], side="right")]
            is_cens = code[rows] == 0
            value[is_cens] += q_pi[np.searchsorted(cens_times, t[rows][is_cens])]
            psi[rows] = value
        return psi


def _newton_step(
    model: _FineGrayModel, b: np.ndarray, lik: float, score: np.ndarray, info: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """One Newton–Raphson step on the pseudo log likelihood.

    ``crr`` output matched plain Newton steps in every case tried. As a
    safeguard, the step is halved if the new value is not finite or drops.
    """
    move = np.linalg.solve(info, score)
    for _ in range(30):
        new_b = b + move
        new_lik, new_score, new_info = model.evaluate(new_b)
        if np.isfinite(new_lik) and new_lik >= lik - 1e-10 * max(abs(lik), 1.0):
            return new_b, new_lik, new_score, new_info
        move = move / 2
    return b, lik, score, info
