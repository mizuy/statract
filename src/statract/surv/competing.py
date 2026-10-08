"""Competing risks: cumulative incidence with Gray's test, and Fine–Gray regression.

``cumulative_incidence`` matches ``cmprsk::cuminc`` and ``timepoints``.
``fine_gray_regression`` matches ``cmprsk::crr`` (fixed covariates) and
``predict.crr``. The loops follow Gray's Fortran so that ties, the variance
terms, and the censoring weights come out the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit
from scipy import stats

from ..design import ColumnRef, Design, build_design, column_series, design_matrix
from ..formula import is_formula
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
    fail_i = failed.astype(np.int64)
    for cause in causes:
        is_cause = np.array([value == cause for value in status], dtype=np.int64)
        for index, group in enumerate(group_labels):
            sel = group_codes == index
            x, f, v = _cinc(t[sel], fail_i[sel], is_cause[sel])
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
            m = 2 * fail_i - is_cause
            score, var = _crstm(t, m, group_codes, strata_codes, float(rho), n_strata, ng)
            statistic = _quadratic_form(score, var)
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
                pl.col("estimate").exp().alias("exp_estimate"),
                pl.col("conf_low").exp().alias("exp_conf_low"),
                pl.col("conf_high").exp().alias("exp_conf_high"),
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
    ici = np.where(np.array([value == cause_key for value in codes]), 1, np.where(cens, 0, 2)).astype(np.int64)
    if group is None:
        icg = np.zeros(t.size, dtype=np.int64)
        ncg = 1
    else:
        labels = np.asarray(group)[order]
        uniq = sorted(set(labels.tolist()), key=_sort_key)
        lookup = {value: i for i, value in enumerate(uniq)}
        icg = np.array([lookup[value] for value in labels.tolist()], dtype=np.int64)
        ncg = len(uniq)
    wt = np.empty((ncg, t.size))
    for k in range(ncg):
        wt[k] = _censoring_weights(t, cens, icg == k)
    if not np.any(ici == 1):
        raise ValueError("no failures of the cause of interest")
    uft = np.unique(t[ici == 1])
    np_ = x.shape[1]
    b = np.zeros(np_)
    converged = False
    for step in range(max_iter + 1):
        lik, score, hess = _crrfsv(t, ici, x, wt, icg, b)
        if np.max(np.abs(score) * np.maximum(np.abs(b), 1)) < max(abs(lik), 1) * gtol:
            converged = True
            break
        if step == max_iter:
            break
        sc = -np.linalg.solve(hess, score)
        bn = b + sc
        fbn = _crrf(t, ici, x, wt, icg, bn)
        halvings = 0
        while not np.isfinite(fbn) or fbn > lik + 1e-4 * float(sc @ score):
            halvings += 1
            sc = sc * 0.5
            bn = b + sc
            fbn = _crrf(t, ici, x, wt, icg, bn)
            if halvings > 20:
                break
        if halvings > 20:
            break
        b = bn
    v_info, v_score = _crrvv(t, ici, x, wt, icg, b)
    inv = np.linalg.inv(v_info)
    covariance = inv @ v_score @ inv.T
    residuals = _crrsr(t, ici, x, wt, icg, b, uft.size)
    null = _crrf(t, ici, x, wt, icg, np.zeros(np_))
    jumps = _crrfit(t, ici, x, wt, icg, b, uft.size)
    return CompetingRisksFit(
        coefficients=b,
        covariance=covariance,
        information=v_info,
        score=-score,
        names=list(design.names),
        n_obs=int(t.size),
        n_missing=int(n_in - t.size),
        log_likelihood=-float(lik),
        null_log_likelihood=-float(null),
        converged=converged,
        event_times=uft,
        baseline_jumps=jumps,
        score_residuals=residuals,
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


def _timepoint_index(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Positions read by ``tpoi``: the corner at or before each time, or -1."""
    n = x.size
    out = np.full(grid.size, -1, dtype=np.int64)
    for i, tp in enumerate(grid):
        if tp > x[n - 1]:
            continue
        if tp == x[n - 1]:
            out[i] = n - 1
            continue
        count = int(np.searchsorted(x[: n - 1], tp, side="right"))
        if count > 0:
            out[i] = count
    return out


def _quadratic_form(score: np.ndarray, var: np.ndarray) -> float:
    rank = np.linalg.matrix_rank(var)
    if rank < var.shape[0]:
        return -1.0
    return float(score @ np.linalg.solve(var, score))


def _censoring_weights(t: np.ndarray, cens: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Kaplan–Meier of censoring in one group, read at ``t-`` for every row.

    This is the ``survfit`` and ``approx(..., method="constant")`` step in ``crr``.
    """
    eps = np.finfo(float).eps
    times = t[sel]
    flags = cens[sel]
    knots = np.unique(times)
    surv = np.empty(knots.size)
    running = 1.0
    for k, knot in enumerate(knots):
        at = times == knot
        n_risk = int(np.sum(times >= knot))
        d = int(np.sum(flags & at))
        if n_risk > 0:
            running *= 1.0 - d / n_risk
        surv[k] = running
    grid = np.concatenate([[min(0.0, knots.min()) - 10 * eps], knots, [knots.max() * (1 + 10 * eps)]])
    values = np.concatenate([[1.0], surv, [0.0]])
    xout = t * (1 - 100 * eps)
    index = np.searchsorted(grid, xout, side="right") - 1
    index = np.clip(index, 0, grid.size - 1)
    return values[index]


@njit(cache=True)
def _cinc(y, ic, icc):
    """Port of ``cinc``: corners, estimates, and variances of one curve."""
    n = y.shape[0]
    size = 2 * int(icc.sum()) + 2
    x = np.zeros(size)
    f = np.zeros(size)
    v = np.zeros(size)
    fk = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    lcnt = 0
    rs = float(n)
    ll = 0
    while ll < n:
        ty = y[ll]
        l = ll
        while l + 1 < n and y[l + 1] == ty:
            l += 1
        nd1 = 0
        nd2 = 0
        for i in range(ll, l + 1):
            nd1 += icc[i]
            nd2 += ic[i] - icc[i]
        nd = nd1 + nd2
        if nd != 0:
            fkn = fk * (rs - nd) / rs
            if nd1 > 0:
                lcnt += 2
                f[lcnt - 1] = f[lcnt - 2]
                f[lcnt] = f[lcnt - 1] + fk * nd1 / rs
            if nd2 > 0 and fkn > 0:
                t5 = 1.0
                if nd2 > 1:
                    t5 = 1.0 - (nd2 - 1.0) / (rs - 1.0)
                t6 = fk * fk * t5 * nd2 / (rs * rs)
                t3 = 1.0 / fkn
                t4 = f[lcnt] / fkn
                v1 += t4 * t4 * t6
                v2 += t3 * t4 * t6
                v3 += t3 * t3 * t6
            if nd1 > 0:
                t5 = 1.0
                if nd1 > 1:
                    t5 = 1.0 - (nd1 - 1.0) / (rs - 1.0)
                t6 = fk * fk * t5 * nd1 / (rs * rs)
                t3 = 0.0
                if fkn > 0:
                    t3 = 1.0 / fkn
                t4 = 1.0 + t3 * f[lcnt]
                v1 += t4 * t4 * t6
                v2 += t3 * t4 * t6
                v3 += t3 * t3 * t6
                t2 = f[lcnt]
                x[lcnt - 1] = y[l]
                x[lcnt] = y[l]
                v[lcnt - 1] = v[lcnt - 2]
                v[lcnt] = v1 + t2 * t2 * v3 - 2 * t2 * v2
            fk = fkn
        rs = float(n - (l + 1))
        ll = l + 1
    lcnt += 1
    x[lcnt] = y[n - 1]
    f[lcnt] = f[lcnt - 1]
    v[lcnt] = v[lcnt - 1]
    return x[: lcnt + 1], f[: lcnt + 1], v[: lcnt + 1]


def _crstm(y, m, ig, ist, rho, nst, ng):
    """Port of ``crstm``: Gray's score and its covariance, summed over strata."""
    ng1 = ng - 1
    score = np.zeros(ng1)
    var = np.zeros((ng1, ng1))
    for ks in range(nst):
        sel = ist == ks
        if not np.any(sel):
            continue
        s, v = _crst(
            np.ascontiguousarray(y[sel]),
            np.ascontiguousarray(m[sel]),
            np.ascontiguousarray(ig[sel]),
            rho,
            ng,
        )
        score += s
        var += v
    return score, var


@njit(cache=True)
def _crst(y, m, ig, rho, ng):
    n = y.shape[0]
    ng1 = ng - 1
    rs = np.zeros(ng)
    for i in range(n):
        rs[ig[i]] += 1
    s = np.zeros(ng1)
    v = np.zeros((ng1, ng1))
    f1m = np.zeros(ng)
    f1 = np.zeros(ng)
    skmm = np.ones(ng)
    skm = np.ones(ng)
    v3 = np.zeros(ng)
    v2 = np.zeros((ng1, ng))
    c = np.zeros((ng, ng))
    a = np.zeros((ng, ng))
    d = np.zeros((3, ng))
    fm = 0.0
    f = 0.0
    ll = 0
    while True:
        lu = ll
        while lu + 1 < n and not (y[lu + 1] > y[ll]):
            lu += 1
        d[:, :] = 0.0
        for i in range(ll, lu + 1):
            d[m[i], ig[i]] += 1
        nd1 = 0.0
        nd2 = 0.0
        for i in range(ng):
            nd1 += d[1, i]
            nd2 += d[2, i]
        if not (nd1 == 0 and nd2 == 0):
            tr = 0.0
            tq = 0.0
            for i in range(ng):
                if rs[i] <= 0:
                    continue
                td = d[1, i] + d[2, i]
                skm[i] = skmm[i] * (rs[i] - td) / rs[i]
                f1[i] = f1m[i] + (skmm[i] * d[1, i]) / rs[i]
                tr += rs[i] / skmm[i]
                tq += rs[i] * (1 - f1m[i]) / skmm[i]
            f = fm + nd1 / tr
            fb = (1 - fm) ** rho
            for i in range(ng):
                for j in range(i, ng):
                    a[i, j] = 0.0
                if rs[i] <= 0:
                    continue
                t1 = rs[i] / skmm[i]
                a[i, i] = fb * t1 * (1 - t1 / tr)
                c[i, i] += a[i, i] * nd1 / (tr * (1 - fm))
                for j in range(i + 1, ng):
                    if rs[j] <= 0:
                        continue
                    a[i, j] = -fb * t1 * rs[j] / (skmm[j] * tr)
                    c[i, j] += a[i, j] * nd1 / (tr * (1 - fm))
            for i in range(1, ng):
                for j in range(i):
                    a[i, j] = a[j, i]
                    c[i, j] = c[j, i]
            for i in range(ng1):
                if rs[i] <= 0:
                    continue
                s[i] += fb * (d[1, i] - nd1 * rs[i] * (1 - f1m[i]) / (skmm[i] * tq))
            if nd1 > 0:
                for k in range(ng):
                    if rs[k] <= 0:
                        continue
                    t4 = 1.0
                    if skm[k] > 0:
                        t4 = 1 - (1 - f) / skm[k]
                    t5 = 1.0
                    if nd1 > 1:
                        t5 = 1 - (nd1 - 1) / (tr * skmm[k] - 1)
                    t3 = t5 * skmm[k] * nd1 / (tr * rs[k])
                    v3[k] += t4 * t4 * t3
                    for i in range(ng1):
                        t1 = a[i, k] - t4 * c[i, k]
                        v2[i, k] += t1 * t4 * t3
                        for j in range(i + 1):
                            t2 = a[j, k] - t4 * c[j, k]
                            v[i, j] += t1 * t2 * t3
            if nd2 != 0:
                for k in range(ng):
                    if skm[k] <= 0 or d[2, k] <= 0:
                        continue
                    t4 = (1 - f) / skm[k]
                    t5 = 1.0
                    if d[2, k] > 1:
                        t5 = 1 - (d[2, k] - 1.0) / (rs[k] - 1.0)
                    t6 = rs[k]
                    t3 = t5 * ((skmm[k] ** 2) * d[2, k]) / (t6 ** 2)
                    v3[k] += t4 * t4 * t3
                    for i in range(ng1):
                        t1 = t4 * c[i, k]
                        v2[i, k] -= t1 * t4 * t3
                        for j in range(i + 1):
                            t2 = t4 * c[j, k]
                            v[i, j] += t1 * t2 * t3
        if lu >= n - 1:
            break
        for i in range(ll, lu + 1):
            rs[ig[i]] -= 1
        fm = f
        for i in range(ng):
            f1m[i] = f1[i]
            skmm[i] = skm[i]
        ll = lu + 1
    for i in range(ng1):
        for j in range(i + 1):
            for k in range(ng):
                v[i, j] += c[i, k] * c[j, k] * v3[k]
                v[i, j] += c[i, k] * v2[j, k]
                v[i, j] += c[j, k] * v2[i, k]
    for i in range(ng1):
        for j in range(i):
            v[j, i] = v[i, j]
    return s, v


@njit(cache=True)
def _lp(x, i, b):
    wk = 0.0
    for k in range(b.shape[0]):
        wk += x[i, k] * b[k]
    return wk


@njit(cache=True)
def _crrfsv(t2, ici, x, wt, icg, b):
    """Port of ``crrfsv``: minus the pseudo log likelihood, its gradient and Hessian."""
    n = t2.shape[0]
    np_ = b.shape[0]
    lik = 0.0
    s = np.zeros(np_)
    v = np.zeros((np_, np_))
    xb = np.zeros(np_)
    xbt = np.zeros(np_)
    vt = np.zeros((np_, np_))
    iuc = n - 1
    while True:
        found = -1
        for i in range(iuc, -1, -1):
            if ici[i] == 1:
                found = i
                break
        if found < 0:
            break
        iuc = found
        cft = t2[iuc]
        twf = 0.0
        itmp = iuc
        for i in range(iuc, -1, -1):
            if t2[i] < cft:
                break
            itmp = i
            if ici[i] == 1:
                twf += 1
                lik -= _lp(x, i, b)
                for j in range(np_):
                    s[j] -= x[i, j]
        iuc = itmp
        xb1 = 0.0
        xb1o = 0.0
        xb[:] = 0.0
        vt[:, :] = 0.0
        for i in range(n):
            if t2[i] < cft:
                if ici[i] <= 1:
                    continue
                twt = np.exp(_lp(x, i, b)) * wt[icg[i], iuc] / wt[icg[i], i]
            else:
                twt = np.exp(_lp(x, i, b))
            xb1 += twt
            for j in range(np_):
                xb[j] += twt * x[i, j]
                xbt[j] = x[i, j] - xb[j] / xb1
            if xb1o > 0:
                w = xb1 * twt / xb1o
                for k in range(np_):
                    for j in range(k, np_):
                        vt[k, j] += w * xbt[k] * xbt[j]
            xb1o = xb1
        lik += twf * np.log(xb1)
        w = twf / xb1
        for i in range(np_):
            s[i] += w * xb[i]
            for j in range(i, np_):
                v[i, j] += w * vt[i, j]
                v[j, i] = v[i, j]
        iuc -= 1
        if iuc < 0:
            break
    return lik, s, v


@njit(cache=True)
def _crrf(t2, ici, x, wt, icg, b):
    """Port of ``crrf``: minus the pseudo log likelihood."""
    n = t2.shape[0]
    lik = 0.0
    iuc = n - 1
    while True:
        found = -1
        for i in range(iuc, -1, -1):
            if ici[i] == 1:
                found = i
                break
        if found < 0:
            break
        iuc = found
        cft = t2[iuc]
        twf = 0.0
        itmp = iuc
        for i in range(iuc, -1, -1):
            if t2[i] < cft:
                break
            itmp = i
            if ici[i] == 1:
                twf += 1
                lik -= _lp(x, i, b)
        iuc = itmp
        xb1 = 0.0
        for i in range(n):
            if t2[i] < cft:
                if ici[i] <= 1:
                    continue
                xb1 += np.exp(_lp(x, i, b)) * wt[icg[i], iuc] / wt[icg[i], i]
            else:
                xb1 += np.exp(_lp(x, i, b))
        lik += twf * np.log(xb1)
        iuc -= 1
        if iuc < 0:
            break
    return lik


@njit(cache=True)
def _crrsr(t2, ici, x, wt, icg, b, ndf):
    """Port of ``crrsr``: score contributions at each distinct event time."""
    n = t2.shape[0]
    np_ = b.shape[0]
    res = np.zeros((ndf, np_))
    xb = np.zeros(np_)
    iuc = n - 1
    ldf = ndf
    while True:
        found = -1
        for i in range(iuc, -1, -1):
            if ici[i] == 1:
                found = i
                break
        if found < 0:
            break
        iuc = found
        ldf -= 1
        cft = t2[iuc]
        twf = 0.0
        itmp = iuc
        for i in range(iuc, -1, -1):
            if t2[i] < cft:
                break
            itmp = i
            if ici[i] == 1:
                twf += 1
                for j in range(np_):
                    res[ldf, j] += x[i, j]
        iuc = itmp
        xb1 = 0.0
        xb[:] = 0.0
        for i in range(n):
            if t2[i] < cft:
                if ici[i] <= 1:
                    continue
                twt = np.exp(_lp(x, i, b)) * wt[icg[i], iuc] / wt[icg[i], i]
            else:
                twt = np.exp(_lp(x, i, b))
            xb1 += twt
            for j in range(np_):
                xb[j] += twt * x[i, j]
        for j in range(np_):
            res[ldf, j] -= twf / xb1 * xb[j]
        iuc -= 1
        if iuc < 0:
            break
    return res


@njit(cache=True)
def _crrfit(t2, ici, x, wt, icg, b, ndf):
    """Port of ``crrfit``: jumps of the baseline cumulative subdistribution hazard."""
    n = t2.shape[0]
    res = np.zeros(ndf)
    iuc = 0
    ldf = -1
    while iuc < n:
        found = -1
        for i in range(iuc, n):
            if ici[i] == 1:
                found = i
                break
        if found < 0:
            break
        iuc = found
        cft = t2[iuc]
        ldf += 1
        twf = 0.0
        itmp = iuc
        for i in range(iuc, n):
            if t2[i] > cft:
                break
            itmp = i
            if ici[i] == 1:
                twf += 1
        iuc = itmp
        xb1 = 0.0
        for i in range(n):
            if t2[i] < cft:
                if ici[i] <= 1:
                    continue
                xb1 += np.exp(_lp(x, i, b)) * wt[icg[i], iuc] / wt[icg[i], i]
            else:
                xb1 += np.exp(_lp(x, i, b))
        res[ldf] += twf / xb1
        iuc += 1
    return res


@njit(cache=True)
def _crrvv(t2, ici, x, wt, icg, b):
    """Port of ``crrvv``: the information and the score variance with the censoring term."""
    n = t2.shape[0]
    np_ = b.shape[0]
    ncg = wt.shape[0]
    v = np.zeros((np_, np_))
    v2 = np.zeros((np_, np_))
    vt = np.zeros((np_, np_))
    ss2 = np.zeros((np_, ncg))
    ss3 = np.zeros((np_, ncg))
    ss4 = np.zeros(ncg)
    qu = np.zeros((np_, ncg))
    st1 = np.zeros(np_)
    st2 = np.zeros(np_)
    xbt = np.zeros(np_)
    icrsk = np.zeros(ncg)
    xb = np.zeros((n, np_ + 1))
    for i in range(n):
        icrsk[icg[i]] += 1
    for i in range(n):
        if ici[i] != 1:
            continue
        for j in range(n):
            if t2[j] < t2[i]:
                if ici[j] <= 1:
                    continue
                twt = np.exp(_lp(x, j, b)) * wt[icg[j], i] / wt[icg[j], j]
            else:
                twt = np.exp(_lp(x, j, b))
            xb[i, 0] += twt
            for k in range(np_):
                xb[i, k + 1] += twt * x[j, k]
    lc = 0
    for i in range(n):
        st1[:] = 0.0
        for j in range(n):
            if ici[j] != 1:
                continue
            if t2[j] <= t2[i]:
                twt = np.exp(_lp(x, i, b))
            elif t2[i] < t2[j] and ici[i] > 1:
                twt = np.exp(_lp(x, i, b)) * wt[icg[i], j] / wt[icg[i], i]
            else:
                continue
            for k in range(np_):
                st1[k] -= (x[i, k] - xb[j, k + 1] / xb[j, 0]) * twt / xb[j, 0]
        if ici[i] == 1:
            for k in range(np_):
                st1[k] += x[i, k] - xb[i, k + 1] / xb[i, 0]
            vt[:, :] = 0.0
            for j in range(n):
                if t2[j] < t2[i]:
                    if ici[j] <= 1:
                        continue
                    twt = np.exp(_lp(x, j, b)) * wt[icg[j], i] / wt[icg[j], j]
                else:
                    twt = np.exp(_lp(x, j, b))
                for k in range(np_):
                    xbt[k] = x[j, k] - xb[i, k + 1] / xb[i, 0]
                for k in range(np_):
                    for j2 in range(k, np_):
                        vt[k, j2] += twt * xbt[k] * xbt[j2]
            for j1 in range(np_):
                for j2 in range(j1, np_):
                    v[j1, j2] += vt[j1, j2] / xb[i, 0]
        if i == 0 or t2[i] > t2[i - 1]:
            # A censored row in this tie block (or the end of the data) needs q(u).
            need = True
            for j in range(i, n):
                if t2[j] > t2[i]:
                    need = False
                    break
                if ici[j] == 0:
                    break
            if need:
                qu[:, :] = 0.0
                for j1 in range(lc, n):
                    if ici[j1] != 1:
                        continue
                    ss4[:] = 0.0
                    ss3[:, :] = 0.0
                    for j2 in range(n):
                        if t2[j2] >= t2[i]:
                            break
                        if ici[j2] <= 1:
                            continue
                        twt = np.exp(_lp(x, j2, b)) * wt[icg[j2], j1] / wt[icg[j2], j2]
                        ss4[icg[j2]] += twt
                        for k in range(np_):
                            ss3[k, icg[j2]] += x[j2, k] * twt
                    g = icg[j1]
                    for k in range(np_):
                        qu[k, g] += (ss3[k, g] - xb[j1, k + 1] * ss4[g] / xb[j1, 0]) / xb[j1, 0]
                for j in range(i, n):
                    if t2[j] > t2[i]:
                        break
                    if ici[j] == 0:
                        g = icg[j]
                        for k in range(np_):
                            ss2[k, g] -= qu[k, g] / icrsk[g] ** 2
        g = icg[i]
        for k in range(np_):
            st2[k] = ss2[k, g]
        if ici[i] == 0:
            for k in range(np_):
                st2[k] += qu[k, g] / icrsk[g]
        for k in range(np_):
            st1[k] += st2[k]
        for j1 in range(np_):
            for j2 in range(j1, np_):
                v2[j1, j2] += st1[j1] * st1[j2]
        if i < n - 1 and t2[i + 1] > t2[i]:
            for j in range(lc, i + 1):
                icrsk[icg[j]] -= 1
            lc = i + 1
    for j1 in range(np_ - 1):
        for j2 in range(j1 + 1, np_):
            v[j2, j1] = v[j1, j2]
            v2[j2, j1] = v2[j1, j2]
    return v, v2
