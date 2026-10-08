"""Classical tests and intervals that match R's ``stats`` package.

``t_test``, ``wilcox_test``, ``mcnemar_test``, ``binom_test``, ``prop_test`` and
``p_adjust`` follow ``t.test``, ``wilcox.test``, ``mcnemar.test``,
``binom.test``, ``prop.test`` and ``p.adjust`` in R 4.3. Missing values are
dropped as R drops them. Exact Wilcoxon p values use the same counting
distributions as R, and the normal-approximation intervals use a port of R's
``uniroot``, so the estimates agree to rounding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

Alternative = Literal["two.sided", "less", "greater"]
_ALTERNATIVES = ("two.sided", "less", "greater")
_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class HTest:
    """Result of a test, shaped like R's ``htest``.

    ``estimates`` keeps R's names (``"mean of x"``, ``"prop 1"``, ...).
    ``estimate`` is the single number ``broom::tidy`` reports: the difference
    for two-sample t tests and two-sample proportion tests, else the first
    estimate.
    """

    method: str
    statistic: float
    statistic_name: str
    p_value: float
    alternative: str | None
    parameter: float | None = None
    parameter_name: str | None = None
    estimates: dict[str, float] = field(default_factory=dict)
    conf_int: tuple[float, float] | None = None
    conf_level: float | None = None
    null_value: float | None = None
    stderr: float | None = None

    @property
    def estimate(self) -> float | None:
        values = list(self.estimates.values())
        if not values:
            return None
        if len(values) == 2 and (self.method.endswith("Two Sample t-test") or self.method.startswith("2-sample test for equality")):
            return values[0] - values[1]
        return values[0]

    def frame(self) -> pl.DataFrame:
        """One row with ``estimate``, ``statistic``, ``p_value``, ``parameter``, the interval, ``method`` and ``alternative``."""
        lo, hi = self.conf_int if self.conf_int is not None else (None, None)
        return pl.DataFrame(
            {
                "estimate": [self.estimate],
                "statistic": [self.statistic],
                "p_value": [self.p_value],
                "parameter": [self.parameter],
                "conf_low": [lo],
                "conf_high": [hi],
                "method": [self.method],
                "alternative": [self.alternative],
            },
            schema={
                "estimate": pl.Float64,
                "statistic": pl.Float64,
                "p_value": pl.Float64,
                "parameter": pl.Float64,
                "conf_low": pl.Float64,
                "conf_high": pl.Float64,
                "method": pl.String,
                "alternative": pl.String,
            },
        )


def _floats(values, name: str) -> np.ndarray:
    if isinstance(values, pl.Series):
        values = values.cast(pl.Float64).to_numpy()
    arr = np.asarray(values, dtype=float).ravel()
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"'{name}' must be numeric")
    return arr


def _check_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")


def _check_level(conf_level: float) -> None:
    if not (np.isfinite(conf_level) and 0 < conf_level < 1):
        raise ValueError("conf_level must be a single number between 0 and 1")


def t_test(
    x,
    y=None,
    *,
    alternative: Alternative = "two.sided",
    mu: float = 0.0,
    paired: bool = False,
    var_equal: bool = False,
    conf_level: float = 0.95,
) -> HTest:
    """Student, Welch and paired t tests like ``t.test``.

    With ``y=None`` it is the one-sample test. With ``paired=True`` it tests the
    differences ``x - y`` on complete pairs. Otherwise it is Welch's test, or the
    pooled-variance test with ``var_equal=True``.
    """
    _check_alternative(alternative)
    _check_level(conf_level)
    xv = _floats(x, "x")
    if y is not None:
        yv = _floats(y, "y")
        if paired:
            if xv.size != yv.size:
                raise ValueError("'x' and 'y' must have the same length")
            ok = ~np.isnan(xv) & ~np.isnan(yv)
            xv = xv[ok] - yv[ok]
            yv = None
        else:
            xv = xv[~np.isnan(xv)]
            yv = yv[~np.isnan(yv)]
    else:
        if paired:
            raise ValueError("'y' is missing for paired test")
        xv = xv[~np.isnan(xv)]
        yv = None
    nx = xv.size
    mx = float(np.mean(xv)) if nx else float("nan")
    vx = float(np.var(xv, ddof=1)) if nx > 1 else float("nan")
    if yv is None:
        if nx < 2:
            raise ValueError("not enough 'x' observations")
        df = nx - 1.0
        stderr = math.sqrt(vx / nx)
        if stderr < 10 * _EPS * abs(mx):
            raise ValueError("data are essentially constant")
        tstat = (mx - mu) / stderr
        method = "Paired t-test" if paired else "One Sample t-test"
        estimates = {"mean difference" if paired else "mean of x": mx}
    else:
        ny = yv.size
        if nx < 1 or (not var_equal and nx < 2):
            raise ValueError("not enough 'x' observations")
        if ny < 1 or (not var_equal and ny < 2):
            raise ValueError("not enough 'y' observations")
        if var_equal and nx + ny < 3:
            raise ValueError("not enough observations")
        my = float(np.mean(yv))
        vy = float(np.var(yv, ddof=1)) if ny > 1 else float("nan")
        method = "Two Sample t-test" if var_equal else "Welch Two Sample t-test"
        estimates = {"mean of x": mx, "mean of y": my}
        if var_equal:
            df = nx + ny - 2.0
            v = 0.0
            if nx > 1:
                v += (nx - 1) * vx
            if ny > 1:
                v += (ny - 1) * vy
            v /= df
            stderr = math.sqrt(v * (1 / nx + 1 / ny))
        else:
            sx = math.sqrt(vx / nx)
            sy = math.sqrt(vy / ny)
            stderr = math.sqrt(sx**2 + sy**2)
            df = stderr**4 / (sx**4 / (nx - 1) + sy**4 / (ny - 1))
        if stderr < 10 * _EPS * max(abs(mx), abs(my)):
            raise ValueError("data are essentially constant")
        tstat = (mx - my - mu) / stderr
    if alternative == "less":
        pval = float(stats.t.cdf(tstat, df))
        cint = (-math.inf, tstat + float(stats.t.ppf(conf_level, df)))
    elif alternative == "greater":
        pval = float(stats.t.sf(tstat, df))
        cint = (tstat - float(stats.t.ppf(conf_level, df)), math.inf)
    else:
        pval = float(2 * stats.t.cdf(-abs(tstat), df))
        q = float(stats.t.ppf(1 - (1 - conf_level) / 2, df))
        cint = (tstat - q, tstat + q)
    cint = (mu + cint[0] * stderr, mu + cint[1] * stderr)
    return HTest(
        method=method,
        statistic=float(tstat),
        statistic_name="t",
        p_value=pval,
        alternative=alternative,
        parameter=float(df),
        parameter_name="df",
        estimates=estimates,
        conf_int=cint,
        conf_level=conf_level,
        null_value=mu,
        stderr=stderr,
    )


# --- exact Wilcoxon distributions (R nmath signrank.c / wilcox.c) -------------


@lru_cache(maxsize=64)
def _signrank_counts(n: int) -> tuple[int, ...]:
    """Number of subsets of 1..n with each sum."""
    counts = [0] * (n * (n + 1) // 2 + 1)
    counts[0] = 1
    top = 0
    for k in range(1, n + 1):
        top += k
        for s in range(top, k - 1, -1):
            counts[s] += counts[s - k]
    return tuple(counts)


@lru_cache(maxsize=256)
def _wilcox_counts(m: int, n: int) -> tuple[int, ...]:
    """Frequencies of the Mann–Whitney U for sample sizes m and n (Gaussian binomial)."""
    poly = [1] + [0] * (m * n)
    for i in range(1, m + 1):
        # multiply by (1 - q^(n+i))
        shift = n + i
        for s in range(m * n, shift - 1, -1):
            poly[s] -= poly[s - shift]
        # divide by (1 - q^i)
        for s in range(i, m * n + 1):
            poly[s] += poly[s - i]
    return tuple(poly)


def _psignrank(x: float, n: int, lower_tail: bool = True) -> float:
    x = float(np.round(x + 1e-7))
    u = n * (n + 1) / 2
    if x < 0:
        return 0.0 if lower_tail else 1.0
    if x >= u:
        return 1.0 if lower_tail else 0.0
    counts = _signrank_counts(n)
    f = math.exp(-n * math.log(2))
    p = 0.0
    if x <= n * (n + 1) / 4:
        for i in range(int(x) + 1):
            p += counts[i] * f
    else:
        x = u - x
        for i in range(math.ceil(x)):
            p += counts[i] * f
        lower_tail = not lower_tail
    return p if lower_tail else 0.5 - p + 0.5


def _qsignrank(x: float, n: int) -> int:
    if x == 0:
        return 0
    if x == 1:
        return n * (n + 1) // 2
    counts = _signrank_counts(n)
    f = math.exp(-n * math.log(2))
    p = 0.0
    q = 0
    if x <= 0.5:
        x = x - 10 * _EPS
        while True:
            p += counts[q] * f
            if p >= x:
                return q
            q += 1
    x = 1 - x + 10 * _EPS
    while True:
        p += counts[q] * f
        if p > x:
            return n * (n + 1) // 2 - q
        q += 1


def _pwilcox(q: float, m: int, n: int, lower_tail: bool = True) -> float:
    q = math.floor(q + 1e-7)
    if q < 0:
        return 0.0 if lower_tail else 1.0
    if q >= m * n:
        return 1.0 if lower_tail else 0.0
    counts = _wilcox_counts(m, n)
    c = float(math.comb(m + n, n))
    p = 0.0
    if q <= m * n / 2:
        for i in range(int(q) + 1):
            p += counts[i] / c
    else:
        q = m * n - q
        for i in range(int(q)):
            p += counts[i] / c
        lower_tail = not lower_tail
    return p if lower_tail else 0.5 - p + 0.5


def _qwilcox(x: float, m: int, n: int) -> int:
    if x == 0:
        return 0
    if x == 1:
        return m * n
    counts = _wilcox_counts(m, n)
    c = float(math.comb(m + n, n))
    p = 0.0
    q = 0
    if x <= 0.5:
        x = x - 10 * _EPS
        while True:
            p += counts[q] / c
            if p >= x:
                return q
            q += 1
    x = 1 - x + 10 * _EPS
    while True:
        p += counts[q] / c
        if p > x:
            return m * n - q
        q += 1


def _zeroin(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, tol: float, maxit: int = 1000) -> float:
    """R's ``R_zeroin2`` (Brent), as ``uniroot`` calls it."""
    c, fc = a, fa
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(maxit + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * _EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def _uniroot(f: Callable[[float], float], lower: float, upper: float, tol: float, f_lower=None, f_upper=None) -> float:
    fl = f(lower) if f_lower is None else f_lower
    fu = f(upper) if f_upper is None else f_upper
    if fl * fu > 0:
        raise ValueError("f() values at end points not of opposite sign")
    return _zeroin(f, lower, upper, fl, fu, tol)


def _tie_sum(ranks: np.ndarray) -> float:
    _, counts = np.unique(ranks, return_counts=True)
    counts = counts.astype(float)
    return float(np.sum(counts**3 - counts))


def _sign(v: float) -> float:
    return float(np.sign(v))


def wilcox_test(
    x,
    y=None,
    *,
    alternative: Alternative = "two.sided",
    mu: float = 0.0,
    paired: bool = False,
    exact: bool | None = None,
    correct: bool = True,
    conf_int: bool = False,
    conf_level: float = 0.95,
    tol_root: float = 1e-4,
) -> HTest:
    """Wilcoxon signed rank and rank sum (Mann–Whitney) tests like ``wilcox.test``.

    ``exact=None`` uses the exact distribution below 50 observations when there
    are no ties (and, for the signed rank test, no zero differences). Otherwise
    it is the normal approximation with the tie correction and, with
    ``correct=True``, the continuity correction. ``conf_int=True`` adds the
    Hodges–Lehmann estimate and its interval.
    """
    _check_alternative(alternative)
    if conf_int:
        _check_level(conf_level)
    if not np.isfinite(mu):
        raise ValueError("'mu' must be a single number")
    xv = _floats(x, "x")
    yv = None
    if y is not None:
        yv = _floats(y, "y")
        if paired:
            if xv.size != yv.size:
                raise ValueError("'x' and 'y' must have the same length")
            ok = ~np.isnan(xv) & ~np.isnan(yv)
            xv = xv[ok] - yv[ok]
            yv = None
        else:
            yv = yv[~np.isnan(yv)]
    elif paired:
        raise ValueError("'y' is missing for paired test")
    xv = xv[~np.isnan(xv)]
    if xv.size < 1:
        raise ValueError("not enough (non-missing) 'x' observations")

    cint: tuple[float, float] | None = None
    estimates: dict[str, float] = {}
    level = conf_level
    if yv is None:
        method = "Wilcoxon signed rank test"
        xv = xv - mu
        zeroes = bool(np.any(xv == 0))
        if zeroes:
            xv = xv[xv != 0]
        n = xv.size
        if exact is None:
            exact = n < 50
        r = stats.rankdata(np.abs(xv))
        statistic = float(np.sum(r[xv > 0]))
        ties = np.unique(r).size != r.size
        if exact and not ties and not zeroes:
            method = "Wilcoxon signed rank exact test"
            if alternative == "two.sided":
                p = (
                    _psignrank(statistic - 1, n, lower_tail=False)
                    if statistic > n * (n + 1) / 4
                    else _psignrank(statistic, n)
                )
                pval = min(2 * p, 1.0)
            elif alternative == "greater":
                pval = _psignrank(statistic - 1, n, lower_tail=False)
            else:
                pval = _psignrank(statistic, n)
            if conf_int:
                xs = xv + mu
                alpha = 1 - conf_level
                diffs = np.add.outer(xs, xs)
                diffs = np.sort(diffs[np.triu_indices(n)]) / 2
                total = n * (n + 1) // 2
                if alternative == "two.sided":
                    qu = _qsignrank(alpha / 2, n) or 1
                    ql = total - qu
                    achieved = 2 * _psignrank(math.trunc(qu) - 1, n)
                    cint = (float(diffs[qu - 1]), float(diffs[ql]))
                elif alternative == "greater":
                    qu = _qsignrank(alpha, n) or 1
                    achieved = _psignrank(math.trunc(qu) - 1, n)
                    cint = (float(diffs[qu - 1]), math.inf)
                else:
                    qu = _qsignrank(alpha, n) or 1
                    ql = total - qu
                    achieved = _psignrank(math.trunc(qu) - 1, n)
                    cint = (-math.inf, float(diffs[ql]))
                if achieved - alpha > alpha / 2:
                    warnings.warn("requested conf.level not achievable", stacklevel=2)
                    level = 1 - float(f"{achieved:.2g}")
                estimates = {"(pseudo)median": float(np.median(diffs))}
        else:
            z = statistic - n * (n + 1) / 4
            sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - _tie_sum(r) / 48)
            correction = 0.0
            if correct:
                correction = {"two.sided": _sign(z) * 0.5, "greater": 0.5, "less": -0.5}[alternative]
                method += " with continuity correction"
            z = (z - correction) / sigma
            pval = _normal_p(z, alternative)
            if conf_int:
                xs = xv + mu
                alpha = 1 - conf_level

                def w_stat(d: float, use_correct: bool = correct) -> float:
                    xd = xs - d
                    xd = xd[xd != 0]
                    nx = xd.size
                    dr = stats.rankdata(np.abs(xd))
                    zd = float(np.sum(dr[xd > 0])) - nx * (nx + 1) / 4
                    sigma_ci = math.sqrt(nx * (nx + 1) * (2 * nx + 1) / 24 - _tie_sum(dr) / 48)
                    if sigma_ci == 0:
                        warnings.warn(
                            "cannot compute confidence interval when all observations are zero or tied", stacklevel=3
                        )
                    corr = {"two.sided": _sign(zd) * 0.5, "greater": 0.5, "less": -0.5}[alternative] if use_correct else 0.0
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return float(np.float64(zd - corr) / np.float64(sigma_ci))

                cint, level, estimates = _wilcox_normal_ci(
                    w_stat, xs, n, alternative, alpha, conf_level, tol_root, "(pseudo)median"
                )
            if exact and ties:
                warnings.warn("cannot compute exact p-value with ties", stacklevel=2)
            if exact and zeroes:
                warnings.warn("cannot compute exact p-value with zeroes", stacklevel=2)
        stat_name = "V"
    else:
        if yv.size < 1:
            raise ValueError("not enough 'y' observations")
        method = "Wilcoxon rank sum test"
        r = stats.rankdata(np.concatenate([xv - mu, yv]))
        nx, ny = xv.size, yv.size
        if exact is None:
            exact = nx < 50 and ny < 50
        statistic = float(np.sum(r[:nx])) - nx * (nx + 1) / 2
        ties = np.unique(r).size != r.size
        if exact and not ties:
            method = "Wilcoxon rank sum exact test"
            if alternative == "two.sided":
                p = (
                    _pwilcox(statistic - 1, nx, ny, lower_tail=False)
                    if statistic > nx * ny / 2
                    else _pwilcox(statistic, nx, ny)
                )
                pval = min(2 * p, 1.0)
            elif alternative == "greater":
                pval = _pwilcox(statistic - 1, nx, ny, lower_tail=False)
            else:
                pval = _pwilcox(statistic, nx, ny)
            if conf_int:
                alpha = 1 - conf_level
                diffs = np.sort(np.subtract.outer(xv, yv).ravel())
                if alternative == "two.sided":
                    qu = _qwilcox(alpha / 2, nx, ny) or 1
                    ql = nx * ny - qu
                    achieved = 2 * _pwilcox(math.trunc(qu) - 1, nx, ny)
                    cint = (float(diffs[qu - 1]), float(diffs[ql]))
                elif alternative == "greater":
                    qu = _qwilcox(alpha, nx, ny) or 1
                    achieved = _pwilcox(math.trunc(qu) - 1, nx, ny)
                    cint = (float(diffs[qu - 1]), math.inf)
                else:
                    qu = _qwilcox(alpha, nx, ny) or 1
                    ql = nx * ny - qu
                    achieved = _pwilcox(math.trunc(qu) - 1, nx, ny)
                    cint = (-math.inf, float(diffs[ql]))
                if achieved - alpha > alpha / 2:
                    warnings.warn("Requested conf.level not achievable", stacklevel=2)
                    level = 1 - achieved
                estimates = {"difference in location": float(np.median(diffs))}
        else:
            nties = _tie_sum(r)
            z = statistic - nx * ny / 2
            sigma = math.sqrt((nx * ny / 12) * ((nx + ny + 1) - nties / ((nx + ny) * (nx + ny - 1))))
            correction = 0.0
            if correct:
                correction = {"two.sided": _sign(z) * 0.5, "greater": 0.5, "less": -0.5}[alternative]
                method += " with continuity correction"
            z = (z - correction) / sigma
            pval = _normal_p(z, alternative)
            if conf_int:
                alpha = 1 - conf_level
                mumin = float(np.min(xv) - np.max(yv))
                mumax = float(np.max(xv) - np.min(yv))

                def w_stat(d: float, use_correct: bool = correct) -> float:
                    dr = stats.rankdata(np.concatenate([xv - d, yv]))
                    dz = float(np.sum(dr[:nx])) - nx * (nx + 1) / 2 - nx * ny / 2
                    corr = {"two.sided": _sign(dz) * 0.5, "greater": 0.5, "less": -0.5}[alternative] if use_correct else 0.0
                    sigma_ci = math.sqrt((nx * ny / 12) * ((nx + ny + 1) - _tie_sum(dr) / ((nx + ny) * (nx + ny - 1))))
                    if sigma_ci == 0:
                        warnings.warn("cannot compute confidence interval when all observations are tied", stacklevel=3)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return float(np.float64(dz - corr) / np.float64(sigma_ci))

                f_min = w_stat(mumin)
                f_max = w_stat(mumax)

                def root(zq: float) -> float:
                    lo = f_min - zq
                    if lo <= 0:
                        return mumin
                    hi = f_max - zq
                    if hi >= 0:
                        return mumax
                    return _zeroin(lambda d: w_stat(d) - zq, mumin, mumax, lo, hi, tol_root)

                if alternative == "two.sided":
                    cint = (root(float(stats.norm.isf(alpha / 2))), root(float(stats.norm.ppf(alpha / 2))))
                elif alternative == "greater":
                    cint = (root(float(stats.norm.isf(alpha))), math.inf)
                else:
                    cint = (-math.inf, root(float(stats.norm.ppf(alpha))))
                # R sets correct <- FALSE before this call, and W reads it.
                estimates = {"difference in location": _uniroot(lambda d: w_stat(d, False), mumin, mumax, tol_root)}
            if exact and ties:
                warnings.warn("cannot compute exact p-value with ties", stacklevel=2)
        stat_name = "W"
    return HTest(
        method=method,
        statistic=statistic,
        statistic_name=stat_name,
        p_value=float(pval),
        alternative=alternative,
        estimates=estimates,
        conf_int=cint,
        conf_level=level if conf_int else None,
        null_value=mu,
    )


def _wilcox_normal_ci(w_stat, xs, n, alternative, alpha, conf_level, tol_root, name):
    level = conf_level
    mumin = float(np.min(xs)) if n > 0 else float("nan")
    mumax = float(np.max(xs)) if n > 0 else float("nan")
    f_max = float("nan")
    if n > 0:
        f_min = w_stat(mumin)
        f_max = w_stat(mumax) if np.isfinite(f_min) else float("nan")
    if n == 0 or not np.isfinite(f_max):
        cint = (-math.inf if alternative == "less" else math.nan, math.inf if alternative == "greater" else math.nan)
        est = {"midrange": (mumin + mumax) / 2} if n > 0 else {}
        return cint, 0.0, est

    def root(zq: float) -> float:
        return _zeroin(lambda d: w_stat(d) - zq, mumin, mumax, f_min - zq, f_max - zq, tol_root)

    median = float(np.median(xs))
    if alternative == "two.sided":
        while True:
            if f_min - stats.norm.isf(alpha / 2) < 0 or f_max - stats.norm.ppf(alpha / 2) > 0:
                alpha *= 2
            else:
                break
        if alpha >= 1 or 1 - conf_level < alpha * 0.75:
            level = 1 - min(1.0, alpha)
            warnings.warn("requested conf.level not achievable", stacklevel=3)
        if alpha < 1:
            cint = (root(float(stats.norm.isf(alpha / 2))), root(float(stats.norm.ppf(alpha / 2))))
        else:
            cint = (median, median)
    elif alternative == "greater":
        while f_min - stats.norm.isf(alpha) < 0:
            alpha *= 2
        if alpha >= 1 or 1 - conf_level < alpha * 0.75:
            level = 1 - min(1.0, alpha)
            warnings.warn("requested conf.level not achievable", stacklevel=3)
        cint = (root(float(stats.norm.isf(alpha))) if alpha < 1 else median, math.inf)
    else:
        while f_max - stats.norm.ppf(alpha / 2) > 0:
            alpha *= 2
        if alpha >= 1 or 1 - conf_level < alpha * 0.75:
            level = 1 - min(1.0, alpha)
            warnings.warn("requested conf.level not achievable", stacklevel=3)
        cint = (-math.inf, root(float(stats.norm.ppf(alpha))) if alpha < 1 else median)
    # R sets correct <- FALSE before this call, and W reads it.
    est = {name: _uniroot(lambda d: w_stat(d, False), mumin, mumax, tol_root)}
    return cint, level, est


def _normal_p(z: float, alternative: str) -> float:
    if alternative == "less":
        return float(stats.norm.cdf(z))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(2 * min(stats.norm.cdf(z), stats.norm.sf(z)))


def mcnemar_test(x, y=None, *, correct: bool = True) -> HTest:
    """McNemar's test for paired categories like ``mcnemar.test``.

    ``x`` is a square table of counts, or ``x`` and ``y`` are paired categorical
    columns. Levels are the sorted distinct values of each column, matched by
    position as R's ``table`` does. Tables larger than 2 x 2 give Bowker's test
    of symmetry. The continuity correction applies only to 2 x 2 tables.
    """
    if y is None:
        table = np.asarray(x, dtype=float)
        if table.ndim != 2:
            raise ValueError("if 'x' is not a matrix, 'y' must be given")
        r = table.shape[0]
        if r < 2 or table.shape[1] != r:
            raise ValueError("'x' must be square with at least two rows and columns")
        if np.any(table < 0) or np.any(np.isnan(table)):
            raise ValueError("all entries of 'x' must be nonnegative and finite")
    else:
        xs = x if isinstance(x, pl.Series) else pl.Series(list(x))
        ys = y if isinstance(y, pl.Series) else pl.Series(list(y))
        if xs.len() != ys.len():
            raise ValueError("'x' and 'y' must have the same length")
        frame = pl.DataFrame({"x": xs, "y": ys})
        frame = frame.filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
        if frame.schema["x"].is_float():
            frame = frame.filter(~pl.col("x").is_nan())
        if frame.schema["y"].is_float():
            frame = frame.filter(~pl.col("y").is_nan())
        lx = _levels(frame["x"], x)
        ly = _levels(frame["y"], y)
        r = len(lx)
        if r < 2 or len(ly) != r:
            raise ValueError("'x' and 'y' must have the same number of levels (minimum 2)")
        ix = {v: i for i, v in enumerate(lx)}
        iy = {v: i for i, v in enumerate(ly)}
        table = np.zeros((r, r))
        for a, b in frame.iter_rows():
            table[ix[a], iy[b]] += 1
    parameter = r * (r - 1) / 2
    method = "McNemar's Chi-squared test"
    if correct and r == 2 and np.any(table - table.T != 0):
        diff = np.abs(table - table.T) - 1
        method += " with continuity correction"
    else:
        diff = table - table.T
    total = table + table.T
    iu = np.triu_indices(r, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = float(np.sum(diff[iu] ** 2 / total[iu]))
    pval = float(stats.chi2.sf(statistic, parameter))
    return HTest(
        method=method,
        statistic=statistic,
        statistic_name="McNemar's chi-squared",
        p_value=pval,
        alternative=None,
        parameter=float(parameter),
        parameter_name="df",
    )


def _levels(series: pl.Series, original) -> list:
    """Factor levels: an Enum or Categorical keeps its order, else sorted distinct values."""
    dtype = original.dtype if isinstance(original, pl.Series) else series.dtype
    if isinstance(dtype, pl.Enum):
        present = set(series.to_list())
        return [v for v in dtype.categories.to_list() if v in present]
    return sorted(series.unique().to_list())


def binom_test(
    x: int | Sequence[int],
    n: int | None = None,
    *,
    p: float = 0.5,
    alternative: Alternative = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """Exact binomial test with the Clopper–Pearson interval, like ``binom.test``.

    ``x`` is the number of successes out of ``n`` trials, or a pair
    ``(successes, failures)``.
    """
    _check_alternative(alternative)
    _check_level(conf_level)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    xr = np.round(xs)
    if np.any(np.isnan(xs) | (xs < 0)) or np.max(np.abs(xs - xr)) > 1e-7:
        raise ValueError("'x' must be nonnegative and integer")
    if xr.size == 2:
        trials = int(xr.sum())
        k = int(xr[0])
    elif xr.size == 1:
        if n is None:
            raise ValueError("'n' must be a positive integer >= 'x'")
        nr = round(n)
        k = int(xr[0])
        if n < 1 or abs(n - nr) > 1e-7 or k > nr:
            raise ValueError("'n' must be a positive integer >= 'x'")
        trials = int(nr)
    else:
        raise ValueError("incorrect length of 'x'")
    if not (0 <= p <= 1):
        raise ValueError("'p' must be a single number between 0 and 1")
    if alternative == "less":
        pval = float(stats.binom.cdf(k, trials, p))
    elif alternative == "greater":
        pval = float(stats.binom.sf(k - 1, trials, p))
    elif p == 0:
        pval = float(k == 0)
    elif p == 1:
        pval = float(k == trials)
    else:
        rel = 1 + 1e-7
        d = stats.binom.pmf(k, trials, p)
        m = trials * p
        if k == m:
            pval = 1.0
        elif k < m:
            i = np.arange(math.ceil(m), trials + 1)
            yy = int(np.sum(stats.binom.pmf(i, trials, p) <= d * rel))
            pval = float(stats.binom.cdf(k, trials, p) + stats.binom.sf(trials - yy, trials, p))
        else:
            i = np.arange(0, math.floor(m) + 1)
            yy = int(np.sum(stats.binom.pmf(i, trials, p) <= d * rel))
            pval = float(stats.binom.cdf(yy - 1, trials, p) + stats.binom.sf(k - 1, trials, p))
        pval = min(pval, 1.0)
    cint = _clopper_pearson(k, trials, conf_level, alternative)
    return HTest(
        method="Exact binomial test",
        statistic=float(k),
        statistic_name="number of successes",
        p_value=pval,
        alternative=alternative,
        parameter=float(trials),
        parameter_name="number of trials",
        estimates={"probability of success": k / trials},
        conf_int=cint,
        conf_level=conf_level,
        null_value=p,
    )


def _clopper_pearson(k: int, n: int, conf_level: float, alternative: str = "two.sided") -> tuple[float, float]:
    def lower(alpha: float) -> float:
        return 0.0 if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))

    def upper(alpha: float) -> float:
        return 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))

    if alternative == "less":
        return (0.0, upper(1 - conf_level))
    if alternative == "greater":
        return (lower(1 - conf_level), 1.0)
    alpha = (1 - conf_level) / 2
    return (lower(alpha), upper(alpha))


def _wilson(k: float, n: float, conf_level: float, alternative: str = "two.sided", yates: float = 0.0):
    """Score interval of ``prop.test``. ``yates=0`` is the plain Wilson interval."""
    z = float(stats.norm.ppf((1 + conf_level) / 2 if alternative == "two.sided" else conf_level))
    est = k / n
    z22n = z**2 / (2 * n)
    pc = est + yates / n
    hi = 1.0 if pc >= 1 else (pc + z22n + z * math.sqrt(pc * (1 - pc) / n + z22n / (2 * n))) / (1 + 2 * z22n)
    pc = est - yates / n
    lo = 0.0 if pc <= 0 else (pc + z22n - z * math.sqrt(pc * (1 - pc) / n + z22n / (2 * n))) / (1 + 2 * z22n)
    if alternative == "greater":
        return (max(lo, 0.0), 1.0)
    if alternative == "less":
        return (0.0, min(hi, 1.0))
    return (max(lo, 0.0), min(hi, 1.0))


def prop_test(
    x,
    n=None,
    *,
    p=None,
    alternative: Alternative = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> HTest:
    """Proportion tests like ``prop.test``.

    One sample: the score test with the Wilson interval (with continuity
    correction unless ``correct=False``). Two samples: the test of equal
    proportions with the interval for the difference. More samples, or given
    ``p`` for several samples: the chi-squared test without an interval.
    ``x`` and ``n`` are counts of successes and trials, or ``x`` is a matrix of
    ``(successes, failures)`` rows and ``n`` is omitted.
    """
    _check_alternative(alternative)
    _check_level(conf_level)
    xa = np.asarray(x, dtype=float)
    if n is None:
        if xa.ndim == 2 and xa.shape[1] == 2:
            na = xa.sum(axis=1)
            xa = xa[:, 0]
        elif xa.ndim == 1 and xa.size == 2:
            na = np.array([xa.sum()])
            xa = xa[:1]
        else:
            raise ValueError("'n' is required unless 'x' is a two-column matrix")
    else:
        xa = np.atleast_1d(xa).astype(float)
        na = np.atleast_1d(np.asarray(n, dtype=float))
        if xa.size != na.size:
            raise ValueError("'x' and 'n' must have the same length")
    pa = None if p is None else np.atleast_1d(np.asarray(p, dtype=float))
    if pa is not None and pa.size != xa.size:
        raise ValueError("'p' must have the same length as 'x' and 'n'")
    ok = ~np.isnan(xa) & ~np.isnan(na)
    xa, na = xa[ok], na[ok]
    if pa is not None:
        pa = pa[ok]
    k = xa.size
    if k < 1:
        raise ValueError("not enough data")
    if np.any(na <= 0):
        raise ValueError("elements of 'n' must be positive")
    if np.any(xa < 0):
        raise ValueError("elements of 'x' must be nonnegative")
    if np.any(xa > na):
        raise ValueError("elements of 'x' must not be greater than those of 'n'")
    if pa is None and k == 1:
        pa = np.array([0.5])
    if pa is not None and np.any((pa <= 0) | (pa >= 1)):
        raise ValueError("elements of 'p' must be in (0,1)")
    if k > 2 or (k == 2 and pa is not None):
        alternative = "two.sided"
    est = xa / na
    names = ["p"] if k == 1 else [f"prop {i + 1}" for i in range(k)]
    estimates = {nm: float(v) for nm, v in zip(names, est)}
    yates = 0.5 if (correct and k <= 2) else 0.0
    cint = None
    delta = float("nan")
    if k == 1:
        yates = min(yates, abs(xa[0] - na[0] * pa[0]))
        cint = _wilson(float(xa[0]), float(na[0]), conf_level, alternative, float(yates))
    elif k == 2 and pa is None:
        delta = float(est[0] - est[1])
        yates = min(yates, abs(delta) / float(np.sum(1 / na)))
        z = float(stats.norm.ppf((1 + conf_level) / 2 if alternative == "two.sided" else conf_level))
        width = z * math.sqrt(float(np.sum(est * (1 - est) / na))) + yates * float(np.sum(1 / na))
        if alternative == "two.sided":
            cint = (max(delta - width, -1.0), min(delta + width, 1.0))
        elif alternative == "greater":
            cint = (max(delta - width, -1.0), 1.0)
        else:
            cint = (-1.0, min(delta + width, 1.0))
    if k == 1:
        method = "1-sample proportions test"
    else:
        method = f"{k}-sample test for {'equality of' if pa is None else 'given'} proportions"
    method += " with continuity correction" if yates else " without continuity correction"
    null_value = None
    if pa is None:
        pooled = float(np.sum(xa) / np.sum(na))
        parameter = k - 1.0
        pmat = np.full(k, pooled)
    else:
        parameter = float(k)
        pmat = pa
        null_value = float(pa[0]) if k == 1 else None
    obs = np.column_stack([xa, na - xa])
    exp = np.column_stack([na * pmat, na * (1 - pmat)])
    if np.any(exp < 5):
        warnings.warn("Chi-squared approximation may be incorrect", stacklevel=2)
    statistic = float(np.sum((np.abs(obs - exp) - yates) ** 2 / exp))
    if alternative == "two.sided":
        pval = float(stats.chi2.sf(statistic, parameter))
    else:
        z = (np.sign(est[0] - pmat[0]) if k == 1 else np.sign(delta)) * math.sqrt(statistic)
        pval = float(stats.norm.cdf(z) if alternative == "less" else stats.norm.sf(z))
    return HTest(
        method=method,
        statistic=statistic,
        statistic_name="X-squared",
        p_value=pval,
        alternative=alternative,
        parameter=parameter,
        parameter_name="df",
        estimates=estimates,
        conf_int=cint,
        conf_level=conf_level if cint is not None else None,
        null_value=null_value,
    )


P_ADJUST_METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")


def p_adjust(
    p,
    method: Literal["holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"] = "holm",
    n: int | None = None,
) -> np.ndarray:
    """Adjust p values for multiple comparisons like ``p.adjust``.

    Missing values stay missing and do not count toward ``n`` unless ``n`` is
    given. ``"fdr"`` is ``"BH"``. Returns a float array in the input order.
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(f"method must be one of {P_ADJUST_METHODS}")
    if method == "fdr":
        method = "BH"
    p0 = _floats(p, "p").copy()
    nna = ~np.isnan(p0)
    pv = p0[nna]
    lp = pv.size
    if n is None:
        n = lp
    if n < lp:
        raise ValueError("n must be at least the number of non-missing p values")
    if n <= 1:
        return p0
    if n == 2 and method == "hommel":
        method = "hochberg"
    if method == "bonferroni":
        out = np.minimum(1.0, n * pv)
    elif method == "holm":
        i = np.arange(1, lp + 1)
        o = np.argsort(pv, kind="stable")
        ro = np.argsort(o, kind="stable")
        out = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * pv[o]))[ro]
    elif method == "hommel":
        q_full = np.concatenate([pv, np.ones(n - lp)]) if n > lp else pv
        i = np.arange(1, n + 1)
        o = np.argsort(q_full, kind="stable")
        ps = q_full[o]
        ro = np.argsort(o, kind="stable")
        q = np.full(n, np.min(n * ps / i))
        pa = q.copy()
        for j in range(n - 1, 1, -1):
            ij = np.arange(0, n - j + 1)
            i2 = np.arange(n - j + 1, n)
            q1 = np.min(j * ps[i2] / np.arange(2, j + 1))
            q[ij] = np.minimum(j * ps[ij], q1)
            q[i2] = q[n - j]
            pa = np.maximum(pa, q)
        out = np.maximum(pa, ps)[ro[:lp] if lp < n else ro]
    elif method in ("hochberg", "BH", "BY"):
        i = np.arange(lp, 0, -1)
        o = _order_decreasing(pv)
        ro = np.argsort(o, kind="stable")
        if method == "hochberg":
            scaled = (n + 1 - i) * pv[o]
        elif method == "BH":
            scaled = n / i * pv[o]
        else:
            scaled = np.sum(1 / np.arange(1, n + 1)) * n / i * pv[o]
        out = np.minimum(1.0, np.minimum.accumulate(scaled))[ro]
    else:
        out = pv
    p0[nna] = out
    return p0


def _order_decreasing(v: np.ndarray) -> np.ndarray:
    """R's ``order(v, decreasing=TRUE)``: ties keep their input order."""
    idx = np.arange(v.size)
    return np.lexsort((idx, -v))
