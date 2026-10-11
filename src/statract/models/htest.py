"""Classical tests and intervals, compared with R's ``stats`` package.

``t_test``, ``wilcox_test``, ``mcnemar_test``, ``binom_test``, ``prop_test`` and
``p_adjust`` are checked against ``t.test``, ``wilcox.test``, ``mcnemar.test``,
``binom.test``, ``prop.test`` and ``p.adjust`` in R 4.3. Missing values are
dropped. Exact Wilcoxon p values count the null distributions with the
textbook generating functions (Hollander, Wolfe and Chicken 2014).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
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


def _check_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")


def _check_level(conf_level: float) -> None:
    if not (np.isfinite(conf_level) and 0 < conf_level < 1):
        raise ValueError("conf_level must be one number in (0, 1)")


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
    """One-sample, paired, Welch and pooled two-sample t tests.

    Inputs: numeric ``x`` and optional ``y``; ``alternative``, ``mu``,
    ``paired``, ``var_equal``, ``conf_level``. Output: :class:`HTest`.

    Student (1908) for the one-sample and pooled statistics; Welch (1947,
    *Biometrika* 34:28) with Satterthwaite's (1946) degrees of freedom for
    unequal variances. Output is compared with R's ``t.test()``.
    """
    _check_alternative(alternative)
    _check_level(conf_level)
    if not np.isfinite(mu):
        raise ValueError("mu must be one finite number")
    if paired:
        if y is None:
            raise ValueError("a paired test needs y")
        xs, ys = _as_float(x), _as_float(y)
        if xs.size != ys.size:
            raise ValueError("paired samples must have the same length")
        both = ~(np.isnan(xs) | np.isnan(ys))
        sample = xs[both] - ys[both]
        y = None
    else:
        sample = _drop_missing(x)

    if y is None:
        size = sample.size
        if size < 2:
            raise ValueError("the t test needs at least 2 non-missing observations")
        estimate = float(np.mean(sample))
        stderr = float(np.sqrt(np.var(sample, ddof=1) / size))
        df = float(size - 1)
        method = "Paired t-test" if paired else "One Sample t-test"
        estimates = {"mean difference" if paired else "mean of x": estimate}
        scale = abs(estimate)
        shift = estimate
    else:
        other = _drop_missing(y)
        nx, ny = sample.size, other.size
        mean_x, mean_y = float(np.mean(sample)) if nx else np.nan, float(np.mean(other)) if ny else np.nan
        if var_equal:
            if nx < 1 or ny < 1 or nx + ny < 3:
                raise ValueError("the pooled t test needs 1 observation per group and 3 in all")
            # Pooled variance; a group of one adds nothing to the sum of squares.
            squares = float(np.sum((sample - mean_x) ** 2) + np.sum((other - mean_y) ** 2))
            pooled = squares / (nx + ny - 2)
            stderr = float(np.sqrt(pooled * (1.0 / nx + 1.0 / ny)))
            df = float(nx + ny - 2)
            method = "Two Sample t-test"
        else:
            if nx < 2 or ny < 2:
                raise ValueError("the Welch t test needs at least 2 observations per group")
            part_x = float(np.var(sample, ddof=1)) / nx
            part_y = float(np.var(other, ddof=1)) / ny
            stderr = float(np.sqrt(part_x + part_y))
            # Satterthwaite (1946) approximate degrees of freedom.
            df = (part_x + part_y) ** 2 / (part_x**2 / (nx - 1) + part_y**2 / (ny - 1))
            method = "Welch Two Sample t-test"
        estimates = {"mean of x": mean_x, "mean of y": mean_y}
        scale = max(abs(mean_x), abs(mean_y))
        shift = mean_x - mean_y
    if not stderr > 10.0 * _EPS * scale:
        raise ValueError("the data are constant up to rounding, so the t statistic is undefined")

    statistic = (shift - mu) / stderr
    if alternative == "less":
        p_value = float(stats.t.cdf(statistic, df))
        bounds = (-np.inf, statistic + float(stats.t.ppf(conf_level, df)))
    elif alternative == "greater":
        p_value = float(stats.t.sf(statistic, df))
        bounds = (statistic - float(stats.t.ppf(conf_level, df)), np.inf)
    else:
        p_value = float(2.0 * stats.t.cdf(-abs(statistic), df))
        half = float(stats.t.ppf(0.5 + conf_level / 2.0, df))
        bounds = (statistic - half, statistic + half)
    conf = (float(mu + bounds[0] * stderr), float(mu + bounds[1] * stderr))
    return HTest(
        method=method,
        statistic=float(statistic),
        statistic_name="t",
        p_value=p_value,
        alternative=alternative,
        parameter=float(df),
        parameter_name="df",
        estimates=estimates,
        conf_int=conf,
        conf_level=conf_level,
        null_value=mu,
        stderr=stderr,
    )


def _as_float(values) -> np.ndarray:
    """Flat float array with None (and polars nulls) as NaN."""
    if isinstance(values, pl.Series):
        values = values.cast(pl.Float64).to_numpy()
    if np.isscalar(values):
        values = [values]
    return np.array([np.nan if v is None else v for v in np.asarray(values, dtype=object).ravel()], dtype=float)


def _drop_missing(values) -> np.ndarray:
    array = _as_float(values)
    return array[~np.isnan(array)]


# --- exact Wilcoxon null distributions ----------------------------------------
#
# Counts come from the generating functions in Hollander, Wolfe and Chicken
# (2014), *Nonparametric Statistical Methods*, 3rd ed., sec. 3.1 and 4.1: the
# signed rank statistic V has generating function prod_{k=1}^{n} (1 + t^k), and
# the Mann–Whitney U has the Gaussian binomial coefficient [m+n choose m]_t.
# Counts are exact Python integers, so tail probabilities are exact up to the
# final division.


@lru_cache(maxsize=64)
def _signrank_counts(n: int) -> tuple[int, ...]:
    """Number of subsets of 1..n with each sum."""
    counts = [1] + [0] * (n * (n + 1) // 2)
    top = 0
    for k in range(1, n + 1):
        top += k
        for s in range(top, k - 1, -1):
            counts[s] += counts[s - k]
    return tuple(counts)


@lru_cache(maxsize=256)
def _wilcox_counts(m: int, n: int) -> tuple[int, ...]:
    """Frequencies of the Mann–Whitney U for sample sizes m and n (Gaussian binomial)."""
    m, n = min(m, n), max(m, n)
    size = m * n + 1
    poly = [1] + [0] * (size - 1)
    # [m+n choose m]_t = prod_{i=1}^{m} (1 - t^{n+i}) / (1 - t^i), applied factor by factor.
    for i in range(1, m + 1):
        shift = n + i
        for s in range(size - 1, shift - 1, -1):
            poly[s] -= poly[s - shift]
        for s in range(i, size):
            poly[s] += poly[s - i]
    return tuple(poly)


def _discrete_cdf(counts: tuple[int, ...], q: float, lower_tail: bool) -> float:
    """``P(X <= q)`` (or ``P(X > q)``) for X with these frequencies on 0, 1, 2, ..."""
    total = sum(counts)
    k = math.floor(q + 1e-7)
    if k < 0:
        below = 0
    elif k >= len(counts) - 1:
        below = total
    else:
        below = sum(counts[: k + 1])
    return (below if lower_tail else total - below) / total


def _discrete_quantile(counts: tuple[int, ...], p: float) -> int:
    """Smallest x with ``P(X <= x) >= p``, with a relative fuzz of 64 eps on ``p``."""
    total = sum(counts)
    target = p * total * (1.0 - 64.0 * _EPS)
    running = 0
    for x, c in enumerate(counts):
        running += c
        if running >= target:
            return x
    return len(counts) - 1


def _psignrank(x: float, n: int, lower_tail: bool = True) -> float:
    """Distribution function of the signed rank statistic (``psignrank``)."""
    return _discrete_cdf(_signrank_counts(n), x, lower_tail)


def _qsignrank(x: float, n: int) -> int:
    """Quantile of the signed rank statistic (``qsignrank``)."""
    return _discrete_quantile(_signrank_counts(n), x)


def _pwilcox(q: float, m: int, n: int, lower_tail: bool = True) -> float:
    """Distribution function of the Mann–Whitney U (``pwilcox``)."""
    return _discrete_cdf(_wilcox_counts(m, n), q, lower_tail)


def _qwilcox(x: float, m: int, n: int) -> int:
    """Quantile of the Mann–Whitney U (``qwilcox``)."""
    return _discrete_quantile(_wilcox_counts(m, n), x)


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
    Hodges–Lehmann estimate and its interval. With the normal approximation
    these are roots found to tolerance ``tol_root``.
    """
    _check_alternative(alternative)
    _check_level(conf_level)
    if not np.isfinite(mu):
        raise ValueError("mu must be one finite number")
    if not tol_root > 0:
        raise ValueError("tol_root must be positive")
    alpha = 1.0 - conf_level
    if y is None or paired:
        if paired:
            if y is None:
                raise ValueError("a paired test needs y")
            xs, ys = _as_float(x), _as_float(y)
            if xs.size != ys.size:
                raise ValueError("paired samples must have the same length")
            both = ~(np.isnan(xs) | np.isnan(ys))
            centre = xs[both] - ys[both]
        else:
            centre = _drop_missing(x)
        if not np.all(np.isfinite(centre)):
            raise ValueError("the observations must be finite")
        diffs = centre - mu
        has_zeros = bool(np.any(diffs == 0))
        nonzero = diffs[diffs != 0]
        size = nonzero.size
        if size == 0:
            raise ValueError("the signed rank test needs at least one difference from mu that is not zero")
        statistic, null_mean, null_var, has_ties = _signed_rank_parts(nonzero)
        use_exact = (size < 50) if exact is None else bool(exact)
        if use_exact and (has_ties or has_zeros):
            warnings.warn(
                "ties or zero differences: the exact signed rank p value is unavailable; using the normal approximation",
                RuntimeWarning,
                stacklevel=2,
            )
            use_exact = False
        if use_exact:
            method = "Wilcoxon signed rank exact test"
            lower = _psignrank(statistic, size)
            upper = _psignrank(statistic - 1, size, lower_tail=False)
            p_value = _exact_p(lower, upper, alternative)
        else:
            method = "Wilcoxon signed rank test"
            z = _standardize(statistic, null_mean, null_var, alternative, correct)
            p_value = _normal_p(z, alternative)
        # Zero differences leave the sample for the interval as well as for
        # the test (Hollander, Wolfe and Chicken 2014, sec. 3.1).
        interval_args = (centre[diffs != 0], None)
        statistic_name = "V"
    else:
        sample, other = _drop_missing(x), _drop_missing(y)
        if sample.size < 1 or other.size < 1:
            raise ValueError("the rank sum test needs at least one observation in each sample")
        if not (np.all(np.isfinite(sample)) and np.all(np.isfinite(other))):
            raise ValueError("the observations must be finite")
        nx, ny = sample.size, other.size
        statistic, null_mean, null_var, has_ties = _rank_sum_parts(sample - mu, other)
        use_exact = (nx < 50 and ny < 50) if exact is None else bool(exact)
        if use_exact and has_ties:
            warnings.warn(
                "ties: the exact rank sum p value is unavailable; using the normal approximation",
                RuntimeWarning,
                stacklevel=2,
            )
            use_exact = False
        if use_exact:
            method = "Wilcoxon rank sum exact test"
            lower = _pwilcox(statistic, nx, ny)
            upper = _pwilcox(statistic - 1, nx, ny, lower_tail=False)
            p_value = _exact_p(lower, upper, alternative)
        else:
            method = "Wilcoxon rank sum test"
            z = _standardize(statistic, null_mean, null_var, alternative, correct)
            p_value = _normal_p(z, alternative)
        interval_args = (sample, other)
        statistic_name = "W"
    if not use_exact and correct:
        method += " with continuity correction"

    interval = level = None
    estimates: dict[str, float] = {}
    if conf_int:
        first, second = interval_args
        if use_exact:
            interval, level, estimates = _wilcox_exact_interval(first, second, alternative, alpha, conf_level)
        elif second is None:
            interval, level, estimates = _signrank_normal_interval(first, alternative, alpha, conf_level, correct, tol_root)
        else:
            interval, level, estimates = _ranksum_normal_interval(first, second, alternative, alpha, conf_level, correct, tol_root)
    return HTest(
        method=method,
        statistic=float(statistic),
        statistic_name=statistic_name,
        p_value=float(p_value),
        alternative=alternative,
        estimates=estimates,
        conf_int=interval,
        conf_level=level,
        null_value=mu,
    )


def _tie_sizes(values: np.ndarray) -> np.ndarray:
    _, counts = np.unique(values, return_counts=True)
    return counts.astype(float)


def _signed_rank_parts(diffs: np.ndarray) -> tuple[float, float, float, bool]:
    """``(V, E0 V, Var0 V, ties)`` for non-zero differences.

    V is the sum of the mid-ranks of ``|d|`` over the positive ``d``. The null
    mean and the tie-corrected null variance are those of Hollander, Wolfe and
    Chicken (2014), sec. 3.1: ``n(n+1)/4`` and
    ``n(n+1)(2n+1)/24 - sum(t^3 - t)/48``.
    """
    size = diffs.size
    magnitude = np.abs(diffs)
    ranks = stats.rankdata(magnitude)
    statistic = float(np.sum(ranks[diffs > 0]))
    ties = _tie_sizes(magnitude)
    mean = size * (size + 1) / 4.0
    var = size * (size + 1) * (2 * size + 1) / 24.0 - float(np.sum(ties**3 - ties)) / 48.0
    return statistic, mean, var, bool(np.any(ties > 1))


def _rank_sum_parts(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float, bool]:
    """``(W, E0 W, Var0 W, ties)`` for the rank sum test of ``first`` against ``second``.

    W is the rank sum of ``first`` in the pooled sample minus ``m(m+1)/2``,
    i.e. the Mann–Whitney (1947) count. Null mean ``mn/2`` and tie-corrected
    variance ``(mn/12)[(N+1) - sum(t^3 - t)/(N(N-1))]`` (Hollander, Wolfe and
    Chicken 2014, sec. 4.1).
    """
    m, n = first.size, second.size
    pooled = np.concatenate([first, second])
    ranks = stats.rankdata(pooled)
    statistic = float(np.sum(ranks[:m]) - m * (m + 1) / 2.0)
    ties = _tie_sizes(pooled)
    total = m + n
    tie_term = float(np.sum(ties**3 - ties)) / (total * (total - 1)) if total > 1 else 0.0
    var = m * n / 12.0 * ((total + 1) - tie_term)
    return statistic, m * n / 2.0, var, bool(np.any(ties > 1))


def _continuity(gap: float, alternative: str) -> float:
    """Continuity correction of 1/2 toward the null in the tested direction."""
    if alternative == "greater":
        return 0.5
    if alternative == "less":
        return -0.5
    return 0.5 * float(np.sign(gap))


def _standardize(statistic: float, mean: float, var: float, alternative: str, correct: bool) -> float:
    gap = statistic - mean
    correction = _continuity(gap, alternative) if correct else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((gap - correction) / np.sqrt(var))


def _normal_p(z: float, alternative: str) -> float:
    if alternative == "less":
        return float(stats.norm.cdf(z))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(2.0 * min(stats.norm.cdf(z), stats.norm.sf(z)))


def _exact_p(lower: float, upper: float, alternative: str) -> float:
    """Exact p from ``P(T <= t)`` and ``P(T >= t)``; two-sided is twice the smaller tail."""
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    return min(1.0, 2.0 * min(lower, upper))


def _wilcox_exact_interval(x, y, alternative, alpha, conf_level):
    """Hodges–Lehmann estimate and exact interval (Bauer 1972; Hollander, Wolfe and Chicken 2014, secs. 3.3, 4.3).

    Signed rank (``y is None``): the ``n(n+1)/2`` Walsh averages
    ``(x_i + x_j)/2, i <= j``. Rank sum: the ``mn`` differences ``x_i - y_j``.
    With these sorted, ``[A_(c), A_(M+1-c)]`` covers the location with
    probability ``1 - 2 P(T <= c - 1)`` (``1 - P(T <= c - 1)`` for one side),
    where T is the null statistic; ``c`` is the null quantile at ``alpha/2``
    (two-sided) or ``alpha``. The estimate is the median of the values
    (Hodges & Lehmann 1963).
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        size = x.size
        rows, cols = np.triu_indices(size)
        values = np.sort((x[rows] + x[cols]) / 2.0)

        def quantile(p):
            return _qsignrank(p, size)

        def lower_tail(q):
            return _psignrank(q, size)

        name = "(pseudo)median"
    else:
        y = np.asarray(y, dtype=float)
        values = np.sort((x[:, None] - y[None, :]).ravel())

        def quantile(p):
            return _qwilcox(p, x.size, y.size)

        def lower_tail(q):
            return _pwilcox(q, x.size, y.size)

        name = "difference in location"
    count = values.size
    tail = alpha / 2.0 if alternative == "two.sided" else alpha
    cut = quantile(tail)
    reached = cut >= 1
    cut = max(cut, 1)
    missed = lower_tail(cut - 1)
    low, high = float(values[cut - 1]), float(values[count - cut])
    if alternative == "two.sided":
        interval = (low, high)
        achieved = 1.0 - 2.0 * missed
    elif alternative == "greater":
        interval = (low, np.inf)
        achieved = 1.0 - missed
    else:
        interval = (-np.inf, high)
        achieved = 1.0 - missed
    level = conf_level
    if not reached:
        # The requested level is above what the sample size allows; report the
        # attainable one. For the signed rank test it is shown to two digits,
        # as R's wilcox.test() prints it.
        level = float(f"{achieved:.2g}") if y is None else achieved
        warnings.warn(f"the requested confidence level is not attainable; using {level}", RuntimeWarning, stacklevel=3)
    return interval, level, {name: float(np.median(values))}


def _signrank_z(x: np.ndarray, shift: float, alternative: str, correct: bool) -> float:
    """Standardized signed rank statistic of ``x - shift`` (zeros dropped, ties corrected)."""
    diffs = x - shift
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    statistic, mean, var, _ = _signed_rank_parts(diffs)
    return _standardize(statistic, mean, var, alternative, correct)


def _ranksum_z(x: np.ndarray, y: np.ndarray, shift: float, alternative: str, correct: bool) -> float:
    """Standardized rank sum statistic of ``x - shift`` against ``y`` (ties corrected)."""
    statistic, mean, var, _ = _rank_sum_parts(x - shift, y)
    return _standardize(statistic, mean, var, alternative, correct)


def _normal_interval(score, low_end, high_end, alternative, alpha, conf_level, correct, tol_root, name):
    """Invert a standardized rank statistic ``score(shift, alternative, correct)``.

    The score is a non-increasing step function of the shift. Hodges and
    Lehmann (1963) take the estimate where the uncorrected score crosses 0;
    the limits are where the (corrected) score equals the normal quantiles
    (Hollander, Wolfe and Chicken 2014, secs. 3.3, 4.3). Roots are found by
    Brent's method to ``tol_root`` within ``[low_end, high_end]``. When the
    score at an end of the range does not reach the quantile, that end is the
    limit and the attainable level is reported with a warning.
    """
    from scipy import optimize

    def root(target, side_alternative, use_correct):
        def gap(shift):
            return score(shift, side_alternative, use_correct) - target

        at_low, at_high = gap(low_end), gap(high_end)
        if at_low == 0:
            return low_end, True
        if at_high == 0:
            return high_end, True
        if at_low > 0 > at_high:
            return float(optimize.brentq(gap, low_end, high_end, xtol=tol_root)), True
        return (low_end if at_low <= 0 else high_end), False

    tail = alpha / 2.0 if alternative == "two.sided" else alpha
    quantile = float(stats.norm.isf(tail))
    low, high = -np.inf, np.inf
    reached = True
    if alternative in ("two.sided", "greater"):
        low, ok = root(quantile, alternative, correct)
        reached &= ok
    if alternative in ("two.sided", "less"):
        high, ok = root(-quantile, alternative, correct)
        reached &= ok
    level = conf_level
    if not reached:
        top = score(low_end, alternative, correct)
        bottom = score(high_end, alternative, correct)
        if alternative == "two.sided":
            level = float(2.0 * stats.norm.cdf(min(top, -bottom)) - 1.0)
        elif alternative == "greater":
            level = float(stats.norm.cdf(top))
        else:
            level = float(stats.norm.cdf(-bottom))
        level = max(level, 0.0)
        warnings.warn(f"the requested confidence level is not attainable; using {level:.4g}", RuntimeWarning, stacklevel=4)
    if low_end == high_end:
        estimate = float(low_end)
    else:
        estimate, found = root(0.0, alternative, False)
        if not found:
            estimate = (low_end + high_end) / 2.0
    return (float(low), float(high)), level, {name: float(estimate)}


def _signrank_normal_interval(x, alternative, alpha, conf_level, correct, tol_root):
    x = np.asarray(x, dtype=float)

    def score(shift, side, use_correct):
        return _signrank_z(x, shift, side, use_correct)

    return _normal_interval(
        score, float(np.min(x)), float(np.max(x)), alternative, alpha, conf_level, correct, tol_root, "(pseudo)median"
    )


def _ranksum_normal_interval(x, y, alternative, alpha, conf_level, correct, tol_root):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def score(shift, side, use_correct):
        return _ranksum_z(x, y, shift, side, use_correct)

    return _normal_interval(
        score,
        float(np.min(x) - np.max(y)),
        float(np.max(x) - np.min(y)),
        alternative,
        alpha,
        conf_level,
        correct,
        tol_root,
        "difference in location",
    )


def mcnemar_test(x, y=None, *, correct: bool = True) -> HTest:
    """McNemar's test for paired categories like ``mcnemar.test``.

    ``x`` is a square table of counts, or ``x`` and ``y`` are paired categorical
    columns. Levels are the sorted distinct values of each column (Enum order
    when given), matched by position. Tables larger than 2 x 2 give Bowker's test
    of symmetry. The continuity correction applies only to 2 x 2 tables.
    """
    if y is None:
        table = np.asarray(x, dtype=float)
        if table.ndim != 2 or table.shape[0] != table.shape[1] or table.shape[0] < 2:
            raise ValueError("x must be a square table with at least 2 rows")
        if not np.all(np.isfinite(table)) or np.any(table < 0):
            raise ValueError("the table must hold finite non-negative counts")
    else:
        table = _paired_table(x, y)
    size = table.shape[0]
    upper_rows, upper_cols = np.triu_indices(size, k=1)
    above = table[upper_rows, upper_cols]
    below = table[upper_cols, upper_rows]
    discordant = above + below
    df = float(size * (size - 1) // 2)
    use_correction = bool(correct) and size == 2
    gap = np.abs(above - below)
    if use_correction:
        # Edwards (1948): move |n12 - n21| one unit toward 0, but not past it.
        gap = np.maximum(gap - 1.0, 0.0)
    # Bowker (1948): sum over i < j of (n_ij - n_ji)^2 / (n_ij + n_ji). Pairs
    # of cells that are both empty carry no information and are left out.
    used = discordant > 0
    if np.any(used):
        statistic = float(np.sum(gap[used] ** 2 / discordant[used]))
        p_value = float(stats.chi2.sf(statistic, df))
    else:
        statistic = p_value = float("nan")
    method = "McNemar's Chi-squared test"
    if use_correction:
        method += " with continuity correction"
    return HTest(
        method=method,
        statistic=statistic,
        statistic_name="McNemar's chi-squared",
        p_value=p_value,
        alternative=None,
        parameter=df,
        parameter_name="df",
    )


def _category_levels(series: pl.Series) -> list:
    """Enum order (present levels only) for a polars Enum, else the sorted distinct values."""
    present = series.drop_nulls()
    if isinstance(series.dtype, pl.Enum):
        seen = set(present.cast(pl.String).to_list())
        return [level for level in series.dtype.categories.to_list() if level in seen]
    return sorted(set(present.to_list()))


def _paired_table(x, y) -> np.ndarray:
    """Cross-tabulate two paired categorical columns; levels are matched by position."""
    first = x if isinstance(x, pl.Series) else pl.Series(list(x))
    second = y if isinstance(y, pl.Series) else pl.Series(list(y))
    if first.len() != second.len():
        raise ValueError("x and y must have the same length")

    def missing(series: pl.Series) -> np.ndarray:
        mask = series.is_null().to_numpy()
        if series.dtype.is_float():
            mask = mask | series.is_nan().fill_null(False).to_numpy()
        return mask

    keep = pl.Series(~(missing(first) | missing(second)))
    first, second = first.filter(keep), second.filter(keep)
    levels_x, levels_y = _category_levels(first), _category_levels(second)
    if len(levels_x) < 2 or len(levels_x) != len(levels_y):
        raise ValueError("x and y must have the same number (at least 2) of levels")
    if isinstance(first.dtype, pl.Enum):
        first = first.cast(pl.String)
    if isinstance(second.dtype, pl.Enum):
        second = second.cast(pl.String)
    code_x = {level: i for i, level in enumerate(levels_x)}
    code_y = {level: i for i, level in enumerate(levels_y)}
    table = np.zeros((len(levels_x), len(levels_y)))
    for a, b in zip(first.to_list(), second.to_list(), strict=True):
        table[code_x[a], code_y[b]] += 1.0
    return table


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
        raise ValueError("x must hold non-negative whole numbers")
    if xr.size == 2:
        trials = int(xr.sum())
        k = int(xr[0])
    elif xr.size == 1:
        if n is None:
            raise ValueError("n must be a whole number no smaller than x")
        nr = round(n)
        k = int(xr[0])
        if n < 1 or abs(n - nr) > 1e-7 or k > nr:
            raise ValueError("n must be a whole number no smaller than x")
        trials = int(nr)
    else:
        raise ValueError("x must be one count (with n) or a pair of counts")
    if not (0 <= p <= 1):
        raise ValueError("p must be one number in [0, 1]")
    if alternative == "less":
        pval = float(stats.binom.cdf(k, trials, p))
    elif alternative == "greater":
        pval = float(stats.binom.sf(k - 1, trials, p))
    elif p == 0:
        pval = float(k == 0)
    elif p == 1:
        pval = float(k == trials)
    else:
        pval = _binom_two_sided_p(k, trials, p)
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


def _binom_two_sided_p(k: int, n: int, p: float) -> float:
    """Two-sided exact binomial p value for ``k`` successes in ``n`` trials, ``0 < p < 1``.

    Minimum-likelihood (point-probability) method of Sterne (1954): the null
    probability of every outcome no more likely than the observed one.
    """
    if k == n * p:
        return 1.0
    observed = float(stats.binom.pmf(k, n, p))
    # Outcomes whose probability equals the observed one up to rounding count
    # as ties (relative fuzz below).
    limit = observed * (1.0 + _TIE_FUZZ)
    # The binomial mass is unimodal, so the outcomes no more likely than k form
    # k's own tail plus a tail on the far side of the mode. Find where the far
    # tail starts and add both tails with the distribution function.
    if k < n * p:
        far = np.arange(math.ceil(n * p), n + 1)
        inside = far[stats.binom.pmf(far, n, p) <= limit]
        far_tail = float(stats.binom.sf(inside[0] - 1, n, p)) if inside.size else 0.0
        near_tail = float(stats.binom.cdf(k, n, p))
    else:
        far = np.arange(0, math.floor(n * p) + 1)
        inside = far[stats.binom.pmf(far, n, p) <= limit]
        far_tail = float(stats.binom.cdf(inside[-1], n, p)) if inside.size else 0.0
        near_tail = float(stats.binom.sf(k - 1, n, p))
    return min(1.0, near_tail + far_tail)


# Relative tolerance under which two binomial probabilities count as equal.
_TIE_FUZZ = 1e-7


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
    """Wilson score interval ``(low, high)`` for ``k`` successes in ``n`` trials,
    with continuity correction ``yates`` (0 = none).

    Wilson (1927), *JASA* 22:209: the limits are the roots in pi of
    ``(phat - pi)^2 = z^2 pi (1 - pi) / n``. The continuity-corrected version is
    Newcombe (1998), *Stat Med* 17:857, method 4, which equals the plain Wilson
    limit evaluated at ``phat + c/n`` (upper) and ``phat - c/n`` (lower).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if alternative == "two.sided":
        z = float(stats.norm.ppf(0.5 + conf_level / 2.0))
    else:
        z = float(stats.norm.ppf(conf_level))
    z2 = z * z

    def limit(share: float, sign: float) -> float:
        # Root of (share - pi)^2 = z^2 pi (1 - pi) / n on the side ``sign``.
        centre = share + z2 / (2.0 * n)
        spread = z * math.sqrt(max(share * (1.0 - share) / n + z2 / (4.0 * n * n), 0.0))
        return (centre + sign * spread) / (1.0 + z2 / n)

    shift = yates / n
    low_share, high_share = k / n - shift, k / n + shift
    low = 0.0 if low_share <= 0.0 else max(0.0, limit(low_share, -1.0))
    high = 1.0 if high_share >= 1.0 else min(1.0, limit(high_share, 1.0))
    if alternative == "greater":
        return (low, 1.0)
    if alternative == "less":
        return (0.0, high)
    return (low, high)


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
    successes, trials = _proportion_counts(x, n)
    groups = successes.size
    if p is not None:
        null = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
        if null.size != groups:
            raise ValueError("p must hold one null proportion per group")
        if np.any(~np.isfinite(null) | (null <= 0) | (null >= 1)):
            raise ValueError("each null proportion must lie strictly between 0 and 1")
    elif groups == 1:
        null = np.array([0.5])
    else:
        null = None
    if not (groups == 1 or (groups == 2 and null is None)):
        alternative = "two.sided"

    shares = successes / trials
    if null is None:
        pooled = successes.sum() / trials.sum()
        expected_successes = trials * pooled
        df = groups - 1
    else:
        expected_successes = trials * null
        df = groups
    # Yates (1934) correction of 1/2 on the count scale, used for at most two
    # groups and never larger than the observed departure from the null
    # (?prop.test): for one group |x - n p0|, for two groups
    # |p1 - p2| / (1/n1 + 1/n2).
    correction = 0.0
    if correct and groups <= 2:
        if groups == 1:
            departure = abs(successes[0] - expected_successes[0])
        else:
            departure = abs(shares[0] - shares[1]) / (1.0 / trials[0] + 1.0 / trials[1])
        correction = min(0.5, float(departure))
    observed = np.concatenate([successes, trials - successes])
    expected = np.concatenate([expected_successes, trials - expected_successes])
    if np.any(expected < 5):
        warnings.warn("some expected counts are below 5, so the chi-squared approximation may be poor", RuntimeWarning, stacklevel=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = float(np.sum((np.abs(observed - expected) - correction) ** 2 / expected))

    if alternative == "two.sided":
        p_value = float(stats.chi2.sf(statistic, df))
    else:
        if groups == 1:
            direction = np.sign(shares[0] - null[0])
        else:
            direction = np.sign(shares[0] - shares[1])
        signed_root = float(direction * math.sqrt(statistic))
        p_value = float(stats.norm.cdf(signed_root) if alternative == "less" else stats.norm.sf(signed_root))

    interval = None
    level = None
    if groups == 1:
        interval = _wilson(float(successes[0]), float(trials[0]), conf_level, alternative, yates=correction)
        level = conf_level
    elif groups == 2 and null is None:
        # Wald interval for p1 - p2 widened by the correction (Fleiss, Levin
        # and Paik 2003, sec. 4.3).
        difference = float(shares[0] - shares[1])
        widen = correction * float(1.0 / trials[0] + 1.0 / trials[1])
        spread = math.sqrt(float(np.sum(shares * (1.0 - shares) / trials)))
        z = float(stats.norm.ppf(conf_level if alternative != "two.sided" else 0.5 + conf_level / 2.0))
        margin = z * spread + widen
        if alternative == "less":
            interval = (-1.0, min(1.0, difference + margin))
        elif alternative == "greater":
            interval = (max(-1.0, difference - margin), 1.0)
        else:
            interval = (max(-1.0, difference - margin), min(1.0, difference + margin))
        level = conf_level

    if groups == 1:
        method = "1-sample proportions test"
        estimates = {"p": float(shares[0])}
    else:
        kind = "equality of proportions" if null is None else "given proportions"
        method = f"{groups}-sample test for {kind}"
        estimates = {f"prop {i + 1}": float(v) for i, v in enumerate(shares)}
    method += " with continuity correction" if correction > 0 else " without continuity correction"
    return HTest(
        method=method,
        statistic=statistic,
        statistic_name="X-squared",
        p_value=p_value,
        alternative=alternative,
        parameter=float(df),
        parameter_name="df",
        estimates=estimates,
        conf_int=interval,
        conf_level=level,
        null_value=float(null[0]) if groups == 1 else None,
    )


def _proportion_counts(x, n) -> tuple[np.ndarray, np.ndarray]:
    """Successes and trials per group, with groups that have a missing count dropped."""
    if n is None:
        table = np.asarray(_as_float(x) if np.ndim(x) <= 1 else np.asarray(x, dtype=float), dtype=float)
        if table.ndim == 1:
            if table.size != 2:
                raise ValueError("without n, x must be a (successes, failures) pair or a k x 2 table")
            table = table.reshape(1, 2)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError("without n, x must be a (successes, failures) pair or a k x 2 table")
        successes, trials = table[:, 0], table.sum(axis=1)
    else:
        successes, trials = _as_float(x), _as_float(n)
        if successes.size != trials.size:
            raise ValueError("x and n must have the same length")
    keep = ~(np.isnan(successes) | np.isnan(trials))
    successes, trials = successes[keep], trials[keep]
    if successes.size == 0:
        raise ValueError("no groups are left after dropping missing counts")
    if np.any(~np.isfinite(trials) | (trials <= 0)):
        raise ValueError("each n must be positive")
    if np.any((successes < 0) | (successes > trials)):
        raise ValueError("each x must lie between 0 and its n")
    return successes, trials


P_ADJUST_METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")


def p_adjust(
    p,
    method: Literal["holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"] = "holm",
    n: int | None = None,
) -> np.ndarray:
    """Adjust p values for multiple comparisons.

    Input: p values (may contain NaN), ``method`` and optional ``n``. Output:
    float array of adjusted p values in the input order.

    Methods: Bonferroni; Holm (1979, *Scand J Stat* 6:65) step-down;
    Hochberg (1988, *Biometrika* 75:800) step-up; Hommel (1988, *Biometrika*
    75:383) closed Simes procedure with adjusted p values as in Wright (1992,
    *Biometrics* 48:1005); Benjamini & Hochberg (1995, *JRSS-B* 57:289,
    ``"BH"``/``"fdr"``); Benjamini & Yekutieli (2001, *Ann Stat* 29:1165,
    ``"BY"``). With ``n`` above the number of non-missing values the missing
    hypotheses count as p = 1. Output is compared with R's ``p.adjust()``.
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(f"method must be one of {P_ADJUST_METHODS}")
    if method == "fdr":
        method = "BH"
    values = np.asarray(p, dtype=float).ravel().copy()
    present = ~np.isnan(values)
    observed = values[present]
    m = observed.size
    total = m if n is None else int(n)
    if total < m:
        raise ValueError("n must be at least the number of non-missing p values")
    if total <= 1:
        return values
    if total == 2 and method == "hommel":
        method = "hochberg"

    order = np.argsort(observed, kind="stable")
    sorted_p = observed[order]
    rank = np.arange(1, m + 1, dtype=float)  # ascending rank among the observed values
    if method == "none":
        adjusted_sorted = sorted_p
    elif method == "bonferroni":
        adjusted_sorted = np.minimum(1.0, total * sorted_p)
    elif method == "holm":
        # Step down: (total - i + 1) p_(i), made non-decreasing.
        adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((total - rank + 1) * sorted_p))
    elif method == "hochberg":
        # Step up: (total - i + 1) p_(i), made non-decreasing from the largest down.
        raw = (total - rank + 1) * sorted_p
        adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    elif method in ("BH", "BY"):
        factor = 1.0 if method == "BH" else float(np.sum(1.0 / np.arange(1, total + 1)))
        raw = factor * total / rank * sorted_p
        adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    else:
        adjusted_sorted = _hommel_sorted(np.concatenate([sorted_p, np.ones(total - m)]))[:m]
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    values[present] = adjusted
    return values


def _hommel_sorted(sorted_p: np.ndarray) -> np.ndarray:
    """Hommel adjusted p values for ascending ``sorted_p`` (Wright 1992, sec. 2).

    The adjusted value of H_(i) is the largest Simes p value over all
    intersections containing it. For each intersection size s the worst case
    is the s largest p values when H_(i) is among them, and otherwise H_(i)
    together with the s - 1 largest; the Simes p value of the s largest is
    ``c_s = min_k s p_(total - s + k) / k``, and of the second kind
    ``min(s p_(i), c_s)``.
    """
    total = sorted_p.size
    adjusted = sorted_p.copy()
    for size in range(total, 1, -1):
        top = sorted_p[total - size:]
        simes = float(np.min(size * top / np.arange(1, size + 1)))
        adjusted[total - size:] = np.maximum(adjusted[total - size:], simes)
        if total - size:
            rest = np.minimum(size * sorted_p[: total - size], simes)
            adjusted[: total - size] = np.maximum(adjusted[: total - size], rest)
    return np.minimum(adjusted, 1.0)

