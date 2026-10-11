"""ROC curves, AUC, DeLong variance and tests, matching R pROC 1.18.5 output.

``roc_curve`` follows ``pROC::roc`` defaults: the first level of ``truth`` is the
control group, the second the case group, ``direction="auto"`` compares the
medians, and thresholds sit halfway between sorted unique scores plus ``-inf``
and ``inf``. Rows with a missing truth or score are dropped (``na.rm=TRUE``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import polars as pl
from scipy import stats

Direction = Literal["auto", "<", ">"]

_COORD_NAMES = (
    "threshold",
    "sensitivity",
    "specificity",
    "accuracy",
    "tn",
    "tp",
    "fn",
    "fp",
    "npv",
    "ppv",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "fdr",
    "1-specificity",
    "1-sensitivity",
    "1-accuracy",
    "1-npv",
    "1-ppv",
    "precision",
    "recall",
    "youden",
    "closest.topleft",
)


def _as_series(x: Any, name: str) -> pl.Series:
    s = x.rename(name) if isinstance(x, pl.Series) else pl.Series(name, x)
    if s.dtype.is_float():
        s = s.fill_nan(None)
    return s


def _thresholds(values: np.ndarray, direction: str) -> np.ndarray:
    """Candidate ROC thresholds (sorted float array, length unique + 1) for
    the pooled scores ``values`` and a direction ``"<"`` or ``">"``.

    The empirical ROC curve only changes between consecutive distinct scores,
    so the cut points are the midpoints between sorted distinct values plus
    ``-inf`` and ``+inf`` (Fawcett 2006, Pattern Recogn Lett 27:861, Sec. 5).
    When two distinct scores are so close that their floating-point midpoint
    collapses onto one of them, the cut point is moved to the neighbour that
    still separates them under the classification rule (``score >= t`` is
    positive for ``"<"``, ``score <= t`` for ``">"``).
    """
    distinct = np.unique(np.asarray(values, dtype=float))
    lower, upper = distinct[:-1], distinct[1:]
    huge = (np.abs(lower) > 1e100) | (np.abs(upper) > 1e100)
    with np.errstate(over="ignore"):
        midpoints = np.where(huge, lower / 2 + upper / 2, (lower + upper) / 2)
    collapsed = (midpoints == lower) | (midpoints == upper)
    if collapsed.any():
        # "<": t must exceed the lower value so that only the upper one is >= t.
        # ">": t must stay below the upper value so that only the lower one is <= t.
        separating = upper if direction == "<" else lower
        midpoints = np.where(collapsed, separating, midpoints)
        if direction == "<":
            ok = (midpoints > lower) & (midpoints <= upper)
        else:
            ok = (midpoints >= lower) & (midpoints < upper)
        if not ok.all():
            raise RuntimeError("thresholds do not separate near-tied predictor values")
    return np.concatenate(([-np.inf], midpoints, [np.inf]))


def _se_sp(
    thresholds: np.ndarray, controls: np.ndarray, cases: np.ndarray, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity and specificity at each threshold."""
    cs = np.sort(cases)
    ct = np.sort(controls)
    if direction == ">":
        tp = np.searchsorted(cs, thresholds, side="right")
        tn = ct.size - np.searchsorted(ct, thresholds, side="right")
    else:
        tp = cs.size - np.searchsorted(cs, thresholds, side="left")
        tn = np.searchsorted(ct, thresholds, side="left")
    return tp / cs.size, tn / ct.size


def _optim_crit(se: np.ndarray, sp: np.ndarray, weights: Sequence[float], method: str) -> np.ndarray:
    """Optimality criterion per curve point for ``method`` ``"youden"`` or
    ``"closest.topleft"`` (larger is better), with ``weights = (cost, prevalence)``.

    With ``r = (1 - prevalence) / (cost * prevalence)`` the cost-weighted
    Youden criterion is ``se + r * sp`` (Youden 1950, Cancer 3:32; Perkins &
    Schisterman 2006, Am J Epidemiol 163:670, eq. for the weighted index) and
    the weighted closest-to-(0, 1) criterion is the negated weighted squared
    distance ``-[(1 - se)^2 + r * (1 - sp)^2]`` (Perkins & Schisterman 2006).
    """
    cost, prevalence = float(weights[0]), float(weights[1])
    if not 0 < prevalence < 1:
        raise ValueError("best_weights[1] (prevalence) must be in (0, 1)")
    ratio = (1 - prevalence) / (cost * prevalence)
    se = np.asarray(se, dtype=float)
    sp = np.asarray(sp, dtype=float)
    if method == "youden":
        return se + ratio * sp
    if method in ("closest.topleft", "topleft"):
        return -((1 - se) ** 2 + ratio * (1 - sp) ** 2)
    raise ValueError("best_method must be 'youden' or 'closest.topleft'")


def _placements(controls: np.ndarray, cases: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """DeLong placement values ``(case_place, control_place)``.

    ``case_place[i]`` is the share of controls scoring below case ``i`` and
    ``control_place[j]`` the share of cases scoring above control ``j``, ties
    counting one half (DeLong, DeLong & Clarke-Pearson 1988, Biometrics 44:837,
    Sec. 2, the components V10 and V01). Both means equal the AUC.

    Computed with mid-ranks as in Sun & Xu (2014, IEEE Signal Process Lett
    21:1389, eqs. 6-7): the mid-rank of a case in the pooled sample minus its
    mid-rank among the cases counts the controls below it (ties as 1/2).
    For direction ``">"`` the scores are negated so that cases score higher.
    """
    sign = -1.0 if direction == ">" else 1.0
    case_scores = sign * np.asarray(cases, dtype=float)
    control_scores = sign * np.asarray(controls, dtype=float)
    n_cases, n_controls = case_scores.size, control_scores.size
    pooled_rank = stats.rankdata(np.concatenate((case_scores, control_scores)), method="average")
    case_rank_pooled, control_rank_pooled = pooled_rank[:n_cases], pooled_rank[n_cases:]
    case_place = (case_rank_pooled - stats.rankdata(case_scores, method="average")) / n_controls
    # controls below-or-tied-half among cases, turned into "cases above".
    control_place = 1.0 - (control_rank_pooled - stats.rankdata(control_scores, method="average")) / n_cases
    return case_place, control_place


def _is_perfect(curve: RocCurve) -> bool:
    """True when some curve point has sensitivity + specificity equal to 2."""
    tol = np.sqrt(np.finfo(float).eps)
    return bool(np.any(np.abs(curve.sensitivities + curve.specificities - 2.0) < tol))


@dataclass(repr=False)
class RocCurve:
    """Empirical ROC curve. Points are in pROC order (specificity increasing)."""

    thresholds: np.ndarray
    sensitivities: np.ndarray
    specificities: np.ndarray
    auc: float
    direction: str
    controls: np.ndarray
    cases: np.ndarray
    levels: tuple[Any, Any]
    response: pl.Series = field(default_factory=lambda: pl.Series([]))
    original_response: pl.Series = field(default_factory=lambda: pl.Series([]))
    original_predictor: pl.Series = field(default_factory=lambda: pl.Series([]))

    def __repr__(self) -> str:
        return (
            f"RocCurve(auc={self.auc:.4f}, direction='{self.direction}', "
            f"controls={self.controls.size}, cases={self.cases.size})"
        )

    def frame(self) -> pl.DataFrame:
        """``threshold``, ``sensitivity``, ``specificity`` per curve point."""
        return pl.DataFrame(
            {
                "threshold": self.thresholds,
                "sensitivity": self.sensitivities,
                "specificity": self.specificities,
            }
        )

    def var_auc(self) -> float:
        """DeLong variance of the AUC; output is compared with R's ``pROC::var``.

        ``S10 / m + S01 / n`` with ``S10`` and ``S01`` the sample variances
        (denominator ``k - 1``) of the case and control placement values
        (DeLong et al. 1988, Biometrics 44:837, Sec. 2; Hanley & McNeil 1982).
        NaN with fewer than two cases or controls.
        """
        if self.cases.size < 2 or self.controls.size < 2:
            return float("nan")
        if _is_perfect(self):
            warnings.warn(
                "the ROC curve is perfect (sensitivity and specificity both reach 1); its DeLong variance is 0",
                UserWarning,
                stacklevel=2,
            )
        case_place, control_place = _placements(self.controls, self.cases, self.direction)
        return float(np.var(case_place, ddof=1) / case_place.size + np.var(control_place, ddof=1) / control_place.size)

    def ci_auc(self, level: float = 0.95, method: Literal["delong"] = "delong") -> tuple[float, float, float]:
        """``(low, auc, high)`` DeLong interval, clipped to [0, 1].

        Normal approximation ``auc +/- z_{(1+level)/2} sqrt(var_auc())``
        (DeLong et al. 1988; Hanley & McNeil 1982). Output is compared with
        R's ``pROC::ci.auc``.
        """
        if method != "delong":
            raise ValueError("method must be 'delong'")
        if not 0 <= level <= 1:
            raise ValueError("level must be in [0, 1]")
        if self.cases.size < 2 or self.controls.size < 2:
            return (float("nan"), float("nan"), float("nan"))
        if _is_perfect(self):
            warnings.warn(
                "the ROC curve is perfect; the DeLong interval shrinks to the single value 1",
                UserWarning,
                stacklevel=2,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            se_auc = np.sqrt(self.var_auc())
        z = stats.norm.ppf((1 + level) / 2)
        low, mid, high = np.clip([self.auc - z * se_auc, self.auc, self.auc + z * se_auc], 0.0, 1.0)
        return (float(low), float(mid), float(high))

    def _calc_coords(
        self, thr: np.ndarray, se: np.ndarray, sp: np.ndarray, weights: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """All ``_COORD_NAMES`` columns at the given curve points.

        Confusion-matrix counts and derived rates as in Fawcett (2006, Pattern
        Recogn Lett 27:861, Sec. 2, Fig. 1); the weighted Youden and
        closest-to-top-left columns use ``_optim_crit`` (Perkins & Schisterman
        2006). Ratios with a zero denominator are NaN.
        """
        se = np.asarray(se, dtype=float)
        sp = np.asarray(sp, dtype=float)
        n_cases, n_controls = self.cases.size, self.controls.size
        tp = se * n_cases
        fn = n_cases - tp
        tn = sp * n_controls
        fp = n_controls - tn
        with np.errstate(divide="ignore", invalid="ignore"):
            npv = tn / (tn + fn)
            ppv = tp / (tp + fp)
            fnr = fn / (tp + fn)
            fdr = fp / (tp + fp)
        accuracy = (tp + tn) / (n_cases + n_controls)
        return {
            "threshold": np.asarray(thr, dtype=float),
            "sensitivity": se,
            "specificity": sp,
            "accuracy": accuracy,
            "tn": tn,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "npv": npv,
            "ppv": ppv,
            "tpr": se,
            "tnr": sp,
            "fpr": 1 - sp,
            "fnr": fnr,
            "fdr": fdr,
            "1-specificity": 1 - sp,
            "1-sensitivity": 1 - se,
            "1-accuracy": 1 - accuracy,
            "1-npv": 1 - npv,
            "1-ppv": 1 - ppv,
            "precision": ppv,
            "recall": se,
            "youden": _optim_crit(se, sp, weights, "youden"),
            "closest.topleft": -_optim_crit(se, sp, weights, "closest.topleft"),
        }

    def coords(
        self,
        x: str | float | Iterable[float] = "best",
        *,
        best_method: Literal["youden", "closest.topleft"] = "youden",
        ret: str | Sequence[str] = ("threshold", "specificity", "sensitivity"),
        best_weights: Sequence[float] = (1.0, 0.5),
    ) -> pl.DataFrame:
        """Coordinates like ``pROC::coords(..., transpose=FALSE)``.

        ``x`` is ``"best"``, ``"all"``, or thresholds. ``ret`` takes pROC names
        (``accuracy``, ``npv``, ``ppv``, ``youden``, ``closest.topleft``, ...) or ``"all"``.
        """
        ret = [ret] if isinstance(ret, str) else list(ret)
        if ret == ["all"]:
            ret = list(_COORD_NAMES)
        bad = [r for r in ret if r not in _COORD_NAMES]
        if bad:
            raise ValueError(f"unknown ret: {bad}")
        if isinstance(x, str):
            if x == "all":
                thr, se, sp = self.thresholds, self.sensitivities, self.specificities
            elif x == "best":
                crit = _optim_crit(self.sensitivities, self.specificities, best_weights, best_method)
                keep = crit == crit.max()
                thr, se, sp = self.thresholds[keep], self.sensitivities[keep], self.specificities[keep]
            else:
                raise ValueError("x must be 'best', 'all', or numeric thresholds")
        else:
            thr = np.atleast_1d(np.asarray(x, dtype=float))
            if thr.size == 0:
                raise ValueError("numeric x has length 0")
            se, sp = _se_sp(thr, self.controls, self.cases, self.direction)
        cols = self._calc_coords(thr, se, sp, best_weights)
        return pl.DataFrame({r: cols[r] for r in ret})


def _build(
    response: pl.Series,
    predictor: pl.Series,
    levels: Sequence[Any] | None,
    direction: Direction,
    original_response: pl.Series,
    original_predictor: pl.Series,
) -> RocCurve:
    if levels is None:
        uniq = response.drop_nulls().unique().sort().to_list()
        if len(uniq) < 2:
            raise ValueError("truth must have two levels")
        if len(uniq) > 2:
            warnings.warn(
                "truth has more than two levels; using the first two. Set levels explicitly.",
                stacklevel=3,
            )
        levels = uniq[:2]
    elif len(levels) != 2:
        raise ValueError("levels must have length 2")
    lev = (levels[0], levels[1])
    keep = (response.is_not_null() & predictor.is_not_null()).to_numpy()
    response, predictor = response.filter(keep), predictor.filter(keep)
    in_lev = response.is_in(list(lev)).to_numpy()
    response, predictor = response.filter(in_lev), predictor.filter(in_lev)
    resp = response.to_numpy()
    pred = predictor.cast(pl.Float64).to_numpy()
    controls = pred[resp == lev[0]]
    cases = pred[resp == lev[1]]
    if controls.size == 0:
        raise ValueError("the control group has no rows after filtering to the two levels")
    if cases.size == 0:
        raise ValueError("the case group has no rows after filtering to the two levels")
    if np.isinf(pred).any():
        raise ValueError("infinite values in score; cannot build a valid ROC curve")
    if direction == "auto":
        direction = "<" if np.median(controls) <= np.median(cases) else ">"
    elif direction not in ("<", ">"):
        raise ValueError("direction must be 'auto', '<' or '>'")
    thr = _thresholds(np.concatenate((controls, cases)), direction)
    se, sp = _se_sp(thr, controls, cases, direction)
    if np.any(np.diff(sp) < 0):
        thr, se, sp = thr[::-1].copy(), se[::-1].copy(), sp[::-1].copy()
    auc = float(np.sum((se[1:] + se[:-1]) / 2 * (sp[1:] - sp[:-1])))
    return RocCurve(
        thresholds=thr,
        sensitivities=se,
        specificities=sp,
        auc=auc,
        direction=direction,
        controls=controls,
        cases=cases,
        levels=lev,
        response=response,
        original_response=original_response,
        original_predictor=original_predictor,
    )


def roc_curve(
    truth: Any,
    score: Any,
    *,
    levels: Sequence[Any] | None = None,
    direction: Direction = "auto",
) -> RocCurve:
    """Build an empirical ROC curve like ``pROC::roc(truth, score)``.

    ``levels`` is ``(control, case)``; by default the sorted unique values of
    ``truth`` (0/1 gives controls 0, cases 1). ``direction="<"`` means controls
    score lower than cases.

    Examples:
        >>> r = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        >>> r.auc
        0.75
    """
    t = _as_series(truth, "truth")
    s = _as_series(score, "score")
    if t.len() != s.len():
        raise ValueError("truth and score must have the same length")
    if not (s.dtype.is_numeric() or s.dtype == pl.Boolean or s.dtype == pl.Null):
        raise ValueError("score must be numeric")
    return _build(t, s, levels, direction, t, s)


@dataclass
class RocTest:
    """Result of ``roc_test``. ``df`` is set only for the unpaired test."""

    statistic: float
    p_value: float
    auc1: float
    auc2: float
    method: str
    paired: bool
    alternative: str
    df: float | None = None
    conf_int: tuple[float, float] | None = None


def _safe_ratio(num: float, den: float) -> float:
    """``num / den`` with 0 / 0 read as 0 (no difference, no spread)."""
    if num == 0 and den == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(num) / np.float64(den))


def _same_values(a: pl.Series, b: pl.Series) -> bool:
    """Equal length and equal values position by position (nulls match nulls)."""
    if a.len() != b.len():
        return False
    return all(
        (x is None and y is None) or (x is not None and y is not None and x == y)
        for x, y in zip(a.to_list(), b.to_list())
    )


def _paired_rocs(roc1: RocCurve, roc2: RocCurve) -> tuple[RocCurve, RocCurve] | None:
    """The two curves restricted to common subjects, or None when unpaired.

    DeLong et al. (1988) compare curves measured on the same subjects. The
    curves count as paired when they have the same levels and the same
    response vector. When only the original (unfiltered) responses agree,
    both curves are rebuilt on the rows usable by both: response present and
    in the levels, and both predictors present. Each curve keeps its own
    levels and direction.
    """
    if list(roc1.levels) != list(roc2.levels):
        return None
    if _same_values(roc1.response, roc2.response):
        return roc1, roc2
    if not _same_values(roc1.original_response, roc2.original_response):
        return None
    response = roc1.original_response
    usable = (
        response.is_not_null()
        & response.is_in(list(roc1.levels))
        & roc1.original_predictor.is_not_null()
        & roc2.original_predictor.is_not_null()
    )
    rebuilt = []
    for curve in (roc1, roc2):
        rebuilt.append(
            _build(
                response.filter(usable),
                curve.original_predictor.filter(usable),
                curve.levels,
                curve.direction,
                curve.original_response,
                curve.original_predictor,
            )
        )
    return rebuilt[0], rebuilt[1]


def roc_test(
    roc1: RocCurve,
    roc2: RocCurve,
    *,
    method: Literal["delong"] = "delong",
    paired: bool | None = None,
    alternative: Literal["two.sided", "less", "greater"] = "two.sided",
    conf_level: float = 0.95,
) -> RocTest:
    """DeLong test of two AUCs like ``pROC::roc.test(..., method="delong")``.

    ``paired=None`` decides as pROC does: the curves are paired when they share
    the same response vector. The paired test gives Z; the unpaired test gives
    a t statistic with Welch-type degrees of freedom.

    Paired: ``Z = (A1 - A2) / sqrt(L S L')`` with ``L = (1, -1)`` and
    ``S = S10 / m + S01 / n`` built from the 2 x 2 sample covariance matrices
    of the placement values (DeLong, DeLong & Clarke-Pearson 1988, Biometrics
    44:837, Sec. 3). Unpaired: ``t = (A1 - A2) / sqrt(V1 + V2)`` with each
    curve's DeLong variance and Welch-Satterthwaite degrees of freedom
    ``(V1 + V2)^2 / (V1^2 / (N1 - 1) + V2^2 / (N2 - 1))``, ``Nk`` the number
    of observations (cases + controls) of curve ``k`` (Welch 1947,
    Biometrika 34:28; Satterthwaite 1946). Output is compared with R's
    ``pROC::roc.test()``.
    """
    if method != "delong":
        raise ValueError("method must be 'delong'")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")
    if roc1.direction != roc2.direction:
        warnings.warn(
            "the two curves have opposite directions; the DeLong test assumes both "
            "curves rank cases above controls in the same way",
            UserWarning,
            stacklevel=2,
        )
    paired_pair = _paired_rocs(roc1, roc2)
    if paired is None:
        paired = paired_pair is not None
    elif paired and paired_pair is None:
        raise ValueError("paired=True but the curves do not share the same response vector")
    elif not paired and paired_pair is not None:
        warnings.warn(
            "the curves share the same response vector and look paired; consider paired=True",
            UserWarning,
            stacklevel=2,
        )
    if _is_perfect(roc1) and _is_perfect(roc2):
        warnings.warn(
            "both ROC curves are perfect; the test of their AUCs carries no information",
            UserWarning,
            stacklevel=2,
        )
    auc1, auc2 = float(roc1.auc), float(roc2.auc)

    if paired:
        assert paired_pair is not None
        c1, c2 = paired_pair
        case1, ctrl1 = _placements(c1.controls, c1.cases, c1.direction)
        case2, ctrl2 = _placements(c2.controls, c2.cases, c2.direction)
        cov_cases = np.cov(np.vstack((case1, case2)), ddof=1)
        cov_controls = np.cov(np.vstack((ctrl1, ctrl2)), ddof=1)
        cov_auc = cov_cases / case1.size + cov_controls / ctrl1.size
        contrast = np.array([1.0, -1.0])
        var_diff = float(contrast @ cov_auc @ contrast)
        diff = float(c1.auc - c2.auc)
        se_diff = np.sqrt(max(var_diff, 0.0))
        statistic = _safe_ratio(diff, se_diff)
        cdf = stats.norm.cdf
        z = stats.norm.ppf(1 - (1 - conf_level) / 2)
        conf_int: tuple[float, float] | None = (
            float(diff - z * se_diff),
            float(diff + z * se_diff),
        )
        df: float | None = None
        label = "DeLong's test for two correlated ROC curves"
        auc1, auc2 = float(c1.auc), float(c2.auc)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            var1, var2 = roc1.var_auc(), roc2.var_auc()
        n1 = roc1.cases.size + roc1.controls.size
        n2 = roc2.cases.size + roc2.controls.size
        statistic = _safe_ratio(auc1 - auc2, np.sqrt(var1 + var2))
        with np.errstate(divide="ignore", invalid="ignore"):
            df = float((var1 + var2) ** 2 / (var1**2 / (n1 - 1) + var2**2 / (n2 - 1)))
        dof = df

        def cdf(x: float) -> float:
            return stats.t.cdf(x, dof)

        conf_int = None
        label = "DeLong's test for two ROC curves"

    if alternative == "two.sided":
        p_value = 2 * cdf(-abs(statistic))
    elif alternative == "greater":
        p_value = cdf(-statistic)
    else:
        p_value = cdf(statistic)
    return RocTest(
        statistic=float(statistic),
        p_value=float(p_value),
        auc1=auc1,
        auc2=auc2,
        method=label,
        paired=bool(paired),
        alternative=alternative,
        df=df,
        conf_int=conf_int,
    )


def plot_roc(
    curves: RocCurve | Mapping[str, RocCurve],
    path: Path,
    *,
    title: str = "ROC curve",
) -> Path:
    """Save ROC curves (sensitivity vs 1 - specificity) with AUC in the legend."""
    from .._core import mpl as _mpl  # noqa: F401

    import matplotlib.pyplot as plt

    if isinstance(curves, RocCurve):
        curves = {"model": curves}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    for name, r in curves.items():
        ax.plot(1 - r.specificities, r.sensitivities, label=f"{name} (AUC {r.auc:.3f})")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
