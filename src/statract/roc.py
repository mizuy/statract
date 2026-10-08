"""ROC curves, AUC, DeLong variance and tests (port of R pROC 1.18.5).

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
    """``roc_utils_thresholds``: midpoints of sorted unique values with near-tie fix."""
    u = np.unique(values)
    lo = np.concatenate(([-np.inf], u))
    hi = np.concatenate((u, [np.inf]))
    with np.errstate(invalid="ignore"):
        t1 = (lo + hi) / 2
        t2 = lo / 2 + hi / 2
    thr = np.where(np.abs(t1) > 1e100, t2, t1)
    for i in np.flatnonzero(np.isin(thr, values)):
        if direction == ">":
            if thr[i] == u[i - 1]:
                pass
            elif thr[i] == u[i]:
                thr[i] = u[i - 1]
            else:
                raise RuntimeError("could not fix near ties in thresholds")
        else:
            if thr[i] == u[i - 1]:
                thr[i] = u[i]
            elif thr[i] == u[i]:
                pass
            else:
                raise RuntimeError("could not fix near ties in thresholds")
    return thr


def _se_sp(
    thresholds: np.ndarray, controls: np.ndarray, cases: np.ndarray, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity and specificity at each threshold (``rocUtilsPerfsAllC``)."""
    cs = np.sort(cases)
    ct = np.sort(controls)
    if direction == ">":
        tp = np.searchsorted(cs, thresholds, side="right")
        tn = ct.size - np.searchsorted(ct, thresholds, side="right")
    else:
        tp = cs.size - np.searchsorted(cs, thresholds, side="left")
        tn = np.searchsorted(ct, thresholds, side="left")
    return tp / cs.size, tn / ct.size


def _placements(controls: np.ndarray, cases: np.ndarray, direction: str) -> tuple[float, np.ndarray, np.ndarray]:
    """DeLong placements ``(theta, X, Y)`` as in ``delongPlacementsCpp``."""
    if direction == ">":
        controls, cases = -controls, -cases
    m, n = cases.size, controls.size
    ct = np.sort(controls)
    cs = np.sort(cases)
    lo = np.searchsorted(ct, cases, side="left")
    x_raw = lo + (np.searchsorted(ct, cases, side="right") - lo) / 2.0
    lo = np.searchsorted(cs, controls, side="left")
    y_raw = lo + (np.searchsorted(cs, controls, side="right") - lo) / 2.0
    theta = float(x_raw.sum() / m / n)
    return theta, x_raw / n, 1.0 - y_raw / m


def _optim_crit(se: np.ndarray, sp: np.ndarray, weights: Sequence[float], method: str) -> np.ndarray:
    w1, w2 = float(weights[0]), float(weights[1])
    if not 0 < w2 < 1:
        raise ValueError("best_weights[1] (prevalence) must be in (0, 1)")
    r = (1 - w2) / (w1 * w2)
    if method == "youden":
        return se + r * sp
    if method in ("closest.topleft", "topleft"):
        return -((1 - se) ** 2 + r * (1 - sp) ** 2)
    raise ValueError("best_method must be 'youden' or 'closest.topleft'")


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

    def _perfect(self) -> bool:
        return bool(abs(np.max(self.sensitivities + self.specificities) - 2) < np.finfo(float).eps ** 0.5)

    def var_auc(self) -> float:
        """DeLong variance of the AUC (``pROC::var``)."""
        m, n = self.cases.size, self.controls.size
        if m <= 1 or n <= 1:
            return float("nan")
        if self._perfect():
            warnings.warn("var() of a ROC curve with AUC == 1 is always 0", stacklevel=2)
        _, x, y = _placements(self.controls, self.cases, self.direction)
        return float(np.var(y, ddof=1) / n + np.var(x, ddof=1) / m)

    def ci_auc(self, level: float = 0.95, method: Literal["delong"] = "delong") -> tuple[float, float, float]:
        """``(low, auc, high)`` DeLong interval, clipped to [0, 1] (``pROC::ci.auc``)."""
        if method != "delong":
            raise ValueError("only method='delong' is supported")
        if not 0 <= level <= 1:
            raise ValueError("level must be in [0, 1]")
        m, n = self.cases.size, self.controls.size
        if m <= 1 or n <= 1:
            return (float("nan"),) * 3
        if self._perfect():
            warnings.warn("ci_auc() of a ROC curve with AUC == 1 is always 1-1", stacklevel=2)
        theta, x, y = _placements(self.controls, self.cases, self.direction)
        s = np.sum((x - theta) ** 2) / (m - 1) / m + np.sum((y - theta) ** 2) / (n - 1) / n
        z = stats.norm.ppf([(1 - level) / 2, 0.5, 1 - (1 - level) / 2])
        ci = np.clip(theta + np.sqrt(s) * z, 0.0, 1.0)
        return float(ci[0]), float(ci[1]), float(ci[2])

    def _calc_coords(self, thr: np.ndarray, se: np.ndarray, sp: np.ndarray, weights: Sequence[float]) -> dict[str, np.ndarray]:
        ncases, ncontrols = self.cases.size, self.controls.size
        tp = se * ncases
        fn = ncases - tp
        tn = sp * ncontrols
        fp = ncontrols - tn
        with np.errstate(divide="ignore", invalid="ignore"):
            npv = tn / (tn + fn)
            ppv = tp / (tp + fp)
            fnr = fn / (tp + fn)
            fdr = fp / (tp + fp)
        accuracy = (tp + tn) / (ncases + ncontrols)
        return {
            "threshold": thr,
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
        raise ValueError("no control observation")
    if cases.size == 0:
        raise ValueError("no case observation")
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


def _paired_rocs(r1: RocCurve, r2: RocCurve) -> tuple[RocCurve, RocCurve] | None:
    """``pROC::are.paired`` with ``return.paired.rocs=TRUE``."""
    if r1.levels != r2.levels:
        return None
    if r1.response.equals(r2.response, check_names=False):
        return r1, r2
    if r1.original_response.len() == r2.original_response.len() and r1.original_response.equals(
        r2.original_response, check_names=False
    ):
        resp = r1.original_response
        exclude = (
            r1.original_predictor.is_null()
            | r2.original_predictor.is_null()
            | resp.is_null()
            | ~resp.is_in(list(r1.levels))
        )
        keep = ~exclude
        out = []
        for r in (r1, r2):
            out.append(
                _build(
                    resp.filter(keep),
                    r.original_predictor.filter(keep),
                    list(r.levels),
                    r.direction,  # type: ignore[arg-type]
                    r.original_response,
                    r.original_predictor,
                )
            )
        return out[0], out[1]
    return None


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
    """
    if method != "delong":
        raise ValueError("only method='delong' is supported")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")
    if roc1._perfect() and roc2._perfect():
        warnings.warn("roc_test() of two ROC curves with AUC == 1 always has p = 1", stacklevel=2)
    pair = _paired_rocs(roc1, roc2)
    if paired is None:
        paired = pair is not None
    elif paired and pair is None:
        raise ValueError("the paired ROC test cannot be applied to unpaired curves")
    elif not paired and pair is not None:
        warnings.warn("the ROC curves seem to be paired; consider a paired test", stacklevel=2)
    if paired:
        assert pair is not None
        roc1, roc2 = pair
    if roc1.direction != roc2.direction:
        warnings.warn("DeLong's test should not be applied to ROC curves with a different direction", stacklevel=2)

    def _p(stat: float, cdf: Any) -> float:
        if alternative == "two.sided":
            return float(2 * cdf(-abs(stat)))
        if alternative == "greater":
            return float(cdf(-stat))
        return float(cdf(stat))

    tr, xr, yr = _placements(roc1.controls, roc1.cases, roc1.direction)
    ts, xs, ys = _placements(roc2.controls, roc2.cases, roc2.direction)
    if paired:
        n, m = roc1.controls.size, roc1.cases.size
        dxr, dxs, dyr, dys = xr - tr, xs - ts, yr - tr, ys - ts
        sx = np.array([[dxr @ dxr, dxr @ dxs], [dxs @ dxr, dxs @ dxs]]) / (m - 1)
        sy = np.array([[dyr @ dyr, dyr @ dys], [dys @ dyr, dys @ dys]]) / (n - 1)
        s = sx / m + sy / n
        lvec = np.array([1.0, -1.0])
        sig = float(np.sqrt(lvec @ s @ lvec))
        d = tr - ts
        with np.errstate(divide="ignore", invalid="ignore"):
            z = d / sig
        if np.isnan(z) and d == 0 and sig == 0:
            z = 0.0
        crit = stats.norm.ppf(1 - (1 - conf_level) / 2)
        return RocTest(
            statistic=float(z),
            p_value=_p(z, stats.norm.cdf),
            auc1=roc1.auc,
            auc2=roc2.auc,
            method="DeLong's test for two correlated ROC curves",
            paired=True,
            alternative=alternative,
            conf_int=(d - crit * sig, d + crit * sig),
        )
    nr, mr = roc1.controls.size, roc1.cases.size
    ns, ms = roc2.controls.size, roc2.cases.size
    sr = np.sum((xr - tr) ** 2) / (mr - 1) / mr + np.sum((yr - tr) ** 2) / (nr - 1) / nr
    ss = np.sum((xs - ts) ** 2) / (ms - 1) / ms + np.sum((ys - ts) ** 2) / (ns - 1) / ns
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (tr - ts) / np.sqrt(sr + ss)
        df = (sr + ss) ** 2 / (sr**2 / (nr + mr - 1) + ss**2 / (ns + ms - 1))
    return RocTest(
        statistic=float(t),
        p_value=_p(t, lambda q: stats.t.cdf(q, df)),
        auc1=roc1.auc,
        auc2=roc2.auc,
        method="DeLong's test for two ROC curves",
        paired=False,
        alternative=alternative,
        df=float(df),
    )


def plot_roc(
    curves: RocCurve | Mapping[str, RocCurve],
    path: Path,
    *,
    title: str = "ROC curve",
) -> Path:
    """Save ROC curves (sensitivity vs 1 - specificity) with AUC in the legend."""
    from . import _mpl as _mpl  # noqa: F401

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
