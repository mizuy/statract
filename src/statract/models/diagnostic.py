"""Diagnostic accuracy of a binary test against a reference standard.

``diagnostic_accuracy`` follows ``epiR::epi.tests`` (epiR 2.0): the same
statistics, proportion intervals (``method=``), log-scale intervals for the
likelihood ratios (Simel et al. 1991) and the diagnostic odds ratio, and the
same plug-in limits for the Youden index and the number needed to diagnose.

The 2×2 table is

============  ===========  ===========
              disease +    disease −
============  ===========  ===========
test +        ``tp``       ``fp``
test −        ``fn``       ``tn``
============  ===========  ===========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

Method = Literal["wilson", "exact", "clopper-pearson", "agresti", "jeffreys"]
_METHODS = ("wilson", "exact", "clopper-pearson", "agresti", "jeffreys")

# (name, epiR name, numerator, denominator) for the plain proportions.
_PROPORTIONS = (
    ("prevalence", "tp", "dpos", "n"),
    ("apparent_prevalence", "ap", "tpos", "n"),
    ("sensitivity", "se", "tp", "dpos"),
    ("specificity", "sp", "tn", "dneg"),
    ("accuracy", "diag.ac", "correct", "n"),
    ("ppv", "pv.pos", "tp", "tpos"),
    ("npv", "pv.neg", "tn", "tneg"),
    ("rule_out", "p.rout", "tneg", "n"),
    ("rule_in", "p.rin", "tpos", "n"),
    ("false_positive_rate", "p.tpdn", "fp", "dneg"),
    ("false_negative_rate", "p.tndp", "fn", "dpos"),
    ("false_discovery_rate", "p.dntp", "fp", "tpos"),
    ("false_omission_rate", "p.dptn", "fn", "tneg"),
)
_ORDER = (
    "apparent_prevalence",
    "prevalence",
    "sensitivity",
    "specificity",
    "accuracy",
    "diagnostic_or",
    "nnd",
    "youden",
    "ppv",
    "npv",
    "lr_pos",
    "lr_neg",
    "rule_out",
    "rule_in",
    "false_positive_rate",
    "false_negative_rate",
    "false_discovery_rate",
    "false_omission_rate",
)
EPIR_NAMES = {name: r for name, r, _, _ in _PROPORTIONS} | {
    "diagnostic_or": "diag.or",
    "nnd": "nndx",
    "youden": "youden",
    "lr_pos": "lr.pos",
    "lr_neg": "lr.neg",
}


def _interval(k: float, n: float, method: str, conf_level: float) -> tuple[float, float]:
    """epiR's ``zwilson`` / ``zexact`` / ``zclopperpearson`` / ``zagresti`` / ``zjeffreys``."""
    alpha = 1 - conf_level
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    if method == "wilson":
        z = stats.norm.ppf(1 - alpha / 2)
        half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / (1 + z**2 / n)
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        return (float(centre - half), float(centre + half))
    if method in ("exact", "clopper-pearson"):
        lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
        return (lo, hi)
    if method == "agresti":
        z = stats.norm.ppf(1 - alpha / 2)
        n_ac = n + z**2
        p_ac = (k + z**2 / 2) / n_ac
        half = z * np.sqrt(p_ac * (1 - p_ac) / n_ac)
        return (float(p_ac - half), float(p_ac + half))
    if method == "jeffreys":
        return (
            float(stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)),
            float(stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)),
        )
    raise ValueError(method)


@dataclass(frozen=True)
class DiagnosticAccuracy:
    """Counts of the 2×2 table and the interval settings. ``tidy()`` gives the statistics."""

    tp: int
    fp: int
    fn: int
    tn: int
    method: str
    conf_level: float

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def table(self) -> pl.DataFrame:
        """The 2×2 table with margins, test result by row and disease by column."""
        return pl.DataFrame(
            {
                "test": ["positive", "negative", "total"],
                "disease_positive": [self.tp, self.fn, self.tp + self.fn],
                "disease_negative": [self.fp, self.tn, self.fp + self.tn],
                "total": [self.tp + self.fp, self.fn + self.tn, self.n],
            }
        )

    def tidy(self) -> pl.DataFrame:
        """One row per statistic: ``statistic``, ``estimate``, ``conf_low``, ``conf_high``.

        Proportions use ``method``. ``lr_pos``, ``lr_neg`` and
        ``diagnostic_or`` use log-scale normal intervals. ``youden`` and
        ``nnd`` (number needed to diagnose, ``1 / youden``) plug the
        sensitivity and specificity limits in, as epiR does. Point estimates
        are the plain ratios for every method (epiR's ``agresti`` reports the
        adjusted proportion instead).
        """
        tp, fp, fn, tn = (np.float64(v) for v in (self.tp, self.fp, self.fn, self.tn))
        counts = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n": tp + fp + fn + tn,
            "dpos": tp + fn,
            "dneg": fp + tn,
            "tpos": tp + fp,
            "tneg": fn + tn,
            "correct": tp + tn,
        }
        rows: dict[str, tuple[float, float, float]] = {}
        for name, _, num, den in _PROPORTIONS:
            k, m = counts[num], counts[den]
            est = k / m if m else np.nan
            lo, hi = _interval(k, m, self.method, self.conf_level)
            rows[name] = (est, lo, hi)
        z = float(stats.norm.ppf(1 - (1 - self.conf_level) / 2))
        se, sp = rows["sensitivity"][0], rows["specificity"][0]
        m1, m0 = counts["dpos"], counts["dneg"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lrp = se / (1 - sp)
            s = np.sqrt((1 - se) / (m1 * se) + sp / (m0 * (1 - sp)))
            rows["lr_pos"] = (lrp, float(np.exp(np.log(lrp) - z * s)), float(np.exp(np.log(lrp) + z * s)))
            lrn = (1 - se) / sp
            s = np.sqrt(se / (m1 * (1 - se)) + (1 - sp) / (m0 * sp))
            rows["lr_neg"] = (lrn, float(np.exp(np.log(lrn) - z * s)), float(np.exp(np.log(lrn) + z * s)))
            dor = (tp * tn) / (fp * fn)
            s = np.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn)
            rows["diagnostic_or"] = (dor, float(np.exp(np.log(dor) - z * s)), float(np.exp(np.log(dor) + z * s)))
            (_, se_lo, se_hi), (_, sp_lo, sp_hi) = rows["sensitivity"], rows["specificity"]
            j = se - (1 - sp)
            j1, j2 = se_lo - (1 - sp_lo), se_hi - (1 - sp_hi)
            rows["youden"] = (j, min(j1, j2), max(j1, j2))
            rows["nnd"] = (1 / j, min(1 / j1, 1 / j2), max(1 / j1, 1 / j2))
        return pl.DataFrame(
            {
                "statistic": list(_ORDER),
                "estimate": [float(rows[k][0]) for k in _ORDER],
                "conf_low": [float(rows[k][1]) for k in _ORDER],
                "conf_high": [float(rows[k][2]) for k in _ORDER],
            }
        )

    def predictive_values(self, prevalence: float | np.ndarray) -> pl.DataFrame:
        """PPV and NPV at other prevalences, from this sensitivity and specificity (Bayes' rule).

        Point estimates only. Use this when the study prevalence differs from
        the setting where the test will be used (for example a case–control
        design).
        """
        prev = np.atleast_1d(np.asarray(prevalence, dtype=float))
        if np.any((prev < 0) | (prev > 1)):
            raise ValueError("prevalence must be in [0, 1]")
        se = self.tp / (self.tp + self.fn)
        sp = self.tn / (self.tn + self.fp)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
            npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
        return pl.DataFrame({"prevalence": prev, "ppv": ppv, "npv": npv})


def _binary(values, name: str) -> np.ndarray:
    if isinstance(values, pl.Series):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr.astype(float)
    arr = arr.astype(float)
    ok = ~np.isnan(arr)
    if not np.all(np.isin(arr[ok], (0.0, 1.0))):
        raise ValueError(f"{name} must be 0/1 or boolean")
    return arr


def diagnostic_accuracy(
    test=None,
    disease=None,
    *,
    tp: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
    tn: int | None = None,
    method: Method = "wilson",
    conf_level: float = 0.95,
) -> DiagnosticAccuracy:
    """Sensitivity, specificity, predictive values, likelihood ratios and more, with intervals.

    Pass the test result and the reference standard as 0/1 or boolean arrays
    (rows with a missing value in either are dropped), or the four counts
    ``tp``, ``fp``, ``fn``, ``tn``. ``method`` is the proportion interval:
    ``"wilson"`` (default), ``"exact"`` / ``"clopper-pearson"``,
    ``"agresti"`` or ``"jeffreys"``, as in ``epiR::epi.tests``.

    ```python
    acc = diagnostic_accuracy(df["risk"] >= 0.05, df["bleed"])
    acc.tidy()                       # sensitivity, specificity, lr_pos, ...
    acc.predictive_values([0.01, 0.05, 0.2])
    ```
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be in (0, 1)")
    counts = (tp, fp, fn, tn)
    if test is not None or disease is not None:
        if test is None or disease is None:
            raise ValueError("pass both test and disease")
        if any(c is not None for c in counts):
            raise ValueError("pass arrays or counts, not both")
        t = _binary(test, "test")
        d = _binary(disease, "disease")
        if t.shape != d.shape:
            raise ValueError("test and disease must have the same length")
        keep = ~(np.isnan(t) | np.isnan(d))
        t, d = t[keep] == 1, d[keep] == 1
        tp, fp = int(np.sum(t & d)), int(np.sum(t & ~d))
        fn, tn = int(np.sum(~t & d)), int(np.sum(~t & ~d))
    else:
        if any(c is None for c in counts):
            raise ValueError("pass test and disease, or all of tp, fp, fn, tn")
        if any(int(c) != c or c < 0 for c in counts):
            raise ValueError("counts must be non-negative integers")
        tp, fp, fn, tn = (int(c) for c in counts)
    return DiagnosticAccuracy(tp=tp, fp=fp, fn=fn, tn=tn, method=method, conf_level=conf_level)
