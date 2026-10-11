"""Cox proportional hazards models.

Partial likelihood is maximized by Newton–Raphson with step halving. Ties use
the Efron correction unless ``ties="breslow"``. Strata keep separate risk sets
and share the coefficients. Cluster-robust covariance is the Lin–Wei estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from numba import njit

from ..models.design import ColumnRef, Design, build_design, column_series, design_matrix
from ..models.fit import Fit
from ..models.formula import is_formula
from .spec import combine_strata, parse_survival_formula

_MAX_ITER = 25
_EPS = 1e-9


@dataclass
class CoxFit:
    """Cox model. The coefficient table matches :class:`statract.models.fit.Fit`."""

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_obs: int
    log_likelihood: float | None
    residual_df: int | None
    family: str
    x: np.ndarray
    y: np.ndarray
    row_index: np.ndarray
    design: Design
    weights: np.ndarray
    offset: np.ndarray
    working_residuals: np.ndarray
    working_weights: np.ndarray
    hat_values: np.ndarray
    dispersion: float
    time: np.ndarray
    event: np.ndarray
    strata: np.ndarray
    entry: np.ndarray
    baseline_time: np.ndarray
    baseline_hazard_values: np.ndarray
    baseline_strata: np.ndarray
    information: np.ndarray
    ties: str
    converged: bool
    deviance: float | None = None

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False, df: float | None = None) -> pl.DataFrame:
        return Fit.tidy(self, level=level, exponentiate=exponentiate, df=np.inf)

    def _ensure_baseline(self) -> None:
        pending = getattr(self, "_baseline_pending", None)
        if pending is None:
            return
        times, hazards, labels = _baseline(*pending)
        self.baseline_time = np.asarray(times)
        self.baseline_hazard_values = np.asarray(hazards, dtype=float)
        self.baseline_strata = np.asarray(labels)
        self._baseline_pending = None

    def baseline_hazard(self) -> pl.DataFrame:
        """Cumulative baseline hazard, matching ``basehaz(centered=FALSE)``.

        One row per distinct follow-up time in each stratum, censored times
        included, so the hazard is flat between deaths.
        """
        self._ensure_baseline()
        times: list[np.ndarray] = []
        hazards: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for stratum in np.unique(self.strata):
            sel = np.flatnonzero(self.baseline_strata == stratum)
            event_times = self.baseline_time[sel].astype(float)
            cumulative = np.cumsum(self.baseline_hazard_values[sel])
            grid = np.unique(self.time[self.strata == stratum])
            at = np.searchsorted(event_times, grid, side="right")
            hazard = np.where(at > 0, np.concatenate([[0.0], cumulative])[at], 0.0)
            times.append(grid)
            hazards.append(hazard)
            labels.append(np.full(len(grid), stratum, dtype=object))
        return pl.DataFrame(
            {
                "time": np.concatenate(times) if times else np.zeros(0),
                "hazard": np.concatenate(hazards) if hazards else np.zeros(0),
                "stratum": np.concatenate(labels).astype(str) if labels else np.zeros(0, dtype=str),
            }
        )

    def survival_curve(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """Subject-specific survival at each row's follow-up time."""
        survival = self.predict(data, kind="survival")
        if data is None:
            time = self.time
        else:
            time = _column_or_raise(data, self._time_name)
        return pl.DataFrame({"time": time, "survival": survival})

    def predict(self, data: pl.DataFrame | None = None, *, kind: str = "linear_predictor", times: np.ndarray | None = None) -> np.ndarray:
        if kind not in {"linear_predictor", "risk", "expected", "survival"}:
            raise ValueError("kind must be linear_predictor, risk, expected, or survival")
        if self.ties == "exact" and kind in {"expected", "survival"}:
            raise ValueError("expected and survival use a baseline hazard; conditional_logit has none")
        if data is None:
            lp = self.x @ self.coefficients + self.offset
        else:
            design = build_design(data, self.design)
            lp = design.x @ self.coefficients
        if kind == "linear_predictor":
            return lp
        if kind == "risk":
            return np.exp(lp)
        if data is None and times is None:
            # predict.coxph: the fitted expected count is status minus the
            # martingale residual, which carries the Efron tie correction.
            expected = self.event.astype(float) - _martingale_residuals(self)
            return expected if kind == "expected" else np.exp(-expected)
        if data is None:
            time = np.asarray(times, dtype=float)
            strata = self.strata
            entry = self.entry
        else:
            time = times if times is not None else _column_or_raise(data, self._time_name)
            strata_names = getattr(self, "_strata_names", None)
            if strata_names:
                strata = combine_strata(data, strata_names, design.row_index)
            else:
                strata = _optional_aligned(data, self._strata_name, design.row_index, "_")
            entry = _optional_aligned(data, self._entry_name, design.row_index, 0.0).astype(float)
            if times is not None:
                time = np.asarray(times, dtype=float)
        if kind == "expected":
            return self._expected(time, entry, lp, strata)
        return np.exp(-self._expected(time, entry, lp, strata))

    def residuals(self, kind: str = "martingale") -> np.ndarray:
        if self.ties == "exact":
            raise ValueError("residuals use the Cox partial likelihood; conditional_logit method='exact' has none")
        allowed = {
            "martingale",
            "deviance",
            "score",
            "schoenfeld",
            "scaled_schoenfeld",
            "dfbeta",
            "dfbetas",
        }
        if kind not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}")
        martingale = _martingale_residuals(self)
        if kind == "martingale":
            return martingale
        if kind == "deviance":
            # residuals.coxph: sign(M) * sqrt(-2 * (M + status * log(status - M)))
            status = self.event.astype(float)
            term = martingale.copy()
            live = status > 0
            gap = np.clip(status[live] - martingale[live], 1e-12, None)
            term[live] = martingale[live] + status[live] * np.log(gap)
            return np.sign(martingale) * np.sqrt(np.clip(-2 * term, 0, None))
        score = self._score_residuals()
        if kind == "score":
            return score
        # dfbeta multiplies by the case weight after the coefficient covariance.
        weighted_score = score * self.weights[:, None]
        if kind == "dfbeta":
            return weighted_score @ self.information_inverse()
        if kind == "dfbetas":
            naive = self.information_inverse()
            se = np.sqrt(np.clip(np.diag(naive), 0, None))
            return (weighted_score @ naive) / se
        schoenfeld = self._schoenfeld()
        if kind == "schoenfeld":
            return schoenfeld
        ndead = schoenfeld.shape[0]
        naive = self.information_inverse()
        return schoenfeld @ naive * ndead + self.coefficients

    def concordance(self) -> pl.DataFrame:
        """Harrell's C and its standard error, matching ``concordance(coxph)``.

        Pairs are compared within strata and pooled. Counting-process rows are
        compared at each death time among the rows at risk then. The standard
        error is the infinitesimal jackknife, summed by cluster when the fit
        has one.
        """
        lp = self.x @ self.coefficients + self.offset
        counting = not bool(np.all(self.entry == 0))
        concordant = discordant = tied = 0.0
        inf_c = np.zeros(self.n_obs)
        inf_d = np.zeros(self.n_obs)
        inf_t = np.zeros(self.n_obs)
        for stratum in np.unique(self.strata):
            idx = np.flatnonzero(self.strata == stratum)
            c, d, t, ic, idd, it = _concordance_counts(
                np.ascontiguousarray(self.time[idx], dtype=np.float64),
                np.ascontiguousarray(self.entry[idx], dtype=np.float64),
                np.ascontiguousarray(self.event[idx], dtype=np.float64),
                np.ascontiguousarray(lp[idx], dtype=np.float64),
                np.ascontiguousarray(self.weights[idx], dtype=np.float64),
                counting,
            )
            concordant += c
            discordant += d
            tied += t
            inf_c[idx] = ic
            inf_d[idx] = idd
            inf_t[idx] = it
        usable = concordant + discordant + tied
        c_index = (concordant + 0.5 * tied) / usable if usable else float("nan")
        if usable:
            somer = (concordant - discordant) / usable
            influence = ((inf_c - inf_d) - (inf_c + inf_d + inf_t) * somer) * self.weights / (2 * usable)
        else:
            influence = np.zeros(self.n_obs)
        cluster = getattr(self, "_cluster", None)
        if cluster is not None:
            _, inverse = np.unique(cluster, return_inverse=True)
            influence = np.bincount(inverse, weights=influence)
        se = float(np.sqrt(np.sum(influence**2)))
        return pl.DataFrame(
            {
                "concordance": [c_index],
                "std_error": [se],
                "concordant": [concordant],
                "discordant": [discordant],
                "tied_risk": [tied],
            }
        )

    def information_inverse(self) -> np.ndarray:
        """Model-based covariance, before any robust or cluster replacement."""
        return np.linalg.pinv(self.information)

    def score_contributions(self) -> np.ndarray:
        if self.ties == "exact":
            raise ValueError("score residuals use the Cox partial likelihood; conditional_logit method='exact' has none")
        return self._score_residuals()

    def bread(self) -> np.ndarray:
        return self.n_obs * self.information_inverse()

    def _expected(self, time: np.ndarray, entry: np.ndarray, lp: np.ndarray, strata: np.ndarray) -> np.ndarray:
        self._ensure_baseline()
        risk = np.exp(lp)
        out = np.zeros(len(time))
        for i, (t, start, stratum, r) in enumerate(zip(time, entry, strata, risk, strict=False)):
            sel = (self.baseline_strata == stratum) & (self.baseline_time <= t) & (self.baseline_time > start)
            out[i] = r * float(self.baseline_hazard_values[sel].sum())
        return out

    def _survival_at(self, time: np.ndarray, lp: np.ndarray, strata: np.ndarray) -> np.ndarray:
        return np.exp(-self._expected(time, np.zeros(len(time)), lp, strata))

    def _score_residuals(self) -> np.ndarray:
        return _score_residuals(self)

    def _schoenfeld(self) -> tuple[np.ndarray, np.ndarray]:
        return _schoenfeld(self)


def cox_ph(
    data: pl.DataFrame,
    time: ColumnRef | None = None,
    event: ColumnRef | None = None,
    predictors: list[ColumnRef] | None = None,
    *,
    strata: ColumnRef | None = None,
    cluster: ColumnRef | None = None,
    entry: ColumnRef | None = None,
    weights: ColumnRef | None = None,
    offset: ColumnRef | None = None,
    ties: str = "efron",
) -> CoxFit:
    """Fit a Cox model.

    ``time`` may be a Wilkinson formula such as ``"Surv(time, status) ~ age + sex"``.
    ``strata(site)`` and ``cluster(id)`` in that formula replace ``strata=`` and
    ``cluster=``. ``Surv(start, stop, status)`` sets the entry time.
    """
    if ties not in {"efron", "breslow"}:
        raise ValueError("ties must be 'efron' or 'breslow'")
    formula_cluster = None
    strata_names: tuple[str, ...] = ()
    if is_formula(time):
        if event is not None or predictors is not None:
            raise ValueError("a Surv formula already names the time, the event, and the predictors")
        parsed = parse_survival_formula(
            data,
            str(time),
            weights=weights,
            offset=offset,
            strata=strata,
            cluster=cluster,
            entry=entry,
            drop_intercept=True,
            allow_counting=True,
            allow_strata=True,
            allow_cluster=True,
        )
        design = parsed.design
        time_a = parsed.time
        event_a = np.asarray(parsed.event, dtype=float) > 0
        weight_a = np.ones(design.n_obs) if weights is None else _numeric_or_one(data, weights, design.row_index)
        offset_a = parsed.offset
        entry_a = parsed.entry
        strata_a = parsed.strata
        formula_cluster = parsed.cluster
        time_name = parsed.time_name
        strata_names = parsed.strata_names
        entry_name = parsed.entry_name
        strata_label = strata_names[0] if len(strata_names) == 1 else None
    else:
        if time is None or event is None or predictors is None:
            raise ValueError("time, event, and predictors are required when time is a column")
        extra: list[ColumnRef] = [time, event]
        for ref in (strata, cluster, entry, weights, offset):
            if ref is not None:
                extra.append(ref)
        design = design_matrix(data, predictors, extra=extra, intercept=False)
        idx = design.row_index
        time_a = np.asarray(column_series(data, time).gather(idx.tolist()).to_numpy(), dtype=float)
        event_a = np.asarray(column_series(data, event).gather(idx.tolist()).to_numpy(), dtype=float) > 0
        weight_a = _numeric_or_one(data, weights, idx)
        offset_a = _numeric_or_zero(data, offset, idx)
        entry_a = _numeric_or_zero(data, entry, idx) if entry is not None else np.zeros(design.n_obs)
        if strata is None:
            strata_a = np.array(["_"] * design.n_obs)
        else:
            strata_a = np.asarray(column_series(data, strata).gather(idx.tolist()).to_list())
        time_name = time if isinstance(time, str) else column_series(data, time).name
        strata_label = strata if isinstance(strata, str) else None
        entry_name = entry if isinstance(entry, str) else None
        if strata_label is not None:
            strata_names = (strata_label,)
    beta, info, converged, ll = _newton(design.x, time_a, event_a, weight_a, offset_a, entry_a, strata_a, ties)
    cov = np.linalg.pinv(info)
    # The baseline hazard is not needed for the coefficients. Build it on first use.
    fit = CoxFit(
        coefficients=beta,
        covariance=cov,
        names=list(design.names),
        n_obs=design.n_obs,
        log_likelihood=ll,
        residual_df=None,
        family="cox",
        x=design.x,
        y=event_a.astype(float),
        row_index=design.row_index,
        design=design,
        weights=weight_a,
        offset=offset_a,
        working_residuals=np.zeros(design.n_obs),
        working_weights=weight_a,
        hat_values=np.zeros(design.n_obs),
        dispersion=1.0,
        deviance=float(-2 * ll),
        time=time_a,
        event=event_a.astype(float),
        strata=strata_a,
        entry=entry_a,
        baseline_time=np.zeros(0),
        baseline_hazard_values=np.zeros(0),
        baseline_strata=np.empty(0, dtype=object),
        information=info,
        ties=ties,
        converged=converged,
    )
    fit._baseline_pending = (design.x, beta, time_a, event_a, weight_a, offset_a, entry_a, strata_a, ties)
    # survival::coxph uses the Lin–Wei variance with a cluster, or with case
    # weights that are not all whole numbers (one cluster per row). Integer
    # weights count as frequency weights and keep the model-based covariance.
    robust_labels = None
    if formula_cluster is not None:
        robust_labels = np.asarray(formula_cluster)
    elif cluster is not None:
        robust_labels = np.asarray(column_series(data, cluster).gather(design.row_index.tolist()).to_list())
    elif weights is not None and bool(np.any(weight_a != np.floor(weight_a))):
        robust_labels = np.arange(design.n_obs)
    explicit_cluster = formula_cluster is not None or cluster is not None
    fit._cluster = robust_labels if explicit_cluster else None
    if robust_labels is not None:
        scores = fit.score_contributions() * weight_a[:, None]
        _, inverse = np.unique(robust_labels, return_inverse=True)
        summed = np.zeros((inverse.max() + 1, scores.shape[1]))
        np.add.at(summed, inverse, scores)
        meat = summed.T @ summed
        fit.covariance = cov @ meat @ cov
    fit._time_name = time_name
    fit._strata_name = strata_label
    fit._strata_names = strata_names
    fit._entry_name = entry_name
    return fit


def proportional_hazards_test(fit: CoxFit, *, time_transform: str = "kaplan_meier") -> pl.DataFrame:
    """Score test of a time-varying coefficient, matching ``cox.zph``.

    ``time_transform`` is ``"kaplan_meier"`` (R's ``"km"``), ``"rank"``,
    ``"identity"``, or ``"log"``.
    """
    g = _centered_time_weights(fit, time_transform)
    u, imat = _zph_score(fit, g)
    p = len(fit.names)
    rows = []
    # One row per model term, as cox.zph(terms=TRUE): a factor's columns are tested jointly.
    for term, cols in _term_columns(fit.design, fit.names):
        idx = list(range(p)) + [p + j for j in cols]
        block = imat[np.ix_(idx, idx)]
        score = np.zeros(len(idx))
        score[p:] = u[[p + j for j in cols]]
        stat = float(score @ np.linalg.solve(block, score))
        df = float(len(cols))
        rows.append({"term": term, "statistic": stat, "df": df, "p_value": float(stats.chi2.sf(stat, df))})
    score = np.zeros(2 * p)
    score[p:] = u[p:]
    stat = float(score @ np.linalg.solve(imat, score))
    rows.append({"term": "global", "statistic": stat, "df": float(p), "p_value": float(stats.chi2.sf(stat, p))})
    return pl.DataFrame(rows)


def _term_columns(design: Design, names: list[str]) -> list[tuple[str, list[int]]]:
    """Group design columns by model term (``stage`` for ``stageII`` and ``stageIII``)."""
    labels: list[str] = []
    if design.recipes is not None:
        recipes = [r for r, name in zip(design.recipes, design.names, strict=False) if name in names]
        labels = [":".join(var for var, _level in recipe) or name for recipe, name in zip(recipes, names, strict=False)]
    else:
        lookup = {}
        for spec in design.predictors:
            if spec.kind == "factor":
                for level in spec.levels:
                    lookup[f"{spec.name}{level}"] = spec.name
            else:
                lookup[spec.name] = spec.name
        labels = [lookup.get(name, name) for name in names]
    groups: dict[str, list[int]] = {}
    for j, label in enumerate(labels):
        groups.setdefault(label, []).append(j)
    return list(groups.items())


def _newton(x, time, event, weights, offset, entry, strata, ties):
    p = x.shape[1]
    beta = np.zeros(p)
    # Right-censored rows are sorted once. Counting-process rows stay on the walk.
    layout = _ordinary_layout(x, time, event, weights, offset, strata) if np.all(entry == 0) else None
    efron = ties == "efron"

    def score(beta_now):
        if layout is None:
            return _partial(beta_now, x, time, event, weights, offset, entry, strata, ties)
        return _score_layout(beta_now, layout, efron)

    ll, grad, hess = score(beta)
    converged = False
    for _ in range(_MAX_ITER):
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        lam = 1.0
        accepted = False
        while lam > 1e-8:
            trial = beta + lam * step
            ll_t, grad_t, hess_t = score(trial)
            if np.isfinite(ll_t) and ll_t >= ll - 1e-8:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        if abs(ll_t - ll) <= _EPS * (abs(ll) + _EPS):
            beta, ll, grad, hess = trial, ll_t, grad_t, hess_t
            converged = True
            break
        beta, ll, grad, hess = trial, ll_t, grad_t, hess_t
    return beta, -hess, converged, ll


class _OrdinaryLayout:
    """Exit-time order of every stratum, concatenated so one kernel scores them all."""

    __slots__ = ("x", "event", "weight", "offset", "starts", "ends", "row0", "group0")

    def __init__(self, x, event, weight, offset, starts, ends, row0, group0):
        self.x = x
        self.event = event
        self.weight = weight
        self.offset = offset
        self.starts = starts
        self.ends = ends
        self.row0 = row0
        self.group0 = group0


def _ordinary_layout(x, time, event, weights, offset, strata):
    pieces_x = []
    pieces_e = []
    pieces_w = []
    pieces_o = []
    starts = []
    ends = []
    row0 = [0]
    group0 = [0]
    row = 0
    groups = 0
    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        if idx.size == 0:
            continue
        order, bounds = _time_blocks(time[idx])
        if bounds.size <= 1:
            continue
        pieces_x.append(np.ascontiguousarray(x[idx][order]))
        pieces_e.append(np.ascontiguousarray(event[idx][order], dtype=np.float64))
        pieces_w.append(np.ascontiguousarray(weights[idx][order], dtype=np.float64))
        pieces_o.append(np.ascontiguousarray(offset[idx][order], dtype=np.float64))
        local_starts = bounds[:-1].astype(np.int64, copy=False)
        local_ends = (bounds[1:] - 1).astype(np.int64, copy=False)
        starts.append(local_starts + row)
        ends.append(local_ends + row)
        row += int(idx.size)
        groups += int(local_starts.size)
        row0.append(row)
        group0.append(groups)
    p = x.shape[1]
    if not pieces_x:
        empty = np.zeros((0, p))
        return _OrdinaryLayout(
            empty,
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
    return _OrdinaryLayout(
        np.concatenate(pieces_x),
        np.concatenate(pieces_e),
        np.concatenate(pieces_w),
        np.concatenate(pieces_o),
        np.concatenate(starts),
        np.concatenate(ends),
        np.asarray(row0, dtype=np.int64),
        np.asarray(group0, dtype=np.int64),
    )


def _score_layout(beta, layout, efron):
    if layout.row0.shape[0] <= 1:
        p = beta.shape[0]
        return 0.0, np.zeros(p), np.zeros((p, p))
    eta = layout.x @ beta + layout.offset
    ex = np.exp(eta) * layout.weight
    return _score_blocks(
        layout.x,
        layout.event,
        layout.weight,
        np.ascontiguousarray(eta),
        np.ascontiguousarray(ex),
        layout.starts,
        layout.ends,
        layout.row0,
        layout.group0,
        efron,
    )


@njit(cache=True)
def _score_blocks(x, event, weight, eta, ex, starts, ends, row0, group0, efron):
    """Partial likelihood of every stratum. Group bounds are positions in ``x``."""
    p = x.shape[1]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    n_st = row0.shape[0] - 1
    for s in range(n_st):
        r0 = row0[s]
        r1 = row0[s + 1]
        g0 = group0[s]
        g1 = group0[s + 1]
        add_ll, add_grad, add_hess = _ordinary_score(
            x[r0:r1],
            event[r0:r1],
            weight[r0:r1],
            eta[r0:r1],
            ex[r0:r1],
            starts[g0:g1] - r0,
            ends[g0:g1] - r0,
            efron,
        )
        ll += add_ll
        grad += add_grad
        hess += add_hess
    return ll, grad, hess


def _partial(beta, x, time, event, weights, offset, entry, strata, ties):
    """Partial likelihood by one backward pass per stratum.

    At each time the risk set is everyone already visited (later exit) who has
    not yet been removed (entry at or after this time). Sums are updated once
    per row.
    """
    eta = x @ beta + offset
    p = beta.shape[0]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    ordinary = bool(np.all(entry == 0))
    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        if idx.size == 0:
            continue
        ll, grad, hess = _stratum_partial(
            ll, grad, hess, x[idx], time[idx], event[idx], weights[idx], eta[idx], entry[idx], ties, ordinary
        )
    return ll, grad, hess


def _stratum_partial(ll, grad, hess, x, time, event, weights, eta, entry, ties, ordinary):
    piece = _RiskScan(x, time, event, weights, eta, entry, ties, ordinary)
    piece.partial(ll, grad, hess)
    return piece.ll, piece.grad, piece.hess


def _time_blocks(time: np.ndarray):
    """Row order and half-open bounds of tied times, ascending."""
    order = np.argsort(time, kind="mergesort")
    ordered = time[order]
    if ordered.size == 0:
        return order, np.zeros(1, dtype=int)
    change = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    return order, np.append(change, ordered.size)


@njit(cache=True)
def _ordinary_score(x, event, weight, eta, ex, starts, ends, efron):
    """Partial likelihood for ordinary survival, walking exit times backward.

    ``x`` through ``ex`` are already ordered by ascending time. ``starts`` and
    ``ends`` are inclusive bounds of each tied time. The Efron sum does not
    depend on the order of deaths inside a tie.
    """
    p = x.shape[1]
    s0 = 0.0
    s1 = np.zeros(p)
    s2 = np.zeros((p, p))
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    sd1 = np.zeros(p)
    sd2 = np.zeros((p, p))
    n_groups = starts.shape[0]
    for g in range(n_groups - 1, -1, -1):
        a = int(starts[g])
        b = int(ends[g])
        for i in range(a, b + 1):
            e = ex[i]
            s0 += e
            xi = x[i]
            for j in range(p):
                s1[j] += xi[j] * e
                for k in range(p):
                    s2[j, k] += xi[j] * xi[k] * e
        deaths = 0
        for i in range(a, b + 1):
            if event[i] > 0.0:
                deaths += 1
        if deaths == 0 or s0 <= 0.0:
            continue
        for j in range(p):
            sd1[j] = 0.0
            for k in range(p):
                sd2[j, k] = 0.0
        sd0 = 0.0
        wsum = 0.0
        for i in range(a, b + 1):
            if event[i] <= 0.0:
                continue
            w = weight[i]
            e = ex[i]
            xi = x[i]
            ll += w * eta[i]
            sd0 += e
            wsum += w
            for j in range(p):
                grad[j] += w * xi[j]
                sd1[j] += e * xi[j]
                for k in range(p):
                    sd2[j, k] += e * xi[j] * xi[k]
        if (not efron) or deaths == 1:
            wd = wsum
            ll -= wd * np.log(s0)
            for j in range(p):
                mean_j = s1[j] / s0
                grad[j] -= wd * mean_j
                for k in range(p):
                    hess[j, k] -= wd * (s2[j, k] / s0 - mean_j * (s1[k] / s0))
            continue
        width = float(deaths)
        wtave = wsum / width
        for step in range(deaths):
            r = step / width
            a0 = s0 - r * sd0
            ll -= wtave * np.log(a0)
            for j in range(p):
                mean_j = (s1[j] - r * sd1[j]) / a0
                grad[j] -= wtave * mean_j
                for k in range(p):
                    mean_k = (s1[k] - r * sd1[k]) / a0
                    hess[j, k] -= wtave * ((s2[j, k] - r * sd2[j, k]) / a0 - mean_j * mean_k)
    return ll, grad, hess


class _RiskScan:
    """Backward risk-set sums for one stratum. Rows are local to the stratum."""

    def __init__(self, x, time, event, weights, eta, entry, ties, ordinary):
        self.x = x
        self.time = time
        self.event = event
        self.weights = weights
        self.eta = eta
        self.entry = entry
        self.ties = ties
        self.ordinary = ordinary
        p = x.shape[1]
        self.s0 = 0.0
        self.s1 = np.zeros(p)
        self.s2 = np.zeros((p, p))
        self.ll = 0.0
        self.grad = np.zeros(p)
        self.hess = np.zeros((p, p))

    def partial(self, ll, grad, hess):
        self.ll, self.grad, self.hess = ll, grad, hess
        if self.ordinary:
            order, bounds = _time_blocks(self.time)
            if bounds.size <= 1:
                return
            ex = np.exp(self.eta) * self.weights
            add_ll, add_grad, add_hess = _ordinary_score(
                np.ascontiguousarray(self.x[order]),
                np.ascontiguousarray(self.event[order], dtype=np.float64),
                np.ascontiguousarray(self.weights[order]),
                np.ascontiguousarray(self.eta[order]),
                np.ascontiguousarray(ex[order]),
                np.ascontiguousarray(bounds[:-1], dtype=np.int64),
                np.ascontiguousarray(bounds[1:] - 1, dtype=np.int64),
                self.ties == "efron",
            )
            self.ll += float(add_ll)
            self.grad += add_grad
            self.hess += add_hess
            return
        self._walk_counting(score=True)

    def baseline(self) -> tuple[list[float], list[float]]:
        times: list[float] = []
        hazards: list[float] = []
        if self.ordinary:
            self._walk_exits(score=False, times=times, hazards=hazards)
        else:
            self._walk_counting(score=False, times=times, hazards=hazards)
        times.reverse()
        hazards.reverse()
        return times, hazards

    def _baseline_from_groups(self, rows, ex, death, group_id, ends, n_death, alive, times, hazards):
        n_groups = len(ends)
        wd = np.bincount(group_id, weights=self.weights[rows] * death, minlength=n_groups)
        risk0 = np.cumsum(ex)
        haz = np.zeros(n_groups)
        simple = alive & ((self.ties == "breslow") | (n_death == 1))
        haz[simple] = wd[simple] / risk0[ends[simple]]
        tied = alive & (self.ties == "efron") & (n_death > 1)
        if np.any(tied):
            _sd0, _sd1, _sd2, _wsum, wtave, sums = self._efron_pack(rows, ex, death, group_id, ends, tied, risk0)
            haz[tied] = wtave * sums[1]
        clock = self.time[rows[ends]]
        alive_at = np.flatnonzero(alive)
        times.extend(float(value) for value in clock[alive_at])
        hazards.extend(float(value) for value in haz[alive_at])

    def _score_simple_groups(self, rows, xx, death, group_id, ends, simple, risk0, s1, s2):
        if not np.any(simple):
            return
        n_groups = len(ends)
        p = xx.shape[1]
        case_w = self.weights[rows] * death
        ll_event = np.bincount(group_id, weights=case_w * self.eta[rows], minlength=n_groups)
        wd = np.bincount(group_id, weights=case_w, minlength=n_groups)
        grad_event = np.column_stack(
            [np.bincount(group_id, weights=case_w * xx[:, j], minlength=n_groups) for j in range(p)]
        )
        use = np.flatnonzero(simple)
        s0 = risk0[ends[use]]
        mean = s1[ends[use]] / s0[:, None]
        self.ll += float(np.sum(ll_event[use] - wd[use] * np.log(s0)))
        self.grad += np.sum(grad_event[use] - wd[use, None] * mean, axis=0)
        moment = s2[ends[use]] / s0[:, None, None] - mean[:, :, None] * mean[:, None, :]
        self.hess -= np.sum(wd[use, None, None] * moment, axis=0)

    def _add(self, rows, sign: float):
        ex = np.exp(self.eta[rows]) * self.weights[rows]
        xx = self.x[rows]
        self.s0 += sign * float(ex.sum())
        self.s1 += sign * (xx.T @ ex)
        self.s2 += sign * ((xx * ex[:, None]).T @ xx)
        return ex

    def _walk_exits(self, *, score: bool, times=None, hazards=None):
        order, bounds = _time_blocks(self.time)
        n_groups = len(bounds) - 1
        if n_groups <= 0:
            return
        # Latest time first, so a cumulative sum is the risk set at that time.
        groups = np.arange(n_groups - 1, -1, -1)
        sizes = bounds[1:] - bounds[:-1]
        sizes = sizes[groups]
        rows = np.concatenate([order[bounds[g] : bounds[g + 1]] for g in groups])
        ends = np.cumsum(sizes) - 1
        ex = np.exp(self.eta[rows]) * self.weights[rows]
        xx = self.x[rows]
        risk0 = np.cumsum(ex)
        death = np.asarray(self.event[rows], dtype=bool)
        group_id = np.repeat(np.arange(n_groups), sizes)
        n_death = np.bincount(group_id, weights=death.astype(float), minlength=n_groups)
        alive = (n_death > 0) & (risk0[ends] > 0)
        if not np.any(alive):
            return
        if not score:
            self._baseline_from_groups(rows, ex, death, group_id, ends, n_death, alive, times, hazards)
            return
        s1 = np.cumsum(xx * ex[:, None], axis=0)
        s2 = np.cumsum(xx[:, :, None] * (xx * ex[:, None])[:, None, :], axis=0)
        simple = alive & ((self.ties == "breslow") | (n_death == 1))
        tied = alive & (self.ties == "efron") & (n_death > 1)
        self._score_simple_groups(rows, xx, death, group_id, ends, simple, risk0, s1, s2)
        if np.any(tied):
            self._score_efron_groups(rows, xx, ex, death, group_id, ends, tied, risk0, s1, s2)

    def _walk_counting(self, *, score: bool, times=None, hazards=None):
        # Add an exit when the clock reaches it, and drop a row whose entry
        # equals the clock before scoring. entry < t is then the risk set.
        exit_order, exit_bounds = _time_blocks(self.time)
        entry_order, entry_bounds = _time_blocks(self.entry)
        exit_at = len(exit_bounds) - 2
        entry_at = len(entry_bounds) - 2
        clocks = np.unique(np.concatenate([self.time, self.entry]))[::-1]
        for tm in clocks:
            added = None
            while exit_at >= 0 and self.time[exit_order[exit_bounds[exit_at]]] == tm:
                rows = exit_order[exit_bounds[exit_at] : exit_bounds[exit_at + 1]]
                added = rows
                self._add(rows, 1.0)
                exit_at -= 1
            while entry_at >= 0 and self.entry[entry_order[entry_bounds[entry_at]]] == tm:
                rows = entry_order[entry_bounds[entry_at] : entry_bounds[entry_at + 1]]
                self._add(rows, -1.0)
                entry_at -= 1
            if added is None:
                continue
            death = self.event[added] & (self.entry[added] < tm)
            if not np.any(death):
                continue
            ex = np.exp(self.eta[added]) * self.weights[added]
            self._at_deaths(added, ex, death, score=score, times=times, hazards=hazards)

    def _efron_pack(self, rows, ex, death, group_id, ends, tied, risk0):
        """Per tied group: risk sums of the deaths, and the Efron step totals.

        The returned step totals are ``sum log(a0)``, ``sum 1/a0``,
        ``sum r/a0``, ``sum 1/a0^2``, ``sum r/a0^2``, ``sum r^2/a0^2``,
        with ``r = k / d``. ``wtave * sum(1/a0)`` is the Efron hazard.
        """
        tied_ids = np.flatnonzero(tied)
        compact = np.full(len(ends), -1, dtype=int)
        compact[tied_ids] = np.arange(len(tied_ids))
        death_rows = np.flatnonzero(tied[group_id] & death)
        g = compact[group_id[death_rows]]
        width = len(tied_ids)
        ex_d = ex[death_rows]
        w_d = self.weights[rows][death_rows]
        x_d = self.x[rows][death_rows]
        p = x_d.shape[1]
        d = np.bincount(g, minlength=width).astype(float)
        sd0 = np.bincount(g, weights=ex_d, minlength=width)
        wsum = np.bincount(g, weights=w_d, minlength=width)
        weighted_x = ex_d[:, None] * x_d
        sd1 = np.column_stack(
            [np.bincount(g, weights=weighted_x[:, j], minlength=width) for j in range(p)]
        )
        sd2 = np.empty((width, p, p))
        for j in range(p):
            sd2[:, j, :] = np.column_stack(
                [np.bincount(g, weights=weighted_x[:, j] * x_d[:, k], minlength=width) for k in range(p)]
            )
        s0 = risk0[ends[tied_ids]]
        wtave = wsum / d
        sums = _efron_step_sums(s0, sd0, d)
        return sd0, sd1, sd2, wsum, wtave, sums

    def _score_efron_groups(self, rows, xx, ex, death, group_id, ends, tied, risk0, s1, s2):
        tied_ids = np.flatnonzero(tied)
        sd0, sd1, sd2, wsum, wtave, sums = self._efron_pack(rows, ex, death, group_id, ends, tied, risk0)
        sum_log, sum_inv, sum_r_over_a, sum_inv2, sum_r_inv2, sum_r2_inv2 = sums
        death_rows = np.flatnonzero(tied[group_id] & death)
        compact = np.full(len(ends), -1, dtype=int)
        compact[tied_ids] = np.arange(len(tied_ids))
        g = compact[group_id[death_rows]]
        width = len(tied_ids)
        w_d = self.weights[rows][death_rows]
        x_d = xx[death_rows]
        eta_d = self.eta[rows][death_rows]
        p = x_d.shape[1]
        self.ll += float(np.sum(w_d * eta_d) - np.sum(wtave * sum_log))
        weighted_design = w_d[:, None] * x_d
        grad_event = np.column_stack(
            [np.bincount(g, weights=weighted_design[:, j], minlength=width) for j in range(p)]
        )
        risk_s1 = s1[ends[tied_ids]]
        mean_sum = risk_s1 * sum_inv[:, None] - sd1 * sum_r_over_a[:, None]
        self.grad += grad_event.sum(axis=0) - (wtave[:, None] * mean_sum).sum(axis=0)
        risk_s2 = s2[ends[tied_ids]]
        sum_a2 = risk_s2 * sum_inv[:, None, None] - sd2 * sum_r_over_a[:, None, None]
        s1_outer = risk_s1[:, :, None] * risk_s1[:, None, :]
        sd1_outer = sd1[:, :, None] * sd1[:, None, :]
        cross = risk_s1[:, :, None] * sd1[:, None, :] + sd1[:, :, None] * risk_s1[:, None, :]
        sum_outer = (
            s1_outer * sum_inv2[:, None, None]
            - cross * sum_r_inv2[:, None, None]
            + sd1_outer * sum_r2_inv2[:, None, None]
        )
        self.hess -= (wtave[:, None, None] * (sum_a2 - sum_outer)).sum(axis=0)
        del wsum

    def _at_deaths(self, rows, ex, death, *, score, times, hazards):
        if self.s0 <= 0:
            return
        if not score:
            hazards.append(_hazard_from_sums(ex, self.weights[rows], death, self.s0, self.ties))
            times.append(float(self.time[rows[0]]))
            return
        d_idx = np.flatnonzero(death)
        eta_d = self.eta[rows][d_idx]
        w_d = self.weights[rows][d_idx]
        x_d = self.x[rows][d_idx]
        ex_d = ex[d_idx]
        self.ll += float(np.sum(w_d * eta_d))
        self.grad += x_d.T @ w_d
        if self.ties == "breslow" or len(d_idx) == 1:
            wd = float(w_d.sum())
            self.ll -= wd * np.log(self.s0)
            mean = self.s1 / self.s0
            self.grad -= wd * mean
            self.hess -= wd * (self.s2 / self.s0 - np.outer(mean, mean))
            return
        d = len(d_idx)
        sd0 = float(ex_d.sum())
        sd1 = x_d.T @ ex_d
        sd2 = (x_d * ex_d[:, None]).T @ x_d
        wtave = float(w_d.sum()) / d
        frac = np.arange(d) / d
        a0 = self.s0 - frac * sd0
        mean = (self.s1 - frac[:, None] * sd1) / a0[:, None]
        self.ll -= wtave * float(np.log(a0).sum())
        self.grad -= wtave * mean.sum(axis=0)
        a2 = self.s2 - frac[:, None, None] * sd2
        moment = a2 / a0[:, None, None] - mean[:, :, None] * mean[:, None, :]
        self.hess -= wtave * moment.sum(axis=0)


def _efron_step_sums(s0, sd0, d):
    """Sums over the Efron fractions ``k / d`` for each tied group."""
    width = len(d)
    max_d = int(d.max()) if width else 0
    empty = np.zeros(width)
    if max_d == 0:
        return empty, empty, empty, empty, empty, empty
    steps = np.arange(max_d, dtype=float)
    mask = steps[None, :] < d[:, None]
    ratio = np.zeros((width, max_d))
    np.divide(steps[None, :], d[:, None], out=ratio, where=d[:, None] > 0)
    level = s0[:, None] - ratio * sd0[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(mask, 1.0 / level, 0.0)
        log_level = np.where(mask, np.log(level), 0.0)
    inv2 = inv * inv
    return (
        log_level.sum(axis=1),
        inv.sum(axis=1),
        (ratio * inv).sum(axis=1),
        inv2.sum(axis=1),
        (ratio * inv2).sum(axis=1),
        ((ratio * ratio) * inv2).sum(axis=1),
    )


def _hazard_from_sums(ex, weights, death, s0, ties) -> float:
    d = int(np.sum(death))
    w_d = weights[death]
    if ties == "efron" and d > 1:
        sd0 = float(ex[death].sum())
        wtave = float(w_d.sum()) / d
        frac = np.arange(d) / d
        return float(np.sum(wtave / (s0 - frac * sd0)))
    return float(w_d.sum()) / s0


def _baseline(x, beta, time, event, weights, offset, entry, strata, ties):
    eta = x @ beta + offset
    times = []
    hazards = []
    labels = []
    ordinary = bool(np.all(entry == 0))
    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        if idx.size == 0:
            continue
        scan = _RiskScan(x[idx], time[idx], event[idx], weights[idx], eta[idx], entry[idx], ties, ordinary)
        stratum_times, stratum_hazards = scan.baseline()
        times.extend(stratum_times)
        hazards.extend(stratum_hazards)
        labels.extend([stratum] * len(stratum_times))
    return np.asarray(times), np.asarray(hazards), np.asarray(labels)


@njit(cache=True)
def _concordance_counts(time, entry, event, lp, weights, counting):
    """Harrell pair counts and per-row influence, in index order.

    With ``counting``, row ``j`` is compared with a death at ``t`` only when it
    is at risk then (``entry[j] < t``).
    """
    n = time.shape[0]
    concordant = 0.0
    discordant = 0.0
    tied = 0.0
    inf_c = np.zeros(n)
    inf_d = np.zeros(n)
    inf_t = np.zeros(n)
    for i in range(n):
        if event[i] <= 0.0:
            continue
        w_i = weights[i]
        wc = 0.0
        wd = 0.0
        wt = 0.0
        sc = 0.0
        sd = 0.0
        st = 0.0
        any_comp = False
        for j in range(n):
            same = time[j] == time[i]
            if not ((time[j] > time[i]) or (same and event[j] <= 0.0)):
                continue
            if counting and not entry[j] < time[i]:
                continue
            any_comp = True
            diff = lp[i] - lp[j]
            w = w_i * weights[j]
            if diff > 0.0:
                wc += w
                sc += weights[j]
                inf_c[j] += w_i
            elif diff < 0.0:
                wd += w
                sd += weights[j]
                inf_d[j] += w_i
            else:
                wt += w
                st += weights[j]
                inf_t[j] += w_i
        if any_comp:
            concordant += wc
            discordant += wd
            tied += wt
            inf_c[i] += sc
            inf_d[i] += sd
            inf_t[i] += st
    return concordant, discordant, tied, inf_c, inf_d, inf_t


# ---------------------------------------------------------------- residuals
#
# All residuals below are built from one table of "pseudo-events" per stratum.
# A death time with d tied deaths is d pseudo-events under the Efron (1977)
# approximation: the k-th one (k = 0..d-1) removes the fraction k/d of the
# tied deaths' risk score from the risk set and carries the mean dying weight.
# Under Breslow ties a death time is one pseudo-event with the full risk set
# and the total dying weight. Therneau & Grambsch (2000), sec. 3.3 and 4.2-4.6.


class _PseudoEvents:
    """Risk-set summaries of one stratum at its pseudo-events.

    ``death_times`` are the distinct death times (ascending). Per pseudo-event
    ``group`` indexes the death time, ``frac`` is k/d, ``hazard`` is the
    baseline hazard increment dLambda_k, ``mean`` the risk-weighted covariate
    mean xbar_k, and ``weight`` the dying weight that event carries.
    """

    __slots__ = ("death_times", "group", "frac", "hazard", "mean", "weight", "variance", "n_deaths", "valid")


def _suffix_sums(key: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Sum of ``values`` over rows with ``key >= at`` for each element of ``at``."""
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    shape = (1,) + values.shape[1:]
    tail = np.concatenate([np.cumsum(values[order][::-1], axis=0)[::-1], np.zeros(shape)], axis=0)
    return tail[np.searchsorted(sorted_key, at, side="left")]


def _pseudo_events(x, time, entry, event, weights, eta, efron: bool, second_moment: bool = False) -> _PseudoEvents:
    """Pseudo-event table of one stratum.

    The risk set at t is ``entry < t <= time``; its sums are the sums over
    ``time >= t`` minus those over ``entry >= t`` (entry < time on every row).
    ``eta`` should already be shifted for numerical safety; every residual
    uses ``exp(eta) * dLambda``, which does not change under a common shift.
    """
    p = x.shape[1]
    risk = weights * np.exp(eta)
    dead = event > 0
    out = _PseudoEvents()
    death_times = np.unique(time[dead])
    out.death_times = death_times
    k_times = death_times.size
    if k_times == 0:
        out.group = np.zeros(0, dtype=int)
        out.frac = np.zeros(0)
        out.hazard = np.zeros(0)
        out.mean = np.zeros((0, p))
        out.weight = np.zeros(0)
        out.variance = np.zeros((0, p, p)) if second_moment else None
        out.n_deaths = np.zeros(0)
        out.valid = np.zeros(0, dtype=bool)
        return out
    row_vals = [risk[:, None], risk[:, None] * x]
    if second_moment:
        row_vals.append((risk[:, None] * x)[:, :, None] * x[:, None, :])
    sums = []
    for v in row_vals:
        sums.append(_suffix_sums(time, v, death_times) - _suffix_sums(entry, v, death_times))
    s0 = sums[0][:, 0]
    s1 = sums[1]
    # sums over the rows dying at each death time
    which = np.searchsorted(death_times, time[dead])
    n_deaths = np.bincount(which, minlength=k_times).astype(float)
    d0 = np.bincount(which, weights=risk[dead], minlength=k_times)
    d1 = np.column_stack([np.bincount(which, weights=risk[dead] * x[dead, j], minlength=k_times) for j in range(p)]) if p else np.zeros((k_times, 0))
    w_dead = np.bincount(which, weights=weights[dead], minlength=k_times)
    if efron:
        n_pseudo = n_deaths.astype(int)
    else:
        n_pseudo = np.ones(k_times, dtype=int)
    group = np.repeat(np.arange(k_times), n_pseudo)
    starts = np.cumsum(n_pseudo) - n_pseudo
    step = np.arange(group.size) - starts[group]
    frac = step / n_pseudo[group]
    weight = w_dead[group] / n_pseudo[group]
    denom = s0[group] - frac * d0[group]
    valid_time = s0 > 0
    valid = valid_time[group] & (denom > 0)
    safe = np.where(valid, denom, 1.0)
    hazard = np.where(valid, weight / safe, 0.0)
    mean = (s1[group] - frac[:, None] * d1[group]) / safe[:, None]
    mean[~valid] = 0.0
    out.group = group
    out.frac = frac
    out.hazard = hazard
    out.mean = mean
    out.weight = weight
    out.n_deaths = n_deaths
    out.valid = valid
    out.variance = None
    if second_moment:
        dd2 = np.zeros((k_times, p, p))
        xd = x[dead]
        rd = risk[dead]
        np.add.at(dd2, which, (rd[:, None] * xd)[:, :, None] * xd[:, None, :])
        second = (sums[2][group] - frac[:, None, None] * dd2[group]) / safe[:, None, None]
        variance = second - mean[:, :, None] * mean[:, None, :]
        variance[~valid] = 0.0
        out.variance = variance
    return out


def _stratum_inputs(fit: CoxFit):
    """Centered design and shifted linear predictor, and the stratum row sets."""
    x = np.asarray(fit.x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    # Residuals are invariant to centering x and to shifting eta; both help precision.
    center = x.mean(axis=0) if x.shape[0] else np.zeros(x.shape[1])
    xc = x - center
    eta = xc @ np.asarray(fit.coefficients, dtype=float) + np.asarray(fit.offset, dtype=float)
    groups = [np.flatnonzero(fit.strata == s) for s in np.unique(fit.strata)]
    return xc, eta, groups


def _exposure(fit: CoxFit, efron: bool):
    """Per-row pieces shared by martingale and score residuals.

    Returns, per row, ``cum_hazard`` = int Y_i dLambda (before multiplying by
    the row's risk score), ``cum_mean`` = int Y_i xbar dLambda, the dN-weighted
    mean ``dn_mean`` (the mean over the pseudo-events of the row's own death
    time; zero for censored rows), and the risk score exp(eta) with the
    stratum shift. For a dying row the k-th pseudo-event of its own death time
    is weighted by (1 - k/d) (Therneau & Grambsch 2000, sec. 3.3).
    """
    xc, eta, groups = _stratum_inputs(fit)
    n, p = xc.shape
    cum_hazard = np.zeros(n)
    cum_mean = np.zeros((n, p))
    dn_mean = np.zeros((n, p))
    score = np.zeros(n)
    time = np.asarray(fit.time, dtype=float)
    entry = np.asarray(fit.entry, dtype=float)
    event = np.asarray(fit.event, dtype=float)
    weights = np.asarray(fit.weights, dtype=float)
    for idx in groups:
        e = eta[idx]
        e = e - e.max() if e.size else e
        score[idx] = np.exp(e)
        table = _pseudo_events(xc[idx], time[idx], entry[idx], event[idx], weights[idx], e, efron)
        k_times = table.death_times.size
        if k_times == 0:
            continue
        g = table.group
        haz = np.bincount(g, weights=table.hazard, minlength=k_times)
        hmean = np.column_stack([np.bincount(g, weights=table.hazard * table.mean[:, j], minlength=k_times) for j in range(p)]) if p else np.zeros((k_times, 0))
        part = 1.0 - table.frac
        haz_die = np.bincount(g, weights=part * table.hazard, minlength=k_times)
        hmean_die = np.column_stack([np.bincount(g, weights=part * table.hazard * table.mean[:, j], minlength=k_times) for j in range(p)]) if p else np.zeros((k_times, 0))
        count = np.bincount(g, minlength=k_times).astype(float)
        avg_mean = np.column_stack([np.bincount(g, weights=table.mean[:, j], minlength=k_times) for j in range(p)]) / count[:, None] if p else np.zeros((k_times, 0))
        cum_h = np.concatenate([[0.0], np.cumsum(haz)])
        cum_m = np.concatenate([np.zeros((1, p)), np.cumsum(hmean, axis=0)], axis=0)
        hi = np.searchsorted(table.death_times, time[idx], side="right")
        lo = np.searchsorted(table.death_times, entry[idx], side="right")
        lo = np.minimum(lo, hi)
        ch = cum_h[hi] - cum_h[lo]
        cm = cum_m[hi] - cum_m[lo]
        died = event[idx] > 0
        own = hi[died] - 1
        ch[died] += haz_die[own] - haz[own]
        cm[died] += hmean_die[own] - hmean[own]
        dm = np.zeros((idx.size, p))
        dm[died] = avg_mean[own]
        cum_hazard[idx] = ch
        cum_mean[idx] = cm
        dn_mean[idx] = dm
    return xc, cum_hazard, cum_mean, dn_mean, score


def _martingale_residuals(fit: CoxFit) -> np.ndarray:
    """Martingale residuals M_i = N_i - int Y_i r_i dLambda_0, one per row.

    Therneau, Grambsch & Fleming (1990), eq. (2.1); the Breslow-type baseline
    hazard with Efron's tie correction as in Therneau & Grambsch (2000),
    sec. 3.3 and 4.2. Case weights enter the hazard but do not multiply the
    residual. Output is compared with R's ``residuals.coxph``.
    """
    _xc, cum_hazard, _cm, _dn, score = _exposure(fit, fit.ties == "efron")
    return np.asarray(fit.event, dtype=float) - score * cum_hazard


def _score_residuals(fit: CoxFit) -> np.ndarray:
    """Score residuals L_ij = int (x_ij - xbar_j(t)) dM_i(t), an n x p array.

    Therneau, Grambsch & Fleming (1990), sec. 3; Lin & Wei (1989); Therneau &
    Grambsch (2000), sec. 4.4. Unweighted per row; the column sums of
    ``weights * L`` are the partial-likelihood score.
    """
    xc, cum_hazard, cum_mean, dn_mean, score = _exposure(fit, fit.ties == "efron")
    event = (np.asarray(fit.event, dtype=float) > 0).astype(float)
    death_part = event[:, None] * (xc - dn_mean)
    hazard_part = score[:, None] * (xc * cum_hazard[:, None] - cum_mean)
    return death_part - hazard_part


def _schoenfeld(fit: CoxFit) -> np.ndarray:
    """Schoenfeld residuals x_i - xbar(t_i), one row per event (d x p).

    Schoenfeld (1982); Therneau & Grambsch (2000), sec. 4.6. Under Efron ties
    the subtracted mean is the average of the d Efron-adjusted means. Rows are
    ordered by stratum (``np.unique`` order), then by time (stable).
    """
    xc, eta, groups = _stratum_inputs(fit)
    p = xc.shape[1]
    time = np.asarray(fit.time, dtype=float)
    entry = np.asarray(fit.entry, dtype=float)
    event = np.asarray(fit.event, dtype=float)
    weights = np.asarray(fit.weights, dtype=float)
    efron = fit.ties == "efron"
    pieces = []
    for idx in groups:
        e = eta[idx]
        e = e - e.max() if e.size else e
        table = _pseudo_events(xc[idx], time[idx], entry[idx], event[idx], weights[idx], e, efron)
        k_times = table.death_times.size
        if k_times == 0:
            continue
        g = table.group
        count = np.bincount(g, minlength=k_times).astype(float)
        avg_mean = np.column_stack([np.bincount(g, weights=table.mean[:, j], minlength=k_times) for j in range(p)]) / count[:, None] if p else np.zeros((k_times, 0))
        local = np.flatnonzero(event[idx] > 0)
        local = local[np.argsort(time[idx][local], kind="mergesort")]
        own = np.searchsorted(table.death_times, time[idx][local])
        pieces.append(xc[idx][local] - avg_mean[own])
    if not pieces:
        return np.zeros((0, p))
    return np.concatenate(pieces, axis=0)



def _schoenfeld_times(fit: CoxFit) -> np.ndarray:
    times = []
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        t = fit.time[idx]
        e = fit.event[idx] > 0
        # Same order as _schoenfeld: by time within the stratum, ties stable.
        times.append(np.sort(t[e], kind="mergesort"))
    return np.concatenate(times) if times else np.zeros(0)


def _time_transform(fit: CoxFit, times: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return times.astype(float)
    if name == "rank":
        return stats.rankdata(times).astype(float)
    if name == "log":
        return np.log(times)
    if name == "kaplan_meier":
        # left-continuous pooled KM at the death times: 1 - S(t-)
        order = np.argsort(fit.time, kind="mergesort")
        t = fit.time[order]
        e = fit.event[order] > 0
        n = len(t)
        survival = 1.0
        km_before = {}
        i = 0
        while i < n:
            tm = t[i]
            j = i
            while j < n and t[j] == tm:
                j += 1
            risk = n - i
            deaths = int(e[i:j].sum())
            km_before[tm] = 1 - survival
            if risk:
                survival *= 1 - deaths / risk
            i = j
        return np.array([km_before[tm] for tm in times])
    raise ValueError("time_transform must be kaplan_meier, rank, identity, or log")


def _centered_time_weights(fit: CoxFit, name: str) -> np.ndarray:
    """Per-row g(t), centered by the mean over events, as in ``cox.zph``."""
    times = fit.time.astype(float)
    if name == "identity":
        g = times.copy()
    elif name == "rank":
        g = stats.rankdata(times).astype(float)
    elif name == "log":
        g = np.log(times)
    elif name == "kaplan_meier":
        g = _kaplan_meier_transform(fit.time, fit.event > 0, fit.entry)
    else:
        raise ValueError("time_transform must be kaplan_meier, rank, identity, or log")
    event = fit.event > 0
    return g - float(g[event].mean())


def _kaplan_meier_transform(time: np.ndarray, event: np.ndarray, entry: np.ndarray | None = None) -> np.ndarray:
    """``1 - S(t-)`` from the pooled Kaplan–Meier, matching ``cox.zph(transform="km")``.

    Counting-process rows are at risk on ``(entry, time]``.
    """
    if entry is not None and not bool(np.all(entry == 0)):
        utimes = np.unique(time)
        sorted_time = np.sort(time)
        sorted_entry = np.sort(entry)
        before_map: dict[float, float] = {}
        survival = 1.0
        for tm in utimes:
            before_map[float(tm)] = 1.0 - survival
            risk = (len(time) - np.searchsorted(sorted_time, tm, side="left")) - (
                len(entry) - np.searchsorted(sorted_entry, tm, side="left")
            )
            deaths = int(np.sum((time == tm) & event))
            if risk > 0:
                survival *= 1.0 - deaths / risk
        return np.array([before_map[float(tm)] for tm in time])
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    e = event[order]
    survival = 1.0
    before: dict[float, float] = {}
    i = 0
    n = len(t)
    while i < n:
        tm = t[i]
        j = i
        while j < n and t[j] == tm:
            j += 1
        before[float(tm)] = 1.0 - survival
        risk = n - i
        deaths = int(e[i:j].sum())
        if risk:
            survival *= 1.0 - deaths / risk
        i = j
    return np.array([before[float(tm)] for tm in time])


def _zph_score(fit: CoxFit, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score and information of the augmented model with covariates x * g(t).

    Grambsch & Therneau (1994), sec. 3; Therneau & Grambsch (2000), sec. 6.2.
    At (beta_hat, 0) the augmented covariate of every row at risk at t is
    x * g(t), so its risk-weighted mean is g(t) xbar(t) and its covariance
    g(t) V(t), g(t)^2 V(t). Efron ties are applied unless ties="breslow".
    """
    xc, eta, groups = _stratum_inputs(fit)
    p = xc.shape[1]
    time = np.asarray(fit.time, dtype=float)
    entry = np.asarray(fit.entry, dtype=float)
    event = np.asarray(fit.event, dtype=float)
    weights = np.asarray(fit.weights, dtype=float)
    g = np.asarray(g, dtype=float)
    efron = fit.ties != "breslow"
    score = np.zeros(2 * p)
    info = np.zeros((2 * p, 2 * p))
    for idx in groups:
        e = eta[idx]
        e = e - e.max() if e.size else e
        table = _pseudo_events(xc[idx], time[idx], entry[idx], event[idx], weights[idx], e, efron, second_moment=True)
        k_times = table.death_times.size
        if k_times == 0:
            continue
        dead = np.flatnonzero(event[idx] > 0)
        own = np.searchsorted(table.death_times, time[idx][dead])
        g_time = np.zeros(k_times)
        g_time[own] = g[idx][dead]
        wd = weights[idx][dead]
        xd = xc[idx][dead]
        gd = g_time[own]
        wt = np.where(table.valid, table.weight, 0.0)
        gk = g_time[table.group]
        score[:p] += wd @ xd - wt @ table.mean
        score[p:] += (wd * gd) @ xd - (wt * gk) @ table.mean
        v = table.variance
        info[:p, :p] += np.tensordot(wt, v, axes=1)
        info[:p, p:] += np.tensordot(wt * gk, v, axes=1)
        info[p:, p:] += np.tensordot(wt * gk * gk, v, axes=1)
    info[p:, :p] = info[:p, p:].T
    return score, info


def _numeric_or_one(data, ref, idx):
    if ref is None:
        return np.ones(len(idx))
    return np.asarray(column_series(data, ref).gather(idx.tolist()).to_numpy(), dtype=float)


def _numeric_or_zero(data, ref, idx):
    if ref is None:
        return np.zeros(len(idx))
    return np.asarray(column_series(data, ref).gather(idx.tolist()).to_numpy(), dtype=float)


def _column_or_raise(data, name):
    if name is None or name not in data.columns:
        raise ValueError("predict on new data needs the original time column or times=")
    return np.asarray(data[name].to_numpy(), dtype=float)


def _optional_aligned(data, name, row_index, default):
    if name is None or name not in getattr(data, "columns", []):
        return np.array([default] * len(row_index))
    return np.asarray(column_series(data, name).gather(row_index.tolist()).to_list())
