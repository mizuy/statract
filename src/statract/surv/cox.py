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
    per row, which is the same recurrence as ``coxfit6``.
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
def _martingale_ordinary(time, event, weights, score, efron):
    """Charged hazard when everyone enters at 0. ``time`` is sorted ascending."""
    m = time.shape[0]
    charged = np.zeros(m)
    suffix = np.zeros(m + 1)
    for k in range(m - 1, -1, -1):
        suffix[k] = suffix[k + 1] + score[k] * weights[k]
    haz_before = 0.0
    i = 0
    while i < m:
        j = i + 1
        while j < m and time[j] == time[i]:
            j += 1
        d = 0
        wtsum = 0.0
        e_denom = 0.0
        for k in range(i, j):
            if event[k] > 0.0:
                d += 1
                wtsum += weights[k]
                e_denom += score[k] * weights[k]
        denom = suffix[i]
        if d == 0 or denom <= 0.0:
            for k in range(i, j):
                charged[k] = haz_before
            i = j
            continue
        if d == 1 or not efron:
            hazard = wtsum / denom
            e_hazard = hazard
        else:
            width = float(d)
            wtave = wtsum / width
            hazard = 0.0
            e_hazard = 0.0
            for step in range(d):
                temp = step / width
                base = denom - temp * e_denom
                hazard += wtave / base
                e_hazard += wtave * (1.0 - temp) / base
        for k in range(i, j):
            if event[k] > 0.0:
                charged[k] = haz_before + e_hazard
            else:
                charged[k] = haz_before + hazard
        haz_before += hazard
        i = j
    return charged


@njit(cache=True)
def _martingale_scan(time, entry, event, weights, score, death_times, efron):
    """Charged hazard for counting-process rows. Death times are sorted."""
    m = time.shape[0]
    charged = np.zeros(m)
    for t_i in range(death_times.shape[0]):
        tm = death_times[t_i]
        d = 0
        denom = 0.0
        wtsum = 0.0
        e_denom = 0.0
        for k in range(m):
            if not (entry[k] < tm and time[k] >= tm):
                continue
            denom += score[k] * weights[k]
            if time[k] == tm and event[k] > 0.0:
                d += 1
                wtsum += weights[k]
                e_denom += score[k] * weights[k]
        if d == 0 or denom <= 0.0:
            continue
        if d == 1 or not efron:
            hazard = wtsum / denom
            e_hazard = hazard
        else:
            width = float(d)
            wtave = wtsum / width
            hazard = 0.0
            e_hazard = 0.0
            for step in range(d):
                temp = step / width
                base = denom - temp * e_denom
                hazard += wtave / base
                e_hazard += wtave * (1.0 - temp) / base
        for k in range(m):
            if not (entry[k] < tm and time[k] >= tm):
                continue
            if time[k] == tm and event[k] > 0.0:
                charged[k] += e_hazard
            else:
                charged[k] += hazard
    return charged


@njit(cache=True)
def _schoenfeld_ordinary(xx, time, event, rs, efron, out):
    """Schoenfeld rows when everyone enters at 0. ``time`` is sorted ascending."""
    m, p = xx.shape
    suffix_rs = np.zeros(m + 1)
    suffix_a = np.zeros((m + 1, p))
    for k in range(m - 1, -1, -1):
        suffix_rs[k] = suffix_rs[k + 1] + rs[k]
        for col in range(p):
            suffix_a[k, col] = suffix_a[k + 1, col] + xx[k, col] * rs[k]
    written = 0
    i = 0
    while i < m:
        j = i + 1
        while j < m and time[j] == time[i]:
            j += 1
        d = 0
        efron_wt = 0.0
        a2 = np.zeros(p)
        for k in range(i, j):
            if event[k]:
                d += 1
                efron_wt += rs[k]
                for col in range(p):
                    a2[col] += xx[k, col] * rs[k]
        if d > 0:
            denom = suffix_rs[i]
            width = float(d)
            mean = np.zeros(p)
            for step in range(d):
                temp = (step / width) if efron else 0.0
                scale = width * (denom - temp * efron_wt)
                for col in range(p):
                    mean[col] += (suffix_a[i, col] - temp * a2[col]) / scale
            for k in range(i, j):
                if event[k]:
                    for col in range(p):
                        out[written, col] = xx[k, col] - mean[col]
                    written += 1
        i = j
    return written


@njit(cache=True)
def _schoenfeld_scan(xx, time, entry, event, rs, efron, out):
    """Schoenfeld rows for counting-process data. ``time`` is sorted ascending."""
    m, p = xx.shape
    written = 0
    i = 0
    while i < m:
        tm = time[i]
        j = i + 1
        while j < m and time[j] == tm:
            j += 1
        any_event = False
        for k in range(i, j):
            if event[k]:
                any_event = True
                break
        if not any_event:
            i = j
            continue
        d = 0
        denom = 0.0
        efron_wt = 0.0
        a = np.zeros(p)
        a2 = np.zeros(p)
        for k in range(m):
            if not (entry[k] < tm and time[k] >= tm):
                continue
            denom += rs[k]
            for col in range(p):
                a[col] += xx[k, col] * rs[k]
            if time[k] == tm and event[k]:
                d += 1
                efron_wt += rs[k]
                for col in range(p):
                    a2[col] += xx[k, col] * rs[k]
        if d > 0:
            width = float(d)
            mean = np.zeros(p)
            for step in range(d):
                temp = (step / width) if efron else 0.0
                scale = width * (denom - temp * efron_wt)
                for col in range(p):
                    mean[col] += (a[col] - temp * a2[col]) / scale
            for k in range(i, j):
                if event[k] and entry[k] < tm:
                    for col in range(p):
                        out[written, col] = xx[k, col] - mean[col]
                    written += 1
        i = j
    return written


@njit(cache=True)
def _score_ordered(x, event, weights, score, time, efron, resid):
    """Score residuals for one right-censored stratum, deaths before censored ties."""
    m, p = x.shape
    denom = 0.0
    cumhaz = 0.0
    a = np.zeros(p)
    xhaz = np.zeros(p)
    death_at = np.empty(m, dtype=np.int64)
    i = m - 1
    while i >= 0:
        newtime = time[i]
        deaths = 0
        e_denom = 0.0
        meanwt = 0.0
        a2 = np.zeros(p)
        while i >= 0 and time[i] == newtime:
            risk = score[i] * weights[i]
            denom += risk
            for col in range(p):
                resid[i, col] = score[i] * (x[i, col] * cumhaz - xhaz[col])
                a[col] += risk * x[i, col]
            if event[i] > 0.0:
                deaths += 1
                e_denom += risk
                meanwt += weights[i]
                for col in range(p):
                    a2[col] += risk * x[i, col]
                death_at[deaths - 1] = i
            i -= 1
        if deaths > 0 and denom > 0.0:
            if deaths < 2 or not efron:
                hazard = meanwt / denom
                cumhaz += hazard
                for s in range(deaths):
                    row = death_at[s]
                    for col in range(p):
                        xbar = a[col] / denom
                        resid[row, col] += x[row, col] - xbar
                for col in range(p):
                    xhaz[col] += (a[col] / denom) * hazard
            else:
                width = float(deaths)
                meanwt = meanwt / width
                for step in range(deaths):
                    downwt = step / width
                    temp = denom - downwt * e_denom
                    hazard = meanwt / temp
                    cumhaz += hazard
                    for col in range(p):
                        xbar = (a[col] - downwt * a2[col]) / temp
                        xhaz[col] += xbar * hazard
                    for s in range(deaths):
                        row = death_at[s]
                        for col in range(p):
                            xbar = (a[col] - downwt * a2[col]) / temp
                            temp2 = x[row, col] - xbar
                            resid[row, col] += temp2 / width
                            resid[row, col] += temp2 * score[row] * hazard * downwt
    for row in range(m):
        for col in range(p):
            resid[row, col] += score[row] * (xhaz[col] - x[row, col] * cumhaz)


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


@njit(cache=True)
def _zph_add(u, imat, a, cmat, denom, weight, timewt):
    p = a.shape[0]
    for col in range(p):
        mean = a[col] / denom
        u[col] -= weight * mean
        u[p + col] -= timewt * weight * mean
    for col in range(p):
        mean_col = a[col] / denom
        for k in range(p):
            temp = weight * (cmat[col, k] / denom - mean_col * (a[k] / denom))
            imat[col, k] += temp
            imat[col, p + k] += temp * timewt
            imat[p + col, p + k] += temp * timewt * timewt


@njit(cache=True)
def _zph_ordered(x, event, weights, eta, time, g, efron, u, imat):
    """One stratum of the ``cox.zph`` score, walking exit times backward."""
    m, p = x.shape
    denom = 0.0
    a = np.zeros(p)
    cmat = np.zeros((p, p))
    i = m - 1
    while i >= 0:
        dtime = time[i]
        ndead = 0
        deadwt = 0.0
        denom2 = 0.0
        a2 = np.zeros(p)
        cmat2 = np.zeros((p, p))
        timewt = g[i]
        while i >= 0 and time[i] == dtime:
            risk = np.exp(eta[i]) * weights[i]
            if event[i] <= 0.0:
                denom += risk
                for col in range(p):
                    a[col] += risk * x[i, col]
                    for k in range(p):
                        cmat[col, k] += risk * x[i, col] * x[i, k]
            else:
                ndead += 1
                deadwt += weights[i]
                denom2 += risk
                for col in range(p):
                    u[col] += weights[i] * x[i, col]
                    u[p + col] += timewt * weights[i] * x[i, col]
                    a2[col] += risk * x[i, col]
                    for k in range(p):
                        cmat2[col, k] += risk * x[i, col] * x[i, k]
            i -= 1
        if ndead <= 0:
            continue
        if not efron:
            denom += denom2
            for col in range(p):
                a[col] += a2[col]
                for k in range(p):
                    cmat[col, k] += cmat2[col, k]
            _zph_add(u, imat, a, cmat, denom, deadwt, timewt)
        else:
            width = float(ndead)
            wtave = deadwt / width
            for _step in range(ndead):
                denom += denom2 / width
                for col in range(p):
                    a[col] += a2[col] / width
                    for k in range(p):
                        cmat[col, k] += cmat2[col, k] / width
                _zph_add(u, imat, a, cmat, denom, wtave, timewt)


@njit(cache=True)
def _score_counting(x, time, entry, event, weights, score, death_times, efron, resid):
    """Score residuals for (entry, time] rows in one stratum (``agscore3``)."""
    m, p = x.shape
    s1 = np.zeros(p)
    d1 = np.zeros(p)
    for t_i in range(death_times.shape[0]):
        tm = death_times[t_i]
        s0 = 0.0
        d0 = 0.0
        d = 0
        wtsum = 0.0
        s1[:] = 0.0
        d1[:] = 0.0
        for k in range(m):
            if not (entry[k] < tm and time[k] >= tm):
                continue
            risk = score[k] * weights[k]
            s0 += risk
            for col in range(p):
                s1[col] += risk * x[k, col]
            if time[k] == tm and event[k] > 0.0:
                d += 1
                wtsum += weights[k]
                d0 += risk
                for col in range(p):
                    d1[col] += risk * x[k, col]
        if d == 0 or s0 <= 0.0:
            continue
        width = float(d) if efron else 1.0
        steps = d if efron else 1
        wtave = wtsum / width
        for step in range(steps):
            frac = step / width if efron else 0.0
            denom = s0 - frac * d0
            hazard = wtave / denom
            for k in range(m):
                if not (entry[k] < tm and time[k] >= tm):
                    continue
                dead = time[k] == tm and event[k] > 0.0
                down = (1.0 - frac) if dead else 1.0
                for col in range(p):
                    xbar = (s1[col] - frac * d1[col]) / denom
                    resid[k, col] -= score[k] * down * hazard * (x[k, col] - xbar)
                    if dead:
                        resid[k, col] += (x[k, col] - xbar) / width


@njit(cache=True)
def _zph_counting(x, time, entry, event, weights, eta, g_death, death_times, efron, u, imat):
    """``cox.zph`` score and information for (entry, time] rows in one stratum."""
    m, p = x.shape
    a = np.zeros(p)
    cmat = np.zeros((p, p))
    a2 = np.zeros(p)
    cmat2 = np.zeros((p, p))
    for t_i in range(death_times.shape[0]):
        tm = death_times[t_i]
        timewt = g_death[t_i]
        denom = 0.0
        denom2 = 0.0
        a[:] = 0.0
        a2[:] = 0.0
        cmat[:, :] = 0.0
        cmat2[:, :] = 0.0
        ndead = 0
        deadwt = 0.0
        for k in range(m):
            if not (entry[k] < tm and time[k] >= tm):
                continue
            risk = np.exp(eta[k]) * weights[k]
            if time[k] == tm and event[k] > 0.0:
                ndead += 1
                deadwt += weights[k]
                denom2 += risk
                for col in range(p):
                    u[col] += weights[k] * x[k, col]
                    u[p + col] += timewt * weights[k] * x[k, col]
                    a2[col] += risk * x[k, col]
                    for j in range(p):
                        cmat2[col, j] += risk * x[k, col] * x[k, j]
            else:
                denom += risk
                for col in range(p):
                    a[col] += risk * x[k, col]
                    for j in range(p):
                        cmat[col, j] += risk * x[k, col] * x[k, j]
        if ndead == 0:
            continue
        if not efron:
            denom += denom2
            for col in range(p):
                a[col] += a2[col]
                for j in range(p):
                    cmat[col, j] += cmat2[col, j]
            _zph_add(u, imat, a, cmat, denom, deadwt, timewt)
        else:
            width = float(ndead)
            wtave = deadwt / width
            for _step in range(ndead):
                denom += denom2 / width
                for col in range(p):
                    a[col] += a2[col] / width
                    for j in range(p):
                        cmat[col, j] += cmat2[col, j] / width
                _zph_add(u, imat, a, cmat, denom, wtave, timewt)


def _martingale_residuals(fit: CoxFit) -> np.ndarray:
    """Martingale residuals from ``agmart3``: status minus score times exposed hazard."""
    score = np.exp(fit.x @ fit.coefficients + fit.offset)
    resid = np.zeros(fit.n_obs)
    efron = fit.ties == "efron"
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        time = np.ascontiguousarray(fit.time[idx], dtype=np.float64)
        entry = np.ascontiguousarray(fit.entry[idx], dtype=np.float64)
        event = np.ascontiguousarray(fit.event[idx], dtype=np.float64)
        weights = np.ascontiguousarray(fit.weights[idx], dtype=np.float64)
        sc = np.ascontiguousarray(score[idx], dtype=np.float64)
        if bool(np.all(entry == 0.0)):
            order = np.argsort(time, kind="mergesort")
            charged = np.empty(len(idx))
            charged[order] = _martingale_ordinary(
                np.ascontiguousarray(time[order]),
                np.ascontiguousarray(event[order]),
                np.ascontiguousarray(weights[order]),
                np.ascontiguousarray(sc[order]),
                efron,
            )
        else:
            death_times = np.ascontiguousarray(np.unique(time[event > 0.0]), dtype=np.float64)
            charged = _martingale_scan(time, entry, event, weights, sc, death_times, efron)
        resid[idx] = event - sc * charged
    return resid


def _score_residuals(fit: CoxFit) -> np.ndarray:
    """Score residuals from ``coxscore2``. Deaths are ordered before censored ties."""
    score = np.exp(fit.x @ fit.coefficients + fit.offset)
    n, p = fit.x.shape
    resid = np.zeros((n, p))
    efron = fit.ties == "efron"
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        if not bool(np.all(fit.entry[idx] == 0.0)):
            block = np.zeros((len(idx), p))
            time = np.ascontiguousarray(fit.time[idx], dtype=np.float64)
            event = np.ascontiguousarray(fit.event[idx], dtype=np.float64)
            _score_counting(
                np.ascontiguousarray(fit.x[idx], dtype=np.float64),
                time,
                np.ascontiguousarray(fit.entry[idx], dtype=np.float64),
                event,
                np.ascontiguousarray(fit.weights[idx], dtype=np.float64),
                np.ascontiguousarray(score[idx], dtype=np.float64),
                np.ascontiguousarray(np.unique(time[event > 0.0]), dtype=np.float64),
                efron,
                block,
            )
            resid[idx] = block
            continue
        local = np.lexsort((-fit.event[idx], fit.time[idx]))
        ix = idx[local]
        block = np.zeros((len(ix), p))
        _score_ordered(
            np.ascontiguousarray(fit.x[ix], dtype=np.float64),
            np.ascontiguousarray(fit.event[ix], dtype=np.float64),
            np.ascontiguousarray(fit.weights[ix], dtype=np.float64),
            np.ascontiguousarray(score[ix], dtype=np.float64),
            np.ascontiguousarray(fit.time[ix], dtype=np.float64),
            efron,
            block,
        )
        resid[ix] = block
    return resid


def _schoenfeld(fit: CoxFit) -> np.ndarray:
    """Schoenfeld residuals, one row per death, in time order within strata."""
    risk_score = np.exp(fit.x @ fit.coefficients + fit.offset) * fit.weights
    p = fit.x.shape[1]
    blocks = []
    efron = fit.ties == "efron"
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        order = np.argsort(fit.time[idx], kind="mergesort")
        ix = idx[order]
        time = np.ascontiguousarray(fit.time[ix], dtype=np.float64)
        entry = np.ascontiguousarray(fit.entry[ix], dtype=np.float64)
        event = np.ascontiguousarray(fit.event[ix] > 0.0)
        xx = np.ascontiguousarray(fit.x[ix], dtype=np.float64)
        rs = np.ascontiguousarray(risk_score[ix], dtype=np.float64)
        out = np.empty((len(ix), p))
        if bool(np.all(entry == 0.0)):
            n_out = _schoenfeld_ordinary(xx, time, event, rs, efron, out)
        else:
            n_out = _schoenfeld_scan(xx, time, entry, event, rs, efron, out)
        if n_out:
            blocks.append(np.array(out[:n_out], copy=True))
    if not blocks:
        return np.zeros((0, p))
    return np.vstack(blocks)


def _schoenfeld_times(fit: CoxFit) -> np.ndarray:
    times = []
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        t = fit.time[idx]
        e = fit.event[idx] > 0
        times.append(t[e])
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
    """Score and 2p information for ``cox.zph`` (``zph1``)."""
    p = fit.x.shape[1]
    # Center covariates, as zph1 does, so the information is better conditioned.
    x = fit.x - fit.x.mean(axis=0)
    eta = fit.x @ fit.coefficients + fit.offset
    u = np.zeros(2 * p)
    imat = np.zeros((2 * p, 2 * p))
    efron = fit.ties != "breslow"
    for stratum in np.unique(fit.strata):
        idx = np.flatnonzero(fit.strata == stratum)
        if not bool(np.all(fit.entry[idx] == 0.0)):
            time = fit.time[idx]
            dead = fit.event[idx] > 0
            death_times, first = np.unique(time[dead], return_index=True)
            _zph_counting(
                np.ascontiguousarray(x[idx], dtype=np.float64),
                np.ascontiguousarray(time, dtype=np.float64),
                np.ascontiguousarray(fit.entry[idx], dtype=np.float64),
                np.ascontiguousarray(fit.event[idx], dtype=np.float64),
                np.ascontiguousarray(fit.weights[idx], dtype=np.float64),
                np.ascontiguousarray(eta[idx], dtype=np.float64),
                np.ascontiguousarray(g[idx][dead][first], dtype=np.float64),
                np.ascontiguousarray(death_times, dtype=np.float64),
                efron,
                u,
                imat,
            )
            continue
        order = np.argsort(fit.time[idx], kind="mergesort")
        ix = idx[order]
        _zph_ordered(
            np.ascontiguousarray(x[ix], dtype=np.float64),
            np.ascontiguousarray(fit.event[ix], dtype=np.float64),
            np.ascontiguousarray(fit.weights[ix], dtype=np.float64),
            np.ascontiguousarray(eta[ix], dtype=np.float64),
            np.ascontiguousarray(fit.time[ix], dtype=np.float64),
            np.ascontiguousarray(g[ix], dtype=np.float64),
            efron,
            u,
            imat,
        )
    imat[p:, :p] = imat[:p, p:].T
    return u, imat


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
