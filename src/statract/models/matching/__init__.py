"""Sample matching in the sense of MatchIt, with a Python calling style.

``distance="logit"`` fits a logistic regression and matches on the propensity
score. ``order="data"`` walks treated units in frame order, so the pairs are
deterministic. Optimal 1:1 matching uses the Hungarian algorithm.
``total_distance`` is the sum of discrepancies on the matched edges. Full
matching, and optimal matching with ``ratio`` above 1, use the same
minimum-cost flow as ``optmatch::fullmatch``. Weights follow MatchIt's
``normalize=TRUE``: within each arm, positive weights sum to the number of
positive-weight units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit
from scipy.optimize import linear_sum_assignment

from ..design import ColumnRef, _factor_codes, _factor_levels, _is_factor, column_series, design_matrix
from ..fit import fit_glm
from .flow import full_match_edges

_METHODS = {"nearest", "exact", "subclass", "cem", "optimal", "full"}
_DISTANCES = {"logit", "probit", "mahalanobis", "robust_mahalanobis", "euclidean", "scaled_euclidean"}
_SD_DENOMS = {"pooled", "treated", "control"}
_BINARY = {"raw", "std"}


@dataclass
class MatchedSample:
    """Matched sample. Row positions refer to the input frame."""

    data: pl.DataFrame
    treatment: str
    covariates: list[str]
    weights: np.ndarray
    distance: np.ndarray
    subclass: np.ndarray
    pair_table: pl.DataFrame
    estimand: str
    method: str
    # Exact, subclass, and CEM store subclass ids. The treated-by-control
    # product is built when pairs() is asked for.
    _expand_pairs: bool = False
    # Exact and CEM do not use the default propensity for their weights.
    # The logit (or the requested link) is fit when a caller reads it.
    _distance_link: str | None = None

    def _ensure_distance(self) -> None:
        if self._distance_link is None:
            return
        dist, rows = _distance_vector(self.data, self.treatment, self.covariates, "logit", self._distance_link)
        self.distance = _align_distance(dist, rows, self.data.height)
        self._distance_link = None

    def frame(self) -> pl.DataFrame:
        """Input rows with ``distance``, ``weights``, and ``subclass``."""
        self._ensure_distance()
        return self.data.with_columns(
            pl.Series("distance", self.distance),
            pl.Series("weights", self.weights),
            pl.Series("subclass", self.subclass),
        )

    def pairs(self) -> pl.DataFrame:
        self._ensure_distance()
        if self._expand_pairs:
            self.pair_table = _pairs_from_subclasses(self)
            self._expand_pairs = False
        return self.pair_table

    def total_distance(self) -> float:
        """Sum of discrepancies on the matched edges.

        Optimal matching minimizes this sum. Full matching minimizes the same
        sum over the edges that define the subclasses.
        """
        table = self.pair_table if not self._expand_pairs else self.pairs()
        if table.height == 0:
            return 0.0
        return float(np.sum(table["distance"].to_numpy()))

    def balance(self, *, sd_denominator: str = "pooled", binary: str = "std") -> pl.DataFrame:
        """Balance before (``*_all``) and after (``*_matched``) matching.

        ``smd_*`` is the treated mean minus the control mean over one fixed
        denominator, computed on the whole sample before matching and shared by
        both rows. ``sd_denominator="pooled"`` (default) is
        ``sqrt((s1^2 + s0^2) / 2)``; ``"treated"`` or ``"control"`` uses that
        group's standard deviation (MatchIt's ATT and ATC default). A column with
        two distinct values is binary: with ``binary="std"`` (default) its
        variance is ``p(1-p)`` (Austin 2009), with ``binary="raw"`` its
        ``smd_*`` is the unstandardized difference (cobalt's default). The
        ``distance`` row is always standardized. ``pair_distance`` is the mean
        absolute within-pair difference over the same denominator.
        """
        if sd_denominator not in _SD_DENOMS:
            raise ValueError(f"sd_denominator must be one of {sorted(_SD_DENOMS)}")
        if binary not in _BINARY:
            raise ValueError("binary must be 'raw' or 'std'")
        self._ensure_distance()
        return _balance(self, sd_denominator=sd_denominator, binary=binary)

    def love_plot(self):
        """Standardized mean differences before and after matching."""
        import matplotlib.pyplot as plt

        table = self.balance()
        labels = table["term"].to_list()[::-1]
        y = np.arange(len(labels))
        fig, ax = plt.subplots()
        ax.scatter(table["smd_all"].to_numpy()[::-1], y, label="all", marker="o")
        ax.scatter(table["smd_matched"].to_numpy()[::-1], y, label="matched", marker="D")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.set_xlabel("standardized mean difference")
        ax.legend()
        fig.tight_layout()
        return fig

    def balance_plot(self, covariate: str, *, kind: str = "density"):
        """Density of one covariate before and after matching."""
        if kind != "density":
            raise ValueError("kind must be 'density'")
        import matplotlib.pyplot as plt

        values = np.asarray(self.data[covariate].to_numpy(), dtype=float)
        treat = np.asarray(self.data[self.treatment].to_numpy(), dtype=float) > 0
        fig, ax = plt.subplots()
        for label, mask in (("treated", treat), ("control", ~treat)):
            ax.hist(values[mask], bins=15, density=True, histtype="step", label=f"all {label}")
            kept = mask & (self.weights > 0)
            if np.any(kept):
                ax.hist(values[kept], bins=15, density=True, histtype="step", label=f"matched {label}")
        ax.set_xlabel(covariate)
        ax.legend()
        fig.tight_layout()
        return fig


def match_sample(
    data: pl.DataFrame,
    treatment: ColumnRef,
    covariates: list[ColumnRef],
    *,
    method: str = "nearest",
    distance: str | np.ndarray | None = None,
    link: str = "logit",
    estimand: str = "ATT",
    exact: list[ColumnRef] | None = None,
    caliper: float | None = None,
    std_caliper: bool = True,
    ratio: int = 1,
    replace: bool = False,
    order: str | None = None,
    discard: str = "none",
    reestimate: bool = False,
    subclass: int = 6,
    cutpoints: str | dict[str, str | int | np.ndarray] = "sturges",
    seed: int = 0,
) -> MatchedSample:
    """Match controls to treated units.

    ``method`` is ``nearest``, ``exact``, ``subclass``, ``cem``, ``optimal``,
    or ``full``. ``distance`` is ``logit``, ``probit``, ``mahalanobis``,
    ``robust_mahalanobis``, ``euclidean``, ``scaled_euclidean``, or a numeric
    vector already aligned with ``data``. Omitting it uses the logit propensity.
    Exact and CEM do not need that score for their weights, so the fit waits
    until ``frame``, ``pairs``, or ``balance`` reads the distance.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if estimand not in {"ATT", "ATC", "ATE"}:
        raise ValueError("estimand must be ATT, ATC, or ATE")
    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    treat_s = column_series(data, treatment)
    treat = np.asarray(treat_s.to_numpy(), dtype=float) > 0
    names = [_column_name(data, col) for col in covariates]
    focal = treat if estimand != "ATC" else ~treat
    # ATC swaps the roles and the returned weights are mapped back.
    role = treat if estimand != "ATC" else ~treat
    requested = "logit" if distance is None else distance
    defer_distance = distance is None and method in {"exact", "cem"} and discard == "none"
    if defer_distance:
        dist = np.full(data.height, np.nan)
        matrix = False
    else:
        dist, dist_rows = _distance_vector(data, treat_s.name, names, requested, link)
        dist = _align_distance(dist, dist_rows, data.height)
        matrix = isinstance(dist, np.ndarray) and dist.ndim == 2
    if method == "subclass" and matrix:
        raise ValueError("subclass matching needs a one-dimensional distance")
    discarded = _discard_mask(dist, role, discard)
    if reestimate and isinstance(requested, str) and requested in {"logit", "probit"} and np.any(discarded):
        kept = data.filter(~pl.Series(discarded))
        dist2, rows2 = _distance_vector(kept, treat_s.name, names, requested, link)
        dist = np.full(data.height, np.nan)
        dist[np.flatnonzero(~discarded)] = dist2
    if order is None:
        if method == "exact" or matrix:
            order = "data"
        else:
            order = "largest" if np.isfinite(dist).any() else "data"
    if matrix and order in {"largest", "smallest"}:
        raise ValueError("order 'largest' and 'smallest' apply to a scalar distance; use 'data'")
    expand_pairs = method in {"exact", "subclass", "cem"}
    if method == "nearest":
        pairs, weights, subclasses = _nearest(
            role, dist, ratio=ratio, replace=replace, order=order, caliper=caliper,
            std_caliper=std_caliper, discarded=discarded, exact=_exact_labels(data, exact), seed=seed,
        )
    elif method == "exact":
        weights, subclasses = _subclass_weights(role, _exact_labels(data, covariates if exact is None else exact), estimand="ATT")
        pairs = None
    elif method == "subclass":
        labels = _quantile_subclass(dist, subclass, discarded)
        weights, subclasses = _subclass_weights(role, labels, estimand="ATT")
        pairs = None
    elif method == "cem":
        labels = _cem_labels(data, names, role, cutpoints)
        weights, subclasses = _subclass_weights(role, labels, estimand="ATT")
        pairs = None
    elif method == "optimal":
        pairs, weights, subclasses = _optimal(role, dist, ratio=ratio, discarded=discarded, exact=_exact_labels(data, exact))
    else:
        pairs, weights, subclasses = _full(role, dist, discarded=discarded)
    if estimand == "ATC":
        weights = _swap_roles(weights, treat)
        if pairs is not None:
            pairs = pairs.rename({"treated": "control", "control": "treated"})
    weights = _normalize_to_counts(weights, treat)
    if pairs is None:
        pair_frame = pl.DataFrame({"treated": [], "control": [], "distance": []})
    elif pairs.height:
        pair_frame = pl.DataFrame(
            {
                "treated": pairs["treated"].to_numpy(),
                "control": pairs["control"].to_numpy(),
                "distance": _pair_distance(dist, pairs),
            }
        )
    else:
        pair_frame = pl.DataFrame({"treated": [], "control": [], "distance": []})
    return MatchedSample(
        data=data,
        treatment=treat_s.name,
        covariates=names,
        weights=weights.astype(float),
        distance=dist,
        subclass=subclasses,
        pair_table=pair_frame,
        estimand=estimand,
        method=method,
        _expand_pairs=expand_pairs,
        _distance_link=link if defer_distance else None,
    )


def _nearest(treat, dist, *, ratio, replace, order, caliper, std_caliper, discarded, exact, seed):
    usable = _usable_rows(dist)
    treated = np.flatnonzero(treat & ~discarded & usable)
    control = np.flatnonzero(~treat & ~discarded & usable)
    treated = _order_units(treated, dist, order, seed)
    limit = None
    if caliper is not None:
        if dist.ndim == 2:
            raise NotImplementedError("caliper on a matrix distance is not implemented")
        scale = float(np.nanstd(dist[~discarded], ddof=1)) if std_caliper else 1.0
        limit = float(caliper) * scale
    if dist.ndim == 1:
        pair_t, pair_c = _nearest_on_line(treated, control, dist, ratio, replace, limit, exact)
    else:
        pair_t, pair_c = _nearest_scan(treated, control, dist, ratio, replace, limit, exact)
    weights = np.zeros(len(treat))
    subclasses = np.zeros(len(treat), dtype=int)
    estimand_weights_from_pairs(treat, pair_t, pair_c, weights, subclasses, replace)
    pairs = pl.DataFrame({"treated": pair_t, "control": pair_c}) if pair_t else pl.DataFrame({"treated": [], "control": []})
    return pairs, weights, subclasses


def _nearest_on_line(treated, control, dist, ratio, replace, limit, exact):
    """Nearest controls on a scalar score, matching ``MatchIt`` 4.5.5.

    Controls are ordered by score and, within a score, by row index, which is
    ``order()`` on the propensity. Each match takes the unused control on the
    lower-score side of the treated unit and the one on the higher-score side,
    then keeps the closer. An equal gap keeps the lower-score side.
    """
    if control.size == 0 or treated.size == 0:
        return [], []
    ctrl, scores = _score_order(control, dist)
    has_exact = exact is not None
    if has_exact:
        codes = np.asarray(exact, dtype=np.int64)
        treated_codes = np.ascontiguousarray(codes[treated])
        ctrl_codes = np.ascontiguousarray(codes[ctrl])
    else:
        treated_codes = np.zeros(treated.shape[0], dtype=np.int64)
        ctrl_codes = np.zeros(ctrl.shape[0], dtype=np.int64)
    out_t, out_c = _match_sorted(
        np.ascontiguousarray(treated, dtype=np.int64),
        np.ascontiguousarray(ctrl, dtype=np.int64),
        np.ascontiguousarray(scores, dtype=np.float64),
        np.ascontiguousarray(dist, dtype=np.float64),
        int(ratio),
        bool(replace),
        limit is not None,
        0.0 if limit is None else float(limit),
        has_exact,
        treated_codes,
        ctrl_codes,
    )
    return out_t.tolist(), out_c.tolist()


@njit(cache=True)
def _eligible_side(ctrl, scores, query, t, used, has_exact, t_code, ctrl_codes, lower):
    """First unused control on one side of ``t`` in the stable score order."""
    n_c = scores.shape[0]
    hi = np.searchsorted(scores, query)
    eq_hi = hi
    while eq_hi < n_c and scores[eq_hi] == query:
        eq_hi += 1
    if lower:
        j = eq_hi - 1
        while j >= hi:
            row = ctrl[j]
            if row < t and used[row] == 0 and (not has_exact or ctrl_codes[j] == t_code):
                return j
            j -= 1
        j = hi - 1
        while j >= 0:
            row = ctrl[j]
            if used[row] == 0 and (not has_exact or ctrl_codes[j] == t_code):
                return j
            j -= 1
        return -1
    j = hi
    while j < eq_hi:
        row = ctrl[j]
        if row > t and used[row] == 0 and (not has_exact or ctrl_codes[j] == t_code):
            return j
        j += 1
    while j < n_c:
        row = ctrl[j]
        if used[row] == 0 and (not has_exact or ctrl_codes[j] == t_code):
            return j
        j += 1
    return -1


@njit(cache=True)
def _match_sorted(treated, ctrl, scores, dist, ratio, replace, has_limit, limit, has_exact, treated_codes, ctrl_codes):
    """Greedy nearest neighbors on a score already sorted by value, then row.

    One match at a time: the nearer of the adjacent unused controls, with the
    lower score winning an equal gap. That is ``nn_matchC_vec`` in MatchIt 4.5.5.
    """
    n_t = treated.shape[0]
    n_rows = dist.shape[0]
    out_t = np.empty(n_t * ratio, dtype=np.int64)
    out_c = np.empty(n_t * ratio, dtype=np.int64)
    count = 0
    used = np.zeros(n_rows, dtype=np.uint8)
    scratch = np.empty(ratio, dtype=np.int64)
    for t_i in range(n_t):
        t = treated[t_i]
        query = dist[t]
        t_code = treated_codes[t_i]
        taken = 0
        local_n = 0
        while taken < ratio:
            left_j = _eligible_side(ctrl, scores, query, t, used, has_exact, t_code, ctrl_codes, True)
            right_j = _eligible_side(ctrl, scores, query, t, used, has_exact, t_code, ctrl_codes, False)
            left_gap = abs(scores[left_j] - query) if left_j >= 0 else 0.0
            right_gap = abs(scores[right_j] - query) if right_j >= 0 else 0.0
            left_ok = left_j >= 0 and (not has_limit or left_gap <= limit + 1e-12)
            right_ok = right_j >= 0 and (not has_limit or right_gap <= limit + 1e-12)
            if left_ok and right_ok:
                best = ctrl[right_j] if right_gap < left_gap else ctrl[left_j]
            elif left_ok:
                best = ctrl[left_j]
            elif right_ok:
                best = ctrl[right_j]
            else:
                break
            out_t[count] = t
            out_c[count] = best
            count += 1
            used[best] = 1
            if replace:
                scratch[local_n] = best
                local_n += 1
            taken += 1
        if replace:
            for k in range(local_n):
                used[scratch[k]] = 0
    return out_t[:count], out_c[:count]


def _score_order(control: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(dist[control], dtype=float)
    order = np.lexsort((control, scores))
    return control[order], scores[order]


def _nearest_scan(treated, control, dist, ratio, replace, limit, exact):
    """Matrix-distance scan. Ties keep the earlier control row.

    A scalar score uses the same lower-score tie break as ``_nearest_on_line``.
    Squared distances are sums of squares, so identical rows stay an exact tie.
    """
    if np.asarray(dist).ndim == 1:
        return _nearest_on_line(treated, control, dist, ratio, replace, limit, exact)
    treated_a = np.ascontiguousarray(treated, dtype=np.int64)
    control_a = np.ascontiguousarray(control, dtype=np.int64)
    matrix = np.ascontiguousarray(dist, dtype=np.float64)
    n_rows = matrix.shape[0]
    if exact is None:
        exact_a = np.zeros(n_rows, dtype=np.int64)
        has_exact = False
    else:
        exact_a = np.ascontiguousarray(exact, dtype=np.int64)
        has_exact = True
    n_t = int(treated_a.shape[0])
    n_c = int(control_a.shape[0])
    wide = n_t * n_c * int(matrix.shape[1]) >= 50_000 and limit is None
    if wide and n_t > 0 and n_c > 0:
        out_t, out_c = _nearest_gemm(treated_a, control_a, matrix, int(ratio), bool(replace), has_exact, exact_a)
    else:
        out_t, out_c = _scan_matrix(
            treated_a,
            control_a,
            matrix,
            int(ratio),
            bool(replace),
            limit is not None,
            0.0 if limit is None else float(limit),
            has_exact,
            exact_a,
        )
    return out_t.tolist(), out_c.tolist()


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``||left[i] - right[j]||^2``. Identical rows stay exactly zero.

    The square is the einsum of the coordinate difference. A sequential sum
    of those squares is a different double and can change a tie. Each tile
    is about a million coordinate slots, so the temporary stays in cache.
    A larger tile is the same bits and slower.
    """
    n_c = right.shape[0]
    width = right.shape[1]
    step = max(1, 1_000_000 // max(n_c * max(width, 1), 1))
    out = np.empty((left.shape[0], n_c), dtype=np.float64)
    for start in range(0, left.shape[0], step):
        block = left[start : start + step]
        diff = block[:, None, :] - right[None, :, :]
        out[start : start + block.shape[0]] = np.einsum("ijk,ijk->ij", diff, diff)
    return out


def _nearest_gemm(treated, control, matrix, ratio, replace, has_exact, exact):
    """Greedy nearest neighbours from sums of squared coordinate differences."""
    ctrl = np.ascontiguousarray(matrix[control])
    n_t = int(treated.shape[0])
    n_c = int(control.shape[0])
    # One distance block stays under a few hundred megabytes. Larger problems
    # are scored a few treated rows at a time.
    block = n_t if n_t * n_c <= 40_000_000 else max(1, 40_000_000 // n_c)
    used = np.zeros(n_c, dtype=np.uint8)
    parts_t = []
    parts_c = []
    simple = ratio == 1 and not has_exact
    for start in range(0, n_t, block):
        rows = treated[start : start + block]
        tv = np.ascontiguousarray(matrix[rows])
        dist2 = np.ascontiguousarray(_squared_euclidean(tv, ctrl))
        if simple:
            chosen = _take_dist2_one(dist2, used, replace)
            keep = chosen >= 0
            out_t = rows[keep]
            out_c = control[chosen[keep]]
        else:
            out_t, out_c = _take_dist2(dist2, np.ascontiguousarray(rows), control, ratio, replace, has_exact, exact, used)
        if out_t.size:
            parts_t.append(np.asarray(out_t, dtype=np.int64))
            parts_c.append(np.asarray(out_c, dtype=np.int64))
    if not parts_t:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(parts_t), np.concatenate(parts_c)


@njit(cache=True)
def _take_dist2_one(dist2, used, replace):
    """One control per treated row.

    The gap is the Euclidean length, so a one-ulp difference in the square that
    disappears under ``sqrt`` stays a tie. An equal length keeps the earlier
    control, as ``which.min`` does in MatchIt 4.5.5.
    """
    n_t, n_c = dist2.shape
    out = np.empty(n_t, dtype=np.int64)
    for t_i in range(n_t):
        best = 1e300
        best_j = -1
        for j in range(n_c):
            if (not replace) and used[j] != 0:
                continue
            gap = np.sqrt(dist2[t_i, j])
            if gap < best:
                best = gap
                best_j = j
        if best_j < 0:
            out[t_i] = -1
            continue
        if not replace:
            used[best_j] = 1
        out[t_i] = best_j
    return out


@njit(cache=True)
def _take_dist2(dist2, treated, control, ratio, replace, has_exact, exact, used):
    """Pick from precomputed squared distances. ``used`` carries over between blocks."""
    n_t = dist2.shape[0]
    n_c = dist2.shape[1]
    out_t = np.empty(n_t * ratio, dtype=np.int64)
    out_c = np.empty(n_t * ratio, dtype=np.int64)
    count = 0
    best_gap = np.empty(ratio, dtype=np.float64)
    best_j = np.empty(ratio, dtype=np.int64)
    for t_i in range(n_t):
        t = treated[t_i]
        t_code = exact[t] if has_exact else 0
        n_best = 0
        for j in range(n_c):
            if (not replace) and used[j] != 0:
                continue
            if has_exact and exact[control[j]] != t_code:
                continue
            gap = np.sqrt(dist2[t_i, j])
            pos = n_best
            for s in range(n_best):
                if gap < best_gap[s] or (gap == best_gap[s] and j < best_j[s]):
                    pos = s
                    break
            if pos == ratio:
                continue
            if n_best < ratio:
                last = n_best
                n_best += 1
            else:
                last = ratio - 1
            for s in range(last, pos, -1):
                best_gap[s] = best_gap[s - 1]
                best_j[s] = best_j[s - 1]
            best_gap[pos] = gap
            best_j[pos] = j
        for s in range(n_best):
            j = best_j[s]
            out_t[count] = t
            out_c[count] = control[j]
            count += 1
            if not replace:
                used[j] = 1
    return out_t[:count], out_c[:count]


@njit(cache=True)
def _scan_matrix(treated, control, dist, ratio, replace, has_limit, limit, has_exact, exact):
    """Nearest ``ratio`` controls for each treated row.

    Equal distances keep the earlier index in ``control``, the same order as a
    stable sort of the gaps. A scalar distance is passed as one column, so the
    Euclidean gap is the absolute difference.
    """
    n_t = treated.shape[0]
    n_c = control.shape[0]
    p = dist.shape[1]
    out_t = np.empty(n_t * ratio, dtype=np.int64)
    out_c = np.empty(n_t * ratio, dtype=np.int64)
    count = 0
    used = np.zeros(n_c, dtype=np.uint8)
    best_gap = np.empty(ratio, dtype=np.float64)
    best_j = np.empty(ratio, dtype=np.int64)
    for t_i in range(n_t):
        t = treated[t_i]
        t_code = exact[t] if has_exact else 0
        n_best = 0
        for j in range(n_c):
            if (not replace) and used[j] != 0:
                continue
            row = control[j]
            if has_exact and exact[row] != t_code:
                continue
            acc = 0.0
            for k in range(p):
                diff = dist[row, k] - dist[t, k]
                acc += diff * diff
            gap = np.sqrt(acc)
            if has_limit and gap > limit + 1e-12:
                continue
            pos = n_best
            for s in range(n_best):
                if gap < best_gap[s] or (gap == best_gap[s] and j < best_j[s]):
                    pos = s
                    break
            if pos == ratio:
                continue
            if n_best < ratio:
                last = n_best
                n_best += 1
            else:
                last = ratio - 1
            for s in range(last, pos, -1):
                best_gap[s] = best_gap[s - 1]
                best_j[s] = best_j[s - 1]
            best_gap[pos] = gap
            best_j[pos] = j
        for s in range(n_best):
            j = best_j[s]
            out_t[count] = t
            out_c[count] = control[j]
            count += 1
            if not replace:
                used[j] = 1
    return out_t[:count], out_c[:count]


def estimand_weights_from_pairs(treat, pair_t, pair_c, weights, subclasses, replace):
    """ATT weights: matched treated units weigh 1, controls weigh their reuse count."""
    weights[treat] = 0
    if not pair_t:
        return True
    pair_t = np.asarray(pair_t, dtype=int)
    pair_c = np.asarray(pair_c, dtype=int)
    treated_ids, inverse = np.unique(pair_t, return_inverse=True)
    weights[treated_ids] = 1
    controls, counts = np.unique(pair_c, return_counts=True)
    weights[controls] = counts.astype(float) if replace or int(counts.max()) > 1 else 1.0
    # Subclass ids follow the sorted treated units. A reused control keeps the
    # id of the later treated unit, which is the last write in that order.
    subclasses[treated_ids] = np.arange(1, len(treated_ids) + 1)
    order = np.argsort(pair_t, kind="mergesort")
    subclasses[pair_c[order]] = inverse[order] + 1
    return True


def _optimal(treat, dist, *, ratio, discarded, exact):
    usable = _usable_rows(dist)
    treated = np.flatnonzero(treat & ~discarded & usable)
    control = np.flatnonzero(~treat & ~discarded & usable)
    if treated.size == 0 or control.size == 0:
        return pl.DataFrame({"treated": [], "control": []}), np.zeros(len(treat)), np.zeros(len(treat), dtype=int)
    cost = _pair_cost(dist, treated, control)
    if exact is not None:
        bad = exact[treated][:, None] != exact[control][None, :]
        cost = cost.copy()
        cost[bad] = np.inf
    if ratio == 1:
        # Rectangular Hungarian assignment. Infinite entries are closed.
        work = np.array(cost, copy=True)
        work[~np.isfinite(work)] = 1e12
        row, col = linear_sum_assignment(work)
        pair_t, pair_c = [], []
        for r, c in zip(row, col, strict=False):
            if not np.isfinite(cost[r, c]) or work[r, c] >= 1e12:
                continue
            pair_t.append(int(treated[r]))
            pair_c.append(int(control[c]))
    else:
        # k:1 optimal matching is full matching with the ratio fixed.
        # Fewer controls than the ratio asks for falls back to 1:1.
        if control.size < ratio * treated.size:
            return _optimal(treat, dist, ratio=1, discarded=discarded, exact=exact)
        edges, _total = full_match_edges(
            cost, min_controls=ratio, max_controls=ratio, mean_controls=float(ratio)
        )
        pair_t = [int(treated[i]) for i, _j in edges]
        pair_c = [int(control[j]) for _i, j in edges]
    weights = np.zeros(len(treat))
    subclasses = np.zeros(len(treat), dtype=int)
    estimand_weights_from_pairs(treat, pair_t, pair_c, weights, subclasses, replace=False)
    pairs = pl.DataFrame({"treated": pair_t, "control": pair_c}) if pair_t else pl.DataFrame({"treated": [], "control": []})
    return pairs, weights, subclasses


def _full(treat, dist, *, discarded):
    """Full matching by minimum-cost flow, then ATT subclass weights."""
    usable = _usable_rows(dist)
    treated = np.flatnonzero(treat & ~discarded & usable)
    control = np.flatnonzero(~treat & ~discarded & usable)
    weights = np.zeros(len(treat))
    subclasses = np.zeros(len(treat), dtype=int)
    empty = pl.DataFrame({"treated": [], "control": []})
    if treated.size == 0 or control.size == 0:
        return empty, weights, subclasses
    edges, _total = full_match_edges(_pair_cost(dist, treated, control))
    pair_t = [int(treated[i]) for i, _j in edges]
    pair_c = [int(control[j]) for _i, j in edges]
    weights, subclasses = _weights_from_full(treat, pair_t, pair_c)
    pairs = pl.DataFrame({"treated": pair_t, "control": pair_c}) if pair_t else empty
    return pairs, weights, subclasses


def _pair_cost(dist: np.ndarray, treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Treated-by-control discrepancy. A scalar distance uses the absolute gap."""
    if dist.ndim == 1:
        return np.abs(dist[treated][:, None] - dist[control][None, :])
    diff = dist[treated][:, None, :] - dist[control][None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _weights_from_full(treat, pair_t, pair_c):
    """ATT full-matching weights from the subclass sizes."""
    weights = np.zeros(len(treat))
    subclasses = np.zeros(len(treat), dtype=int)
    if not pair_t:
        return weights, subclasses
    # Subclass id is the treated unit when several controls share it,
    # or the control when several treated share it.
    groups: dict[int, list[int]] = {}
    partner: dict[int, int] = {}
    for t, c in zip(pair_t, pair_c, strict=False):
        partner.setdefault(t, c)
    # Build connected pairs as subclasses keyed by the first treated.
    subclass_of: dict[int, int] = {}
    next_id = 1
    members_t: dict[int, list[int]] = {}
    members_c: dict[int, list[int]] = {}
    for t, c in zip(pair_t, pair_c, strict=False):
        sid = subclass_of.get(t) or subclass_of.get(c)
        if sid is None:
            sid = next_id
            next_id += 1
        subclass_of[t] = sid
        subclass_of[c] = sid
        members_t.setdefault(sid, [])
        members_c.setdefault(sid, [])
        if t not in members_t[sid]:
            members_t[sid].append(t)
        if c not in members_c[sid]:
            members_c[sid].append(c)
    for sid, ts in members_t.items():
        cs = members_c[sid]
        for t in ts:
            subclasses[t] = sid
            weights[t] = 1.0
        for c in cs:
            subclasses[c] = sid
            weights[c] = len(ts) / len(cs)
    return weights, subclasses


def _subclass_weights(treat, labels, *, estimand):
    """Weights for exact, subclass, and CEM. ``labels`` <= 0 means unmatched.

    A cell contributes only when it holds both classes. Weights are the
    within-cell size ratio. The pair list is not built here.
    """
    weights = np.zeros(len(treat))
    subclasses = np.zeros(len(treat), dtype=np.int64)
    positive = np.asarray(labels) > 0
    if not np.any(positive):
        return weights, subclasses
    cell = np.asarray(labels, dtype=np.int64)
    uniq, inverse_pos = np.unique(cell[positive], return_inverse=True)
    role = np.asarray(treat, dtype=bool)[positive]
    n_role = np.bincount(inverse_pos, weights=role.astype(float), minlength=len(uniq))
    n_other = np.bincount(inverse_pos, weights=(~role).astype(float), minlength=len(uniq))
    both = (n_role > 0) & (n_other > 0)
    if estimand == "ATC":
        role_w = np.divide(n_other, n_role, out=np.zeros_like(n_role), where=n_role > 0)
        other_w = np.ones_like(n_other)
    else:
        role_w = np.ones_like(n_role)
        other_w = np.divide(n_role, n_other, out=np.zeros_like(n_other), where=n_other > 0)
    role_w = np.where(both, role_w, 0.0)
    other_w = np.where(both, other_w, 0.0)
    weights[positive] = np.where(role, role_w[inverse_pos], other_w[inverse_pos])
    kept = cell.copy()
    kept[~positive] = 0
    kept[positive] = np.where(both[inverse_pos], cell[positive], 0)
    subclasses[:] = kept
    return weights, subclasses


def _pairs_from_subclasses(matched: MatchedSample) -> pl.DataFrame:
    """Cartesian pairs inside each subclass, in the same row order as the old loop."""
    treat = np.asarray(matched.data[matched.treatment].to_numpy(), dtype=float) > 0
    labels = np.asarray(matched.subclass)
    treated_parts: list[np.ndarray] = []
    control_parts: list[np.ndarray] = []
    # ATC used to iterate the focal (control) arm on the outside, then rename.
    outer_treated = matched.estimand != "ATC"
    for label in np.unique(labels):
        if int(label) <= 0:
            continue
        members = np.flatnonzero(labels == label)
        treated = members[treat[members]]
        control = members[~treat[members]]
        if treated.size == 0 or control.size == 0:
            continue
        if outer_treated:
            treated_parts.append(np.repeat(treated, control.size))
            control_parts.append(np.tile(control, treated.size))
        else:
            treated_parts.append(np.tile(treated, control.size))
            control_parts.append(np.repeat(control, treated.size))
    if not treated_parts:
        return pl.DataFrame({"treated": [], "control": [], "distance": []})
    treated_id = np.concatenate(treated_parts)
    control_id = np.concatenate(control_parts)
    return pl.DataFrame(
        {
            "treated": treated_id,
            "control": control_id,
            "distance": _pair_distance_ids(matched.distance, treated_id, control_id),
        }
    )


def _quantile_subclass(dist, n_sub, discarded):
    labels = np.zeros(len(dist), dtype=int)
    usable = np.isfinite(dist) & ~discarded
    if int(usable.sum()) < n_sub:
        labels[usable] = 1
        return labels
    edges = np.quantile(dist[usable], np.linspace(0, 1, n_sub + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # right-closed bins, matching cut(..., include.lowest=TRUE) closely enough
    bin_id = np.searchsorted(edges, dist, side="right") - 1
    labels[usable] = bin_id[usable] + 1
    return labels


def _cem_labels(data, names, treat, cutpoints):
    codes = []
    for name in names:
        values = np.asarray(data[name].to_numpy())
        if np.issubdtype(values.dtype, np.number):
            bins = cutpoints.get(name, "sturges") if isinstance(cutpoints, dict) else cutpoints
            codes.append(_coarsen(values.astype(float), bins))
        else:
            levels, inverse = np.unique(values.astype(str), return_inverse=True)
            codes.append(inverse.astype(int))
    if not codes:
        return np.ones(len(treat), dtype=int)
    stacked = np.ascontiguousarray(np.column_stack(codes), dtype=np.int64)
    if stacked.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort(tuple(stacked[:, i] for i in range(stacked.shape[1] - 1, -1, -1)))
    sorted_rows = stacked[order]
    change = np.empty(len(sorted_rows), dtype=bool)
    change[0] = True
    if len(sorted_rows) > 1:
        change[1:] = np.any(sorted_rows[1:] != sorted_rows[:-1], axis=1)
    group_sorted = np.cumsum(change) - 1
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = group_sorted
    n_groups = int(group_sorted[-1]) + 1
    role = np.asarray(treat, dtype=float)
    n_role = np.bincount(inverse, weights=role, minlength=n_groups)
    n_other = np.bincount(inverse, weights=1.0 - role, minlength=n_groups)
    keep = (n_role > 0) & (n_other > 0)
    remap = np.zeros(n_groups, dtype=np.int64)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int64)
    return remap[inverse]


def _coarsen(values: np.ndarray, bins) -> np.ndarray:
    if isinstance(bins, str):
        if bins == "sturges":
            n_bins = int(np.ceil(np.log2(len(values)) + 1))
        elif bins == "scott":
            sigma = float(np.std(values, ddof=1))
            width = 3.5 * sigma / (len(values) ** (1 / 3)) if sigma > 0 else 1.0
            n_bins = max(int(np.ceil((values.max() - values.min()) / width)), 1)
        else:
            raise ValueError("cutpoints string must be sturges or scott")
        edges = np.linspace(values.min(), values.max(), n_bins + 1)
    elif isinstance(bins, (int, np.integer)):
        edges = np.linspace(values.min(), values.max(), int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    edges = edges.astype(float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.searchsorted(edges, values, side="right")


def _distance_vector(data, treatment, covariates, distance, link):
    if not isinstance(distance, str):
        values = np.asarray(distance, dtype=float)
        return values, np.arange(len(values))
    if distance not in _DISTANCES:
        raise ValueError(f"distance must be one of {sorted(_DISTANCES)} or a numeric vector")
    if distance in {"logit", "probit"}:
        family_link = distance if link == "logit" else link
        if family_link == "logit":
            fit = fit_glm(data, treatment, covariates, family="binomial")
            return fit.predict(kind="response"), fit.row_index
        return _probit_probability(data, treatment, covariates)
    # Mahalanobis uses every factor level, divided by sqrt(2). The other
    # matrix distances keep treatment-contrast dummies.
    if distance == "mahalanobis":
        raw = _matchit_covariates(data, covariates)
    else:
        raw = _covariate_matrix(data, covariates)
    x = _drop_constant(raw)
    treat = np.asarray(column_series(data, treatment).to_numpy(), dtype=float) > 0
    if distance == "euclidean":
        coords = x
    elif distance == "scaled_euclidean":
        coords = x / _pooled_sd(x, treat)
    elif distance == "mahalanobis":
        coords = _mahalanobis_coordinates(x, treat)
    elif distance == "robust_mahalanobis":
        coords = _robust_mahalanobis_coordinates(x, treat)
    else:
        raise ValueError(distance)
    return coords, np.arange(data.height)


def _probit_probability(data, treatment, covariates):
    design = design_matrix(data, covariates, extra=[treatment])
    y = np.asarray(column_series(data, treatment).gather(design.row_index.tolist()).to_numpy(), dtype=float)
    eta = design.x @ _probit_coefficients(y, design.x)
    return stats_norm_cdf(eta), design.row_index


def _probit_coefficients(y, x, *, tol=1e-10, maxiter=100):
    """Probit regression by IRLS, started from ``glm``'s mean (y + 0.5) / 2.

    Iteration stops when no coefficient moves by more than ``tol`` relative to
    its size, which lands within rounding of the maximum likelihood estimate.
    """
    from scipy.stats import norm

    eps = np.finfo(float).eps
    eta = norm.ppf((y + 0.5) / 2.0)
    mu = norm.cdf(eta)
    coef = np.zeros(x.shape[1])
    for _ in range(maxiter):
        mu_eta = np.maximum(norm.pdf(eta), eps)
        variance = np.clip(mu * (1.0 - mu), eps, None)
        z = eta + (y - mu) / mu_eta
        sw = mu_eta / np.sqrt(variance)
        previous = coef
        coef = np.linalg.lstsq(x * sw[:, None], z * sw, rcond=None)[0]
        eta = x @ coef
        mu = np.clip(norm.cdf(eta), eps, 1.0 - eps)
        if np.all(np.abs(coef - previous) <= tol * (1.0 + np.abs(coef))):
            break
    return coef


def stats_norm_cdf(eta):
    from scipy.stats import norm

    return norm.cdf(eta)


def _align_distance(dist, rows, n):
    if dist.ndim == 2:
        return dist
    if len(dist) == n and np.array_equal(rows, np.arange(n)):
        return dist.astype(float)
    out = np.full(n, np.nan)
    out[np.asarray(rows, dtype=int)] = dist
    return out


def _order_units(units, dist, order, seed):
    if order == "data":
        return units
    if order == "largest":
        return units[np.argsort(-dist[units], kind="mergesort")]
    if order == "smallest":
        return units[np.argsort(dist[units], kind="mergesort")]
    if order == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(units))
        return units[perm]
    raise ValueError("order must be largest, smallest, random, or data")


def _discard_mask(dist, treat, discard):
    mask = np.zeros(len(treat), dtype=bool)
    if discard == "none" or dist.ndim == 2:
        return mask
    finite = np.isfinite(dist)
    t = dist[treat & finite]
    c = dist[~treat & finite]
    if t.size == 0 or c.size == 0:
        return mask
    lo, hi = max(t.min(), c.min()), min(t.max(), c.max())
    outside = finite & ((dist < lo) | (dist > hi))
    if discard == "both":
        mask |= outside
    elif discard == "treated":
        mask |= outside & treat
    elif discard == "control":
        mask |= outside & ~treat
    elif discard != "none":
        raise ValueError("discard must be none, both, treated, or control")
    return mask


def _exact_labels(data, exact):
    if not exact:
        return None
    parts = []
    for col in exact:
        series = column_series(data, col)
        values = series.to_numpy()
        _, inverse = np.unique(values.astype(str), return_inverse=True)
        parts.append(inverse)
    return np.column_stack(parts).dot(np.array([1, *np.cumprod([len(np.unique(p)) for p in parts[:-1]])])) if False else _pack(parts)


def _pack(parts: list[np.ndarray]) -> np.ndarray:
    code = np.zeros(len(parts[0]), dtype=int)
    stride = 1
    for part in parts:
        code = code + (part + 1) * stride
        stride *= int(part.max()) + 2
    return code


def _column_name(data, col: ColumnRef) -> str:
    return col if isinstance(col, str) else column_series(data, col).name


def _normalize_to_counts(weights: np.ndarray, treat: np.ndarray) -> np.ndarray:
    """MatchIt's ``normalize=TRUE``: each arm's positive weights sum to its size."""
    out = np.asarray(weights, dtype=float).copy()
    for flag in (True, False):
        mask = (treat == flag) & (out > 0)
        total = out[mask].sum()
        if total > 0:
            out[mask] *= mask.sum() / total
    return out


def _swap_roles(weights: np.ndarray, treat: np.ndarray) -> np.ndarray:
    return weights


def _pair_distance_ids(dist: np.ndarray, treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    if treated.size == 0:
        return np.zeros(0)
    if dist.ndim != 1:
        diff = dist[control] - dist[treated]
        return np.sqrt(np.sum(diff * diff, axis=1))
    return np.abs(dist[treated] - dist[control])


def _subclass_pair_balance(values, treat, labels, denom: float) -> float:
    """Mean |treated - control| over the subclass product, without building it."""
    values = np.asarray(values, dtype=float)
    total = 0.0
    count = 0
    for label in np.unique(labels):
        if int(label) <= 0:
            continue
        members = np.flatnonzero(labels == label)
        treated = values[members[treat[members]]]
        control = values[members[~treat[members]]]
        if treated.size == 0 or control.size == 0:
            continue
        if not (np.isfinite(treated).all() and np.isfinite(control).all()):
            return float("nan")
        total += _sum_abs_outer(treated, control)
        count += int(treated.size * control.size)
    if count == 0:
        return float("nan")
    if not (denom > 0 and np.isfinite(denom)):
        return float("nan")
    return float((total / count) / denom)


def _sum_abs_outer(treated: np.ndarray, control: np.ndarray) -> float:
    ordered = np.sort(control)
    prefix = np.cumsum(ordered)
    n_control = ordered.size
    total_control = float(prefix[-1])
    at = np.searchsorted(ordered, treated, side="right")
    sum_left = np.where(at > 0, prefix[at - 1], 0.0)
    sum_right = total_control - sum_left
    return float(np.sum(at * treated - sum_left + sum_right - (n_control - at) * treated))


def _pair_distance(dist, pairs: pl.DataFrame) -> np.ndarray:
    if pairs.height == 0:
        return np.zeros(0)
    return _pair_distance_ids(dist, pairs["treated"].to_numpy(), pairs["control"].to_numpy())


def _usable_rows(dist: np.ndarray) -> np.ndarray:
    if dist.ndim == 1:
        return np.isfinite(dist)
    return np.all(np.isfinite(dist), axis=1)


def _gaps(dist: np.ndarray, index: int, others: np.ndarray) -> np.ndarray:
    if dist.ndim == 1:
        return np.abs(dist[index] - dist[others])
    diff = dist[others] - dist[index]
    return np.sqrt(np.sum(diff * diff, axis=1))


def _matchit_covariates(data: pl.DataFrame, names: list[str]) -> np.ndarray:
    """Covariates as MatchIt ``get.covs.matrix.for.dist``.

    A factor is every level as an indicator divided by ``sqrt(2)``. Numeric
    columns stay as they are. A null leaves that row missing.
    """
    n = data.height
    blocks: list[np.ndarray] = []
    indicator = 1.0 / np.sqrt(2.0)
    for name in names:
        series = column_series(data, name)
        if not _is_factor(series):
            blocks.append(np.asarray(series.to_numpy(), dtype=float).reshape(-1, 1))
            continue
        levels = _factor_levels(series, None)
        width = len(levels)
        block = np.full((n, max(width, 1)), np.nan)
        if width == 0:
            blocks.append(block)
            continue
        present = series.is_not_null().to_numpy()
        if not np.any(present):
            blocks.append(block)
            continue
        filled = series.fill_null(series.drop_nulls()[0]) if not bool(present.all()) else series
        codes = _factor_codes(filled, levels)
        rows = np.flatnonzero(present)
        block[rows] = 0.0
        block[rows, codes[rows]] = indicator
        blocks.append(block)
    if not blocks:
        return np.zeros((n, 0))
    return np.column_stack(blocks)


def _covariate_matrix(data: pl.DataFrame, names: list[str]) -> np.ndarray:
    """Numeric columns, with factors expanded as treatment-contrast dummies.

    Rows dropped for a null are left as missing, so they stay unmatched.
    """
    design = design_matrix(data, names, intercept=False)
    out = np.full((data.height, design.x.shape[1]), np.nan)
    out[np.asarray(design.row_index, dtype=int)] = design.x
    return out


def _drop_constant(x: np.ndarray) -> np.ndarray:
    span = np.nanmax(x, axis=0) - np.nanmin(x, axis=0)
    keep = span >= np.sqrt(np.finfo(float).eps)
    if not np.any(keep):
        return x[:, :1]
    return x[:, keep]


def _pooled_sd(x: np.ndarray, treat: np.ndarray) -> np.ndarray:
    """MatchIt pooled standard deviation, proportional contribution."""
    n = x.shape[0]
    groups = int(np.any(treat)) + int(np.any(~treat))
    out = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        col = np.array(x[:, j], dtype=float, copy=True)
        binary = np.all((col == 0) | (col == 1))
        if binary:
            total = 0.0
            for flag in (True, False):
                part = col[treat == flag]
                ni = part.size
                if ni == 0:
                    continue
                sxi = float(part.sum())
                total += sxi * (1.0 - sxi / ni) / n
            out[j] = np.sqrt(total)
        else:
            for flag in (True, False):
                mask = treat == flag
                if np.any(mask):
                    col[mask] -= col[mask].mean()
            out[j] = np.sqrt(np.sum(col**2) / (n - groups))
    out[~np.isfinite(out) | (out == 0)] = 1.0
    return out


def _longdouble_sum(values: np.ndarray) -> np.longdouble:
    """Sequential long-double sum, the order R's ``sum`` uses.

    ``np.sum`` adds in pairs. That sum differs by an ulp once the column is
    a few hundred rows, and the Mahalanobis factor then picks another control.
    ``cumsum`` is a running total, so the last entry is the sequential sum.
    """
    flat = np.asarray(values, dtype=np.longdouble).reshape(-1)
    if flat.size == 0:
        return np.longdouble(0)
    return np.cumsum(flat, dtype=np.longdouble)[-1]


def _r_colmean(values: np.ndarray) -> float:
    """``colMeans``: one long-double sum, then a double."""
    return float(_longdouble_sum(values) / values.shape[0])


def _r_mean(values: np.ndarray) -> float:
    """``mean`` / the two-pass mean inside ``cov``."""
    n = values.shape[0]
    total = _longdouble_sum(values)
    tmp = total / n
    if np.isfinite(float(tmp)):
        adjust = _longdouble_sum(np.asarray(values, dtype=np.longdouble) - tmp)
        tmp = tmp + adjust / n
    return float(tmp)


def _r_scale(x: np.ndarray) -> np.ndarray:
    """``scale``: ``colMeans``, then ``sqrt(sum(v^2) / (n - 1))``.

    The square is a double and ``sum`` accumulates those squares in long
    double, then returns a double. The division and square root stay in
    double. A long-double square root moves one scaled column by an ulp and
    flips the Mahalanobis pivot.
    """
    n, width = x.shape
    center = np.empty(width)
    for column in range(width):
        center[column] = _r_colmean(x[:, column])
    centered = x - center
    scale = np.empty(width)
    denom = max(1, n - 1)
    for column in range(width):
        values = centered[:, column]
        total = _longdouble_sum(values * values)
        scale[column] = np.sqrt(float(total) / denom)
    scale[scale == 0] = 1.0
    return centered / scale


def _pooled_cov(x: np.ndarray, treat: np.ndarray) -> np.ndarray:
    """Pooled within-group covariance, in the order ``cov`` uses.

    Group means are removed with ``mean``, then ``cov`` recenters and sums
    each product in long double. A float64 matrix product changes the
    null-space pivot of a singular factor design and the nearest control.
    Each product is a long-double dot, which adds in the same order as the
    sequential sum. A pairwise reduction does not.
    """
    n, width = x.shape
    centered = np.array(x, dtype=np.float64, copy=True)
    for flag in (True, False):
        rows = np.flatnonzero(treat == flag)
        if rows.size == 0:
            continue
        for column in range(width):
            values = centered[rows, column]
            centered[rows, column] = values - _r_mean(values)
    means = np.empty(width)
    for column in range(width):
        means[column] = _r_mean(centered[:, column])
    dev = np.ascontiguousarray(centered, dtype=np.longdouble)
    dev -= np.asarray(means, dtype=np.longdouble)
    gram = np.empty((width, width))
    n1 = n - 1
    for left in range(width):
        column = dev[:, left]
        for right in range(left + 1):
            total = np.dot(column, dev[:, right])
            value = float(total / n1)
            gram[left, right] = value
            gram[right, left] = value
    groups = int(np.unique(treat).size)
    return gram * (n - 1) / (n - groups)


_REFERENCE_LAPACK = (
    "/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3",
    "/usr/lib/x86_64-linux-gnu/liblapack.so.3",
)
_REFERENCE_BLAS = (
    "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3",
    "/usr/lib/x86_64-linux-gnu/libblas.so.3",
)


def _cdll(paths: tuple[str, ...]):
    import ctypes

    for path in paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _reference_linear_algebra():
    """Reference LAPACK and BLAS, the pair R uses for ``svd`` and ``chol``.

    OpenBLAS moves the tiniest pivot of a singular Mahalanobis factor and
    changes which control is nearest. The handles are cached on the function.
    """
    cached = getattr(_reference_linear_algebra, "cached", None)
    if cached is not None:
        return cached
    lapack = _cdll(_REFERENCE_LAPACK)
    blas = _cdll(_REFERENCE_BLAS)
    if lapack is None or blas is None:
        _reference_linear_algebra.cached = None
        return None
    _reference_linear_algebra.cached = (lapack.dgesdd_, lapack.dpstrf_, blas.dgemm_)
    return _reference_linear_algebra.cached


def _generalized_inverse(sigma: np.ndarray) -> np.ndarray:
    """``MASS::ginv`` with MatchIt's singular-value cutoff.

    The decomposition and the following product go through reference LAPACK
    and BLAS. Another SVD driver leaves a different null-space factor.
    """
    import ctypes

    routines = _reference_linear_algebra()
    matrix = np.array(sigma, dtype=np.float64, order="F")
    if routines is None:
        from scipy.linalg import svd

        u, singular, vt = svd(matrix, full_matrices=False, lapack_driver="gesdd")
        keep = singular > max(1e-8 * float(singular[0]), 0.0) if singular.size else np.array([], dtype=bool)
        if not np.any(keep):
            return np.zeros_like(matrix)
        return vt[keep].T @ ((1.0 / singular[keep])[:, None] * u[:, keep].T)
    dgesdd, _dpstrf, dgemm = routines
    height, width = matrix.shape
    thin = min(height, width)
    singular = np.zeros(thin)
    u = np.zeros((height, thin), order="F")
    vt = np.zeros((thin, width), order="F")
    iwork = np.zeros(8 * thin, dtype=np.int32)
    job = ctypes.c_char(b"S")

    def _call(values: np.ndarray, work: np.ndarray, lwork: int) -> None:
        info = ctypes.c_int(0)
        dgesdd(
            ctypes.byref(job),
            ctypes.byref(ctypes.c_int(height)),
            ctypes.byref(ctypes.c_int(width)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(height)),
            singular.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(height)),
            vt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(thin)),
            work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(lwork)),
            iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(info),
            ctypes.c_size_t(1),
        )
        if info.value != 0:
            raise np.linalg.LinAlgError(f"dgesdd failed with info {info.value}")

    probe = np.zeros(1)
    _call(np.array(matrix, order="F", copy=True), probe, -1)
    _call(matrix, np.zeros(int(probe[0])), int(probe[0]))
    if singular.size == 0:
        return np.array(sigma, dtype=float, copy=True)
    keep = singular > max(1e-8 * float(singular[0]), 0.0)
    if not np.any(keep):
        return np.zeros((height, width))
    left = np.asfortranarray(u[:, keep])
    right = np.asfortranarray(vt.T[:, keep])
    scaled = np.asfortranarray((left * (1.0 / singular[keep])).T)
    inverse = np.zeros((height, height), order="F")
    side = right.shape[0]
    rank = right.shape[1]
    dgemm(
        ctypes.byref(ctypes.c_char(b"N")),
        ctypes.byref(ctypes.c_char(b"N")),
        ctypes.byref(ctypes.c_int(side)),
        ctypes.byref(ctypes.c_int(side)),
        ctypes.byref(ctypes.c_int(rank)),
        ctypes.byref(ctypes.c_double(1.0)),
        right.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(side)),
        scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(rank)),
        ctypes.byref(ctypes.c_double(0.0)),
        inverse.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(side)),
        ctypes.c_size_t(1),
        ctypes.c_size_t(1),
    )
    return inverse


def _pivoted_chol_coordinates(x: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """``tcrossprod(X, chol(inv, pivot=TRUE)[, order(pivot)])``.

    Reference LAPACK's pivoted factor, then reference BLAS for the product.
    The trailing columns after the numerical rank stay in the factor, as in
    MatchIt 4.5.5.
    """
    import ctypes

    routines = _reference_linear_algebra()
    if routines is None:
        from scipy.linalg.lapack import dpstrf

        factor, pivot, _rank, info = dpstrf(np.array(inv, dtype=np.float64, order="F"))
        if info < 0:
            raise np.linalg.LinAlgError("pivoted Cholesky failed")
        return np.asarray(x, dtype=np.float64) @ np.triu(factor)[:, np.argsort(pivot)].T
    _dgesdd, dpstrf, dgemm = routines
    factor = np.array(inv, dtype=np.float64, order="F")
    side = factor.shape[0]
    pivot = np.zeros(side, dtype=np.int32)
    work = np.zeros(2 * side)
    rank = ctypes.c_int(0)
    info = ctypes.c_int(0)
    dpstrf(
        ctypes.byref(ctypes.c_char(b"U")),
        ctypes.byref(ctypes.c_int(side)),
        factor.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(side)),
        pivot.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(rank),
        ctypes.byref(ctypes.c_double(-1.0)),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info),
        ctypes.c_size_t(1),
    )
    if info.value < 0:
        raise np.linalg.LinAlgError("pivoted Cholesky failed")
    unpivoted = np.asfortranarray(np.triu(factor)[:, np.argsort(pivot)])
    rows = np.asfortranarray(np.asarray(x, dtype=np.float64))
    n_rows, width = rows.shape
    out = np.zeros((n_rows, width), order="F")
    dgemm(
        ctypes.byref(ctypes.c_char(b"N")),
        ctypes.byref(ctypes.c_char(b"T")),
        ctypes.byref(ctypes.c_int(n_rows)),
        ctypes.byref(ctypes.c_int(width)),
        ctypes.byref(ctypes.c_int(width)),
        ctypes.byref(ctypes.c_double(1.0)),
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(n_rows)),
        unpivoted.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(width)),
        ctypes.byref(ctypes.c_double(0.0)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ctypes.c_int(n_rows)),
        ctypes.c_size_t(1),
        ctypes.c_size_t(1),
    )
    return out


def _mahalanobize(x: np.ndarray, var: np.ndarray) -> np.ndarray:
    det = float(np.linalg.det(var)) if var.size > 1 else float(var.reshape(-1)[0])
    try:
        inv = np.linalg.inv(var) if det > 1e-8 else np.linalg.pinv(var)
        factor = np.linalg.cholesky(inv)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(var)
        vals, vecs = np.linalg.eigh(inv)
        vals = np.clip(vals, 0, None)
        factor = vecs * np.sqrt(vals)
    return x @ factor


def _mahalanobis_coordinates(x: np.ndarray, treat: np.ndarray) -> np.ndarray:
    """MatchIt 4.5.5 coordinates: scale, pooled covariance, pivoted Cholesky.

    ``det > 1e-8`` uses ``solve``. Otherwise the inverse is ``MASS::ginv``.
    The factor is not truncated after the numerical rank, matching
    ``mahalanobize`` in that release.
    """
    scaled = _r_scale(np.asarray(x, dtype=np.float64))
    var = _pooled_cov(scaled, treat)
    det = float(np.linalg.det(var)) if var.size > 1 else float(var.reshape(-1)[0])
    inv = None
    if det > 1e-8:
        try:
            inv = np.linalg.inv(var)
        except np.linalg.LinAlgError:
            inv = None
    if inv is None:
        inv = _generalized_inverse(var)
    return _pivoted_chol_coordinates(scaled, inv)


def _robust_mahalanobis_coordinates(x: np.ndarray, _treat: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = np.column_stack([rankdata(x[:, j], method="average") for j in range(x.shape[1])])
    var = np.atleast_2d(np.cov(ranks, rowvar=False, ddof=1))
    multiplier = np.std(np.arange(1, x.shape[0] + 1), ddof=1) / np.sqrt(np.diag(var))
    var = var * np.outer(multiplier, multiplier)
    return _mahalanobize(ranks, var)


def _balance(matched: MatchedSample, *, sd_denominator: str = "pooled", binary: str = "std") -> pl.DataFrame:
    treat = np.asarray(matched.data[matched.treatment].to_numpy(), dtype=float) > 0
    if matched.estimand == "ATC":
        focal = ~treat
    else:
        focal = treat
    rows = []
    columns = ["distance", *matched.covariates]
    for name in columns:
        if name == "distance":
            values = matched.distance
        else:
            raw = matched.data[name].to_numpy()
            if not np.issubdtype(np.asarray(raw).dtype, np.number):
                continue
            values = np.asarray(raw, dtype=float)
        is_binary = name != "distance" and np.unique(values[np.isfinite(values)]).size == 2
        if is_binary and binary == "raw":
            denom = 1.0
        else:
            denom = _smd_denominator(values, treat, sd_denominator, is_binary)
        everyone = np.ones(len(focal), dtype=bool)
        all_row = _one_balance(values, focal, everyone, denom=denom)
        matched_row = _one_balance(values, focal, everyone, weights=matched.weights, denom=denom)
        if matched._expand_pairs:
            pair = _subclass_pair_balance(values, treat, matched.subclass, denom)
        else:
            pair = _pair_balance(values, matched.pair_table, denom)
        rows.append({"term": name, **_prefix(all_row, "all"), **_prefix(matched_row, "matched"), "pair_distance": pair})
    return pl.DataFrame(rows)


def _smd_denominator(values, treat, how: str, is_binary: bool) -> float:
    """Standard deviation for the SMD, from the whole sample before matching."""
    finite = np.isfinite(values)

    def var(mask):
        x = values[finite & mask]
        if is_binary:
            # p(1-p) on the 0/1 scale, rescaled to the two observed values.
            lo, hi = np.unique(values[finite])
            p = float(np.mean(x == hi)) if x.size else float("nan")
            return p * (1.0 - p) * (hi - lo) ** 2
        return float(np.var(x, ddof=1)) if x.size >= 2 else float("nan")

    if how == "treated":
        v = var(treat)
    elif how == "control":
        v = var(~treat)
    else:
        v = (var(treat) + var(~treat)) / 2.0
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")


def _prefix(row: dict, name: str) -> dict:
    return {f"{key}_{name}" if key != "n" else f"n_{name}": value for key, value in row.items()}


def _one_balance(values, focal, mask, weights=None, *, denom: float):
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    finite = mask & np.isfinite(values)
    ft = finite & focal & (w > 0)
    fc = finite & ~focal & (w > 0)
    wt, wc = w[ft], w[fc]
    mt = _wmean(values[ft], wt)
    mc = _wmean(values[fc], wc)
    # One denominator from the unmatched sample for both rows, so the change
    # in SMD reflects the change in means, not in spread.
    smd = (mt - mc) / denom if denom > 0 else float("nan")
    vt = _wvar(values[ft], np.ones(ft.sum()))
    vc = _wvar(values[fc], np.ones(fc.sum()))
    ratio = vt / vc if vc > 0 else float("nan")
    e_mean, e_max = _ecdf_diff(values[finite & focal], values[finite & ~focal], w[finite & focal], w[finite & ~focal])
    return {
        "mean_treated": mt,
        "mean_control": mc,
        "smd": smd,
        "variance_ratio": ratio,
        "ecdf_mean": e_mean,
        "ecdf_max": e_max,
        "n": int(mask.sum()),
    }


def _pair_balance(values, pairs: pl.DataFrame, denom: float) -> float:
    if pairs.height == 0:
        return float("nan")
    diff = np.abs(values[pairs["treated"].to_numpy()] - values[pairs["control"].to_numpy()])
    if not (denom > 0 and np.isfinite(denom)):
        return float("nan")
    return float(diff.mean() / denom)


def _wmean(values, weights):
    if values.size == 0 or weights.sum() == 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _wvar(values, weights):
    if values.size < 2:
        return float("nan")
    mean = np.average(values, weights=weights)
    # MatchIt uses the unweighted variance inside a matched group when weights are constant.
    if np.allclose(weights, weights[0]):
        return float(np.var(values, ddof=1))
    return float(np.average((values - mean) ** 2, weights=weights))


def _wsd(values, weights):
    var = _wvar(values, weights)
    return float(np.sqrt(var)) if np.isfinite(var) else float("nan")


def _ecdf_diff(left, right, wl, wr):
    """Mean and max |F_treated - F_control| at the observed values, as ``qqsum``."""
    if left.size == 0 or right.size == 0:
        return float("nan"), float("nan")
    values = np.concatenate([left, right])
    group = np.concatenate([np.ones(left.size), np.zeros(right.size)])
    weights = np.concatenate([wl, wr]).astype(float)
    for flag in (1.0, 0.0):
        mask = group == flag
        total = weights[mask].sum()
        if total > 0:
            weights[mask] /= total
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    signed = weights[order]
    signed[group[order] == group[order][0]] *= -1
    running = np.abs(np.cumsum(signed))
    keep = np.empty(len(ordered), dtype=bool)
    keep[-1] = True
    if len(ordered) > 1:
        keep[:-1] = np.diff(ordered) != 0
    kept = running[keep]
    return float(kept.mean()), float(kept.max())
