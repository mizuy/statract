"""Wilkinson formulas for the column-wise estimators.

The grammar follows R's ``formula``: ``+``, ``-``, ``*``, ``:``, ``/``,
``%in%``, ``^``, ``0`` / ``1``, ``.``, ``I()``, ``offset()``, and calls such
as ``log(x)``. ``(1 | g)`` and ``(1 + x | g)`` are the random-effect extension
used by ``fit_mixed``. Expansion follows ``terms``: term order is by degree,
factor order inside a term is the order names first appear, and treatment
contrasts follow ``model.matrix`` (including the first factor when the
intercept is removed).

``model_matrix`` is the shared entry point. ``fit_ols``, ``fit_glm``, and
``fit_mixed`` accept the same strings. ``cox_ph``, ``accelerated_failure``,
and ``fine_gray`` accept ``Surv(time, status) ~ ...``. The column interface
stays available.
The language, column names, and contrast rules are written up in
``docs/models/formula.md``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from .design import Design, Predictor, _factor_levels, _is_factor, _level_key, column_series

_TOKEN = re.compile(
    r"\s+"
    r"|(?P<in>%in%)"
    r"|(?P<num>\d+(?:\.\d*)?)"
    r"|(?P<op>\|\||[+\-*/:^~()|,])"
    r"|(?P<name>`(?:\\`|[^`])+`|[A-Za-z_.][A-Za-z0-9_.]*)"
)
_FUNCS = {"log", "exp", "sqrt", "log10", "abs", "I"}
_PREC = {"^": 4, "*": 3, "/": 3, "+": 2, "-": 2}


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Col:
    name: str


@dataclass(frozen=True)
class Bin:
    op: str
    left: Arith
    right: Arith


@dataclass(frozen=True)
class Unary:
    op: str
    expr: Arith


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple[Arith, ...]


Arith = Num | Col | Bin | Unary | Call
Term = frozenset[str]


@dataclass(frozen=True)
class RandomEffect:
    """One ``(slopes | group)`` term."""

    intercept: bool
    slopes: tuple[str, ...]
    group: str
    correlated: bool
    slope_exprs: tuple[tuple[str, Arith], ...] = ()


@dataclass(frozen=True)
class SurvResponse:
    """``Surv(time, status)`` or ``Surv(start, stop, status)``."""

    time: str
    event: str
    entry: str | None = None


@dataclass
class FormulaModel:
    """Evaluated formula aligned to ``design.row_index``."""

    design: Design
    y: np.ndarray
    offset: np.ndarray | None
    random_effects: tuple[RandomEffect, ...]
    response_name: str | None
    surv: SurvResponse | None = None
    strata: tuple[str, ...] = ()
    cluster: str | None = None


@dataclass
class _State:
    data: pl.DataFrame
    response_symbols: set[str]
    variables: list[str] = field(default_factory=list)
    computed: dict[str, Arith] = field(default_factory=dict)
    offsets: list[tuple[str, Arith]] = field(default_factory=list)
    random: list[RandomEffect] = field(default_factory=list)
    intercept: bool = True
    parity: bool = True
    surv: SurvResponse | None = None
    strata: list[str] = field(default_factory=list)
    cluster: str | None = None


def model_matrix(formula: str, data: pl.DataFrame) -> FormulaModel:
    """Build the response, fixed-effects matrix, offset, and random effects."""
    if data.height == 0:
        raise ValueError("data has no rows")
    if not isinstance(formula, str) or formula.count("~") < 1:
        raise ValueError("formula must contain '~'")
    response, rhs = _split_formula(formula)
    _check_random_placement(rhs)
    _check_special_placement(rhs)
    surv = _as_surv(response)
    state = _State(data=data, response_symbols=set(_columns_in(response)), surv=surv)
    response_label = _deparse(response)
    _install(state, response_label)
    if isinstance(response, Call) and surv is None:
        state.computed[response_label] = response
    terms = _encode(rhs, state)
    offset_names = {label for label, _expr in state.offsets}
    terms = [term for term in terms if not (term & offset_names)]
    terms = _by_degree(terms)
    dropped = False
    kept: list[Term] = []
    for term in terms:
        if term == frozenset({response_label}):
            dropped = True
            continue
        kept.append(term)
    if dropped:
        warnings.warn(
            "the response appeared on the right-hand side and was dropped",
            UserWarning,
            stacklevel=2,
        )
    if not kept and not state.intercept and not state.random and state.surv is None:
        raise ValueError("formula has no columns")
    return _materialize(response, response_label, kept, state)


def formula_model_matrix(formula: str, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Response vector and fixed-effects matrix. Random effects are rejected."""
    built = model_matrix(formula, data)
    if built.random_effects:
        raise ValueError("random effects belong in fit_mixed, for example (1 | group)")
    reject_survival_syntax(built)
    return built.y, built.design.x


def reject_survival_syntax(built: FormulaModel) -> None:
    """``Surv`` formulas are for the survival estimators."""
    if built.surv is not None or built.strata or built.cluster is not None:
        raise ValueError(
            "Surv(), strata(), and cluster() belong in cox_ph, conditional_logit, accelerated_failure, or fine_gray"
        )


def is_formula(value: object) -> bool:
    return isinstance(value, str) and "~" in value


def rebuild_formula_design(
    data: pl.DataFrame,
    design: Design,
    *,
    extra: Sequence[str] | None = None,
) -> Design:
    """Rebuild a Wilkinson design on new rows."""
    assert design.recipes is not None
    computed = design.computed or {}
    variables = sorted(
        {name for recipe in design.recipes for name, _level in recipe if name not in computed}
    )
    computed_inputs: list[str] = []
    for expr in computed.values():
        computed_inputs.extend(_columns_in(expr))
    needed = list(dict.fromkeys([*variables, *computed_inputs, *(extra or ())]))
    keep = np.ones(data.height, dtype=bool)
    cache: dict[str, np.ndarray] = {}
    for name in needed:
        series = column_series(data, name)
        keep &= series.is_not_null().to_numpy()
        cache[name] = series.to_numpy()
    levels = design.factor_levels or {}
    for name, allowed in levels.items():
        if name not in cache:
            series = column_series(data, name)
            keep &= series.is_not_null().to_numpy()
            cache[name] = series.to_numpy()
        keys = [_cell_key(value) for value in _kept(cache[name], keep)]
        unknown = sorted({key for key in keys if key not in allowed})
        if unknown:
            raise ValueError(f"factor {name!r} has unseen levels {unknown}")
    computed_values = {label: _eval_arith(expr, cache, keep) for label, expr in computed.items()}
    columns = []
    for recipe in design.recipes:
        column = np.ones(int(keep.sum()), dtype=float)
        for name, level in recipe:
            if level is None:
                column = column * computed_values.get(name, _numeric_kept(cache[name], keep))
            else:
                keys = [_cell_key(value) for value in _kept(cache[name], keep)]
                column = column * np.asarray([key == level for key in keys], dtype=float)
        columns.append(column)
    x = np.column_stack(columns) if columns else np.zeros((int(keep.sum()), 0))
    return Design(
        x=x,
        names=list(design.names),
        row_index=np.flatnonzero(keep).astype(np.int64),
        predictors=list(design.predictors),
        intercept=design.intercept,
        recipes=design.recipes,
        factor_levels=design.factor_levels,
        computed=design.computed,
    )


def _materialize(response: Arith, response_label: str, terms: list[Term], state: _State) -> FormulaModel:
    inputs: set[str] = set(state.response_symbols)
    for term in terms:
        inputs.update(term)
    for _label, expr in state.offsets:
        inputs.update(_columns_in(expr))
    inputs.update(state.strata)
    if state.cluster is not None:
        inputs.add(state.cluster)
    for effect in state.random:
        inputs.add(effect.group)
        expressed = {name for name, _expr in effect.slope_exprs}
        for name in effect.slopes:
            if name not in expressed:
                inputs.add(name)
        for _name, expr in effect.slope_exprs:
            inputs.update(_columns_in(expr))
    computed = {
        name: expr
        for name, expr in state.computed.items()
        if name != response_label and any(name in term for term in terms)
    }
    inputs -= set(computed)
    inputs -= {label for label, _expr in state.offsets}
    inputs.discard(response_label)
    keep = np.ones(state.data.height, dtype=bool)
    cache: dict[str, np.ndarray] = {}
    series_of: dict[str, pl.Series] = {}
    for name in inputs:
        series = _require_column(state.data, name)
        series_of[name] = series
        keep &= series.is_not_null().to_numpy()
        cache[name] = series.to_numpy()
    for expr in list(computed.values()) + [expr for _label, expr in state.offsets] + [response]:
        for name in _columns_in(expr):
            if name in cache:
                continue
            series = _require_column(state.data, name)
            series_of[name] = series
            keep &= series.is_not_null().to_numpy()
            cache[name] = series.to_numpy()
    for effect in state.random:
        for name, expr in effect.slope_exprs:
            for col in _columns_in(expr):
                if col in cache:
                    continue
                series = _require_column(state.data, col)
                keep &= series.is_not_null().to_numpy()
                cache[col] = series.to_numpy()
    if state.surv is None:
        y = _eval_arith(response, cache, keep)
    else:
        y = _event_values(cache[state.surv.event], keep)
    offset = None
    if state.offsets:
        offset = np.zeros(int(keep.sum()), dtype=float)
        for _label, expr in state.offsets:
            offset = offset + _eval_arith(expr, cache, keep)
    used = set().union(*terms) if terms else set()
    factor_levels: dict[str, tuple[str, ...]] = {}
    for name in state.variables:
        if name not in used:
            continue
        series = series_of.get(name)
        if series is None and name in state.data.columns:
            series = state.data.get_column(name)
            series_of[name] = series
        if series is None or not _formula_is_factor(series):
            continue
        observed = _kept(cache[name], keep) if name in cache else series.drop_nulls().to_numpy()
        factor_levels[name] = _formula_levels(series, observed)
    codes = _contrast_codes(terms, state.variables, state.intercept, factor_levels)
    names: list[str] = ["(Intercept)"] if state.intercept else []
    recipes: list[tuple[tuple[str, str | None], ...]] = [()] if state.intercept else []
    columns: list[np.ndarray] = [np.ones(int(keep.sum()), dtype=float)] if state.intercept else []
    predictors: list[Predictor] = []
    for index, term in enumerate(terms):
        pieces = _pieces_for(term, state.variables, codes[index], factor_levels, computed)
        for combo in _cartesian(pieces):
            label = ":".join(_piece_name(symbol, level) for symbol, level in combo)
            column = np.ones(int(keep.sum()), dtype=float)
            for symbol, level in combo:
                if level is None:
                    values = _symbol_values(symbol, computed, cache, keep)
                else:
                    keys = [_cell_key(value) for value in _kept(cache[symbol], keep)]
                    values = np.asarray([key == level for key in keys], dtype=float)
                column = column * values
            names.append(label)
            recipes.append(tuple(combo))
            columns.append(column)
        if len(term) == 1:
            symbol = next(var for var in state.variables if var in term)
            if symbol in factor_levels:
                levels = factor_levels[symbol]
                predictors.append(
                    Predictor(
                        name=symbol,
                        kind="factor",
                        levels=levels,
                        reference=levels[0] if levels else None,
                    )
                )
            elif symbol not in computed:
                predictors.append(Predictor(name=symbol, kind="numeric"))
    if not columns:
        if not state.random and state.surv is None:
            raise ValueError("formula has no columns")
        x = np.zeros((int(keep.sum()), 0))
    else:
        x = np.column_stack(columns)
    design = Design(
        x=x,
        names=names,
        row_index=np.flatnonzero(keep).astype(np.int64),
        predictors=predictors,
        intercept=state.intercept,
        recipes=tuple(recipes),
        factor_levels=factor_levels or None,
        computed=computed or None,
    )
    response_name = response.name if isinstance(response, Col) else None
    return FormulaModel(
        design=design,
        y=y,
        offset=offset,
        random_effects=tuple(state.random),
        response_name=response_name,
        surv=state.surv,
        strata=tuple(state.strata),
        cluster=state.cluster,
    )


def _contrast_codes(
    terms: list[Term],
    variables: list[str],
    intercept: bool,
    factor_levels: dict[str, tuple[str, ...]],
) -> list[dict[str, int]]:
    """1 keeps treatment contrasts. 2 keeps every level."""
    codes: list[dict[str, int]] = []
    for index, term in enumerate(terms):
        code: dict[str, int] = {}
        for var in term:
            margin = term - {var}
            if not margin or any(margin <= prev for prev in terms[:index]):
                code[var] = 1
            else:
                code[var] = 2
        codes.append(code)
    if not intercept:
        for index, term in enumerate(terms):
            for var in variables[1:]:
                if var in factor_levels and var in term:
                    codes[index][var] = 2
                    return codes
    return codes


def _encode(node: object, state: _State) -> list[Term]:
    if node is None:
        return []
    if isinstance(node, Col):
        if node.name == ".":
            return _dot(state)
        _install(state, node.name)
        return [frozenset({node.name})]
    if isinstance(node, Num):
        if node.value == 1:
            state.intercept = state.parity
            return []
        if node.value == 0:
            state.intercept = not state.parity
            return []
        raise ValueError("only 0 and 1 are numeric literals in a formula; wrap arithmetic in I()")
    if isinstance(node, Call):
        if node.fn == "Surv":
            raise ValueError("Surv() is the response, for example Surv(time, status) ~ age + sex")
        if node.fn == "strata":
            if not node.args:
                raise ValueError("strata() needs a column")
            for arg in node.args:
                if not isinstance(arg, Col) or arg.name == ".":
                    raise ValueError("strata() takes column names")
                _require_column(state.data, arg.name)
                if arg.name not in state.strata:
                    state.strata.append(arg.name)
            return []
        if node.fn == "cluster":
            if len(node.args) != 1 or not isinstance(node.args[0], Col) or node.args[0].name == ".":
                raise ValueError("cluster() takes one column name")
            name = node.args[0].name
            _require_column(state.data, name)
            if state.cluster is not None and state.cluster != name:
                raise ValueError("cluster() can be named once")
            state.cluster = name
            return []
        if node.fn == "offset":
            if len(node.args) != 1:
                raise ValueError("offset() takes one expression")
            label = _deparse(node)
            _install(state, label)
            state.offsets.append((label, node.args[0]))
            return [frozenset({label})]
        if node.fn not in _FUNCS:
            raise ValueError(f"function {node.fn}() is not available in formulas")
        if len(node.args) != 1:
            raise ValueError(f"{node.fn}() takes one expression")
        label = _deparse(node)
        _install(state, label)
        state.computed[label] = node
        return [frozenset({label})]
    if isinstance(node, tuple) and node and node[0] == "random":
        _op, inner, group, correlated = node
        state.random.append(_random_effect(inner, str(group), bool(correlated), state.data))
        return []
    if isinstance(node, tuple):
        op = node[0]
        if op == "+":
            return _trim(_encode(node[1], state) + _encode(node[2], state))
        if op == "u-":
            return _delete(None, node[1], state)
        if op == "-":
            return _delete(node[1], node[2], state)
        if op == "*":
            left = _encode(node[1], state)
            right = _encode(node[2], state)
            crossed = [a | b for a in left for b in right]
            return _trim(left + right + crossed)
        if op == ":":
            left = _encode(node[1], state)
            right = _encode(node[2], state)
            return _trim([a | b for a in left for b in right])
        if op == "in":
            left = _encode(node[1], state)
            right = _encode(node[2], state)
            common: Term = frozenset()
            for term in right:
                common = common | term
            return _trim([common | term for term in left])
        if op == "/":
            left = _encode(node[1], state)
            right = _encode(node[2], state)
            common = frozenset()
            for term in left:
                common = common | term
            return _trim(left + [common | term for term in right])
        if op == "^":
            power = node[2]
            if not isinstance(power, int) or power <= 1:
                raise ValueError("formula power must be an integer greater than 1")
            left = _encode(node[1], state)
            right = left
            produced = left
            for _ in range(1, power):
                produced = _trim([a | b for a in left for b in right])
                right = produced
            return produced
    raise ValueError(f"cannot evaluate formula node {node!r}")


def _delete(left_node: object, right_node: object, state: _State) -> list[Term]:
    left = [] if left_node is None else _encode(left_node, state)
    state.parity = not state.parity
    right = _encode(right_node, state)
    state.parity = not state.parity
    for term in right:
        if not term:
            state.intercept = False
    drop = set(right)
    return [term for term in left if term not in drop]


def _trim(terms: list[Term]) -> list[Term]:
    seen: set[Term] = set()
    kept: list[Term] = []
    for term in terms:
        if not term or term in seen:
            continue
        seen.add(term)
        kept.append(term)
    return kept


def _by_degree(terms: list[Term]) -> list[Term]:
    return sorted(terms, key=len)


def _dot(state: _State) -> list[Term]:
    terms: list[Term] = []
    seen: set[str] = set()
    for name in state.data.columns:
        if name in state.response_symbols or name in seen:
            continue
        seen.add(name)
        _install(state, name)
        terms.append(frozenset({name}))
    return terms


def _install(state: _State, name: str) -> None:
    if name not in state.variables:
        state.variables.append(name)


def _random_effect(inner: object, group: str, correlated: bool, data: pl.DataFrame) -> RandomEffect:
    if not group or group in {".", "|", "||"}:
        raise ValueError("the grouping term must be a column name")
    state = _State(data=data, response_symbols=set())
    terms = _by_degree(_encode(inner, state))
    slopes: list[str] = []
    exprs: list[tuple[str, Arith]] = []
    for term in terms:
        ordered = [var for var in state.variables if var in term]
        if len(ordered) != 1:
            raise ValueError("random slopes must be single columns")
        name = ordered[0]
        if name in state.computed and state.computed[name].fn == "offset":
            raise ValueError("offset() is not a random slope")
        slopes.append(name)
        if name in state.computed:
            exprs.append((name, state.computed[name]))
    if state.offsets:
        raise ValueError("offset() is not a random slope")
    return RandomEffect(
        intercept=state.intercept,
        slopes=tuple(slopes),
        group=group,
        correlated=correlated,
        slope_exprs=tuple(exprs),
    )


def _as_surv(response: Arith) -> SurvResponse | None:
    if not isinstance(response, Call) or response.fn != "Surv":
        return None
    names: list[str] = []
    for arg in response.args:
        if not isinstance(arg, Col) or arg.name == ".":
            raise ValueError("Surv() takes column names, for example Surv(time, status)")
        names.append(arg.name)
    if len(names) == 2:
        return SurvResponse(time=names[0], event=names[1])
    if len(names) == 3:
        return SurvResponse(time=names[1], event=names[2], entry=names[0])
    raise ValueError("Surv() takes Surv(time, status) or Surv(start, stop, status)")


def _event_values(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    kept = _kept(values, keep)
    try:
        return np.asarray(kept, dtype=float)
    except (TypeError, ValueError):
        return np.zeros(int(np.asarray(keep).sum()), dtype=float)


def _contains_call(node: object, names: set[str]) -> bool:
    if isinstance(node, Call):
        return node.fn in names or any(_contains_call(arg, names) for arg in node.args)
    if isinstance(node, Unary):
        return _contains_call(node.expr, names)
    if isinstance(node, Bin):
        return _contains_call(node.left, names) or _contains_call(node.right, names)
    if isinstance(node, tuple):
        return any(_contains_call(child, names) for child in node[1:])
    return False


def _check_special_placement(node: object) -> None:
    specials = {"strata", "cluster"}
    if not _contains_call(node, specials):
        return
    if isinstance(node, Call) and node.fn in specials:
        return
    if isinstance(node, tuple) and node and node[0] in {"+", "u-"}:
        if node[0] == "u-" and _contains_call(node[1], specials):
            raise ValueError("strata() and cluster() only combine with other terms using '+'")
        for child in node[1:]:
            _check_special_placement(child)
        return
    if isinstance(node, tuple) and node[0] == "-":
        if _contains_call(node[2], specials):
            raise ValueError("cannot subtract strata() or cluster()")
        _check_special_placement(node[1])
        return
    raise ValueError("strata() and cluster() only combine with other terms using '+'")


def _check_random_placement(node: object) -> None:
    if not _contains_random(node):
        return
    if isinstance(node, tuple) and node and node[0] == "random":
        return
    if isinstance(node, tuple) and node[0] in {"+", "u-"}:
        if node[0] == "u-" and _contains_random(node[1]):
            raise ValueError("a random effect only combines with other terms using '+'")
        for child in node[1:]:
            _check_random_placement(child)
        return
    if isinstance(node, tuple) and node[0] == "-":
        if _contains_random(node[2]):
            raise ValueError("cannot subtract a random effect")
        _check_random_placement(node[1])
        return
    raise ValueError("a random effect only combines with other terms using '+'")


def _contains_random(node: object) -> bool:
    if isinstance(node, tuple) and node and node[0] == "random":
        return True
    if isinstance(node, tuple):
        return any(_contains_random(child) for child in node[1:])
    return False


def _formula_is_factor(series: pl.Series) -> bool:
    return series.dtype == pl.Boolean or _is_factor(series)


def _formula_levels(series: pl.Series, observed: np.ndarray) -> tuple[str, ...]:
    if series.dtype == pl.Boolean:
        return ("FALSE", "TRUE")
    if isinstance(series.dtype, (pl.Enum, pl.Categorical)):
        levels = _factor_levels(series, None)
    else:
        levels = sorted({_cell_key(value) for value in observed})
    if len(levels) < 2:
        raise ValueError(f"factor {series.name!r} needs at least two levels after dropping nulls")
    return tuple(levels)


def _cell_key(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    return _level_key(value)


def _cartesian(
    pieces: list[list[tuple[str, str | None]]],
) -> list[tuple[tuple[str, str | None], ...]]:
    combos: list[tuple[tuple[str, str | None], ...]] = [()]
    for group in reversed(pieces):
        combos = [(*tail, head) for tail in combos for head in group]
    return [tuple(reversed(combo)) for combo in combos]


def _piece_name(symbol: str, level: str | None) -> str:
    return symbol if level is None else f"{symbol}{level}"


def _symbol_values(
    symbol: str,
    computed: dict[str, Arith],
    cache: dict[str, np.ndarray],
    keep: np.ndarray,
) -> np.ndarray:
    if symbol in computed:
        return _eval_arith(computed[symbol], cache, keep)
    return _numeric_kept(cache[symbol], keep)


def _numeric_kept(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.asarray(_kept(values, keep), dtype=float)


def _kept(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.asarray(values)[keep]


def _eval_arith(expr: Arith, cache: dict[str, np.ndarray], keep: np.ndarray) -> np.ndarray:
    if isinstance(expr, Num):
        return np.full(int(keep.sum()), expr.value, dtype=float)
    if isinstance(expr, Col):
        if expr.name == ".":
            raise ValueError("'.' is only valid as its own formula term")
        if expr.name not in cache:
            raise KeyError(f"column {expr.name!r} is not in the frame")
        return _numeric_kept(cache[expr.name], keep)
    if isinstance(expr, Unary):
        value = _eval_arith(expr.expr, cache, keep)
        return value if expr.op == "+" else -value
    if isinstance(expr, Bin):
        left = _eval_arith(expr.left, cache, keep)
        right = _eval_arith(expr.right, cache, keep)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "^":
            return left**right
        raise ValueError(f"unknown arithmetic operator {expr.op}")
    if isinstance(expr, Call):
        if len(expr.args) != 1:
            raise ValueError(f"{expr.fn}() takes one expression")
        arg = _eval_arith(expr.args[0], cache, keep)
        if expr.fn == "I":
            return arg
        func = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "log10": np.log10, "abs": np.abs}[expr.fn]
        return np.asarray(func(arg), dtype=float)
    raise ValueError(f"cannot evaluate {expr!r}")


def _columns_in(expr: Arith) -> list[str]:
    if isinstance(expr, Col):
        return [] if expr.name == "." else [expr.name]
    if isinstance(expr, Num):
        return []
    if isinstance(expr, Unary):
        return _columns_in(expr.expr)
    if isinstance(expr, Bin):
        return _columns_in(expr.left) + _columns_in(expr.right)
    if isinstance(expr, Call):
        names: list[str] = []
        for arg in expr.args:
            names.extend(_columns_in(arg))
        return names
    return []


def _deparse(expr: Arith) -> str:
    return _deparse_at(expr, 0, False)


def _deparse_at(expr: Arith, parent: int, right: bool) -> str:
    if isinstance(expr, Num):
        text = str(expr.value)
        return text[:-2] if text.endswith(".0") else text
    if isinstance(expr, Col):
        return expr.name
    if isinstance(expr, Unary):
        body = _deparse_at(expr.expr, 5, False)
        text = f"{expr.op}{body}"
        return f"({text})" if parent > 5 or (parent == 5 and right) else text
    if isinstance(expr, Bin):
        prec = _PREC[expr.op]
        left = _deparse_at(expr.left, prec, False)
        child_right = _deparse_at(expr.right, prec, True)
        if expr.op in {"+", "-"}:
            text = f"{left} {expr.op} {child_right}"
        elif expr.op == "*":
            text = f"{left} * {child_right}"
        elif expr.op == "/":
            text = f"{left}/{child_right}"
        else:
            text = f"{left}^{child_right}"
        # ``^`` is right-associative. The others group to the left, so a same-
        # precedence expression on the right needs parentheses.
        if expr.op == "^":
            needs = parent > prec or (parent == prec and not right)
        else:
            needs = parent > prec or (parent == prec and right)
        return f"({text})" if needs else text
    if isinstance(expr, Call):
        args = ", ".join(_deparse_at(arg, 0, False) for arg in expr.args)
        return f"{expr.fn}({args})"
    raise ValueError(f"cannot deparse {expr!r}")


def _require_column(data: pl.DataFrame, name: str) -> pl.Series:
    if name not in data.columns:
        raise KeyError(f"column {name!r} is not in the frame")
    return data.get_column(name)


class _Parser:
    def __init__(self, formula: str):
        self.tokens = _tokenize(formula)
        self.index = 0

    def parse_rhs(self) -> object:
        if self._peek() is None:
            raise ValueError("formula is missing a right-hand side")
        node = self._parse_sum()
        if self._peek() is not None:
            raise ValueError(f"unexpected {self._peek()!r} in formula")
        return node

    def _parse_sum(self) -> object:
        if self._peek() == "+":
            self._pop()
            node: object = self._parse_product()
        elif self._peek() == "-":
            self._pop()
            node = ("u-", self._parse_product())
        else:
            node = self._parse_product()
        while self._peek() in {"+", "-"}:
            op = self._pop()
            node = (op, node, self._parse_product())
        return node

    def _parse_product(self) -> object:
        node = self._parse_in()
        while self._peek() in {"*", "/"}:
            op = self._pop()
            node = (op, node, self._parse_in())
        return node

    def _parse_in(self) -> object:
        node = self._parse_colon()
        while self._peek() == "%in%":
            self._pop()
            node = ("in", node, self._parse_colon())
        return node

    def _parse_colon(self) -> object:
        node = self._parse_power()
        while self._peek() == ":":
            self._pop()
            node = (":", node, self._parse_power())
        return node

    def _parse_power(self) -> object:
        node = self._parse_primary()
        if self._peek() == "^":
            self._pop()
            number = self._parse_power_int()
            node = ("^", node, number)
        return node

    def _parse_power_int(self) -> int:
        if self._peek() == "(":
            self._pop()
            number = self._parse_power_int()
            self._expect(")")
            return number
        token = self._pop()
        if not isinstance(token, str) or not re.fullmatch(r"\d+", token):
            raise ValueError("formula power must be an integer greater than 1")
        return int(token)

    def _parse_primary(self) -> object:
        tok = self._pop()
        if tok is None:
            raise ValueError("formula ended early")
        if tok == "(":
            if self._bar_ahead():
                return self._parse_random()
            node = self._parse_sum()
            self._expect(")")
            return node
        if isinstance(tok, str) and re.fullmatch(r"\d+(?:\.\d+)?", tok):
            return Num(float(tok))
        if isinstance(tok, str) and tok.startswith("`"):
            return Col(_unquote(tok))
        if isinstance(tok, str) and re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", tok):
            if self._peek() == "(":
                return self._parse_call(tok)
            return Col(tok)
        raise ValueError(f"unexpected {tok!r} in formula")

    def _parse_call(self, name: str) -> Call:
        self._expect("(")
        args: list[Arith] = []
        if self._peek() != ")":
            args.append(self._parse_arith())
            while self._peek() == ",":
                self._pop()
                args.append(self._parse_arith())
        self._expect(")")
        return Call(name, tuple(args))

    def _parse_random(self) -> tuple:
        inner = self._parse_sum()
        correlated = True
        if self._peek() == "||":
            self._pop()
            correlated = False
        elif self._peek() == "|":
            self._pop()
        else:
            raise ValueError("random-effect term is missing '|'")
        group = self._pop()
        if not isinstance(group, str) or group in {"|", "||", "(", ")"}:
            raise ValueError("the grouping term must be a column name")
        if group.startswith("`"):
            group = _unquote(group)
        self._expect(")")
        return ("random", inner, group, correlated)

    def _parse_arith(self) -> Arith:
        return self._parse_arith_sum()

    def _parse_arith_sum(self) -> Arith:
        node = self._parse_arith_product()
        while self._peek() in {"+", "-"}:
            op = self._pop()
            node = Bin(op, node, self._parse_arith_product())
        return node

    def _parse_arith_product(self) -> Arith:
        node = self._parse_arith_power()
        while self._peek() in {"*", "/"}:
            op = self._pop()
            node = Bin(op, node, self._parse_arith_power())
        return node

    def _parse_arith_power(self) -> Arith:
        node = self._parse_arith_unary()
        if self._peek() == "^":
            self._pop()
            node = Bin("^", node, self._parse_arith_unary())
        return node

    def _parse_arith_unary(self) -> Arith:
        if self._peek() in {"+", "-"}:
            op = self._pop()
            return Unary(op, self._parse_arith_unary())
        return self._parse_arith_primary()

    def _parse_arith_primary(self) -> Arith:
        tok = self._pop()
        if tok == "(":
            node = self._parse_arith()
            self._expect(")")
            return node
        if isinstance(tok, str) and re.fullmatch(r"\d+(?:\.\d+)?", tok):
            return Num(float(tok))
        if isinstance(tok, str) and tok.startswith("`"):
            return Col(_unquote(tok))
        if isinstance(tok, str) and re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", tok):
            if self._peek() == "(":
                return self._parse_call(tok)
            return Col(tok)
        raise ValueError(f"unexpected {tok!r} in an expression")

    def _bar_ahead(self) -> bool:
        depth = 0
        for tok in self.tokens[self.index :]:
            if tok == "(":
                depth += 1
            elif tok == ")":
                if depth == 0:
                    return False
                depth -= 1
            elif tok in {"|", "||"} and depth == 0:
                return True
        return False

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _pop(self) -> str | None:
        tok = self._peek()
        if tok is not None:
            self.index += 1
        return tok

    def _expect(self, token: str) -> None:
        got = self._pop()
        if got != token:
            raise ValueError(f"expected {token!r}, got {got!r}")


def _split_formula(formula: str) -> tuple[Arith, object]:
    tokens = _tokenize(formula)
    depth = 0
    split_at = None
    for i, tok in enumerate(tokens):
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif tok == "~" and depth == 0:
            if split_at is not None:
                raise ValueError("formula must contain one '~'")
            split_at = i
    if split_at is None:
        raise ValueError("formula must contain '~'")
    left = _Parser("")
    left.tokens = tokens[:split_at]
    right = _Parser("")
    right.tokens = tokens[split_at + 1 :]
    if not left.tokens:
        raise ValueError("formula is missing a response")
    response = left._parse_arith()
    if left._peek() is not None:
        raise ValueError("the response must be a column or a function of columns")
    return response, right.parse_rhs()


def _tokenize(formula: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(formula):
        match = _TOKEN.match(formula, pos)
        if match is None:
            raise ValueError(f"unexpected {formula[pos]!r} in formula")
        pos = match.end()
        if match.group(0).isspace():
            continue
        kind = match.lastgroup
        if kind == "in":
            tokens.append("%in%")
        elif kind == "num":
            tokens.append(match.group("num"))
        elif kind == "op":
            tokens.append(match.group("op"))
        else:
            tokens.append(match.group("name"))
    return tokens


def _unquote(token: str) -> str:
    return token[1:-1].replace("\\`", "`").replace("\\\\", "\\")


def _pieces_for(
    term: Term,
    variables: list[str],
    code: dict[str, int],
    factor_levels: dict[str, tuple[str, ...]],
    computed: dict[str, Arith],
) -> list[list[tuple[str, str | None]]]:
    pieces: list[list[tuple[str, str | None]]] = []
    for symbol in variables:
        if symbol not in term:
            continue
        if symbol in computed or symbol not in factor_levels:
            pieces.append([(symbol, None)])
            continue
        levels = factor_levels[symbol]
        if len(levels) < 2:
            raise ValueError(f"factor {symbol!r} needs at least two levels after dropping nulls")
        chosen = levels[1:] if code.get(symbol, 1) == 1 else levels
        if not chosen:
            raise ValueError(f"factor {symbol!r} needs at least two levels after dropping nulls")
        pieces.append([(symbol, level) for level in chosen])
    return pieces
