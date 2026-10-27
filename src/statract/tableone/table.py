"""Table One generation for medical research studies.

This module provides functionality to create Table One, a standard format for
presenting baseline characteristics in medical research papers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import NamedTuple, TypeAlias, TypedDict

import polars as pl
from great_tables import GT, loc, style as gt_style, html

import numpy as np

from . import agg
from . import stat

logger = getLogger(__name__)

TableOneParam: TypeAlias = NamedTuple(
    "TableOneParam",
    [("column", pl.Expr), ("aggfunc", agg.AggFunc | str | None), ("statfunc", stat.StatFunc | str | None)],
)


class TableOneError(Exception):
    """Base exception for tableone module."""


class ColumnNotFoundError(TableOneError):
    """Raised when a specified column is not found in the dataframe."""


class InvalidParameterError(TableOneError):
    """Raised when invalid parameters are provided."""


class TableOneStyle(TypedDict, total=False):
    """Styling options for `tableone`.

    All fields are optional with default values.
    """

    table_width: str
    font_size: str
    column_labels_padding_horizontal: str
    data_row_padding: str
    indent_display: str


def _default_tableone_style() -> TableOneStyle:
    """Return default styling options for `tableone`."""
    return {
        "table_width": "300px",
        "font_size": "12px",
        "column_labels_padding_horizontal": "8px",
        "data_row_padding": "6px",
        "indent_display": "　",  # full-width space
    }


def _tableone_group_counts(df: pl.DataFrame, hue: list[pl.Expr] | None) -> dict[str, int]:
    """Return {group_key: n} where group_key matches tableone's column keys."""
    if hue is None:
        return {"All": df.height}

    grouped = df.group_by(hue, maintain_order=True).len()
    counts: dict[str, int] = {}
    for row in grouped.iter_rows(named=True):
        counts[_get_hue_key(hue, row)] = int(row["len"])
    return counts


def _merge_tableone_style(style: TableOneStyle | None) -> TableOneStyle:
    default_style = _default_tableone_style()
    if style is None:
        return default_style
    return {**default_style, **style}


def _reorder_table_columns(
    table: pl.DataFrame,
    column_order: Sequence[str] | None,
) -> pl.DataFrame:
    if not column_order:
        return table
    preferred = ["name", *[c for c in column_order if c in table.columns and c != "name"]]
    rest = [c for c in table.columns if c not in preferred]
    return table.select(preferred + rest)


def tableone_gt_from_frame(
    table: pl.DataFrame,
    *,
    style: TableOneStyle | None = None,
    group_labels: dict[str, str] | None = None,
    column_order: Sequence[str] | None = None,
    column_labels: dict[str, str] | None = None,
    column_spanners: Sequence[tuple[str, Sequence[str]]] | None = None,
    source_df: pl.DataFrame | None = None,
    hue: str | pl.Expr | list[pl.Expr] | list[str] | None = None,
) -> GT:
    """Turn a DataFrame like `tableone_raw` output into a styled `great_tables.GT`.

    When every ``name`` has the form ``"Group | item"``, each group becomes a
    bold heading row and its items (and their category levels) are indented
    under it.

    The column order is set by reordering the DataFrame with ``column_order`` before building the GT.
    ``cols_move_to_start`` is not used because it pushes the stub (``name``) to the right edge.

    Args:
        table: A frame with a ``name`` column and one column per group, such as
            `tableone_raw` output or two of them joined side by side.
        style: Overrides for font size, table width, and padding.
        group_labels: Header label for each value column. When set, the
            ``n`` headers from ``source_df`` are not added.
        column_order: Order of the value columns.
        column_labels: Header labels for some columns, applied last (after
            ``group_labels`` or the ``n`` headers).
        column_spanners: ``(label, columns)`` pairs. Each puts one spanner
            label over those columns; names not in the table are skipped.
        source_df: The data the table came from. With ``hue``, it adds the
            group sizes to the headers, as `tableone` does.
        hue: Group column(s) of ``source_df``.
    """
    if "name" not in table.columns:
        raise InvalidParameterError("table must have a 'name' column")

    style = _merge_tableone_style(style)
    table = _reorder_table_columns(table, column_order)
    result_visible = table.with_columns(
        pl.col("name").str.replace("^  ", style["indent_display"], literal=False).alias("name"),
    )
    # ``Group | item`` はグループ名だけの見出し行にし、項目は字下げする。
    has_row_group = bool(
        result_visible.height
        and result_visible.select(pl.col("name").str.contains(" | ", literal=True).all()).item()
    )
    if has_row_group:
        split = pl.col("name").str.split_exact(" | ", 1)
        item = split.struct.field("field_1")
        # カテゴリの要素（``"  "`` 始まり）は、変数名の行よりさらに一段下げる。
        child = item.str.starts_with("  ")
        indent = style["indent_display"]
        result_visible = result_visible.with_columns(
            split.struct.field("field_0").alias("row_group"),
            pl.when(child)
            .then(pl.lit(indent * 2) + item.str.strip_chars_start())
            .otherwise(pl.lit(indent) + item)
            .alias("name"),
        )

    labels: dict[str, str] = {"name": ""}
    value_cols = [c for c in result_visible.columns if c not in {"name", "row_group"}]
    if group_labels is not None:
        for col in value_cols:
            labels[col] = group_labels.get(col, col)
    elif source_df is not None:
        normalized_hue = _normalize_hue(hue)
        counts = _tableone_group_counts(source_df, normalized_hue)
        for col in value_cols:
            if col == "All":
                labels[col] = html(f"All<br>(N={source_df.height})")
            else:
                n = counts.get(col)
                labels[col] = html(f"{col}<br>(n={n})") if n is not None else col
    else:
        for col in value_cols:
            labels[col] = col
    if column_labels is not None:
        for col, label in column_labels.items():
            if col in labels:
                labels[col] = label

    if has_row_group:
        gt = GT(result_visible, rowname_col="name", groupname_col="row_group")
        options: dict[str, object] = {"stub_row_group_font_weight": "bold"}
    else:
        # ``tab_stub()`` はコンストラクタで付けた stub を外し、name を本文の先頭列にする。
        gt = GT(result_visible, rowname_col="name").tab_stub()
        options = {}
    gt = (
        gt.cols_label(**labels)
        .cols_align(align="center", columns=value_cols)
        .tab_options(
            table_font_size=style["font_size"],
            table_width=style["table_width"],
            column_labels_padding_horizontal=style["column_labels_padding_horizontal"],
            data_row_padding=style["data_row_padding"],
            **options,
        )
    )
    if not has_row_group:
        gt = gt.cols_align(align="left", columns="name").tab_style(
            gt_style.text(weight="bold"),
            loc.body(columns="name", rows=~pl.col("name").str.starts_with(style["indent_display"])),
        )
    for label, cols in column_spanners or ():
        present = [col for col in cols if col in value_cols]
        if present:
            gt = gt.tab_spanner(label=label, columns=present)
    return gt


def tableone(
    df: pl.DataFrame,
    params: dict[str, TableOneParam],
    hue: str | pl.Expr | list[pl.Expr] | list[str] | None = None,
    *,
    add_all: bool = False,
    add_pvalue: bool = False,
    add_smd: bool = False,
    group_labels: dict[str, str] | None = None,
    style: TableOneStyle | None = None,
    column_order: Sequence[str] | None = None,
) -> GT:
    """Generate Table One and return a styled `great_tables.GT`.

    The rows are those of `tableone_raw`, which takes the same ``df``,
    ``params``, ``hue``, ``add_all``, ``add_pvalue``, and ``add_smd``. The
    headers show the size of each group.

    Args:
        df: Input data.
        params: Display name → ``(column, aggfunc)`` or
            ``(column, aggfunc, statfunc)``, as in `tableone_raw`.
        hue: Group column(s).
        add_all: Add an "All" column.
        add_pvalue: Add a "P value" column.
        add_smd: Add an "SMD" column.
        group_labels: Header label for each value column, in place of the
            ``n`` headers.
        style: Overrides for font size, table width, and padding.
        column_order: Order of the value columns.

    Notes:
    - Category sub-rows are detected by the fixed prefix `"  "` (two spaces).
    - For display, the prefix is replaced by a full-width space by default.
    """
    result = tableone_raw(df, params, hue=hue, add_all=add_all, add_pvalue=add_pvalue, add_smd=add_smd)
    return tableone_gt_from_frame(
        result,
        style=style,
        group_labels=group_labels,
        column_order=column_order,
        source_df=df,
        hue=hue,
    )


def write_tableone_artifacts(
    out_dir: Path | str,
    stem: str,
    *,
    df: pl.DataFrame | None = None,
    params: dict | None = None,
    table: pl.DataFrame | None = None,
    hue: str | pl.Expr | list[pl.Expr] | list[str] | None = None,
    add_all: bool = False,
    add_pvalue: bool = False,
    add_smd: bool = False,
    style: TableOneStyle | None = None,
    group_labels: dict[str, str] | None = None,
    column_order: Sequence[str] | None = None,
    column_labels: dict[str, str] | None = None,
    column_spanners: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> list[Path]:
    """Write a Table One as CSV / companion / GT HTML / GT Markdown fragment.

    Args:
        out_dir: Output folder. Created if missing.
        stem: File name stem (for example ``table_baseline_method`` →
            ``.csv`` / ``_csv.md`` / ``.html`` / ``_gt.md``).
        df: Input data passed to `tableone_raw`. Required with ``params`` when
            ``table`` is not given. With ``table`` and ``hue``, it adds the
            group sizes to the headers.
        params: Rows, as in `tableone_raw`.
        table: A Table One DataFrame that is already built (such as Pre/Post
            side by side). When given, ``df``/``params`` are not needed.
        hue: Group column(s).
        add_all: Add an "All" column.
        add_pvalue: Add a "P value" column.
        add_smd: Add an "SMD" column.
        style: Overrides for font size, table width, and padding. Defaults to
            a table sized to its content.
        group_labels: Header label for each value column.
        column_order: Order of the value columns.
        column_labels: Header labels for some columns, applied last.
        column_spanners: ``(label, columns)`` pairs, one spanner label over
            each set of columns.

    Returns:
        The written paths: CSV, CSV companion, HTML, and GT Markdown fragment.
    """
    from ..report.artifacts import write_csv_companion

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if style is None:
        # Fit columns to their content; a fixed width spreads them out on the docs pages.
        style = {"table_width": "auto", "font_size": "12px", "data_row_padding": "3px"}

    if table is None:
        if df is None or params is None:
            raise InvalidParameterError("Provide df+params, or a prebuilt table=")
        table = tableone_raw(df, params, hue=hue, add_all=add_all, add_pvalue=add_pvalue, add_smd=add_smd)
    table = _reorder_table_columns(table, column_order)
    gt = tableone_gt_from_frame(
        table,
        style=style,
        group_labels=group_labels,
        column_order=column_order,
        column_labels=column_labels,
        column_spanners=column_spanners,
        source_df=df,
        hue=hue,
    )

    csv_path = out / f"{stem}.csv"
    table.write_csv(csv_path)
    written: list[Path] = [csv_path, write_csv_companion(csv_path)]

    html_path = out / f"{stem}.html"
    gt.write_raw_html(html_path, make_page=True)
    written.append(html_path)

    gt_md = out / f"{stem}_gt.md"
    gt_md.write_text(_embeddable_gt_html(gt.as_raw_html(), f"tableone-{stem}"), encoding="utf-8")
    written.append(gt_md)
    return written


def _embeddable_gt_html(html: str, table_id: str) -> str:
    """Make a GT fragment safe to paste into another HTML page.

    GT draws a random id each call and emits two rules that are not scoped to
    it (``tbody, tfoot, tr, td, th`` and a bare ``tr``), which would restyle
    every table on the host page. Use a fixed id and scope those rules. Blank
    lines are dropped so Markdown keeps the fragment as one HTML block. CSS
    lines that start with ``#`` get a leading space: when the fragment is
    included in indented Markdown (a content tab), they would otherwise be
    read as headings, which breaks the styles and the page's table of contents.
    """
    match = re.search(r'<div id="([^"]+)"', html)
    if match is None:
        return html
    html = html.replace(match.group(1), table_id)
    tid = f"#{table_id}"
    html = html.replace(
        f"{tid} thead, tbody, tfoot, tr, td, th {{",
        f"{tid} thead, {tid} tbody, {tid} tfoot, {tid} tr, {tid} td, {tid} th {{",
    )
    html = re.sub(r"^\s*tr \{", f"{tid} tr {{", html, flags=re.MULTILINE)
    lines = [line for line in html.splitlines() if line.strip()]
    return "\n".join(f" {line}" if line.startswith("#") else line for line in lines) + "\n"


def _normalize_column(column: str | pl.Expr) -> pl.Expr:
    """Normalize column to pl.Expr."""
    if isinstance(column, str):
        return pl.col(column)
    return column


def _normalize_params(params: dict[str, TableOneParam | tuple]) -> dict[str, TableOneParam]:
    """Normalize params to dict[str, TableOneParam]."""
    normalized = {}
    for name, param in params.items():
        if isinstance(param, tuple):
            if len(param) == 2:
                column, aggfunc = param
                statfunc = None
            elif len(param) == 3:
                column, aggfunc, statfunc = param
            else:
                raise InvalidParameterError(f"Invalid param tuple length: {len(param)}")
            normalized[name] = TableOneParam(
                column=_normalize_column(column),
                aggfunc=aggfunc,
                statfunc=statfunc,
            )
        elif isinstance(param, TableOneParam):
            normalized[name] = TableOneParam(
                column=_normalize_column(param.column),
                aggfunc=param.aggfunc,
                statfunc=param.statfunc,
            )
        else:
            raise InvalidParameterError(f"Invalid param type: {type(param)}")
    return normalized


def _normalize_hue(hue: str | pl.Expr | list[pl.Expr] | list[str] | None) -> list[pl.Expr] | None:
    """Normalize hue to list[pl.Expr]."""
    if hue is None:
        return None
    if isinstance(hue, str):
        return [pl.col(hue)]
    if isinstance(hue, pl.Expr):
        return [hue]
    if isinstance(hue, list):
        return [_normalize_column(h) if isinstance(h, str) else h for h in hue]
    raise InvalidParameterError(f"Invalid hue type: {type(hue)}")


def _get_aggfunc(aggfunc: agg.AggFunc | str | None, column: pl.Expr) -> agg.AggFunc | None:  # noqa: ARG001
    """Get aggregation function from string or return as-is."""
    if aggfunc is None:
        return None
    if isinstance(aggfunc, str):
        # Map string to function
        aggfunc_map = {
            "mean_sd": agg.agg_mean_sd,
            "median_range": agg.agg_median_range,
            "median_iqr": agg.agg_median_iqr,
            "range": agg.agg_range,
            "median": agg.agg_median,
            "category": agg.agg_category,
            "category_n": agg.agg_category_n,
            "category_np": agg.agg_category_np,
            "category_p": agg.agg_category_p,
            "category_pnn": agg.agg_category_pnn,
            "ratio_np": agg.agg_ratio_np,
            "ratio": agg.agg_ratio,
            "ratio_n": agg.agg_ratio_n,
            "ratio_pnn": agg.agg_ratio_pnn,
            "count": agg.agg_count,
            "nunique": agg.agg_nunique,
        }
        if aggfunc == "auto":
            raise InvalidParameterError(
                "aggfunc='auto' is no longer supported. Use aggfunc=None to enable auto-selection.",
            )
        if aggfunc not in aggfunc_map:
            raise InvalidParameterError(f"Unknown aggregation function: {aggfunc}")
        return aggfunc_map[aggfunc]
    return aggfunc


def _callable_func(aggfunc: agg.AggFunc) -> agg.AggFunc:
    """``functools.partial`` なら包んでいる関数を返す。"""
    return getattr(aggfunc, "func", aggfunc)


def _bound_enum_categories(aggfunc: agg.AggFunc) -> list[str] | None:
    """partial に束縛した ``enum_categories``。文字列列でも並びを固定するために使う。"""
    keywords = getattr(aggfunc, "keywords", None) or {}
    cats = keywords.get("enum_categories")
    if not cats:
        return None
    return list(cats)


def _is_category_aggfunc(aggfunc: agg.AggFunc) -> bool:
    """Check if aggregation function is a category function."""
    category_funcs = {
        agg.agg_category,
        agg.agg_category_n,
        agg.agg_category_np,
        agg.agg_category_p,
        agg.agg_category_pnn,
        agg.agg_bool_category,
    }
    return _callable_func(aggfunc) in category_funcs


def _get_enum_categories(df: pl.DataFrame, column: pl.Expr) -> list[str] | None:
    """
    Get enum categories if the column is an enum type.

    Args:
        df: DataFrame containing the column
        column: Column expression

    Returns:
        List of enum category values in order, or None if not an enum type.
    """
    # Get dtype by selecting the column
    dtype = df.select(column.alias("_temp")).schema["_temp"]

    if isinstance(dtype, pl.Enum):
        return dtype.categories.to_list()
    return None


def _get_auto_aggfunc(column: pl.Expr, df: pl.DataFrame) -> agg.AggFunc:
    """Get automatic aggregation function based on column dtype."""
    # Get dtype by selecting the column
    dtype = df.select(column.alias("_temp")).schema["_temp"]

    if dtype.is_numeric():
        return agg.agg_mean_sd
    elif dtype == pl.Boolean:
        return agg.agg_ratio_np
    elif dtype in (pl.Categorical, pl.String):
        return agg.agg_category
    else:
        return agg.agg_count


def _resolve_statfunc(
    statfunc: stat.StatFunc | str | None,
    column: pl.Expr,
    hue: pl.Expr,
    df: pl.DataFrame,
) -> stat.StatFunc | None:
    """Resolve statfunc from string/callable/None.

    Notes:
    - statfunc=None means auto-selection (based on column/hue types).
    - statfunc='auto' is no longer supported.
    """
    if statfunc is None:
        return stat.stat_auto
    if isinstance(statfunc, str):
        if statfunc == "auto":
            raise InvalidParameterError(
                "statfunc='auto' is no longer supported. Use statfunc=None to enable auto-selection.",
            )
        if statfunc in {"-", "none", "None"}:
            return None
        statfunc_map: dict[str, stat.StatFunc] = {
            "anova": stat.stat_anova,
            "chisq": stat.stat_chisq,
            "fisher": stat.stat_fisher,
            "kruskal": stat.stat_kruskal,
            "auto": stat.stat_auto,
        }
        if statfunc not in statfunc_map:
            raise InvalidParameterError(f"Unknown statistical test function: {statfunc}")
        return statfunc_map[statfunc]
    return statfunc


_MEDIAN_AGGFUNCS = {agg.agg_median, agg.agg_median_iqr, agg.agg_median_range}


def _hue_series(df: pl.DataFrame, hue: list[pl.Expr]) -> pl.Series:
    """One group label per row. Several hue columns combine as in the column headers."""
    if len(hue) == 1:
        return df.select(hue[0]).to_series()
    parts = df.select([h.cast(pl.String) for h in hue])
    return parts.select(pl.concat_str(pl.all(), separator=" / ").alias("_hue")).to_series()


def _compute_pvalue(df: pl.DataFrame, param: TableOneParam, hue: list[pl.Expr]) -> str:
    """Compute the p-value string for a param across the hue groups."""
    statfunc = param.statfunc
    if statfunc is None and _get_aggfunc(param.aggfunc, param.column) in _MEDIAN_AGGFUNCS:
        # A median summary gets a rank test, as R tableone does for nonnormal variables.
        statfunc = stat.stat_kruskal
    try:
        resolved = _resolve_statfunc(statfunc, param.column, hue[0], df)
    except ValueError:
        # e.g. stat_auto cannot decide
        return "-"
    if resolved is None:
        return "-"
    try:
        # Extract Series from DataFrame before calling stat function
        col_series = df.select(param.column).to_series()
        hue_series = _hue_series(df, hue)
        return resolved(col_series, hue_series)
    except Exception:  # noqa: BLE001
        return "Failed"


def _compute_smd(df: pl.DataFrame, param: TableOneParam, hue: list[pl.Expr]) -> str:
    """SMD string for a param across the hue groups (mean of pairs when more than two)."""
    aggfunc = _get_aggfunc(param.aggfunc, param.column) or _get_auto_aggfunc(param.column, df)
    col_series = df.select(param.column).to_series()
    if aggfunc in {agg.agg_count, agg.agg_nunique, agg.agg_size}:
        return ""
    categorical = True if _is_category_aggfunc(aggfunc) else None
    try:
        value = stat.standardized_difference(col_series, _hue_series(df, hue), categorical=categorical)
    except Exception:  # noqa: BLE001
        return "Failed"
    return "-" if not np.isfinite(value) else f"{value:.3f}"


def _add_column_to_rows(
    all_rows: list[tuple[str, dict[str, str]]],
    column: str,
    values: dict[str, str],
) -> list[tuple[str, dict[str, str]]]:
    """Attach ``column`` to header rows (row_name == param_name)."""
    return [
        (
            row_name,
            (row_data | {column: values[row_name]}) if row_name in values else row_data,
        )
        for row_name, row_data in all_rows
    ]


def _get_hue_key(hue: list[pl.Expr], row: dict) -> str:
    """Get hue key from row data."""
    hue_values = tuple(row[k.meta.output_name()] for k in hue)
    return " / ".join(str(v) for v in hue_values) if len(hue_values) > 1 else str(hue_values[0])


def _agg_expr_with_enum_categories(
    df: pl.DataFrame,
    column: pl.Expr,
    aggfunc: agg.AggFunc,
) -> tuple[pl.Expr, list[str] | None]:
    """Build aggregation expression, passing enum_categories when applicable.

    Returns (aggregation_expr, enum_categories).
    """
    column_enum = _get_enum_categories(df, column)
    bound_enum = _bound_enum_categories(aggfunc)
    enum_categories = column_enum if column_enum is not None else bound_enum
    if column_enum is None or bound_enum is not None:
        return aggfunc(column), enum_categories

    def aggfunc_with_enum(col: pl.Expr) -> pl.Expr:
        return aggfunc(col, enum_categories=column_enum)  # type: ignore[misc]

    return aggfunc_with_enum(column), column_enum


def _grouped_results(df: pl.DataFrame, agg_expr: pl.Expr, hue: list[pl.Expr] | None) -> dict[str, object]:
    """Evaluate aggregation, returning {group_key: raw_value}. 'All' is used for no hue."""
    if hue is None:
        result = df.select(agg_expr.alias("_result"))
        if result.height == 0:
            return {"All": None}
        # For list/struct/scalar, to_list()[0] gives python value.
        values = result["_result"].to_list()
        return {"All": values[0] if values else None}

    grouped = df.group_by(hue, maintain_order=True).agg(agg_expr.alias("_result"))
    out: dict[str, object] = {}
    for row in grouped.iter_rows(named=True):
        out[_get_hue_key(hue, row)] = row["_result"]
    return out


def _normalize_category_list(value: object) -> list[dict]:
    """Normalize Polars list-of-struct result into list[dict]."""
    if value is None:
        return []

    items: object
    if isinstance(value, list):
        items = value
    elif hasattr(value, "to_list"):
        try:
            items = value.to_list()  # type: ignore[no-untyped-call]
        except Exception:  # noqa: BLE001
            return []
    elif hasattr(value, "__iter__"):
        items = value
    else:
        return []

    try:
        return [x for x in items if isinstance(x, dict)]  # type: ignore[operator]
    except Exception:  # noqa: BLE001
        return []


def _compute_aggregation_category(
    df: pl.DataFrame,
    param_name: str,
    column: pl.Expr,
    aggfunc: agg.AggFunc,
    hue: list[pl.Expr] | None,
) -> list[tuple[str, dict[str, str]]]:
    """Compute category aggregation with/without hue."""
    agg_expr, enum_categories = _agg_expr_with_enum_categories(df, column, aggfunc)
    grouped = _grouped_results(df, agg_expr, hue)

    all_categories: set[str] = set()
    group_results: dict[str, dict[str, str]] = {}
    for group_key, raw in grouped.items():
        cat_map: dict[str, str] = {}
        for item in _normalize_category_list(raw):
            values = list(item.values())
            if len(values) < 2:
                continue
            cat_value = values[0]
            formatted_value = values[1]
            cat_map[str(cat_value)] = str(formatted_value)
            all_categories.add(str(cat_value))
        group_results[group_key] = cat_map

    if enum_categories is not None:
        categories_to_sort = []
        categories_to_sort.extend(cat for cat in enum_categories if cat in all_categories)
        categories_to_sort.extend(sorted(all_categories - set(enum_categories)))
    else:
        categories_to_sort = sorted(all_categories)

    rows: list[tuple[str, dict[str, str]]] = []
    rows.append((param_name, dict.fromkeys(group_results, "")))
    group_keys = list(group_results)
    rows.extend(
        (
            f"  {cat_value}",
            {group_key: group_results[group_key].get(cat_value, "0") for group_key in group_keys},
        )
        for cat_value in categories_to_sort
    )
    return rows


def _compute_aggregation_regular(
    df: pl.DataFrame,
    param_name: str,
    column: pl.Expr,
    aggfunc: agg.AggFunc,
    hue: list[pl.Expr] | None,
) -> list[tuple[str, dict[str, str]]]:
    """Compute regular aggregation with/without hue."""
    agg_expr = aggfunc(column)
    grouped = _grouped_results(df, agg_expr, hue)
    return [
        (
            param_name,
            {k: ("" if v is None else str(v)) for k, v in grouped.items()},
        ),
    ]


def _compute_aggregation(
    df: pl.DataFrame,
    param_name: str,
    param: TableOneParam,
    hue: list[pl.Expr] | None,
) -> list[tuple[str, dict[str, str]]]:
    """
    Compute aggregation for a single parameter.

    Returns:
        List of tuples: (row_name, {hue_value: formatted_value})
        For category variables, returns multiple tuples (one per category)
        For regular variables, returns single tuple
    """
    column = param.column
    aggfunc = _get_aggfunc(param.aggfunc, column)

    # Handle auto selection
    if aggfunc is None:
        aggfunc = _get_auto_aggfunc(column, df)

    # Check if column exists
    col_name = column.meta.output_name()
    if col_name and col_name not in df.columns:
        raise ColumnNotFoundError(f"Column '{col_name}' not found in dataframe")

    is_category = _is_category_aggfunc(aggfunc)

    if is_category:
        return _compute_aggregation_category(df, param_name, column, aggfunc, hue)
    return _compute_aggregation_regular(df, param_name, column, aggfunc, hue)


def _format_table(rows: list[tuple[str, dict[str, str]]]) -> pl.DataFrame:
    """
    Format table rows into a Polars DataFrame.

    This function converts a list of row tuples into a structured DataFrame
    with "name" column and data columns for each hue group. The "All" column
    (if present) is placed first, followed by other hue columns in alphabetical order.

    Args:
        rows: List of tuples where each tuple contains:
            - row_name (str): The name/label for the row (e.g., "Age", "  Category1")
            - row_data (dict[str, str]): Dictionary mapping hue keys to formatted values
              (e.g., {"All": "30.5", "Group A": "28.0", "Group B": "33.0"})

    Returns:
        pl.DataFrame with columns:
            - name: Row names/labels
            - [hue keys]: Data columns, with "All" first if present, then alphabetical

    Examples:
        Example 1: Simple case with single "All" column
        >>> rows = [
        ...     ("Age", {"All": "30.5 (25-40)"}),
        ...     ("Height", {"All": "170.0 (165-175)"}),
        ... ]
        >>> result = _format_table(rows)
        >>> # Returns DataFrame:
        >>> # ┌────────┬──────────────┐
        >>> # │ name   ┆ All          │
        >>> # │ ---    ┆ ---          │
        >>> # │ str    ┆ str          │
        >>> # ╞════════╪══════════════╡
        >>> # │ Age    ┆ 30.5 (25-40) │
        >>> # │ Height ┆ 170.0 (165-175)│
        >>> # └────────┴──────────────┘

        Example 2: With hue groups and "All" column
        >>> rows = [
        ...     ("Age", {"All": "30.5", "Group A": "28.0", "Group B": "33.0"}),
        ...     ("Height", {"All": "170.0", "Group A": "168.0", "Group B": "172.0"}),
        ... ]
        >>> result = _format_table(rows)
        >>> # Returns DataFrame with columns: name, All, Group A, Group B
        >>> # "All" column is placed first

        Example 3: Category variable with sub-items
        >>> rows = [
        ...     ("Category", {"All": "", "Group A": "", "Group B": ""}),
        ...     ("  X", {"All": "5", "Group A": "2", "Group B": "3"}),
        ...     ("  Y", {"All": "3", "Group A": "1", "Group B": "2"}),
        ... ]
        >>> result = _format_table(rows)
        >>> # Returns DataFrame with indented category sub-items preserved
        >>> # ┌──────────┬─────┬──────────┬──────────┐
        >>> # │ name     ┆ All ┆ Group A  ┆ Group B  │
        >>> # │ ---      ┆ --- ┆ ---      ┆ ---      │
        >>> # │ str      ┆ str ┆ str      ┆ str      │
        >>> # ╞══════════╪═════╪══════════╪══════════╡
        >>> # │ Category ┆     ┆          ┆          │
        >>> # │   X      ┆ 5   ┆ 2        ┆ 3        │
        >>> # │   Y      ┆ 3   ┆ 1        ┆ 2        │
        >>> # └──────────┴─────┴──────────┴──────────┘

        Example 4: Empty rows
        >>> result = _format_table([])
        >>> # Returns empty DataFrame with only "name" column
    """
    if not rows:
        return pl.DataFrame({"name": []})

    # Get all hue keys from all rows
    all_hue_keys = set()
    for _, row_data in rows:
        all_hue_keys.update(row_data.keys())

    # Sort hue keys ("All" first, then alphabetical), but always place "P value" last
    # when add_pvalue=True injected it.
    hue_keys = sorted(all_hue_keys, key=lambda x: (x != "All", x))
    trailing = [k for k in ("P value", "SMD") if k in hue_keys]
    hue_keys = [k for k in hue_keys if k not in trailing] + trailing

    # Build DataFrame
    name_col = [name for name, _ in rows]
    data_dict = {"name": name_col}
    for hue_key in hue_keys:
        data_dict[hue_key] = [row_data.get(hue_key, "") for _, row_data in rows]

    return pl.DataFrame(data_dict)


def tableone_raw(  # noqa: C901
    df: pl.DataFrame,
    params: dict[str, TableOneParam],
    hue: str | pl.Expr | list[pl.Expr] | list[str] | None = None,
    *,
    add_all: bool = False,
    add_pvalue: bool = False,
    add_smd: bool = False,
) -> pl.DataFrame:
    """Table One generation for medical research studies.

    This module provides functionality to create Table One, a standard format for
    presenting baseline characteristics in medical research papers.

    Args:
      df: Input Polars DataFrame containing the study data.
      params: Dictionary mapping display names to TableOneParam objects or tuples.
              Each tuple should be (column, aggfunc) or (column, aggfunc, statfunc).
              - column: Column name (str) or expression (pl.Expr)
              - aggfunc: Aggregation function or string identifier (e.g., "mean_sd", "median_range", "category")
              - statfunc: Statistical test function (optional)
      hue: Column name(s) for stratification/grouping. Can be:
           - None: Compute overall statistics (creates "All" column)
           - str: Single column name
           - pl.Expr: Single column expression
           - list[str] | list[pl.Expr]: Multiple columns for nested grouping
      add_all: bool = False
        If True and hue is specified, add an "All" column with overall statistics
        as the leftmost column. Has no effect when hue is None.
      add_pvalue: bool = False
        Add a "P value" column. ``statfunc=None`` picks Fisher for categories,
        Kruskal–Wallis for median summaries, and ANOVA for other numbers.
        Several hue columns are crossed into one set of groups.
      add_smd: bool = False
        Add an "SMD" column, the standardized mean difference as R tableone
        reports it (the mean of the pairwise SMDs with more than two groups).

    Returns:
        Polars DataFrame with Table One format

    Examples:
        >>> import polars as pl
        >>> from statract import tableone_raw
        >>> from statract.tableone import agg
        >>>
        >>> # Create sample data
        >>> df = pl.DataFrame({
        ...     "age": [45, 50, 55, 60, 65],
        ...     "gender": ["M", "F", "M", "F", "M"],
        ...     "treatment": ["A", "A", "B", "B", "A"]
        ... })
        >>>
        >>> # Define parameters
        >>> params = {
        ...     "Age": ("age", agg.agg_mean_sd),
        ...     "Gender": ("gender", agg.agg_category)
        ... }
        >>>
        >>> # Generate Table One without grouping
        >>> result = tableone_raw(df, params)
        >>> print(result)
        >>>
        >>> # Generate Table One with grouping by treatment
        >>> result = tableone_raw(df, params, hue="treatment")
        >>> print(result)
        >>>
        >>> # Add "All" column and p-values
        >>> result = tableone_raw(df, params, hue="treatment", add_all=True, add_pvalue=True)
        >>> print(result)
    """

    # Validate inputs
    if not isinstance(df, pl.DataFrame) or df.is_empty():
        raise InvalidParameterError("df must be a non-empty polars DataFrame")
    if not isinstance(params, dict) or not params:
        raise InvalidParameterError("params must be a non-empty dictionary")

    # Normalize params and hue
    normalized_params = _normalize_params(params)
    normalized_hue = _normalize_hue(hue)

    # Compute aggregations for all parameters
    all_rows = []

    if normalized_hue is not None and add_all:
        # When hue is specified and add_all is True, compute both overall and grouped statistics
        for param_name, param in normalized_params.items():
            # Compute overall statistics (hue=None)
            overall_rows = _compute_aggregation(df, param_name, param, None)
            # Compute grouped statistics (hue=normalized_hue)
            grouped_rows = _compute_aggregation(df, param_name, param, normalized_hue)

            # Merge overall and grouped statistics
            # Create a mapping from row name to row data for quick lookup
            grouped_dict = {row_name: row_data for row_name, row_data in grouped_rows}

            # Process each overall row and merge with grouped data
            for row_name, overall_data in overall_rows:
                merged_data = overall_data.copy()  # Start with "All" column
                # Add grouped columns if this row exists in grouped data
                if row_name in grouped_dict:
                    merged_data.update(grouped_dict[row_name])
                all_rows.append((row_name, merged_data))

            # Add any grouped rows that weren't in overall (shouldn't happen, but just in case)
            for row_name, grouped_data in grouped_rows:
                if row_name not in {r[0] for r in overall_rows}:
                    merged_data = {"All": ""}
                    merged_data.update(grouped_data)
                    all_rows.append((row_name, merged_data))
    else:
        # Normal case: compute aggregations with or without hue
        for param_name, param in normalized_params.items():
            rows = _compute_aggregation(df, param_name, param, normalized_hue)
            all_rows.extend(rows)

    # P values and SMDs compare the hue groups. Several hue columns are crossed,
    # so the test compares the same groups as the table's columns.
    if add_pvalue and normalized_hue is not None:
        pvalues = {name: _compute_pvalue(df, param, normalized_hue) for name, param in normalized_params.items()}
        all_rows = _add_column_to_rows(all_rows, "P value", pvalues)
    if add_smd and normalized_hue is not None:
        smds = {name: _compute_smd(df, param, normalized_hue) for name, param in normalized_params.items()}
        all_rows = _add_column_to_rows(all_rows, "SMD", smds)

    # Format and return
    return _format_table(all_rows)
