"""R bridge utilities for statistical analysis.

This module provides utilities for interfacing with R via rpy2, including
rpy2_arrow.polars for efficient data transfer between Python (Polars) and R.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import rpy2
    import rpy2_arrow.polars as rpy2polars
    from rpy2 import robjects
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects.vectors import ListVector, Vector

# Populated on first ``_require_rpy2()`` call (not at ``import statract.r``).
rpy2: Any = None
robjects: Any = None
RRuntimeError: Any = None
ListVector: Any = None
Vector: Any = None
rpy2polars: Any = None
_rpy2_loaded = False


class RNotAvailableError(RuntimeError):
    """Raised when the ``statract[r]`` extra or a working R installation is missing."""


_R_INSTALL_HINT = (
    "R integration requires the statract[r] extra (rpy2 and rpy2-arrow) and a working R installation."
)


class RconsoleBase:
    def __init__(self) -> None:
        self._consolewrite_print_backup: Any = None
        self._consolewrite_print_warnerror: Any = None

    def on(self) -> tuple[Callable[[str], None], Callable[[str], None]]:
        raise NotImplementedError

    def off(self) -> None:
        pass

    def __enter__(self) -> RconsoleBase:
        rpy2 = _require_rpy2()
        p, w = self.on()
        self._consolewrite_print_backup = rpy2.rinterface_lib.callbacks.consolewrite_print
        self._consolewrite_print_warnerror = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        rpy2.rinterface_lib.callbacks.consolewrite_print = p  # type: ignore[attr-defined]
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = w  # type: ignore[attr-defined]
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        rpy2 = _require_rpy2()
        self.off()
        rpy2.rinterface_lib.callbacks.consolewrite_print = self._consolewrite_print_backup  # type: ignore[attr-defined]
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = self._consolewrite_print_warnerror  # type: ignore[attr-defined]


class RconsoleSuppress(RconsoleBase):
    def __init__(self) -> None:
        super().__init__()
        self.console_buff: list[str] = []

    def on(self) -> tuple[Callable[[str], None], Callable[[str], None]]:
        def f(x: str) -> None:
            self.console_buff.append(x)

        return f, f

    def getvalue(self) -> str:
        return "".join(self.console_buff)

    def clear(self) -> None:
        self.console_buff = []

    def pop(self) -> str:
        ret = self.getvalue()
        self.clear()
        return ret


def _require_rpy2() -> Any:
    global rpy2, robjects, RRuntimeError, ListVector, Vector, rpy2polars, _rpy2_loaded
    if not _rpy2_loaded:
        from . import _rpy2_env  # noqa: F401 — RPY2_CFFI_MODE=ABI before rpy2 import

        try:
            import rpy2 as _rpy2_mod
            from rpy2 import robjects as _robjects
            from rpy2.rinterface_lib.embedded import RRuntimeError as _RRuntimeError
            from rpy2.robjects.vectors import ListVector as _ListVector, Vector as _Vector

            rpy2 = _rpy2_mod
            robjects = _robjects
            RRuntimeError = _RRuntimeError
            ListVector = _ListVector
            Vector = _Vector
        except ImportError:
            rpy2 = None
            robjects = None
            RRuntimeError = None
            ListVector = None
            Vector = None

        try:
            import rpy2_arrow.polars as _rpy2polars

            rpy2polars = _rpy2polars
        except ImportError:
            rpy2polars = None

        _rpy2_loaded = True

    if rpy2 is None:
        raise RNotAvailableError(_R_INSTALL_HINT)
    return rpy2


def get_enum_levels(df: pl.DataFrame) -> dict[str, list[str]]:
    """Get Enum column names and their category order from a Polars DataFrame.

    Returns a dict mapping column name to list of category strings in definition order.
    Only columns with dtype pl.Enum are included.

    Args:
        df: Polars DataFrame

    Returns:
        Dict of column name -> list of category strings (empty if no Enum columns).

    Examples:
        >>> df = pl.DataFrame({"x": ["a", "b"]}).with_columns(
        ...     pl.col("x").cast(pl.Enum(["a", "b", "c"]))
        ... )
        >>> get_enum_levels(df)
        {'x': ['a', 'b', 'c']}
    """
    result: dict[str, list[str]] = {}
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Enum):
            result[name] = dtype.categories.to_list()
    return result


def _apply_enum_levels_to_tibble(r_tibble: Any, enum_levels: dict[str, list[str]]) -> Any:
    """Apply Polars enum category order to an R tibble by re-setting factor levels."""
    if robjects is None or not enum_levels:
        return r_tibble
    colnames = list(r_tibble.colnames)
    for nm, levels in enum_levels.items():
        if nm not in colnames:
            continue
        idx = colnames.index(nm)
        r_tibble[idx] = robjects.r("factor")(r_tibble.rx2(nm), levels=robjects.StrVector(levels))
    return r_tibble


def run(command: str, *, show: bool = False, return_console: bool = False) -> str | None:
    """
    Run an R command via rpy2.

    This function suppresses R console output by default, but when an R error occurs,
    it re-attaches the suppressed console output to the exception for debugging.
    """
    _require_rpy2()
    if robjects is None or RRuntimeError is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    with RconsoleSuppress() as con:
        try:
            ret = robjects.r(command)
        except RRuntimeError as e:
            # Attach suppressed console output to the exception.
            e.add_note(con.getvalue())
            raise

    if show:
        print(con.pop())

    if return_console:
        return ret, con.getvalue()
    return ret


def init() -> None:
    """Load endolab R resources if present."""
    rdir = Path(__file__).parent
    init_r = rdir / "init.R"
    if init_r.exists():
        run(init_r.read_text())
    io_r = rdir / "io.R"
    if io_r.exists():
        run(io_r.read_text())
    forest_r = rdir / "forest.R"
    if forest_r.exists():
        run(forest_r.read_text())


def has_r_package(pkg: str) -> bool:
    """
    Return True if R has the given package installed (quietly).
    """
    try:
        # requireNamespace returns invisible(TRUE/FALSE), so we need to explicitly convert to logical
        r = run(f'as.logical(requireNamespace("{pkg}", quietly = TRUE))')
        # r is an R logical vector of length 1, or None if invisible
        if r is None:
            # If None, try to get the value directly from R
            r = run(f'requireNamespace("{pkg}", quietly = TRUE)')
        if r is None:
            return False
        # Handle both vector and scalar cases
        if hasattr(r, "__getitem__") and len(r) > 0:
            return bool(r[0])
        return bool(r)
    except Exception:  # noqa: BLE001
        return False


def assign_dfs(dfs: dict[str, pl.DataFrame]) -> None:
    """
    Assign a dictionary of polars DataFrames into the R global environment.
    """
    for name, df in dfs.items():
        assign(name, df)


def _convert_value_to_r(v: Any) -> Any:
    """Recursively convert Python value to R object."""
    _require_rpy2()
    if robjects is None or ListVector is None or rpy2polars is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    if v is None:
        return robjects.NULL
    if isinstance(v, pl.DataFrame):
        # Convert Polars DataFrame to R polars DataFrame first
        with rpy2polars.converter.context():
            r_polars_df = robjects.conversion.py2rpy(v)
        # Then convert to tibble
        temp_name = f"_temp_polars_{abs(hash(str(v)))}"
        try:
            robjects.globalenv[temp_name] = r_polars_df
            # Convert polars DataFrame to tibble using as_tibble
            r_tibble = robjects.r("tibble::as_tibble")(robjects.globalenv[temp_name])
            # Preserve Enum/Category order by re-applying factor levels
            enum_levels = get_enum_levels(v)
            return _apply_enum_levels_to_tibble(r_tibble, enum_levels)
        finally:
            try:
                del robjects.globalenv[temp_name]
            except Exception:  # noqa: BLE001
                pass
    elif isinstance(v, dict):
        # Recursively convert nested dict
        r_dict = {}
        for nk, nv in v.items():
            r_dict[str(nk)] = _convert_value_to_r(nv)
        return ListVector(r_dict)
    elif isinstance(v, list):
        # Recursively convert list
        if len(v) == 0:
            return robjects.vectors.StrVector([])
        # Convert each element
        r_items = [_convert_value_to_r(x) for x in v]
        # Use ListVector with numeric names for lists
        return ListVector({str(i): item for i, item in enumerate(r_items)})
    else:
        # Basic types
        return robjects.conversion.py2rpy(v)


def _assign_recursive(name: str, obj: Any, *, is_top_level: bool = True) -> None:
    """
    Recursively assign Python object to R, handling nested structures.

    This is an internal helper function for assign().
    """
    _require_rpy2()
    if robjects is None or ListVector is None or rpy2polars is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    if obj is None:
        robjects.globalenv[name] = robjects.NULL
        return

    # Handle Polars DataFrame - convert to tibble in R
    if isinstance(obj, pl.DataFrame):
        # Convert Polars DataFrame to R polars DataFrame first
        with rpy2polars.converter.context():
            r_polars_df = robjects.conversion.py2rpy(obj)
        # Then convert to tibble and assign
        temp_name = f"_temp_polars_{abs(hash(str(obj)))}"
        try:
            robjects.globalenv[temp_name] = r_polars_df
            # Convert polars DataFrame to tibble using as_tibble
            r_tibble = robjects.r("tibble::as_tibble")(robjects.globalenv[temp_name])
            # Preserve Enum/Category order by re-applying factor levels
            enum_levels = get_enum_levels(obj)
            r_tibble = _apply_enum_levels_to_tibble(r_tibble, enum_levels)
            robjects.globalenv[name] = r_tibble
        finally:
            try:
                del robjects.globalenv[temp_name]
            except Exception:  # noqa: BLE001
                pass
        return

    # Handle dict (recursively)
    if isinstance(obj, dict):
        r_dict = {}
        with rpy2polars.converter.context():
            for k, v in obj.items():
                r_dict[str(k)] = _convert_value_to_r(v)

        r_list = ListVector(r_dict)
        robjects.globalenv[name] = r_list
        return

    # Handle list (recursively)
    if isinstance(obj, list):
        r_list_items = []
        with rpy2polars.converter.context():
            for item in obj:
                r_list_items.append(_convert_value_to_r(item))

        # Create R list (ListVector with numeric names)
        r_list = ListVector({str(i): item for i, item in enumerate(r_list_items)})
        robjects.globalenv[name] = r_list
        return

    # Handle basic types
    if isinstance(obj, str):
        escaped = _escape_r_string(obj)
        run(f'{name} <- "{escaped}"')
    elif isinstance(obj, bool):
        run(f"{name} <- {'TRUE' if obj else 'FALSE'}")
    elif isinstance(obj, (int, float)):
        run(f"{name} <- {obj}")
    else:
        # Final fallback: try to assign as-is
        robjects.r.assign(name, obj)


def assign(name: str, obj: Any) -> None:
    """
    Assign a Python object into the R global environment.

    Supports recursive structures:
    - entity = int | float | str | bool | pl.DataFrame | list[entity] | dict[entity]

    Args:
        name: Variable name in R global environment
        obj: Python object to assign

    Examples:
        >>> import polars as pl
        >>> from statract.r import assign
        >>>
        >>> # Assign polars DataFrame
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> assign("my_df", df)
        >>>
        >>> # Assign nested structures
        >>> data = {
        ...     "df1": pl.DataFrame({"x": [1, 2]}),
        ...     "nested": {
        ...         "df2": pl.DataFrame({"y": [3, 4]}),
        ...         "list": [pl.DataFrame({"z": [5, 6]})]
        ...     }
        ... }
        >>> assign("my_data", data)
    """
    _assign_recursive(name, obj, is_top_level=True)


def assign_via_parquet(name: str, df: pl.DataFrame) -> None:
    """Assign a Polars DataFrame to R via a temporary parquet file.

    Uses R ``arrow`` or ``polars`` to read the file instead of rpy2_arrow in-process
    conversion, which can trigger heap corruption (malloc errors) on some macOS +
    ``RPY2_CFFI_MODE=ABI`` setups.
    """
    import tempfile

    _require_rpy2()
    if robjects is None:
        raise RNotAvailableError(_R_INSTALL_HINT)
    if not isinstance(df, pl.DataFrame):
        raise TypeError("assign_via_parquet expects a Polars DataFrame")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        df.write_parquet(path)
        path_r = str(path.resolve()).replace("\\", "/")
        run(
            f"""
            .assign_parquet_path <- "{path_r}"
            if (requireNamespace("arrow", quietly = TRUE)) {{
              assign("{name}", as.data.frame(arrow::read_parquet(.assign_parquet_path)), envir = .GlobalEnv)
            }} else if (requireNamespace("polars", quietly = TRUE)) {{
              assign("{name}", as.data.frame(polars::pl$read_parquet(.assign_parquet_path)$to_data_frame()), envir = .GlobalEnv)
            }} else {{
              stop("assign_via_parquet requires R package 'arrow' or 'polars'")
            }}
            rm(.assign_parquet_path)
            """,
        )
        enum_levels = get_enum_levels(df)
        if enum_levels:
            r_df = robjects.globalenv[name]
            r_df = _apply_enum_levels_to_tibble(r_df, enum_levels)
            robjects.globalenv[name] = r_df


# def as_tibble_preserve_enum(
#     df: pl.DataFrame,
#     r_df_name: str = "df",
#     tibble_name: str = "tdf",
# ) -> None:
#     """Create an R tibble from the Polars DataFrame already in R, preserving Enum order.

#     Use this when you have put a Polars DataFrame into R (e.g. via rpy2polars)
#     and want a tibble with factor levels in the same order as the Polars Enum.

#     The Polars DataFrame must already exist in R under `r_df_name`. This function
#     creates `tibble_name <- as_tibble(r_df_name)` and then re-applies factor levels
#     from the Python schema so that R factor order matches the Enum definition order.

#     Args:
#         df: The same Polars DataFrame (used only to read Enum schema).
#         r_df_name: Name of the variable in R holding the polars DataFrame.
#         tibble_name: Name of the tibble variable to create in R.

#     Examples:
#         >>> import polars as pl
#         >>> import rpy2.robjects as robjects
#         >>> import rpy2_arrow.polars as rpy2polars
#         >>> from statract.r import as_tibble_preserve_enum
#         >>>
#         >>> df = pl.DataFrame({"depth": ["Tis", "T1a"]}).with_columns(
#         ...     pl.col("depth").cast(pl.Enum(["Tis", "T1a", "T1b", "T2"]))
#         ... )
#         >>> with rpy2polars.converter.context():
#         ...     robjects.globalenv["df"] = df
#         >>> as_tibble_preserve_enum(df, r_df_name="df", tibble_name="tdf")
#         >>> # In R: levels(tdf$depth) is now c("Tis", "T1a", "T1b", "T2")
#     """
#     _require_rpy2()
#     if robjects is None or rpy2polars is None:
#         raise RNotAvailableError(_R_INSTALL_HINT)
#     enum_levels = get_enum_levels(df)
#     run(f"library(tibble); {tibble_name} <- as_tibble({r_df_name})")
#     r_tibble = robjects.globalenv[tibble_name]
#     r_tibble = _apply_enum_levels_to_tibble(r_tibble, enum_levels)
#     robjects.globalenv[tibble_name] = r_tibble


def save_session(data_path: Path | str) -> None:
    """Save R session to .RData file.

    Saves all variables in the current R global environment to a .RData file.

    Args:
        data_path: Path to save .RData file

    Examples:
        >>> from pathlib import Path
        >>> from statract.r import save_session
        >>>
        >>> # Save current R session
        >>> save_session(Path("cache/build.RData"))
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    assert not data_path.is_dir(), f"Cannot save to directory: {data_path}"
    _require_rpy2()
    if robjects is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    robjects.r(f"save.image('{data_path}')")


def get(name: str) -> Any:
    """
    Get an R object from the global environment and convert to appropriate Python type.

    Supports:
    - data.frame/tibble -> polars DataFrame (via rpy2_arrow.polars)
    - Other R objects -> appropriate Python types

    Args:
        name: Variable name in R global environment

    Returns:
        Python object converted from R object

    Examples:
        >>> from statract.r import get
        >>>
        >>> # Get polars DataFrame from R
        >>> df = get("my_df")
        >>>
        >>> # Get other R objects
        >>> my_list = get("my_list")
    """
    _require_rpy2()
    if robjects is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    if name not in robjects.globalenv:
        raise KeyError(f"Variable '{name}' not found in R global environment")

    robj = robjects.globalenv[name]

    # Recursively convert (handles data.frame/tibble/polars DataFrame automatically)
    return _get_recursive(robj)


def _get_recursive(robj: Any) -> Any:
    """
    Recursively convert R object to Python, handling nested structures.

    This is an internal helper function for get().
    Uses rpy2polars.converter.context() for the entire conversion process.
    """
    _require_rpy2()
    if robjects is None or ListVector is None or Vector is None or rpy2polars is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    # Use rpy2polars.converter.context() for the entire conversion
    with rpy2polars.converter.context():
        return _get_recursive_impl(robj)


def _get_recursive_impl(robj: Any) -> Any:
    """
    Internal implementation of recursive conversion.
    Should be called within rpy2polars.converter.context().
    """
    _require_rpy2()
    if robjects is None or ListVector is None or Vector is None:
        raise RNotAvailableError(_R_INSTALL_HINT)

    # Check if it's a data.frame/tibble or polars DataFrame
    is_data_frame = False
    is_polars_df = False

    try:
        # Check if it's a data.frame or tibble
        result = robjects.r("inherits")(robj, "data.frame")
        is_data_frame = bool(result[0]) if hasattr(result, "__getitem__") and len(result) > 0 else bool(result)
    except Exception:  # noqa: BLE001
        pass

    if not is_data_frame:
        # Check if it's a polars DataFrame
        try:
            rclass = getattr(robj, "rclass", None)
            if rclass is not None:
                class_names = [str(x) for x in rclass]
                if "polars_data_frame" in class_names or "polars_object" in class_names:
                    is_data_frame = True
                    is_polars_df = True
        except Exception:  # noqa: BLE001
            pass

    if is_data_frame:
        if is_polars_df:
            # R polars DataFrame: convert directly to Python Polars DataFrame
            return robjects.conversion.rpy2py(robj)
        else:
            # It's a data.frame/tibble, convert to polars DataFrame
            # Assign to temporary variable first, then convert using as_polars_df
            temp_ref = f"_temp_ref_{abs(hash(str(robj)))}"
            temp_polars = f"_temp_polars_{abs(hash(str(robj)))}"
            try:
                # polars library is loaded in init.R
                # Assign R object to temporary variable
                robjects.globalenv[temp_ref] = robj
                # Convert to polars DataFrame using polars::as_polars_df
                robjects.globalenv[temp_polars] = robjects.r("polars::as_polars_df")(robjects.globalenv[temp_ref])
                result = robjects.conversion.rpy2py(robjects.globalenv[temp_polars])
                return result
            finally:
                # Clean up temporary variables
                try:
                    del robjects.globalenv[temp_ref]
                except Exception:  # noqa: BLE001
                    pass
                try:
                    del robjects.globalenv[temp_polars]
                except Exception:  # noqa: BLE001
                    pass

    # Named list (ListVector): convert to dict or list
    if isinstance(robj, ListVector):
        # Check if names exist and are not NULL
        try:
            names = robj.names
            has_names = names is not None and str(names) != "NULL" and len(names) > 0
        except (AttributeError, TypeError):
            has_names = False

        if has_names:
            # Check if names are numeric (0, 1, 2, ...) - indicates unnamed list
            names_are_numeric = all(str(name).isdigit() for name in names)
            if names_are_numeric:
                # Unnamed list: convert to Python list
                return [_get_recursive_impl(robj.rx2(name)) for name in names]

            # Named list: convert to dict
            return {str(name): _get_recursive_impl(robj.rx2(name)) for name in names}
        else:
            # ListVector without names: convert to Python list using indices
            return [_get_recursive_impl(robj[i]) for i in range(len(robj))]

    # Vector (unnamed list): convert to list
    if isinstance(robj, Vector) and not isinstance(robj, ListVector):
        result = [_get_recursive_impl(robj[i]) for i in range(len(robj))]
        return result[0] if len(result) == 1 else result

    # Default conversion
    converted = robjects.conversion.rpy2py(robj)
    if isinstance(converted, Vector) and len(converted) == 1:
        return converted[0]
    return converted


def _escape_r_string(value: str) -> str:
    """Escape special characters in R string literals."""
    return value.replace('"', '\\"').replace("\\", "\\\\")
