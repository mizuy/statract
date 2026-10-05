from __future__ import annotations

import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest


def _require_rpy2_and_r_arrow():
    rpy2 = pytest.importorskip("rpy2")
    pytest.importorskip("rpy2_arrow")
    from statract import r as pr

    # Initialize R before checking for packages
    pr.init()

    # Check for rpy2_arrow.polars support
    try:
        import rpy2_arrow.polars as rpy2polars  # noqa: F401
    except ImportError:
        pytest.skip("rpy2_arrow.polars is required for endolab R bridge tests")
    return rpy2, pr


def test_run_basic():
    """Test basic R command execution."""
    _, pr = _require_rpy2_and_r_arrow()

    # Test simple calculation
    result = pr.run("2 + 2")
    assert result is not None

    # Test variable assignment and retrieval
    pr.run("x <- 42")
    result = pr.run("x")
    assert result[0] == 42


def test_run_with_show():
    """Test run with show=True."""
    _, pr = _require_rpy2_and_r_arrow()

    # Should not raise
    pr.run("cat('test output')", show=True)


def test_run_with_return_console():
    """Test run with return_console=True."""
    _, pr = _require_rpy2_and_r_arrow()

    result, console = pr.run("cat('test'); 42", return_console=True)
    assert result is not None
    assert "test" in console


def test_run_error_handling():
    """Test that R errors are properly raised."""
    _, pr = _require_rpy2_and_r_arrow()

    from rpy2.rinterface_lib.embedded import RRuntimeError

    with pytest.raises(RRuntimeError, match="test error"):
        pr.run("stop('test error')")


def test_init():
    """Test R initialization."""
    _, pr = _require_rpy2_and_r_arrow()

    # Should not raise
    pr.init()


def test_has_r_package():
    """Test checking for R package availability."""
    _, pr = _require_rpy2_and_r_arrow()

    # Test with existing package
    assert pr.has_r_package("base") is True
    assert pr.has_r_package("arrow") is True

    # Test with non-existent package
    assert pr.has_r_package("nonexistent_package_xyz123") is False


def test_assign_polars_dataframe():
    """Test assigning a single Polars DataFrame to R."""
    _, pr = _require_rpy2_and_r_arrow()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    pr.assign("test_df", df)

    # Verify in R
    result = pr.run("nrow(test_df)")
    assert result[0] == 3

    result = pr.run("ncol(test_df)")
    assert result[0] == 2


def test_assign_polars_dataframe_initializes_r_resources_in_fresh_process():
    """A cold R bridge must run library(polars) before Polars assignment."""
    pytest.importorskip("rpy2")
    pytest.importorskip("rpy2_arrow")
    code = """
import polars as pl
from statract import r as pr

pr.assign("cold_df", pl.DataFrame({"x": [1]}))
attached = pr.run('"polars" %in% (.packages())')
assert bool(attached[0]) is True
result = pr.run("nrow(cold_df)")
assert result[0] == 1
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_cold_assign_sources_library_polars_before_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a live R session, assigning Polars still sources init.R first."""
    from statract import r as pr

    order: list[str] = []

    class _Global:
        def __getitem__(self, key: str) -> str:
            return "stored"

        def __setitem__(self, key: str, value: object) -> None:
            return None

        def __delitem__(self, key: str) -> None:
            return None

    class _Conversion:
        @staticmethod
        def py2rpy(value: object) -> str:
            order.append("py2rpy")
            return "r-df"

    class _Robjects:
        globalenv = _Global()
        conversion = _Conversion()

        @staticmethod
        def r(command: str):
            assert command == "tibble::as_tibble"
            return lambda obj: "tibble"

    class _Context:
        def __enter__(self) -> _Context:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    class _Converter:
        @staticmethod
        def context() -> _Context:
            return _Context()

    class _Rpy2Polars:
        converter = _Converter()

    def fake_run(command: str, *, show: bool = False, return_console: bool = False) -> None:
        order.append(command)

    monkeypatch.setattr(pr, "_r_resources_initialized", False)
    monkeypatch.setattr(pr, "_r_resources_initializing", False)
    monkeypatch.setattr(pr, "_rpy2_loaded", True)
    monkeypatch.setattr(pr, "rpy2", object())
    monkeypatch.setattr(pr, "robjects", _Robjects())
    monkeypatch.setattr(pr, "RRuntimeError", RuntimeError)
    monkeypatch.setattr(pr, "ListVector", dict)
    monkeypatch.setattr(pr, "rpy2polars", _Rpy2Polars())
    monkeypatch.setattr(pr, "run", fake_run)

    pr.assign("cold_df", pl.DataFrame({"x": [1]}))

    assert order[0].lstrip().startswith("library(tidyverse)")
    assert "library(polars)" in order[0]
    assert order.index("py2rpy") > 0
    sourced = len(order)
    pr.assign("cold_df_2", pl.DataFrame({"x": [2]}))
    assert order[sourced:] == ["py2rpy"]


def test_assign_dfs():
    """Test assigning multiple DataFrames to R."""
    _, pr = _require_rpy2_and_r_arrow()

    df1 = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    df2 = pl.DataFrame({"id": [3, 4], "y": [30, 40]})

    dfs = {
        "df1": df1,
        "df2": df2,
    }

    pr.assign_dfs(dfs)

    # Verify both DataFrames in R
    result = pr.run("nrow(df1)")
    assert result[0] == 2

    result = pr.run("nrow(df2)")
    assert result[0] == 2


def test_assign_polars_dataframe_with_enum():
    """Test assigning DataFrame with Enum column."""
    _, pr = _require_rpy2_and_r_arrow()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "group": pl.Series(["a", "b", "a"], dtype=pl.Enum(["a", "b", "c"])),
        }
    )

    pr.assign("df_enum", df)

    # Verify by getting back and checking Enum is preserved (may become Categorical via tibble)
    df_out = pr.get("df_enum")
    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == (3, 2)
    assert isinstance(df_out.schema["group"], (pl.Enum, pl.Categorical))
    if isinstance(df_out.schema["group"], pl.Enum):
        assert df_out.schema["group"].categories.to_list() == ["a", "b", "c"]


def test_assign_polars_dataframe_with_datetime():
    """Test assigning DataFrame with datetime column."""
    _, pr = _require_rpy2_and_r_arrow()

    df = pl.DataFrame(
        {
            "id": [1, 2],
            "ts": pl.Series(
                [dt.datetime(2024, 1, 1, 9, 0, 0), dt.datetime(2024, 1, 2, 9, 0, 0)],
                dtype=pl.Datetime(time_zone="UTC"),
            ),
        }
    )

    pr.assign("df_dt", df)

    # Verify by getting back and checking datetime is preserved
    df_out = pr.get("df_dt")
    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == (2, 2)
    assert isinstance(df_out.schema["ts"], pl.Datetime)


def test_get_polars_dataframe():
    """Test retrieving DataFrame from R."""
    _, pr = _require_rpy2_and_r_arrow()

    # Create a DataFrame in R
    pr.run("""
        test_r_df <- data.frame(
            id = c(1, 2, 3),
            value = c(10, 20, 30)
        )
    """)

    df = pr.get("test_r_df")

    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    assert "id" in df.columns
    assert "value" in df.columns
    assert df["id"].to_list() == [1, 2, 3]
    assert df["value"].to_list() == [10, 20, 30]


def test_get_polars_dataframe_with_factor():
    """Test retrieving DataFrame with factor column from R."""
    _, pr = _require_rpy2_and_r_arrow()

    # Create a DataFrame with factor in R
    pr.run("""
        test_r_factor <- data.frame(
            id = c(1, 2, 3),
            group = factor(c("a", "b", "a"), levels = c("a", "b", "c"), ordered = TRUE)
        )
    """)

    df = pr.get("test_r_factor")

    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    # Factor should be converted to Enum or Categorical
    assert isinstance(df.schema["group"], (pl.Enum, pl.Categorical))
    if isinstance(df.schema["group"], pl.Enum):
        assert df.schema["group"].categories.to_list() == ["a", "b", "c"]
    else:
        # Categorical case
        assert list(df["group"].unique().sort()) == ["a", "b"]


def test_save_session():
    """Test saving R session to file."""
    _, pr = _require_rpy2_and_r_arrow()

    # Create some variables in R
    pr.run("x <- 42")
    pr.run('y <- "test"')

    with tempfile.TemporaryDirectory() as tmpdir:
        rdata_path = Path(tmpdir) / "test.RData"
        pr.save_session(rdata_path)

        # Verify file was created
        assert rdata_path.exists()

        # Clear R environment
        pr.run("rm(list = ls())")

        # Load session back
        pr.run(f"load('{rdata_path}')")

        # Verify variables are restored
        result = pr.run("x")
        assert result[0] == 42

        result = pr.run("y")
        assert str(result[0]) == "test"


def test_save_session_with_dataframe():
    """Test saving R session with DataFrame."""
    _, pr = _require_rpy2_and_r_arrow()

    # Use data.frame instead of polars DataFrame for RData compatibility
    pr.run("""
        test_df <- data.frame(
            id = c(1, 2, 3),
            value = c(10, 20, 30)
        )
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        rdata_path = Path(tmpdir) / "test.RData"
        pr.save_session(rdata_path)

        assert rdata_path.exists()

        # Clear and reload
        pr.run("rm(list = ls())")
        pr.run(f"load('{rdata_path}')")

        # Verify DataFrame is restored - use get() to retrieve as polars DataFrame
        df_restored = pr.get("test_df")
        assert isinstance(df_restored, pl.DataFrame)
        assert df_restored.shape == (3, 2)




def test_roundtrip_enum_and_datetime_tz():
    """Test roundtrip of Enum and datetime with timezone."""
    _, pr = _require_rpy2_and_r_arrow()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": pl.Series(["b", "a", "b", None], dtype=pl.Enum(["a", "b", "c"])),
            "ts": pl.Series(
                [
                    dt.datetime(2024, 1, 1, 9, 0, 0),
                    dt.datetime(2024, 1, 2, 9, 0, 0),
                    None,
                    dt.datetime(2024, 1, 4, 9, 0, 0),
                ],
                dtype=pl.Datetime(time_zone="UTC"),
            ),
        }
    )

    pr.assign("df_in", df)
    df_out = pr.get("df_in")

    assert df_out.shape == df.shape

    # Enum contract: preserve levels order (may become Categorical via tibble)
    assert isinstance(df_out.schema["group"], (pl.Enum, pl.Categorical))
    if isinstance(df_out.schema["group"], pl.Enum):
        assert df_out.schema["group"].categories.to_list() == ["a", "b", "c"]

    # Datetime contract: preserve timezone if possible.
    assert isinstance(df_out.schema["ts"], pl.Datetime)
    assert df_out.schema["ts"].time_zone == "UTC"


def test_roundtrip_r_factor_becomes_enum():
    """Test that R factor becomes Enum in Polars."""
    _, pr = _require_rpy2_and_r_arrow()

    # Create a data.frame (not polars DataFrame) with factor in R
    pr.run("""
        df_factor <- data.frame(
            x = c("b", "a", "b", NA),
            stringsAsFactors = FALSE
        )
        df_factor$x <- factor(df_factor$x, levels=c("a","b","c"), ordered=TRUE)
    """)

    df_out = pr.get("df_factor")
    assert isinstance(df_out, pl.DataFrame)
    assert isinstance(df_out.schema["x"], (pl.Enum, pl.Categorical))
    if isinstance(df_out.schema["x"], pl.Enum):
        assert df_out.schema["x"].categories.to_list() == ["a", "b", "c"]


# Comprehensive tests for assign and get functions
def test_assign_get_roundtrip_basic_dataframe():
    """Test roundtrip: Python -> R -> Python with basic DataFrame."""
    _, pr = _require_rpy2_and_r_arrow()

    df_in = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "score": [85.5, 90.0, 88.5, 92.0],
        }
    )

    pr.assign("test_df", df_in)
    df_out = pr.get("test_df")

    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == df_in.shape
    assert df_out.columns == df_in.columns
    assert df_out["id"].to_list() == df_in["id"].to_list()
    assert df_out["name"].to_list() == df_in["name"].to_list()
    assert df_out["age"].to_list() == df_in["age"].to_list()
    assert df_out["score"].to_list() == df_in["score"].to_list()


def test_assign_get_roundtrip_with_enum():
    """Test roundtrip: Python -> R -> Python with Enum column."""
    _, pr = _require_rpy2_and_r_arrow()

    df_in = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "status": pl.Series(
                ["active", "inactive", "active", "pending"],
                dtype=pl.Enum(["active", "inactive", "pending", "deleted"]),
            ),
        }
    )

    pr.assign("test_df_enum", df_in)
    df_out = pr.get("test_df_enum")

    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == df_in.shape
    # Enum should be preserved (may become Categorical via tibble)
    assert isinstance(df_out.schema["status"], (pl.Enum, pl.Categorical))
    if isinstance(df_out.schema["status"], pl.Enum):
        assert df_out.schema["status"].categories.to_list() == ["active", "inactive", "pending", "deleted"]
    assert df_out["status"].to_list() == df_in["status"].to_list()


def test_assign_get_roundtrip_with_categorical():
    """Test roundtrip: Python -> R -> Python with Categorical column."""
    _, pr = _require_rpy2_and_r_arrow()

    df_in = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": pl.Series(["A", "B", "A"], dtype=pl.Categorical),
        }
    )

    pr.assign("test_df_cat", df_in)
    df_out = pr.get("test_df_cat")

    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == df_in.shape
    # Categorical should be converted to Enum or remain Categorical
    assert df_out["category"].to_list() == df_in["category"].to_list()


def test_assign_get_roundtrip_with_datetime():
    """Test roundtrip: Python -> R -> Python with Datetime column."""
    _, pr = _require_rpy2_and_r_arrow()

    df_in = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "timestamp": pl.Series(
                [
                    dt.datetime(2024, 1, 1, 12, 0, 0),
                    dt.datetime(2024, 1, 2, 12, 0, 0),
                    dt.datetime(2024, 1, 3, 12, 0, 0),
                ],
                dtype=pl.Datetime(time_zone="UTC"),
            ),
        }
    )

    pr.assign("test_df_dt", df_in)
    df_out = pr.get("test_df_dt")

    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == df_in.shape
    # Datetime should be preserved
    assert isinstance(df_out.schema["timestamp"], pl.Datetime)
    # Values should match (allowing for timezone conversion)
    assert len(df_out["timestamp"]) == len(df_in["timestamp"])


def test_assign_get_roundtrip_with_mixed_types():
    """Test roundtrip: Python -> R -> Python with mixed column types."""
    _, pr = _require_rpy2_and_r_arrow()

    df_in = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "status": pl.Series(
                ["active", "inactive", "active"],
                dtype=pl.Enum(["active", "inactive", "pending"]),
            ),
            "score": [85.5, 90.0, 88.5],
            "timestamp": pl.Series(
                [
                    dt.datetime(2024, 1, 1, 12, 0, 0),
                    dt.datetime(2024, 1, 2, 12, 0, 0),
                    dt.datetime(2024, 1, 3, 12, 0, 0),
                ],
                dtype=pl.Datetime(time_zone="UTC"),
            ),
            "is_active": [True, False, True],
        }
    )

    pr.assign("test_df_mixed", df_in)
    df_out = pr.get("test_df_mixed")

    assert isinstance(df_out, pl.DataFrame)
    assert df_out.shape == df_in.shape
    assert df_out.columns == df_in.columns
    assert df_out["id"].to_list() == df_in["id"].to_list()
    assert df_out["name"].to_list() == df_in["name"].to_list()
    assert df_out["score"].to_list() == df_in["score"].to_list()
    assert df_out["is_active"].to_list() == df_in["is_active"].to_list()
    # Enum should be preserved (may become Categorical via tibble)
    assert isinstance(df_out.schema["status"], (pl.Enum, pl.Categorical))


def test_assign_get_basic_types():
    """Test assign and get with basic Python types."""
    _, pr = _require_rpy2_and_r_arrow()

    # Test integer
    pr.assign("test_int", 42)
    result = pr.get("test_int")
    # R returns scalars as vectors, so extract first element if it's a vector
    if hasattr(result, "__getitem__") and hasattr(result, "__len__") and len(result) == 1:
        result = result[0]
    assert result == 42 or abs(float(result) - 42) < 1e-10

    # Test float
    pr.assign("test_float", 3.14)
    result = pr.get("test_float")
    if hasattr(result, "__getitem__") and hasattr(result, "__len__") and len(result) == 1:
        result = result[0]
    assert abs(float(result) - 3.14) < 1e-10

    # Test string
    pr.assign("test_str", "hello")
    result = pr.get("test_str")
    if hasattr(result, "__getitem__") and hasattr(result, "__len__") and len(result) == 1:
        result = result[0]
    assert str(result) == "hello"

    # Test boolean
    pr.assign("test_bool", True)
    result = pr.get("test_bool")
    if hasattr(result, "__getitem__") and hasattr(result, "__len__") and len(result) == 1:
        result = result[0]
    assert result is True or result == 1


def test_assign_get_list():
    """Test assign and get with list."""
    _, pr = _require_rpy2_and_r_arrow()

    test_list = [1, 2, 3, 4, 5]
    pr.assign("test_list", test_list)
    result = pr.get("test_list")

    # R converts list to vector, so check if it's a vector-like object
    if hasattr(result, "__getitem__"):
        assert len(result) == len(test_list)
        assert list(result)[:5] == test_list
    else:
        assert result == test_list


def test_assign_get_dict():
    """Test assign and get with dict."""
    _, pr = _require_rpy2_and_r_arrow()

    test_dict = {"a": 1, "b": 2, "c": 3}
    pr.assign("test_dict", test_dict)
    result = pr.get("test_dict")

    # R converts dict to named list
    if hasattr(result, "names"):
        assert "a" in result.names
        assert "b" in result.names
        assert "c" in result.names




def test_get_nonexistent_variable():
    """Test get with nonexistent variable raises error."""
    _, pr = _require_rpy2_and_r_arrow()

    with pytest.raises(KeyError, match="not found"):
        pr.get("nonexistent_variable_xyz123")


def test_assign_get_dict_dataframes():
    """Test assign and get with dict[pl.DataFrame]."""
    _, pr = _require_rpy2_and_r_arrow()

    dfs_in = {
        "df1": pl.DataFrame({"id": [1, 2], "x": [10, 20]}),
        "df2": pl.DataFrame({"id": [3, 4], "y": [30, 40]}),
        "df3": pl.DataFrame(
            {
                "id": [5, 6],
                "status": pl.Series(
                    ["active", "inactive"],
                    dtype=pl.Enum(["active", "inactive", "pending"]),
                ),
            }
        ),
    }

    # Assign dict[pl.DataFrame]
    pr.assign("test_dict", dfs_in)

    # Verify in R
    result = pr.run("names(test_dict)")
    keys = list(result)
    assert "df1" in keys
    assert "df2" in keys
    assert "df3" in keys

    # Get back as dict[pl.DataFrame]
    dfs_out = pr.get("test_dict")

    assert isinstance(dfs_out, dict)
    assert set(dfs_out.keys()) == {"df1", "df2", "df3"}

    # Verify each DataFrame
    assert isinstance(dfs_out["df1"], pl.DataFrame)
    assert dfs_out["df1"].shape == (2, 2)
    assert dfs_out["df1"].columns == ["id", "x"]
    assert dfs_out["df1"]["id"].to_list() == [1, 2]
    assert dfs_out["df1"]["x"].to_list() == [10, 20]

    assert isinstance(dfs_out["df2"], pl.DataFrame)
    assert dfs_out["df2"].shape == (2, 2)
    assert dfs_out["df2"].columns == ["id", "y"]
    assert dfs_out["df2"]["id"].to_list() == [3, 4]
    assert dfs_out["df2"]["y"].to_list() == [30, 40]

    assert isinstance(dfs_out["df3"], pl.DataFrame)
    assert dfs_out["df3"].shape == (2, 2)
    assert dfs_out["df3"].columns == ["id", "status"]
    # Enum should be preserved (may become Categorical via tibble)
    assert isinstance(dfs_out["df3"].schema["status"], (pl.Enum, pl.Categorical))
    if isinstance(dfs_out["df3"].schema["status"], pl.Enum):
        assert dfs_out["df3"].schema["status"].categories.to_list() == ["active", "inactive", "pending"]


def test_assign_get_roundtrip_dict_dataframes():
    """Test roundtrip: Python -> R -> Python with dict[pl.DataFrame]."""
    _, pr = _require_rpy2_and_r_arrow()

    dfs_in = {
        "df1": pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
        "df2": pl.DataFrame(
            {
                "id": [4, 5, 6],
                "status": pl.Series(
                    ["a", "b", "a"],
                    dtype=pl.Enum(["a", "b", "c"]),
                ),
            }
        ),
    }

    # Roundtrip
    pr.assign("test_dict", dfs_in)
    dfs_out = pr.get("test_dict")

    assert isinstance(dfs_out, dict)
    assert len(dfs_out) == len(dfs_in)
    assert set(dfs_out.keys()) == set(dfs_in.keys())

    # Verify df1
    assert dfs_out["df1"].shape == dfs_in["df1"].shape
    assert dfs_out["df1"].columns == dfs_in["df1"].columns
    assert dfs_out["df1"]["id"].to_list() == dfs_in["df1"]["id"].to_list()
    assert dfs_out["df1"]["value"].to_list() == dfs_in["df1"]["value"].to_list()

    # Verify df2 with Enum
    assert dfs_out["df2"].shape == dfs_in["df2"].shape
    assert dfs_out["df2"].columns == dfs_in["df2"].columns
    assert isinstance(dfs_out["df2"].schema["status"], (pl.Enum, pl.Categorical))
    if isinstance(dfs_out["df2"].schema["status"], pl.Enum):
        assert dfs_out["df2"].schema["status"].categories.to_list() == ["a", "b", "c"]
    assert dfs_out["df2"]["status"].to_list() == dfs_in["df2"]["status"].to_list()


def test_assign_get_recursive_structure():
    """Test assign and get with recursive structure: entity = int | float | str | bool | pl.DataFrame | list[entity] | dict[entity]."""
    _, pr = _require_rpy2_and_r_arrow()

    # Complex recursive structure
    data_in = {
        "df1": pl.DataFrame({"id": [1, 2], "x": [10, 20]}),
        "nested": {
            "df2": pl.DataFrame({"id": [3, 4], "y": [30, 40]}),
            "list": [
                pl.DataFrame({"id": [5], "z": [50]}),
                "string_value",
                42,
                True,
            ],
            "simple_dict": {
                "a": 1,
                "b": 2.5,
                "c": "test",
            },
            "nested_list": [
                [1, 2, 3],
                ["a", "b", "c"],
            ],
        },
        "simple_list": [1, 2, 3],
        "value": 100,
        "float_value": 3.14,
        "bool_value": True,
        "string_value": "hello",
    }

    # Assign
    pr.assign("test_recursive", data_in)

    # Get back
    data_out = pr.get("test_recursive")

    # Verify structure
    assert isinstance(data_out, dict)
    assert set(data_out.keys()) == set(data_in.keys())

    # Verify top-level DataFrame
    assert isinstance(data_out["df1"], pl.DataFrame)
    assert data_out["df1"].shape == data_in["df1"].shape

    # Verify nested dict
    assert isinstance(data_out["nested"], dict)
    nested = data_out["nested"]
    assert isinstance(nested["df2"], pl.DataFrame)
    assert nested["df2"].shape == data_in["nested"]["df2"].shape

    # Verify nested list
    assert isinstance(nested["list"], list)
    assert len(nested["list"]) == 4
    assert isinstance(nested["list"][0], pl.DataFrame)
    assert nested["list"][0].shape == (1, 2)
    assert nested["list"][1] == "string_value"
    assert nested["list"][2] == 42
    assert nested["list"][3] is True

    # Verify nested simple_dict
    assert isinstance(nested["simple_dict"], dict)
    assert nested["simple_dict"]["a"] == 1
    assert nested["simple_dict"]["b"] == 2.5
    assert nested["simple_dict"]["c"] == "test"

    # Verify nested_list
    assert isinstance(nested["nested_list"], list)
    assert len(nested["nested_list"]) == 2
    assert nested["nested_list"][0] == [1, 2, 3] or nested["nested_list"][0] == [1.0, 2.0, 3.0]
    assert nested["nested_list"][1] == ["a", "b", "c"]

    # Verify simple_list
    assert isinstance(data_out["simple_list"], list)
    assert len(data_out["simple_list"]) == 3

    # Verify scalar values
    assert isinstance(data_out["value"], (int, float))
    assert data_out["value"] == 100
    assert isinstance(data_out["float_value"], (int, float))
    assert abs(data_out["float_value"] - 3.14) < 0.01
    assert data_out["bool_value"] is True
    assert data_out["string_value"] == "hello"


def test_assign_get_roundtrip_recursive_structure():
    """Test roundtrip with complex recursive structure."""
    _, pr = _require_rpy2_and_r_arrow()

    data_in = {
        "level1": {
            "df": pl.DataFrame(
                {
                    "id": [1, 2],
                    "status": pl.Series(
                        ["active", "inactive"],
                        dtype=pl.Enum(["active", "inactive", "pending"]),
                    ),
                }
            ),
            "list": [
                {
                    "nested_df": pl.DataFrame({"x": [1, 2, 3]}),
                    "value": 42,
                },
                "string",
            ],
            "value": 100,
        },
        "top_level_df": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
    }

    # Roundtrip
    pr.assign("test_complex", data_in)
    data_out = pr.get("test_complex")

    # Verify structure
    assert isinstance(data_out, dict)
    assert "level1" in data_out
    assert "top_level_df" in data_out

    # Verify level1
    level1 = data_out["level1"]
    assert isinstance(level1, dict)
    assert isinstance(level1["df"], pl.DataFrame)
    assert isinstance(level1["df"].schema["status"], (pl.Enum, pl.Categorical))
    assert isinstance(level1["list"], list)
    assert isinstance(level1["list"][0], dict)
    assert isinstance(level1["list"][0]["nested_df"], pl.DataFrame)
    assert level1["list"][0]["value"] == 42
    assert level1["list"][1] == "string"
    assert level1["value"] == 100

    # Verify top_level_df
    assert isinstance(data_out["top_level_df"], pl.DataFrame)
    assert data_out["top_level_df"].shape == (3, 2)
