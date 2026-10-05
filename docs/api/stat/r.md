# r

R bridge utilities for statistical analysis.

This module provides utilities for interfacing with R via rpy2, including rpy2_arrow.polars for efficient data transfer between Python (Polars) and R.

::: statract.r

## Overview

The `r` module provides a bridge between Python and R, enabling seamless data exchange using Polars DataFrames. It uses `rpy2_arrow.polars` for efficient conversion between Python Polars DataFrames and R polars DataFrames.

## Key Features

- **Data Exchange**: Convert Polars DataFrames between Python and R
- **Recursive Structures**: Support for nested dictionaries and lists containing DataFrames
- **Type Preservation**: Maintains Enum, Categorical, and Datetime types during conversion
- **R Session Management**: Save and load R sessions

## Requirements

The bridge is an optional extra. `import statract` does not load `rpy2` or start R.

- `statract[r]` (`rpy2` and `rpy2-arrow`), for example `pip install statract[r]` or `uv sync --extra r`
- A working R installation with the `polars` and `tidyverse` packages

Calling `run`, `assign`, or `get` without that extra raises `RNotAvailableError`.

## Supported Data Types

The `assign` and `get` functions support recursive data structures defined as:

```python
entity = int | float | str | bool | pl.DataFrame | list[entity] | dict[entity]
```

### Python → R Conversions (`assign`)

| Python Type | R Type | Conversion Method |
|------------|--------|-------------------|
| `int` | `integer` | Direct assignment |
| `float` | `numeric` | Direct assignment |
| `str` | `character` | Escaped string assignment |
| `bool` | `logical` | `TRUE`/`FALSE` |
| `pl.DataFrame` | `tbl_df` (tibble) | Via `rpy2_arrow.polars` → `as_tibble()` |
| `list[entity]` | `list` | Recursive conversion |
| `dict[entity]` | `list` (named) | Recursive conversion to named list |

### R → Python Conversions (`get`)

| R Type | Python Type | Conversion Method |
|--------|-------------|-------------------|
| `integer` | `int` | Direct conversion |
| `numeric` | `float` | Direct conversion |
| `character` | `str` | Direct conversion |
| `logical` | `bool` | Direct conversion |
| `tbl_df` (tibble) | `pl.DataFrame` | Via `polars::as_polars_df` → `rpy2_arrow.polars` |
| `data.frame` | `pl.DataFrame` | Via `polars::as_polars_df` → `rpy2_arrow.polars` |
| `polars_data_frame` | `pl.DataFrame` | Direct via `rpy2_arrow.polars` |
| `factor` | `pl.Enum` or `pl.Categorical` | Factor levels preserved |
| `POSIXct` | `pl.Datetime` | Timezone preserved when possible |
| `list` (named) | `dict` | Recursive conversion |
| `list` (unnamed) | `list` | Recursive conversion |

### Special Type Handling

- **Enum/Categorical**: Polars Enum columns are converted to R factors (via tibble), and may return as Enum or Categorical depending on the conversion path
- **Datetime**: Timezone information is preserved when possible during roundtrip conversion
- **Nested Structures**: Dictionaries and lists can be nested arbitrarily deep, containing any supported type

## Usage

### Basic Setup

```python
from statract.r import init, assign, get, run

# Initialize R environment (loads init.R and io.R)
init()
```

### Assigning Basic Types

```python
from statract.r import assign

# Assign scalar values
assign("x", 42)                    # integer
assign("y", 3.14)                   # numeric
assign("name", "Alice")             # character
assign("is_active", True)           # logical
```

### Assigning Polars DataFrames

```python
import polars as pl
from statract.r import assign, run

# Create a Polars DataFrame
df = pl.DataFrame({
    "id": [1, 2, 3],
    "value": [10.5, 20.3, 30.7],
    "name": ["Alice", "Bob", "Charlie"]
})

# Assign to R (stored as tibble)
assign("my_df", df)

# Verify in R
run("class(my_df)")  # Returns: ['tbl_df', 'tbl', 'data.frame']
run("nrow(my_df)")    # Returns: 3
```

### Assigning DataFrames with Special Types

```python
import polars as pl
import datetime as dt
from statract.r import assign

# DataFrame with Enum, Categorical, and Datetime
df = pl.DataFrame({
    "id": [1, 2, 3],
    "status": pl.Series(
        ["active", "inactive", "active"],
        dtype=pl.Enum(["active", "inactive", "pending"])
    ),
    "category": pl.Series(
        ["A", "B", "A"],
        dtype=pl.Categorical
    ),
    "timestamp": pl.Series(
        [
            dt.datetime(2024, 1, 1, 9, 0, 0),
            dt.datetime(2024, 1, 2, 9, 0, 0),
            dt.datetime(2024, 1, 3, 9, 0, 0)
        ],
        dtype=pl.Datetime(time_zone="UTC")
    )
})

assign("df_special", df)
```

### Assigning Lists

```python
from statract.r import assign
import polars as pl

# List of mixed types
my_list = [
    pl.DataFrame({"x": [1, 2, 3]}),
    "string",
    42,
    [1, 2, 3]
]

assign("my_list", my_list)
```

### Assigning Dictionaries

```python
from statract.r import assign
import polars as pl

# Dictionary with mixed types
my_dict = {
    "metadata": {
        "version": "1.0",
        "count": 100
    },
    "data": {
        "df1": pl.DataFrame({"x": [1, 2, 3]}),
        "df2": pl.DataFrame({"y": [4, 5, 6]})
    },
    "settings": {
        "threshold": 0.05,
        "enabled": True
    }
}

assign("my_dict", my_dict)
```

### Assigning Nested Structures

```python
from statract.r import assign
import polars as pl

# Deeply nested structure
nested = {
    "level1": {
        "df": pl.DataFrame({"id": [1, 2], "value": [10, 20]}),
        "list": [
            {
                "nested_df": pl.DataFrame({"x": [1, 2]}),
                "value": 42
            },
            "string_item"
        ],
        "metadata": {
            "version": "2.0"
        }
    },
    "level2": [
        pl.DataFrame({"y": [3, 4]}),
        {"key": "value"}
    ]
}

assign("nested_data", nested)
```

### Getting Data from R

```python
from statract.r import get, run
import polars as pl

# Get a tibble/data.frame (converted to Polars DataFrame)
df = get("my_df")
assert isinstance(df, pl.DataFrame)

# Get a list
my_list = get("my_list")
assert isinstance(my_list, list)

# Get a dictionary
my_dict = get("my_dict")
assert isinstance(my_dict, dict)

# Get nested structures
nested = get("nested_data")
assert isinstance(nested, dict)
assert isinstance(nested["level1"]["df"], pl.DataFrame)
```

### Getting R polars DataFrames

```python
from statract.r import get, run
import polars as pl

# Create R polars DataFrame
run("""
    library(polars)
    df_r_polars <- pl$DataFrame(x = 1:3, y = 4:6)
""")

# Get directly as Polars DataFrame (no intermediate conversion)
df = get("df_r_polars")
assert isinstance(df, pl.DataFrame)
```

### Running R Commands

```python
from statract.r import run

# Run R code (output suppressed by default)
result = run("x <- 42")
result = run("mean(c(1, 2, 3, 4, 5))")

# Show output
run("print('Hello from R')", show=True)

# Get console output
result, console = run("cat('Output')", return_console=True)
```

### Saving and Loading R Sessions

```python
from pathlib import Path
from statract.r import save_session, run

# Save current R session
save_session(Path("cache/build.RData"))

# Clear R environment
run("rm(list = ls())")

# Load session back
run("load('cache/build.RData')")
```

## Complete Examples

### Example 1: Statistical Analysis Workflow

```python
import polars as pl
from statract.r import init, assign, get, run

init()

# Prepare data in Python
df = pl.DataFrame({
    "id": range(1, 101),
    "group": pl.Series(
        ["A"] * 50 + ["B"] * 50,
        dtype=pl.Enum(["A", "B"])
    ),
    "value": [i * 0.1 for i in range(100)]
})

# Send to R
assign("df", df)

# Perform statistical analysis in R
run("""
    library(dplyr)
    result <- df %>%
        group_by(group) %>%
        summarise(
            mean_value = mean(value),
            sd_value = sd(value),
            n = n()
        )
""")

# Get results back
result_df = get("result")
print(result_df)
```

### Example 2: Working with Factors

```python
from statract.r import init, assign, get, run
import polars as pl

init()

# Create DataFrame with factor-like data
df = pl.DataFrame({
    "treatment": pl.Series(
        ["Control", "Treatment", "Control", "Treatment"],
        dtype=pl.Enum(["Control", "Treatment"])
    ),
    "outcome": [10, 15, 12, 18]
})

assign("df", df)

# Convert to factor in R and perform analysis
run("""
    df$treatment <- factor(df$treatment, levels = c("Control", "Treatment"))
    model <- lm(outcome ~ treatment, data = df)
    summary(model)
""")

# Get back the modified DataFrame
df_out = get("df")
# treatment column may be Enum or Categorical
```

### Example 3: Complex Nested Data Structure

```python
from statract.r import init, assign, get
import polars as pl

init()

# Complex nested structure
analysis_data = {
    "config": {
        "alpha": 0.05,
        "method": "coxph",
        "enabled": True
    },
    "datasets": {
        "training": pl.DataFrame({
            "id": [1, 2, 3],
            "time": [10, 20, 30],
            "event": [1, 1, 0]
        }),
        "validation": pl.DataFrame({
            "id": [4, 5, 6],
            "time": [15, 25, 35],
            "event": [1, 0, 1]
        })
    },
    "results": [
        {"model": "model1", "aic": 123.45},
        {"model": "model2", "aic": 120.30}
    ]
}

# Send to R
assign("analysis", analysis_data)

# Access in R (example)
# analysis$datasets$training
# analysis$config$alpha

# Get back
result = get("analysis")
assert isinstance(result["datasets"]["training"], pl.DataFrame)
assert result["config"]["alpha"] == 0.05
```

### Example 4: Roundtrip with Type Preservation

```python
import polars as pl
import datetime as dt
from statract.r import init, assign, get, run

init()

# Create DataFrame with various types
df = pl.DataFrame({
    "id": [1, 2, 3],
    "group": pl.Series(["A", "B", "A"], dtype=pl.Enum(["A", "B", "C"])),
    "timestamp": pl.Series(
        [
            dt.datetime(2024, 1, 1, 9, 0, 0),
            dt.datetime(2024, 1, 2, 9, 0, 0),
            dt.datetime(2024, 1, 3, 9, 0, 0)
        ],
        dtype=pl.Datetime(time_zone="UTC")
    ),
    "value": [10.5, 20.3, 30.7]
})

# Send to R
assign("df", df)

# Verify it's a tibble
run("class(df)")  # ['tbl_df', 'tbl', 'data.frame']

# Get back
df_out = get("df")

# Check types
assert isinstance(df_out, pl.DataFrame)
assert isinstance(df_out.schema["group"], (pl.Enum, pl.Categorical))
assert isinstance(df_out.schema["timestamp"], pl.Datetime)
assert df_out.schema["timestamp"].time_zone == "UTC"
```

## Error Handling

The module raises `RNotAvailableError` when R or required packages are not available:

```python
from statract.r import RNotAvailableError

try:
    assign("x", 42)
except RNotAvailableError as e:
    print(f"R not available: {e}")
```

When getting a non-existent variable:

```python
from statract.r import get

try:
    result = get("nonexistent_variable")
except KeyError as e:
    print(f"Variable not found: {e}")
```

## Notes

- The `polars` and `tidyverse` R packages are automatically loaded via `init.R`
- When assigning Polars DataFrames to R, they are stored as tibbles for compatibility with R's tidyverse ecosystem
- R `data.frame` and `tibble` objects are automatically converted to Polars DataFrames when retrieved
- R polars DataFrames are directly converted to Python Polars DataFrames (no intermediate tibble conversion)
- Enum columns may be converted to Categorical during roundtrip conversion via tibble
- RData files may not preserve polars DataFrames; use `data.frame` for RData compatibility
- Column access in R tibbles should use `[["column"]]` syntax rather than `$column` for polars DataFrames