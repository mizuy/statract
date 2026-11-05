# tableone — Overview

This is the baseline table. `tableone` takes a Polars table and returns a great-tables Table 1. `tableone_raw` returns the same content as a DataFrame. Aggregation lives in `statract.tableone.agg`, and tests between groups live in `statract.tableone.stat`.

| Layer | Module | Role |
|----|------------|------|
| Table | `statract.tableone.table` | `tableone`, `tableone_raw`, `tableone_gt_from_frame` |
| Aggregation | `statract.tableone.agg` | Mean, median, proportion, category counts |
| Tests | `statract.tableone.stat` | Chi-squared, Fisher, ANOVA, Kruskal–Wallis, SMD, odds ratio. Used by `add_pvalue=True` and `add_smd=True` |

`hue` is the group column. Group columns are in the order of the values: numbers sort as numbers, a `pl.Enum` keeps its level order, and String and Categorical sort as text. Rows whose `hue` is null are left out of the group columns, the header counts, the P value and the SMD; the `All` column (`add_all=True`) still counts every row. Child rows of a category have names that start with two ASCII spaces. In the display, these become full-width spaces. Levels of a `pl.Enum` stay as rows even when their count is 0. A level that one group lacks shows a zero in the same format as the other cells, such as `0 (0.0%)`. A P value is `-` when fewer than two groups have data for that row. If you pass `None` as the summary, String, Categorical and Enum columns get one row per level (Enum in its level order). `agg_bool_category` shows a Boolean column as the rows `false` and `true`.

```python
import polars as pl
from statract import agg_category_n, tableone

frame = pl.DataFrame(
    {
        "group": ["A", "A", "B"],
        "category": pl.Series(["X", "Y", "X"], dtype=pl.Enum(["X", "Y", "Z"])),
    }
)
tableone(
    frame,
    {"Category": ("category", agg_category_n)},
    hue="group",
    add_all=True,
)
```

`add_pvalue=True` adds a column of P values between groups. If you leave out `statfunc`, categories use Fisher, median summaries (such as `agg_median_iqr`) use Kruskal–Wallis, and other numeric rows use ANOVA. You can also pick a test by name, as in `statfunc="kruskal"`. If you pass several columns to `hue`, the test uses the same group combinations as the table columns. Fisher on tables larger than 2×2 uses Monte Carlo. The seed is fixed, so the same table gives the same P value. P values follow the NEJM style: two decimals from 0.01 up, three decimals from 0.001 to below 0.01, and `P<.001` below that. When a test cannot be done or gives no P value (for example ANOVA on a constant variable), the cell shows `-`.

`add_smd=True` adds a column of standardized differences (SMD), the same as R's tableone. For numeric rows, it is the difference in means between 2 groups divided by the square root of the mean of the sample variances. For categories, it is the multinomial version by Yang and Dalton. With 3 or more groups, it is the mean over all pairs. The standalone function is `standardized_difference`. The denominator is computed within the table you pass. So the SMD in a Table 1 after matching may not match a balance table (which fixes the denominator before adjustment). A numeric 0/1 column uses the continuous formula when summarized by mean, and the $p(1-p)$ formula when summarized as a category. See the [stat implementation notes](../api/tableone/stat.md#smd) for details.

To change the headers, `write_tableone_artifacts` and `tableone_gt_from_frame` take `column_labels={"0": "Control"}` and `column_spanners=[("Before matching", ["0", "1"])]`, which puts one label over several columns. If every row name has the form `"Group | item"` (such as `"Background | Age"`), each group becomes a bold heading row with its items indented below. Category level rows (`"  F"`) do not need the prefix; they go under the item above them. `tableone_gt_from_frame` also styles a table you built yourself, such as two `tableone_raw` results joined side by side.

Tables on public data are on each page of the [examples](../examples/index.md). The API pages are [tableone](../api/tableone/tableone.md), [agg](../api/tableone/agg.md), and [stat](../api/tableone/stat.md). For estimation, see [models](../models/overview.md).
