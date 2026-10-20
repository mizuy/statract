# tableone — Overview

This is the baseline table. `tableone` takes a Polars table and returns a great-tables Table 1. `tableone_raw` returns the same content as a DataFrame. Aggregation lives in `statract.tableone.agg`, and tests between groups live in `statract.tableone.stat`.

| Layer | Module | Role |
|----|------------|------|
| Table | `statract.tableone.table` | `tableone`, `tableone_raw`, `tableone_gt_from_frame` |
| Aggregation | `statract.tableone.agg` | Mean, median, proportion, category counts |
| Tests | `statract.tableone.stat` | Chi-squared, Fisher, ANOVA, Kruskal–Wallis, SMD, odds ratio. Used by `add_pvalue=True` and `add_smd=True` |

`hue` is the group column. Child rows of a category have names that start with two ASCII spaces. In the display, these become full-width spaces. Levels of a `pl.Enum` stay as rows even when their count is 0.

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

`add_pvalue=True` adds a column of P values between groups. If you leave out `statfunc`, categories use Fisher, median summaries (such as `agg_median_iqr`) use Kruskal–Wallis, and other numeric rows use ANOVA. You can also pick a test by name, as in `statfunc="kruskal"`. If you pass several columns to `hue`, the test uses the same group combinations as the table columns. Fisher on tables larger than 2×2 uses Monte Carlo. The seed is fixed, so the same table gives the same P value.

`add_smd=True` adds a column of standardized differences (SMD), the same as R's tableone. For numeric rows, it is the difference in means between 2 groups divided by the square root of the mean of the sample variances. For categories, it is the multinomial version by Yang and Dalton. With 3 or more groups, it is the mean over all pairs. The standalone function is `standardized_difference`. The denominator is computed within the table you pass. So the SMD in a Table 1 after matching may not match a balance table (which fixes the denominator before adjustment). A numeric 0/1 column uses the continuous formula when summarized by mean, and the $p(1-p)$ formula when summarized as a category. See the [stat implementation notes](../api/tableone/stat.md#smd) for details.

Tables on public data are on each page of the [examples](../examples/index.md). The API pages are [tableone](../api/tableone/tableone.md), [agg](../api/tableone/agg.md), and [stat](../api/tableone/stat.md). For estimation, see [models](../models/overview.md).
