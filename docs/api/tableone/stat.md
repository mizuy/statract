# stat

## Implementation notes: SMD { #smd }

`standardized_difference` (the SMD column of `tableone(..., add_smd=True)`) uses the same formula as R's tableone. For numeric variables it is $|\bar x_1 - \bar x_0| / \sqrt{(s_1^2 + s_0^2)/2}$, with sample variances. For categorical variables it is the multinomial version of Yang and Dalton (2012). With 2 levels, this reduces to the formula using the variance $p(1-p)$ of the proportion $p$ (Austin 2009).

The denominator is computed each time within the table you pass. For the sample before adjustment, it matches `smd_all` from `match_sample(...).balance()` and `diff_unadjusted` from `propensity_weights(...).balance()` (except for the sign). A Table 1 built from the matched sample recomputes the denominator with the matched variances. So it may not match `smd_matched` / `diff_adjusted` in the balance table, which fixes the denominator before adjustment. Use the balance table to check balance. The reasons are in the [weighting implementation notes](../models/weighting.md#smd).

For numeric 0/1 columns, the summary function decides how they are handled. Numeric summaries such as the mean treat them as continuous and use the sample variance (`ddof=1`). This is the same as leaving them out of `factorVars` in R's tableone. Categorical summaries (such as `agg_category_n`) or `categorical=True` use $p(1-p)$, which matches the binary rows of the balance table. The two variances differ only by a factor of $n/(n-1)$ in each group.

::: statract.tableone.stat

