# Wilkinson formula

`fit_ols`, `fit_glm`, `fit_mixed`, and `model_matrix` accept the same strings. Expansion follows R's `terms` and `model.matrix` (treatment contrasts). Column names and matrices are pinned in `tests/r_oracle/fixtures/wilkinson.json`. Models with main effects only can also be called with a list of column names.

```python
from statract import fit_ols, model_matrix

fit_ols(data, "y ~ x * stage")
built = model_matrix("y ~ x * stage", data)
built.design.names  # column names
built.y              # response
built.offset         # sum of offset() terms. None if there are none
```

When you pass a formula, leave out `predictors`, `groups`, and `slopes`. Call examples are in [Models](overview.md). The mapping to R is in [Mapping to R packages](vs-r.md).

## Support overview

The reference is the set of operators and functions in formulaic's [Formula Grammar](https://matthew.wardrop.casa/formulaic/latest/guides/grammar/). The meaning of expansion follows R's `terms` and `model.matrix`.

### Operators

| Syntax | Support | Notes |
|------|------|------|
| `` `column name` `` | Yes | Column names with spaces or symbols |
| `(...)` | Yes | Changes precedence |
| `.` | Yes | The remaining columns of the data frame, excluding columns on the left side |
| `^` | Yes | Interactions up to that order. The exponent is an integer of 2 or more. `x^2` is the term `x` |
| `:` | Yes | Interaction. Does not scale columns by numbers |
| `*` | Yes | Main effects of both sides and their interaction |
| `/` | Yes | Nesting. `a / b` is `a + b %in% a` |
| `%in%` | Yes | Product only. `b %in% a` does not include the main effect of `a` |
| `+` `-` | Yes | Add a term, remove a term |
| Unary `+` `-` | Yes | R rules. `-1` removes the intercept and `-0` adds it |
| `~` | Yes | Exactly one. A left side is required |
| `0` `1` | Yes | Toggle the intercept. Other numeric literals are an error |
| `"..."` | No | String literals |
| `{...}` | No | Embedded Python expressions |
| Arbitrary function calls | No | Only the functions in the table below |
| `**` | No | Use `^` for the same role |
| Numbers such as `2.5:x` | No | Do arithmetic inside `I()` |
| `\|` to split design matrices | No | Multiple matrices from the Formula package. `(1 \| g)` is a random effect, see below |
| `[. ~ .]` | No | Multi-stage formulas |
| Omitting `~` | No | Right-side-only formulas |

### Functions

| Function | Support | Notes |
|------|------|------|
| `I(...)` | Yes | `+` `-` `*` `/` `^` inside are arithmetic. One argument. Column names follow R's deparse, as in `I(x^2)` |
| `log` `log10` `exp` `sqrt` `abs` | Yes | One argument. Usable on the left or right side |
| `offset(...)` | Yes | Not a design matrix column. Added to `FormulaModel.offset`. formulaic does not have this function. `fit_mixed` accepts it for binomial, Poisson, and negative binomial |
| `Surv(time, status)` `Surv(start, stop, status)` | Yes | Left side for `cox_ph`, `accelerated_failure`, and `fine_gray`. `Surv(start, stop, status)` is for `cox_ph` and `fine_gray` |
| `strata(...)` | Yes | Strata for `cox_ph` and `conditional_logit`. Not a design matrix column |
| `cluster(...)` | Yes | Cluster-robust variance for `cox_ph`. `conditional_logit` uses it with `method="efron"` and `method="breslow"`. Not available with exact |
| `(1 \| g)` `(1 + x \| g)` `(0 + x \| g)` | Yes | lme4 random effects. Not in formulaic's table. Estimated by `fit_mixed`. Slopes are one numeric column. Binomial, Poisson, and negative binomial also allow multiple terms (crossed, nested) |
| `ar1(time + 0 \| g)` | Yes | glmmTMB AR(1) structure. `time` is a factor of time points (for a numeric column, levels in ascending order). Laplace engine only |
| `Q(...)` | No | Use backquotes for column names |
| `C(...)` | No | The column type decides whether it is a factor. The contrast type is not specified |
| `center` `scale` `standardize` | No | |
| `lag` | No | |
| `poly` | No | For squares, use `I(x^2)` |
| `ns(x, df = 3)` `bs(x, df = 5)` | Yes | R's `splines`. Arguments are `df`, `knots = c(...)`, `intercept`, `Boundary.knots`, and also `degree` for `bs`. Positional arguments follow R's order. Knots are set from the non-missing values when the formula is first evaluated, and prediction uses the same knots (R's `makepredictcall`). Column names match R, as in `ns(x, df = 3)1` |
| `rcs(x, 4)` `rcs(x, c(20, 40, 60))` | Yes | `rms::rcs` (`Hmisc::rcspline.eval`, `norm=2`). Number or positions of knots. Default is 5. Column names are `rcs(x, 4)x`, `rcs(x, 4)x'` |
| `cs` `cr` `cc` `te` | No | Do not write penalized smooths in the formula. Use `smooth` |
| `hashed` | No | |
| `log2` `exp2` `exp10` | No | |
| `np` and caller-side functions | No | There is no way to pass a namespace into the formula |
| Comma-separated arguments and `name = value` | Partial | `Surv(time, status)`, `strata(a, b)`, spline arguments. Spline arguments can only be numbers, `c(...)`, and `TRUE` / `FALSE`. `poly(x, 2)` and `log(x, 10)` are not supported |
| `\|\|` | No | `(x \|\| g)` can be parsed. `fit_mixed` raises an error for uncorrelated slopes |
| `s` | No | Use `smooth` for smooths |

### Differences from formulaic

The following 3 points follow R. formulaic sorts terms alphabetically, keeps the intercept in `(x - 1)`, and reduces matrix rank the patsy way.

- Terms are ordered by degree, then by order of appearance in the formula.
- Neither `(x - 1)` nor `x - 1` has an intercept.
- Contrasts follow the local rule for treatment contrasts.

`+` on the left side is numeric addition. The response of `x + z ~ y` is one column, `x + z`. It does not become multiple response columns.

## Left and right sides

The form is `response ~ right side`. There is exactly one `~`. The left side is a column name, a function from the table above, or arithmetic on them. The response of `log(y) ~ x` is `log(y)`.

The operators on the right side, from highest precedence, are:

| Precedence | Operator | Expansion |
|------|--------|------|
| 1 | `^` | Repeats the product of the same term set. The exponent is an integer of 2 or more |
| 2 | `:` | Pairs the terms on each side and takes the union of their variables |
| 3 | `%in%` | Combines all terms on the right and multiplies each term on the left by the result |
| 4 | `*` `/` | `*` gives the left, the right, and their product. `/` gives the left, plus the whole left multiplied by each term on the right |
| 5 | `+` `-` | Adds terms. `-` removes the right terms from the left |

`a * b:c` is `a * (b:c)`. Whitespace is ignored. Wrap column names that are not valid identifiers in `` `column name` ``.

## Columns created by operators

The levels are I, II, III for `stage` and A, B for `arm`. They are strings, so they use sorted order, and the references are I and A.

| Formula | Columns |
|----|----|
| `y ~ x + stage` | `(Intercept)`, `x`, `stageII`, `stageIII` |
| `y ~ x * stage` | The above plus `x:stageII`, `x:stageIII` |
| `y ~ x:stage` | `(Intercept)`, `x:stageI`, `x:stageII`, `x:stageIII` |
| `y ~ stage:x` | `(Intercept)`, `stageI:x`, `stageII:x`, `stageIII:x` |
| `y ~ stage/arm` | `(Intercept)`, `stageII`, `stageIII`, `stageI:armB`, `stageII:armB`, `stageIII:armB` |
| `y ~ stage + arm %in% stage` | Same columns as `y ~ stage/arm` |
| `y ~ arm %in% stage` | `(Intercept)` and `armA:stageI` through `armB:stageIII`. No main effects |
| `y ~ (x + stage + arm)^2` | Main effects and two-way interactions. No three-way interaction |
| `y ~ x^2` | `(Intercept)`, `x` |
| `y ~ I(x^2)` | `(Intercept)`, `I(x^2)` |

`*` keeps main effects. `:` and `%in%` give the product only. `a/b` is `a + b %in% a`. In `stage + arm:stage`, variables are registered from the left of the formula, so the product is named `stageI:armB`. In `arm:stage`, `arm` comes first, so the names are in the order `armA:stageI`.

`x^2` is the term `x`. The squared column is `I(x^2)`. `(x + stage)^2` gives the same terms as `x * stage`. An exponent of 1 or less, or a non-integer exponent, is an error.

When the same term appears twice, the first one is kept. The columns of `y ~ x * x` are `(Intercept)`, `x`.

## Intercept

By default, the first column is `(Intercept)`. `0`, `1`, and `-` toggle the intercept.

| Formula | Columns |
|----|----|
| `y ~ 1` | `(Intercept)` |
| `y ~ x - 1` | `x` |
| `y ~ 0 + x` | `x` |
| `y ~ -1 + x + stage` | `x`, `stageI`, `stageII`, `stageIII` |
| `y ~ (x - 1)` | `x` |
| `y ~ x - 0` | `(Intercept)`, `x` |
| `y ~ x * 0` | `x` |
| `y ~ x + z - (z - 1)` | `(Intercept)`, `x` |

`y ~ 0` and `y ~ -1` have no columns, so they are an error when there are no random effects. A leading `-stage` removes `stage`, which is not yet in the formula. The following `.` then adds `stage`, so `y ~ -stage + .` keeps `stageII`.

## Term order and contrasts

After expansion, terms are stably sorted by degree (the number of names they contain). Terms of the same degree keep their order of appearance in the formula. Names within a term are in the order they first appear from the left of the formula. Interaction column names are joined with `:`. In the Cartesian product, the first factor of the term varies fastest.

Factors use treatment contrasts. Main effects drop the first level. Dummy names concatenate the column name and the level. The intercept is named `(Intercept)`.

If the rest of a term, after removing one variable, is not contained in an earlier term, that variable keeps all levels.

- `y ~ x:stage` has no earlier `stage` main effect, so it has all levels, starting with `x:stageI`.
- In `y ~ x:stage + stage`, `stage` comes first by degree. The columns are `(Intercept)`, `stageII`, `stageIII`, `x:stageI`, `x:stageII`, `x:stageIII`.

When there is no intercept, the first factor, counting from the variables registered after the response, gets all levels. Only one factor changes.

| Formula | Columns |
|----|----|
| `y ~ x + stage - 1` | `x`, `stageI`, `stageII`, `stageIII` |
| `y ~ stage - 1 + x` | `stageI`, `stageII`, `stageIII`, `x` |
| `y ~ stage + arm - 1` | `stageI`, `stageII`, `stageIII`, `armB` |
| `y ~ (stage - 1) * x` | `stageI`, `stageII`, `stageIII`, `x`, `stageII:x`, `stageIII:x` |
| `y ~ b - 1` | `bFALSE`, `bTRUE` |

## Column types

| Type | Handling in a formula |
|----|----------------|
| Float, integer | One numeric column |
| String | Levels are the sorted observed values after dropping missing values. The first is the reference |
| `pl.Enum`, `pl.Categorical` | Defined order. The first is the reference |
| Boolean | Levels `FALSE`, `TRUE`. The dummy in `y ~ b + x` is `bTRUE` |

A factor with fewer than 2 levels is an error. In `design_matrix` with a list of column names, booleans and integers are 0/1 numbers. In a formula, booleans become factors.

## Functions and offset

The functions allowed on the right and left sides are `I`, `log`, `exp`, `sqrt`, `log10`, and `abs`. Each gives one numeric column and takes one argument. Column names follow R's deparse.

| Written in the formula | Column name |
|----------------|------|
| `I(x^2)` | `I(x^2)` |
| `I(x * z)` | `I(x * z)` |
| `I(x/z)` | `I(x/z)` |
| `I(x + z)` | `I(x + z)` |
| `I((x + z)^2)` | `I((x + z)^2)` |
| `log(w + x)` | `log(w + x)` |
| `I(-x)` | `I(-x)` |

The value of `offset()` goes into `FormulaModel.offset` as the sum of its arguments. No design matrix column is created for it. The columns of both `y ~ offset(z) + x` and `y ~ x * offset(z)` are `(Intercept)`, `x`, and the offset is `z`. `offset(z)` does not remove `z` from the dot.

`fit_ols` and `fit_glm` add the formula's `offset()` and the `offset=` argument. `fit_mixed` does the same for binomial, Poisson, and negative binomial; Gaussian and gamma raise an error. `predict()` on the fitted rows adds the offset to the linear predictor. `predict(data)` on new data is the product of the design matrix and the coefficients.

## Dot

`.` is the remaining columns of the data frame, excluding columns named on the left side. They follow the column order of the data frame. Names already on the right side are skipped. `log(y) ~ .` excludes `y`. `y ~ x + .` keeps `x` only once.

If the right side has a main effect equal to the response, that term is dropped with a `UserWarning`.

## Missing values and new data

Rows with missing values in any column the formula uses are dropped together. Levels are set from the remaining rows. Missing values in `weights=` and `offset=` also drop the same rows afterward.

When you pass new data to `predict`, columns are rebuilt with the recipe from fitting. Levels not seen during fitting are an error.

## Random effects

`(1 | g)` is lme4 syntax, and `model_matrix` puts it in `random_effects`. `fit_ols`, `fit_glm`, and `formula_model_matrix` raise an error when there are random effects. Estimation is done by `fit_mixed`.

Random effects are joined to other terms with `+`. Combining them with `*`, `:`, `/`, `^`, `%in%`, or subtraction is an error. A slope is one numeric column. There is also one grouping column. For Gaussian and gamma, `fit_mixed` accepts one grouping. For binomial (logit), Poisson, and negative binomial, you can write multiple terms, as in `(1 | patient) + (1 | examiner)`. `cox_mixed` takes random effects in a `Surv` formula. It also takes `(1 | hospital:ward)`, grouped by the combinations of the two columns, and `(1 | hospital/ward)`, which is `(1 | hospital) + (1 | hospital:ward)` as in lme4. `fit_mixed` and `gamm` do not take these yet.

- `y ~ x + (1 | g)` has fixed effects `x` and the intercept, and a random intercept.
- `y ~ x + (1 + x | g)` has a random intercept and a random slope for `x`. They are correlated. Use `method="ml"` for maximum likelihood.
- `y ~ (x | g)` has a fixed intercept, and a random intercept and random slope for `x`.
- `y ~ 0 + (1 | g)` has no fixed effects and a random intercept.
- `y ~ 0 + x + (0 + x | g)` has fixed effect `x` and a random slope for `x`.

`(x || g)` can be parsed. `fit_mixed` raises an error for uncorrelated slopes. Factor or boolean slopes are also errors, and so is `offset()` with Gaussian or gamma.

`cox_ph`, `accelerated_failure`, and `fine_gray` accept `Surv(time, status) ~ age + sex`. Cox removes the intercept column, and factors still use treatment contrasts. `strata(site)` and `cluster(id)` can go in a Cox formula. `conditional_logit` accepts `bleed ~ eolm + strata(set)` and removes the intercept. The default `method="exact"` is the combinatorial likelihood within matched sets. `Surv(start, stop, status)` is the counting process form, accepted by `cox_ph` and `fine_gray`. The right side of `fine_gray` is the covariates to keep, and `cause` is a separate argument. Accelerated failure time keeps the intercept. Smooths use `smooth`. Calls with column names are still supported.

## Return value

`model_matrix(formula, data)` returns a `FormulaModel`.

| Attribute | Content |
|------|------|
| `design` | Fixed-effects `Design`. `names` are column names, `x` is the matrix, `row_index` is the remaining rows |
| `y` | Vector from evaluating the left side |
| `offset` | Sum of `offset()` terms. `None` if there are none |
| `random_effects` | Sequence of `(slope \| group)` terms. Each element has `intercept`, `slopes`, `group`, `correlated` |
| `response_name` | Name when the left side is a column name. `None` when it is a function |

Pass random effects to `fit_mixed` (the same formula works for Gaussian and for binomial `family="binomial"`). For fixed effects only, use `fit_glm`. The experimental `glmm_gpboost` is in an optional extra and uses the same `model_matrix` expansion.

## Splines {#splines}

```python
from statract import cox_ph, fit_glm, plot_spline_effect, spline_effect, spline_test

fit = fit_glm(data, "death ~ rcs(age, 4) + sex", family="binomial")
spline_test(fit, "rcs(age, 4)")  # Wald chi-square for the whole term and its nonlinear part (anova.rms)
curve = spline_effect(fit, data, "age", at=range(40, 86), reference=60, exponentiate=True)
plot_spline_effect(curve)  # OR curve with 95% interval, relative to age 60
```

`spline_effect` gives the same numbers as `rms::contrast(fit, list(age = at), list(age = 60))`. As with `datadist`, other variables are fixed at the median for numeric variables and at the most common level for factors (change this with `adjust=`). With Cox, the same function gives an HR curve. The nonlinear part in `spline_test` is for `rcs` only. Interaction columns are not included in the term.
