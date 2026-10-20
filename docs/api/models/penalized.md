# penalized

Penalized regression (ridge, lasso, elastic net). The objective and usage match R's `glmnet` 4.1-8. `family` is `"gaussian"`, `"binomial"` (0/1 outcome), `"poisson"`, or `"cox"` (Breslow ties). `alpha=1` is the lasso, `alpha=0` is ridge, and values in between are the elastic net.

The objective is the same as in the glmnet help. It is the negative log-likelihood with weights scaled to sum to 1, plus `lambda * sum(v_j * ((1 - alpha) / 2 * b_j^2 + alpha * |b_j|))`. `v_j` is `penalty_factor` scaled to sum to the number of columns. By default, columns are standardized with the 1/n variance for fitting, and coefficients are returned on the original scale. The intercept is not penalized.

The solver is coordinate descent written from the papers (Friedman et al. 2010; Simon et al. 2011 for Cox; Tibshirani et al. 2012 for the strong rule). It does not use glmnet code. glmnet is used only to check results. The lambda sequence has `n_lambda` values from `lambda_max` down to `lambda_min_ratio` times that. It uses warm starts and the strong rule, and checks the excluded columns with the KKT conditions. Binomial, Poisson, and Cox are solved by IRLS (iteratively reweighted least squares). Each problem is solved 1e-4 times tighter than glmnet's `thresh`. So the difference from R is glmnet's own convergence error (about 1e-4 to 1e-3 in the coefficients). The automatic lambda sequence stops when the gain in the fraction of deviance explained falls below 1e-5 times that fraction. The inner loop uses numba.

Factors in the formula become dummy columns, as in `model.matrix`. `y ~ x - 1` drops the intercept (glmnet's `intercept=FALSE`). For `offset(log(exposure))`, write it in the formula or pass a column name to `offset=`.

```python
from statract import cv_penalized, penalized_regression

fit = penalized_regression(df, "y ~ age + sex + stage + bmi", family="binomial", alpha=0.5)
fit.path()                       # df and dev_ratio for each lambda
fit.coef(0.02)                   # coefficients at lambda = 0.02 (intercept first)
fit.tidy(0.02)                   # table of lambda, term, estimate
fit.predict(new_df, 0.02, kind="response")

# pass penalty_factor by column name; 0 means no penalty, inf means excluded
penalized_regression(df, "y ~ age + sex + stage", penalty_factor={"age": 0.0})

# cross-validation; passing foldid gives the same result as R
cv = cv_penalized(df, "y ~ age + sex + stage + bmi", family="binomial", foldid="fold", measure="auc")
cv.lambda_min, cv.lambda_1se
cv.table()                       # lambda, cvm, cvsd, cvup, cvlo, nzero
cv.coef("1se")
cv.predict(new_df, "min", kind="class")

# Cox
penalized_regression(df, "Surv(time, status) ~ age + sex + stage", family="cox")
```

`coef` and `predict` interpolate linearly in lambda between the lambda values on the path (`exact=FALSE`). Values outside the path are clamped to the ends. In `cv_penalized`, each fold is refit with its own automatic lambda sequence and then interpolated at the full-data lambda values for prediction. The default `measure` is `"mse"` for gaussian and `"deviance"` for the others.

Differences from R are listed in [models and R package mapping](../../models/vs-r.md).

::: statract.models.penalized
