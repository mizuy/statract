# linear_tests

Wald tests of coefficients, Wald and likelihood ratio tests of nested models, OLS diagnostics, and `check_collinearity` (GVIF).

`profile_interval(fit, level=0.95)` gives profile likelihood intervals for the coefficients of `fit_glm`. It is the counterpart of R's `confint()` (`MASS::confint.glm`, and `stats::confint.glm` from R 4.4). It fixes a coefficient, refits the rest, divides the difference in deviance by the dispersion parameter, and solves directly with Brent's method for the point where the square root equals the normal quantile. The dispersion parameter is 1 for binomial and Poisson, and the `summary.glm` value for Gaussian and gamma. R fits a spline over a grid and reads the limits from it, so results agree with R within 1e-3 times the standard error (at most 2.5e-4 times in the fixtures). They agree with the exact roots to 1e-7. On a side where the deviance levels off due to separation, the limit is infinite (R gives `NA`).

::: statract.models.linear_tests
