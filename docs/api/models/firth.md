# firth

Logistic regression with Firth's penalized likelihood. The model is the same as R's `logistf::logistf`. It adds the Jeffreys prior penalty `0.5 log det I(beta)` to the log-likelihood (Firth 1993, Heinze and Schemper 2002). Estimates stay finite even with separation (no events, or only events, in some group).

The iterations are Newton–Raphson with the modified score of Heinze and Schemper (2002). The code was written from the papers and does not use logistf code. Each step moves a coefficient by at most `maxstep` (default 5), and a step is halved up to `maxhs` (default 0) times. These defaults and the 1e-5 convergence criterion follow the help page of `logistf.control()`. The covariance is the inverse of the Fisher information matrix `X'WX` at the estimates, the same quantity as `vcov(logistf)`.

The default for `tidy()` is `pl = TRUE` in `logistf`. Intervals are profile penalized likelihood intervals, and p-values come from penalized likelihood ratio tests. `statistic` is a chi-square with 1 degree of freedom. Even in fits with a coefficient held fixed, the penalty is computed from the full design matrix (Heinze and Schemper 2002, as in `logistf`). `tidy(method="wald")` gives the Wald z and intervals of `pl = FALSE`. `glance()` is the overall likelihood ratio test. It compares against a model that estimates only the intercept and sets the other coefficients to 0.

```python
from statract import fit_firth

fit = fit_firth(df, "event ~ clip + drug + size")
fit.tidy()                     # profile intervals and penalized likelihood ratio p-values
fit.tidy(exponentiate=True)    # estimates and intervals as odds ratios
fit.tidy(method="wald")        # Wald z and intervals
fit.glance()                   # overall likelihood ratio test
fit.predict(new_df)            # predicted probabilities; kind="link" gives log odds
```

`weights=` are frequency weights. The interval limits are solved directly with Brent's method, as the points where the difference in profile penalized likelihood equals the chi-square quantile. They converge to the same points as the Venzon–Moolgavkar method in `logistf` (which stops at 1e-5). When several covariates overlap with complete separation, the penalized likelihood for a fixed coefficient can have two peaks. In that case the interval is taken from the peak reached by following continuously from the estimate.

::: statract.models.firth
