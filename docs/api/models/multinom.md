# multinom

Multinomial logistic regression (baseline-category logit). It matches `nnet::multinom`. For numerical agreement with R, see [vs R](../../models/vs-r.md).

The first level of the response is the reference. Each non-reference level has its own row of coefficients. Fitting uses Newton's method with the exact Hessian and a tight convergence criterion. `multinom` uses BFGS and stops at `reltol = 1e-8`, so estimates may differ from R's defaults by about 1e-4. The log-likelihood here is the same or higher. SEs come from the inverse Hessian, as in `summary(multinom)`.

```python
from statract import multinomial_regression

fit = multinomial_regression(df, "subtype ~ age + sex", levels=["A", "B", "C"])
fit.tidy(exponentiate=True)  # relative risk ratios; column y_level is the level
fit.probabilities()
```

::: statract.models.multinom
