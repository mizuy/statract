# ordinal

Ordinal logistic regression (proportional odds, cumulative link) and the Brant test. It matches `MASS::polr` and `ordinal::clm`. For numerical agreement with R, see [vs R](../../models/vs-r.md).

The model is `F^{-1}(P(Y <= k)) = zeta_k - x'beta`. A positive `beta` shifts the response toward higher levels. The thresholds `zeta` replace the intercept, so an intercept in the formula is dropped. Links are `logit` (default, `logistic` in `polr`), `probit`, `cloglog`, `loglog`, and `cauchit`. The order of levels comes from `levels=`, then the order defined in `pl.Enum`, then the sorted values (numbers in numeric order).

```python
from statract import brant_test, ordinal_regression

fit = ordinal_regression(df, "grade ~ age + stage", levels=["none", "mild", "moderate", "severe"])
fit.tidy(exponentiate=True)  # coefficients as cumulative odds ratios; thresholds in the same table
fit.probabilities()          # predicted probability of each level
brant_test(fit).table        # test of proportional odds (overall and per coefficient)
```

::: statract.models.ordinal
