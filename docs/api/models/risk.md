# risk

Risk ratios for binary outcomes. `method="poisson"` is the modified Poisson regression of Zou (2004). It fits a Poisson GLM to the 0/1 outcome and uses the HC0 sandwich variance. It matches `glm(family = poisson)` and `sandwich::vcovHC(type = "HC0")`. `cluster=` gives `vcovCL(type = "HC0")`. `method="log-binomial"` is `glm(family = binomial(link = "log"))`. Fitting uses iteratively reweighted least squares (McCullagh & Nelder 1989). A step that leaves `0 < mu < 1` is halved toward the previous coefficients. If the first step leaves the range, it stops and asks for `start=`, as R does.

```python
from statract import fit_risk_ratio

rr = fit_risk_ratio(df, "event ~ arm + age + sex")  # modified Poisson, HC0
rr.tidy()                                           # estimate and interval as risk ratios
fit_risk_ratio(df, "event ~ arm + age", cluster="site")
fit_risk_ratio(df, "event ~ arm + age", method="log-binomial", start=[-1.0, 0.0, 0.0])
```

::: statract.models.risk
