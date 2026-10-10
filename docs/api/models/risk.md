# risk

二値の結果のリスク比です。`method="poisson"` は Zou (2004) の修正ポアソン回帰で、0/1 の結果にポアソン GLM を当て、HC0 のサンドイッチ分散を使います。`glm(family = poisson)` と `sandwich::vcovHC(type = "HC0")` に合わせています。`cluster=` は `vcovCL(type = "HC0")` です。`method="log-binomial"` は `glm(family = binomial(link = "log"))` です。反復は `glm.fit` と同じで、`0 < mu < 1` を外れた一歩は前の係数に向けて半分にします。最初の一歩が外れたら、R と同じく `start=` を求めて止まります。

```python
from statract import fit_risk_ratio

rr = fit_risk_ratio(df, "event ~ arm + age + sex")  # 修正ポアソン、HC0
rr.tidy()                                           # estimate と区間はリスク比
fit_risk_ratio(df, "event ~ arm + age", cluster="site")
fit_risk_ratio(df, "event ~ arm + age", method="log-binomial", start=[-1.0, 0.0, 0.0])
```

::: statract.models.risk
