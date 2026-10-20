# evalue

E-values for unmeasured confounding (VanderWeele and Ding 2017). An E-value is the minimum strength, on the risk ratio scale, that an unmeasured confounder would need in its association with both the exposure and the outcome to fully explain away the result. The formulas are the same as in the R `EValue` package (`evalues.RR`, `evalues.OR`, `evalues.HR`, `evalues.MD`).

```python
from statract import e_value, e_value_curve

e_value(0.498, 0.33, 0.755)                         # risk ratio and interval
e_value(1.8, 1.2, 2.7, measure="OR", rare=False)   # odds ratio for a common outcome is converted to sqrt(OR)
e_value(0.7, 0.55, 0.9, measure="HR", rare=True)
e_value(0.4, measure="SMD", se=0.12)                # standardized mean difference
e_value_curve(2.01, [2.5, 3.0, 4.0])                # curve of combinations that fully explain the result
```

For the interval, only the limit closer to 1 gets an E-value. When the interval crosses 1, the E-value is 1.

::: statract.models.evalue
