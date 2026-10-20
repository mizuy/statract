# rmst

Restricted mean survival time (RMST). The formulas follow Royston and Parmar (2013), Uno et al. (2014), and the textbook by Klein and Moeschberger. Results match the unadjusted two-group comparison in `survRM2::rmst2`. For numerical agreement with R, see [vs R](../../models/vs-r.md).

RMST is the area under the Kaplan–Meier curve up to `tau`. The variance is the Greenwood-type sum of Klein and Moeschberger. No `n / (n - 1)` factor is applied. This value matches `se(rmean)` from `summary(survfit(...), rmean = tau)` in survival 3.5. The restricted mean time lost (RMTL) is `tau - RMST`, and its SE is the same as for RMST.

There are 3 contrasts: the RMST difference (normal interval), and the RMST ratio and RMTL ratio (interval and test on the log scale). The direction is "other / `reference`". The default `tau` is the same as in `rmst2`. If the shorter group ends with an event (the curve reaches 0), it is the larger of the 2 groups' last observed times. Otherwise it is the smaller one.

```python
from statract import restricted_mean_survival

res = restricted_mean_survival(df, "time", "status", by="arm", tau=5)
res.arms       # RMST, SE, interval, and RMTL for each group
res.contrasts  # RMST difference, RMST ratio, RMTL ratio
```

::: statract.surv.rmst
