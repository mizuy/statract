# roc

ROC curves, AUC, DeLong variance and confidence intervals, and the DeLong test for 2 AUCs. Defaults are the same as R's pROC 1.18.5 (first level is controls, second level is cases, direction set automatically by the medians, thresholds at midpoints of unique values plus ±Inf, missing values dropped). For numerical agreement with R, see [vs R](../../models/vs-r.md).

```python
from statract import roc_curve, roc_test

r1 = roc_curve(df["y"], df["score1"])
r1.auc, r1.ci_auc(), r1.var_auc()
r1.coords("best", best_method="youden", ret=["threshold", "specificity", "sensitivity"])
roc_test(r1, roc_curve(df["y"], df["score2"]))  # same response, so paired DeLong
```

::: statract.models.roc
