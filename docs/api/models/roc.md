# roc

ROC 曲線、AUC、DeLong の分散と信頼区間、2 本の AUC の DeLong 検定です。R の pROC 1.18.5 と同じ既定（水準は 1 つ目が対照、2 つ目が症例、向きは中央値で自動、閾値は一意な値の中点と ±Inf、欠測は除く）です。R との数値対応は [vs R](../../models/vs-r.md) です。

```python
from statract import roc_curve, roc_test

r1 = roc_curve(df["y"], df["score1"])
r1.auc, r1.ci_auc(), r1.var_auc()
r1.coords("best", best_method="youden", ret=["threshold", "specificity", "sensitivity"])
roc_test(r1, roc_curve(df["y"], df["score2"]))  # 同じ応答なので対応ありの DeLong
```

::: statract.roc
