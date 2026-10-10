# rmst

制限付き平均生存時間（RMST）です。`survRM2::rmst2` の調整なしの 2 群比較に合わせています。R との数値対応は [vs R](../../models/vs-r.md) です。

RMST は Kaplan–Meier 曲線の `tau` までの面積です。分散は `survRM2::rmst1` と同じ Greenwood 型の和で、`n / (n - 1)` の係数は付けません。この値は survival 3.5 の `summary(survfit(...), rmean = tau)` の `se(rmean)` と一致します。制限付き平均喪失時間（RMTL）は `tau - RMST` で、SE は RMST と同じです。

対比は 3 つです。RMST の差（正規の区間）、RMST の比と RMTL の比（対数で区間と検定）です。比べる向きは「もう一方 / `reference`」です。`tau` の既定は 2 群の最終観察時刻の小さいほうです。

```python
from statract import restricted_mean_survival

res = restricted_mean_survival(df, "time", "status", by="arm", tau=5)
res.arms       # 群ごとの RMST、SE、区間、RMTL
res.contrasts  # RMST difference、RMST ratio、RMTL ratio
```

::: statract.surv.rmst
