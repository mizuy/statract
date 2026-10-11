# diagnostic

二値の検査を参照基準と比べた診断精度です。感度、特異度、陽性・陰性的中率、尤度比、診断オッズ比、Youden 指数などに信頼区間をつけます。`epiR::epi.tests` に合わせています。割合の区間は `method=` で選びます。既定は Wilson のスコア区間で、`"exact"`（Clopper–Pearson）、`"agresti"`、`"jeffreys"` も使えます。尤度比と診断オッズ比は対数をとった正規近似の区間です。

```python
from statract import diagnostic_accuracy

acc = diagnostic_accuracy(df["risk"] >= 0.05, df["bleed"])  # 検査、参照基準（0/1 か bool）
acc.table()                                                  # 2×2 表
acc.tidy()                                                   # 統計量ごとに推定値と区間
acc.predictive_values([0.01, 0.05, 0.20])                    # 有病割合を変えたときの PPV と NPV
diagnostic_accuracy(tp=23, fp=259, fn=12, tn=706, method="exact")
```

`tidy()` の `statistic` と epiR の名前の対応です。

| statract | epiR | 意味 |
|----------|------|------|
| `apparent_prevalence` | `ap` | 検査陽性の割合 |
| `prevalence` | `tp` | 有病割合 |
| `sensitivity` / `specificity` | `se` / `sp` | 感度 / 特異度 |
| `accuracy` | `diag.ac` | 正診率 |
| `diagnostic_or` | `diag.or` | 診断オッズ比 |
| `nnd` | `nndx` | 診断に必要な数（1 / Youden 指数） |
| `youden` | `youden` | Youden 指数 |
| `ppv` / `npv` | `pv.pos` / `pv.neg` | 陽性的中率 / 陰性的中率 |
| `lr_pos` / `lr_neg` | `lr.pos` / `lr.neg` | 陽性尤度比 / 陰性尤度比 |
| `rule_out` / `rule_in` | `p.rout` / `p.rin` | 検査陰性 / 陽性の割合 |
| `false_positive_rate` / `false_negative_rate` | `p.tpdn` / `p.tndp` | 偽陽性率 / 偽陰性率 |
| `false_discovery_rate` / `false_omission_rate` | `p.dntp` / `p.dptn` | 陽性のうち病気でない割合 / 陰性のうち病気の割合 |

::: statract.models.diagnostic
