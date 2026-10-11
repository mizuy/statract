# evalue

未測定交絡についての E-value です（VanderWeele と Ding 2017）。結果を説明しきるのに、未測定の交絡因子が曝露と結果の両方にどれだけ強く関連している必要があるかを、リスク比で表します。式は R の `EValue` パッケージ（`evalues.RR`、`evalues.OR`、`evalues.HR`、`evalues.MD`）と同じです。

```python
from statract import e_value, e_value_curve

e_value(0.498, 0.33, 0.755)                         # リスク比と区間
e_value(1.8, 1.2, 2.7, measure="OR", rare=False)   # よくある結果のオッズ比は sqrt(OR) に直す
e_value(0.7, 0.55, 0.9, measure="HR", rare=True)
e_value(0.4, measure="SMD", se=0.12)                # 標準化平均差
e_value_curve(2.01, [2.5, 3.0, 4.0])                # 結果を説明しきる組み合わせの曲線
```

区間は 1 に近い側の端だけに E-value をつけます。区間が 1 をまたぐときは 1 です。

::: statract.models.evalue
