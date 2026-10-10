# ordinal

順序ロジスティック回帰（比例オッズ、累積リンク）と Brant 検定です。`MASS::polr` と `ordinal::clm` に合わせています。R との数値対応は [vs R](../../models/vs-r.md) です。

モデルは `F^{-1}(P(Y <= k)) = zeta_k - x'beta` です。`beta` が正なら、応答は上の水準に寄ります。切片は閾値 `zeta` が代わるので、式に書いても落とします。リンクは `logit`（既定、`polr` の `logistic`）、`probit`、`cloglog`、`loglog`、`cauchit` です。水準の順は `levels=`、`pl.Enum` の定義順、値のソート順（数値は数の順）の順に決まります。

```python
from statract import brant_test, ordinal_regression

fit = ordinal_regression(df, "grade ~ age + stage", levels=["none", "mild", "moderate", "severe"])
fit.tidy(exponentiate=True)  # 係数は累積オッズ比、閾値も同じ表
fit.probabilities()          # 水準ごとの予測確率
brant_test(fit).table        # 比例オッズの検定（全体と係数ごと）
```

::: statract.models.ordinal
