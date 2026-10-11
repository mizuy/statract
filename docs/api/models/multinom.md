# multinom

多項ロジスティック回帰（基準カテゴリ logit）です。`nnet::multinom` に合わせています。R との数値対応は [vs R](../../models/vs-r.md) です。

応答の最初の水準が基準です。基準以外の水準ごとに係数の行があります。推定は正確な Hessian の Newton 法で、きつく収束させます。`multinom` は BFGS で `reltol = 1e-8` で止まるので、R の既定の推定値とは 1e-4 程度ずれることがあります。対数尤度はこちらが同じか高くなります。SE は `summary(multinom)` と同じく Hessian の逆行列から取ります。

```python
from statract import multinomial_regression

fit = multinomial_regression(df, "subtype ~ age + sex", levels=["A", "B", "C"])
fit.tidy(exponentiate=True)  # 相対リスク比。列 y_level が水準
fit.probabilities()
```

::: statract.models.multinom
