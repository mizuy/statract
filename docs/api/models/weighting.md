# weighting

二値の処置の逆確率重み付け（IPTW）です。公開名は `propensity_weights`、`PropensityWeights`、`balance_table` です。

- 傾向スコアは `fit_glm` のロジスティック回帰です。重みの式、`stabilize`、`trim` は `WeightIt::weightit(method = "glm")` と `WeightIt::trim` に合わせています。
- `balance()` と `balance_table` は `cobalt::bal.tab` の表です。因子は水準ごとに分けます。2 値は割合の差、連続は SMD と分散比です。
- `effective_sample_size()` は群ごとの Kish の有効標本サイズ `(Σw)² / Σw²` です。
- 図は [`plot_love`](../viz/balance.md) です。

アウトカムのモデルはここにありません。`frame()` の `weights` 列を `fit_glm(..., weights="weights")` と `hc_covariance(fit, "HC0")`、または `cox_ph(..., weights="weights")` に渡します。使い方は [Models](../../models/overview.md) の「逆確率重み付け」です。

::: statract.models.weighting
