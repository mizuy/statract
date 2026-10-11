# firth

Firth の罰則付き尤度によるロジスティック回帰です。モデルは R の `logistf::logistf` と同じです。対数尤度に Jeffreys 事前分布の罰 `0.5 log det I(beta)` を足します（Firth 1993、Heinze と Schemper 2002）。分離（ある群で事象がまったく起きない、またはすべて起きる）があっても、推定値は有限です。

反復は、Heinze と Schemper（2002）の修正スコアによる Newton–Raphson 法です。論文から書いたもので、logistf のコードは使っていません。一歩で動く係数は最大 `maxstep`（既定 5）で、`maxhs`（既定 0）回まで半分にします。既定の値と 1e-5 の収束の基準は、`logistf.control()` のヘルプの説明に合わせています。共分散は推定値での Fisher 情報行列 `X'WX` の逆行列で、`vcov(logistf)` と同じ量です。

`tidy()` の既定は `logistf` の `pl = TRUE` です。区間はプロファイル罰則付き尤度の区間、p 値は罰則付き尤度比検定です。`statistic` は 1 自由度のカイ二乗です。係数を固定した当てはめでも、罰は全体の設計行列で計算します（Heinze と Schemper 2002。`logistf` と同じ）。`tidy(method="wald")` は `pl = FALSE` の Wald の z と区間です。`glance()` は全体の尤度比検定です。比べるのは切片だけを推定し、ほかの係数を 0 にしたモデルです。

```python
from statract import fit_firth

fit = fit_firth(df, "event ~ clip + drug + size")
fit.tidy()                     # プロファイル区間と罰則付き尤度比の p 値
fit.tidy(exponentiate=True)    # 推定値と区間はオッズ比
fit.tidy(method="wald")        # Wald の z と区間
fit.glance()                   # 全体の尤度比検定
fit.predict(new_df)            # 予測確率。kind="link" で対数オッズ
```

`weights=` は度数の重みです。区間の端は、プロファイル罰則付き尤度の差がカイ二乗の分位点になる点を Brent 法で直接解きます。`logistf` の Venzon–Moolgavkar 法（1e-5 で停止）と同じ点に収束します。完全分離に共変量がいくつも重なると、固定した係数ごとの罰則付き尤度に山が二つできることがあります。このときの区間は、推定値から連続にたどった山の値です。

::: statract.models.firth
