# penalized

ペナルティ付き回帰（リッジ、lasso、elastic net）です。目的関数と使い方は R の `glmnet` 4.1-8 に合わせています。`family` は `"gaussian"`、`"binomial"`（0/1 の結果）、`"poisson"`、`"cox"`（Breslow の同順位）です。`alpha=1` が lasso、`alpha=0` がリッジ、その間が elastic net です。

目的関数は glmnet のヘルプにあるものと同じです。重みを和 1 にそろえた負の対数尤度に、`lambda * sum(v_j * ((1 - alpha) / 2 * b_j^2 + alpha * |b_j|))` を足します。`v_j` は `penalty_factor` を列数の和にそろえたものです。既定では列を 1/n の分散で標準化して当て、係数は元の尺度で返します。切片には罰則をかけません。

解き方は、論文から書いた座標降下法です（Friedman ら 2010、Cox は Simon ら 2011、strong rule は Tibshirani ら 2012）。glmnet のコードは使っていません。glmnet は結果の確認にだけ使います。lambda 列は `lambda_max` から `lambda_min_ratio` 倍まで `n_lambda` 個です。warm start と strong rule を使い、外した列は KKT 条件で確かめます。二項、ポアソン、Cox は IRLS（重み付き最小二乗のくり返し）で解きます。各問題は glmnet の `thresh` より 1e-4 倍きつく解きます。そのため R との差は、glmnet 自身の収束誤差（係数で 1e-4 から 1e-3 程度）です。自動の lambda 列は、説明できた逸脱度の割合の伸びがその割合の 1e-5 倍を下回ると止まります。内側のループは numba です。

式の因子は `model.matrix` と同じくダミー列になります。`y ~ x - 1` は切片なし（glmnet の `intercept=FALSE`）です。`offset(log(exposure))` は式に書くか、`offset=` に列名を渡します。

```python
from statract import cv_penalized, penalized_regression

fit = penalized_regression(df, "y ~ age + sex + stage + bmi", family="binomial", alpha=0.5)
fit.path()                       # lambda ごとの df と dev_ratio
fit.coef(0.02)                   # lambda = 0.02 の係数（切片が先頭）
fit.tidy(0.02)                   # lambda、term、estimate の表
fit.predict(new_df, 0.02, kind="response")

# penalty_factor は列名で渡す。0 は罰則なし、inf は除外
penalized_regression(df, "y ~ age + sex + stage", penalty_factor={"age": 0.0})

# 交差検証。foldid を渡すと R と同じ結果になる
cv = cv_penalized(df, "y ~ age + sex + stage + bmi", family="binomial", foldid="fold", measure="auc")
cv.lambda_min, cv.lambda_1se
cv.table()                       # lambda、cvm、cvsd、cvup、cvlo、nzero
cv.coef("1se")
cv.predict(new_df, "min", kind="class")

# Cox
penalized_regression(df, "Surv(time, status) ~ age + sex + stage", family="cox")
```

`coef` と `predict` は、経路の lambda の間を lambda について線形に補間します（`exact=FALSE`）。経路の外の値は端にそろえます。`cv_penalized` の各分割は、分割ごとに自動の lambda 列で当て直し、全体の lambda で補間して予測します。`measure` の既定は gaussian が `"mse"`、ほかは `"deviance"` です。

R と違うところは [models と R パッケージの対応](../../models/vs-r.md) にまとめています。

::: statract.models.penalized
