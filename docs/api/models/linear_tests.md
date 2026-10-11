# linear_tests

係数の Wald、入れ子モデルの Wald と尤度比、OLS の診断、`check_collinearity`（GVIF）です。

`profile_interval(fit, level=0.95)` は `fit_glm` の係数のプロファイル尤度区間です。R の `confint()`（`MASS::confint.glm`、R 4.4 からは `stats::confint.glm`）にあたります。係数を固定してほかを当て直した逸脱度の差を分散パラメータで割り、その平方根が正規分位点になる点を Brent 法で直接解きます。二項とポアソンの分散パラメータは 1、ガウスとガンマは `summary.glm` の値です。R は格子点の上にスプラインを引いて読むので、R とは標準誤差の 1e-3 倍以内（fixture では最大 2.5e-4 倍）で一致します。根そのものとは 1e-7 で一致します。分離で逸脱度が頭打ちになる側は無限大です（R は `NA`）。

::: statract.models.linear_tests
