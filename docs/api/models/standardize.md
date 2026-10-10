# standardize

GLM のあとの回帰標準化（g-computation）です。曝露を各値に置き換えて全員の予測を平均し、調整したリスク差、リスク比、オッズ比を出します。分散は共変量を固定した delta 法（`marginaleffects::avg_comparisons`）か、共変量のばらつきも含むサンドイッチ（`stdReg2::standardize_glm` の推定量）です。使い方は [Models](../../models/overview.md)、R との数値対応は [vs R](../../models/vs-r.md) です。Cox モデルの標準化は [surv](../surv/surv.md) の `standardize_cox` です。

::: statract.models.standardize
