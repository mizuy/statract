# viz — 概要

推定結果の図です。係数・OR・HR の forest、Kaplan–Meier、較正と決定曲線、Plotly の補助です。数値の推定は [models](../models/overview.md) です。

| 図 | 入口 | モジュール |
|----|------|------------|
| 係数 / OR / HR | `plot_forest(..., layout="table")` | `statract.viz.forest` |
| Kaplan–Meier | `plot_survival` | `statract.viz.km` |
| 条件付き推論木 | `plot_tree` | `statract.viz.tree` |
| 変量効果 | `plot_random_effects` | `statract.models.glmm_extras` |
| 較正 / DCA | `plot_calibration`、`plot_dca` | `statract.models.probability` |
| Love plot | `plot_love`、`PropensityWeights.love_plot`、`MatchedSample.love_plot` | `statract.viz.balance`、`statract.models.matching` |
| funnel / 画像の連結 | `funnel_plot`、`concat_images` | `statract.viz.misc` |

論文の白黒は forest と Kaplan–Meier とも `style="bw"` です。係数図は `layout="table"` の forest を使います。

生存曲線の数値（`survival_curve`、Cox、AFT、Fine–Gray）は `statract.surv` です。`statract.viz.km` は図だけです。

API は [forest](../api/viz/forest.md)、[km](../api/viz/km.md)、[tree](../api/viz/tree.md)、[balance](../api/viz/balance.md)、[misc](../api/viz/misc.md) です。通しの図は [解析例](../examples/index.md) にあります。
