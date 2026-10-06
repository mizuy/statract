# viz — 概要

推定結果の図です。係数・OR・HR の forest、Kaplan–Meier、較正と決定曲線、Plotly の補助です。数値の推定は [models](../models/overview.md) です。

| 図 | 入口 | モジュール |
|----|------|------------|
| 係数 / OR / HR | `plot_forest(..., layout="table")` | `statract.forest` |
| Kaplan–Meier | `plot_survival`、`plot_survival_grid` | `statract.survival` |
| 変量効果 | `plot_random_effects` | `statract.regression` |
| 較正 / DCA | `plot_calibration`、`plot_dca` | `statract.probability` |
| Love plot | `MatchResult.love_plot` | `statract.matching` |
| funnel / 画像の連結 | `funnel_plot`、`concat_images` | `statract.figure` |

論文の白黒は forest と Kaplan–Meier とも `style="bw"` です。`layout="table"` の forest が係数図の正本で、`statract.figure.hr_forest` は古い呼び出しの互換です。

生存曲線の数値（`survival_curve`、Cox、AFT、Fine–Gray）は `statract.surv` です。`statract.survival` は図だけです。

API は [forest](../api/viz/forest.md)、[survival](../api/viz/survival.md)、[figure](../api/viz/figure.md) です。通しの図は [解析例](../examples/index.md) にあります。
