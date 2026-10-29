# viz — Overview

Figures for estimation results: forest plots of coefficients, ORs, and HRs, Kaplan–Meier curves, calibration and decision curves, and Plotly helpers. For numeric estimation, see [models](../models/overview.md).

| Figure | Entry point | Module |
|----|------|------------|
| Coefficients / OR / HR | `plot_forest(..., layout="table")` | `statract.viz.forest` |
| Kaplan–Meier | `plot_survival` | `statract.viz.km` |
| Cox diagnostics | `plot_loglog`, `plot_cox_residuals`, `write_cox_diagnostic_suite` | `statract.surv.diagnostics` |
| Spline effect (OR / HR curve) | `plot_spline_effect` | `statract.viz.effect` |
| ROC | `plot_roc` | `statract.models.roc` |
| Conditional inference tree | `plot_tree` | `statract.viz.tree` |
| Random effects | `plot_random_effects` | `statract.models.glmm_extras` |
| Calibration / DCA | `plot_calibration`, `plot_dca` | `statract.models.probability` |
| Bootstrap calibration curve | `plot_calibration_curve` | `statract.models.validation` |
| Love plot | `plot_love`, `PropensityWeights.love_plot`, `MatchedSample.love_plot` | `statract.viz.balance`, `statract.models.matching` |
| Funnel / image concatenation | `funnel_plot`, `concat_images` | `statract.viz.misc` |

For black-and-white figures in a paper, use `style="bw"` for both forest and Kaplan–Meier plots. For coefficient plots, use the forest plot with `layout="table"`. `plot_forest(..., panel="model")` stacks one panel per value of that column, with aligned table columns.

The numbers behind survival curves (`survival_curve`, Cox, AFT, Fine–Gray) are in `statract.surv`. `statract.viz.km` only draws the figures.

The API pages are [forest](../api/viz/forest.md), [km](../api/viz/km.md), [tree](../api/viz/tree.md), [balance](../api/viz/balance.md), and [misc](../api/viz/misc.md). Full worked figures are in the [examples](../examples/index.md).
