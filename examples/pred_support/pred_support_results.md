# pred_support_results

解釈は [pred_support_discussion.md](pred_support_discussion.md)。

## 対象集団の流れ

@import "pred_support_out/text_flowchart.md"

@import "pred_support_out/n.md"

@import "pred_support_out/split.md"

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by 180-day death | [pred_support_out/table1.csv](pred_support_out/table1.csv) |

@import "pred_support_out/table1_csv.md"

## 学習標本の多変量 GLM（OR）

@import "pred_support_out/glm_full_train_csv.md"

![Multivariable GLM forest](pred_support_out/figures/glm_full_forest.png)

## ホールドアウト較正 / Brier

@import "pred_support_out/text_calibration.md"

@import "pred_support_out/table_brier_csv.md"

![Calibration (validation)](pred_support_out/figures/fig_calibration.png)

## 見かけ vs 検証較正

@import "pred_support_out/table_calibration_apparent_csv.md"

![Calibration apparent vs validation](pred_support_out/figures/fig_calibration_apparent.png)

## 決定曲線（DCA）

@import "pred_support_out/text_dca.md"

@import "pred_support_out/table_dca_csv.md"

![Decision curve](pred_support_out/figures/fig_dca.png)

## 閾値性能

@import "pred_support_out/binary_perf_val_csv.md"

@import "pred_support_out/threshold_tradeoff_val_csv.md"
