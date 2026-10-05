# cox_retinopathy_results

解釈は [cox_retinopathy_discussion.md](cox_retinopathy_discussion.md)。

## 対象

@import "cox_retinopathy_out/n.md"

## Table 1（ベースライン特性、眼単位、`hue=trt_label`）

| 内容 | CSV |
|---|---|
| Table 1 by `trt_label` | [cox_retinopathy_out/table1.csv](cox_retinopathy_out/table1.csv) |

@import "cox_retinopathy_out/table1_csv.md"

## Kaplan–Meier（number-at-risk 付き）

@import "cox_retinopathy_out/km_at_csv.md"

![KM by trt](cox_retinopathy_out/figures/km_trt.png)

## Log-rank

@import "cox_retinopathy_out/logrank.md"

@import "cox_retinopathy_out/logrank_counts_csv.md"

## Cox PH（Lin–Wei sandwich）

@import "cox_retinopathy_out/cox_n.md"

@import "cox_retinopathy_out/cox_tidy_csv.md"

@import "cox_retinopathy_out/cox_se_compare_csv.md"

![Cox forest (HR, table+forest)](cox_retinopathy_out/figures/cox_forest.png)

## 比例ハザード検定

@import "cox_retinopathy_out/ph_test.md"

@import "cox_retinopathy_out/ph_test_csv.md"

## Cox 診断

![Scaled Schoenfeld](cox_retinopathy_out/figures/cox_schoenfeld.png)

![Log-log](cox_retinopathy_out/figures/cox_loglog.png)

![dfbeta](cox_retinopathy_out/figures/cox_dfbeta.png)

![Martingale](cox_retinopathy_out/figures/cox_martingale.png)

![Deviance](cox_retinopathy_out/figures/cox_deviance.png)
