# surv_colon_results

解釈は [surv_colon_discussion.md](surv_colon_discussion.md)。

## 対象集団の流れ

@import "surv_colon_out/text_flowchart.md"

@import "surv_colon_out/mermaid_flowchart.md"

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by `rx` | [surv_colon_out/table1.csv](surv_colon_out/table1.csv) |

@import "surv_colon_out/table1_csv.md"

## Kaplan–Meier（number-at-risk 付き）

@import "surv_colon_out/km_at_3y_csv.md"

![KM by rx](surv_colon_out/figures/km_rx.png)

## Log-rank

@import "surv_colon_out/logrank.md"

@import "surv_colon_out/logrank_counts_csv.md"

## Cox PH

@import "surv_colon_out/cox_n.md"

@import "surv_colon_out/cox_tidy_csv.md"

![Cox forest (HR, table+forest)](surv_colon_out/figures/cox_forest.png)

## 比例ハザード検定

@import "surv_colon_out/ph_test.md"

@import "surv_colon_out/ph_test_csv.md"

## Cox 診断

![Scaled Schoenfeld](surv_colon_out/figures/cox_schoenfeld.png)

![Log-log](surv_colon_out/figures/cox_loglog.png)

![dfbeta](surv_colon_out/figures/cox_dfbeta.png)

![Martingale](surv_colon_out/figures/cox_martingale.png)

![Deviance](surv_colon_out/figures/cox_deviance.png)
