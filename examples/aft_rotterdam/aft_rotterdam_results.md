# aft_rotterdam_results

解釈は [aft_rotterdam_discussion.md](aft_rotterdam_discussion.md)。

## 対象集団

@import "aft_rotterdam_out/n.md"

除外なし（flowchart 省略）。

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by hormone therapy | [aft_rotterdam_out/table1.csv](aft_rotterdam_out/table1.csv) |

@import "aft_rotterdam_out/table1_csv.md"

## Kaplan–Meier（number-at-risk 付き）

@import "aft_rotterdam_out/km_at_csv.md"

![KM by hormon](aft_rotterdam_out/figures/km_hormon.png)

## Log-rank

@import "aft_rotterdam_out/logrank.md"

@import "aft_rotterdam_out/logrank_counts_csv.md"

## Nelson–Aalen（指定時点）

@import "aft_rotterdam_out/nelson_aalen_at_csv.md"

## Cox PH

@import "aft_rotterdam_out/ph_test.md"

@import "aft_rotterdam_out/cox_tidy_csv.md"

![Cox forest (HR, table+forest)](aft_rotterdam_out/figures/cox_forest.png)

## 比例ハザード検定

@import "aft_rotterdam_out/ph_test_csv.md"

## Cox 診断

![Scaled Schoenfeld](aft_rotterdam_out/figures/cox_schoenfeld.png)

![Log-log](aft_rotterdam_out/figures/cox_loglog.png)

![dfbeta](aft_rotterdam_out/figures/cox_dfbeta.png)

![Martingale](aft_rotterdam_out/figures/cox_martingale.png)

![Deviance](aft_rotterdam_out/figures/cox_deviance.png)

## Cox 内部検証（bootstrap B=200）

@import "aft_rotterdam_out/cox_validate.md"

@import "aft_rotterdam_out/cox_validate_csv.md"

@import "aft_rotterdam_out/cox_calibrate_5y_csv.md"

![Cox calibration at 5 years](aft_rotterdam_out/figures/cox_calibrate_5y.png)

## Weibull AFT

@import "aft_rotterdam_out/aft_n.md"

@import "aft_rotterdam_out/aft_weibull_tidy_csv.md"

@import "aft_rotterdam_out/aft_lognormal_tidy_csv.md"
