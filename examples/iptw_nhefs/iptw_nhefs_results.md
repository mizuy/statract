# iptw_nhefs_results

解釈は [iptw_nhefs_discussion.md](iptw_nhefs_discussion.md)。

## 対象集団の流れ

@import "iptw_nhefs_out/text_flowchart.md"

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by `qsmk` | [iptw_nhefs_out/table1.csv](iptw_nhefs_out/table1.csv) |

@import "iptw_nhefs_out/table1_csv.md"

## 傾向スコアと安定化重み

@import "iptw_nhefs_out/iptw_n.md"

@import "iptw_nhefs_out/iptw_weight_summary_csv.md"

@import "iptw_nhefs_out/ps_glm_csv.md"

## 未調整 OLS

@import "iptw_nhefs_out/ols_unadjusted_csv.md"

## IPTW ATE（HC3）

@import "iptw_nhefs_out/ols_iptw_hc3_csv.md"

![IPTW OLS forest (HC3)](iptw_nhefs_out/figures/ols_iptw_forest.png)

## 多重代入感度（MICE + Rubin）

@import "iptw_nhefs_out/mi_note.md"

@import "iptw_nhefs_out/mi_vs_cc_csv.md"

@import "iptw_nhefs_out/mi_pooled_csv.md"

## CEM 感度（バランスのみ・ATE ではない）

@import "iptw_nhefs_out/cem_sensitivity.md"

![CEM Love plot](iptw_nhefs_out/figures/cem_love.png)
