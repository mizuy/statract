# lmm_pbcseq_results

解釈は [lmm_pbcseq_discussion.md](lmm_pbcseq_discussion.md)。

## 対象集団

@import "lmm_pbcseq_out/n.md"

除外なし（flowchart 省略）。

@import "lmm_pbcseq_out/visit_n.md"

## Table 1（患者単位・ベースライン）

| 内容 | CSV |
|---|---|
| Table 1 by `trt_label` | [lmm_pbcseq_out/table1.csv](lmm_pbcseq_out/table1.csv) |

@import "lmm_pbcseq_out/table1_csv.md"

## ガウス LMM（変量切片）

@import "lmm_pbcseq_out/lmm_fixed_csv.md"

@import "lmm_pbcseq_out/lmm_variance_csv.md"

## OLS + 患者クラスタ SE

未調整分散:

@import "lmm_pbcseq_out/ols_naive_csv.md"

クラスタ SE:

@import "lmm_pbcseq_out/ols_cluster_csv.md"

## GAM（病日の集団平滑・主解析）

@import "lmm_pbcseq_out/gam_n.md"

@import "lmm_pbcseq_out/gam_smooth_csv.md"

![GAM of log bilirubin vs day](lmm_pbcseq_out/figures/gam_day.png)

## GAMM（`gamm`: s(day_years) + dp + (1 | id)、bili > 2 mg/dL）

@import "lmm_pbcseq_out/gamm_n.md"

@import "lmm_pbcseq_out/gamm_tidy_csv.md"

@import "lmm_pbcseq_out/gamm_compare_csv.md"

![GAMM vs gam vs GLMM day effect](lmm_pbcseq_out/figures/gamm_day.png)
