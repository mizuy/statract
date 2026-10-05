# logit_indo_results

解釈は [logit_indo_discussion.md](logit_indo_discussion.md)。

## 対象集団

@import "logit_indo_out/n.md"

除外なし（flowchart 省略）。

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by `rx` | [logit_indo_out/table1.csv](logit_indo_out/table1.csv) |

@import "logit_indo_out/table1_csv.md"

## 施設変量 GLMM（主解析）

@import "logit_indo_out/glmm.md"

@import "logit_indo_out/glmm_tidy_csv.md"

![GLMM fixed effects forest](logit_indo_out/figures/glmm_fixed_forest.png)

## 施設変量効果（BLUP）

@import "logit_indo_out/glmm_random_effects_csv.md"

![Site BLUPs](logit_indo_out/figures/glmm_site_blups.png)

## Median odds ratio

@import "logit_indo_out/glmm_mor_csv.md"

## 二項 GLM（固定効果比較）

@import "logit_indo_out/glm_n.md"

未調整:

@import "logit_indo_out/glm_unadjusted_csv.md"

調整:

@import "logit_indo_out/glm_adjusted_csv.md"

![Adjusted GLM forest (OR)](logit_indo_out/figures/glm_adjusted_forest.png)
