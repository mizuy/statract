# psm_rhc_results

解釈は [psm_rhc_discussion.md](psm_rhc_discussion.md)。

## 対象集団

@import "psm_rhc_out/n.md"

除外なし（flowchart 省略）。

## Table 1（未マッチ）

| 内容 | CSV |
|---|---|
| Unmatched Table 1 | [psm_rhc_out/table1_unmatched.csv](psm_rhc_out/table1_unmatched.csv) |

@import "psm_rhc_out/table1_unmatched_csv.md"

## マッチのバランス

@import "psm_rhc_out/match_n.md"

@import "psm_rhc_out/balance_csv.md"

![Love plot](psm_rhc_out/figures/love_plot.png)

## Table 1（マッチ後、weights>0）

@import "psm_rhc_out/table1_matched_csv.md"

## ロジスティック（30 日死亡）

未マッチ:

@import "psm_rhc_out/glm_unmatched_csv.md"

マッチ後:

@import "psm_rhc_out/glm_matched_csv.md"

![Matched logistic forest (OR)](psm_rhc_out/figures/glm_matched_forest.png)
