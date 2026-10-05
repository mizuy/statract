# statract — 概要

列名で呼ぶ統計関数です。データは Polars の `DataFrame`、結果の表も Polars です。`fit_ols`、`fit_glm`、`fit_mixed` は Wilkinson 式（`y ~ x * stage`、`y ~ x + (1 | g)`）も受けます。`cox_ph`、`accelerated_failure`、`fine_gray` は `Surv(time, status) ~ age + sex` も受けます。`conditional_logit` は `y ~ x + strata(set)` も受けます。文法は [Wilkinson 式](formula.md) です。列名のリストでも同じモデルを呼べます。

| 層 | モジュール | R での近いもの |
|----|------------|----------------|
| 設計行列 | `statract.design`、`model_matrix` | `model.matrix` の treatment contrast |
| 線形モデル | `statract.fit` | `lm` / `glm` |
| 共分散 | `statract.covariance` | sandwich |
| 線形の検定 | `statract.linear_tests` | lmtest |
| 生存時間 | `statract.surv` | survival |
| マッチング | `statract.matching` | MatchIt |
| 加法モデル | `statract.gam` | mgcv。gaussian / binomial / poisson / gamma、`cr` / `tp` / `cc` / `ps` / `re`、テンソル、`ti`、`by`、重み、offset |
| 条件付き推論木 | `statract.tree` | `partykit::ctree`。数値の応答、二次形式、Šidák 調整 |
| 線形・一般化線形混合 | `statract.mixed` | `lmer` / `glmer`。エンジンは lme-python（lme-rs） |

公開名は `statract` からまとめて import できます。数値の対応は [R パッケージとの対応](vs-r.md)、公開データでの **速度と差の一覧** は [Benchmarks](benchmarks.md)、呼び出し例は [Quickstart](quickstart.md)、式の文法は [Wilkinson 式](formula.md)、ベンチの設計は [benchmark-plan](benchmark-plan.md)、残っている差の潰し方は [次の速度改善](speed-plan.md) です。実データの一連の流れは [Examples / Gallery](examples.md) です。

`import statract` は R を起動しません。R は `tests/r_oracle/scripts/generate.R` が fixture を書くときだけ使います。pytest はその JSON を読みます。

## 共通の呼び方

- 第 1 引数はデータ。説明変数は列名のリストか、結果を左辺に置いた Wilkinson 式です。
- カテゴリの最初の水準が参照です。文字列はソート順、`pl.Enum` は定義順です。論理値は `FALSE` / `TRUE` です。ダミー名は `列名 + 水準` で、切片は `(Intercept)` です。交互作用は `:` でつなぎます。
- 欠損がある行は、そのモデルが使う列についてまとめて落とします。
- 係数表 `tidy()` の列は `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` です。
- 予測の種類は `kind` です。グループは `by` です。

## 既存の関数

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival`、`tableone` はそのまま使えます。`sm_summary2df` は呼び出すと `DeprecationWarning` を出します（置き換え先は `Fit.tidy`）。マッチングは `match_sample`、係数表は `fit_glm` / `Fit.tidy` です。

`fit_mixed` は混合モデルの公開入口です。既定エンジンは lme-python（lme-rs）。ガウスは `lmer`（`method="reml"` / `"ml"`）、二項・ポアソン・ガンマは `glmer`（`family=`）。Wilkinson 式 `(1 | g)` を受けます。lme-python は `fit_mixed` を呼んだときに読みます。ガウスの以前のエンジン mixedlm-rs は `engine="mixedlm_rs"`（`statract[mixedlm]`）です。gpboost の `glmm_gpboost` は optional extra の実験実装で、GLMM の本体ではありません。glmmTMB のゼロ過剰や分散モデルは対象外です。
