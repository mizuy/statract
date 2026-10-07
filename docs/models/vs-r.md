# models と R パッケージの対応

呼び方は Python の関数と列名です。同じ標本で R と数値が一致することを、`tests/r_oracle` の fixture で確認します。pytest の実行時に R は要りません。fixture の再生成は `Rscript tests/r_oracle/scripts/generate.R` です。

使い方とモジュールの分け方は [Models](overview.md)、式の文法は [Wilkinson 式](formula.md) です。公開データでの秒数と最大誤差の一覧は [Benchmarks](benchmarks.md) です。ガウスの `lmer` は `fit_mixed`（既定エンジン lme-python）です。二項 `glmer` も同じ関数（`family="binomial"`）ですが、この表の fixture 許容差はまだガウス LMM だけです。

## 役割の対応

| 役割 | R | `statract` | テストが比べる量 |
|------|---|----------------|------------------|
| 最小二乗 | `lm` | `fit_ols` | 係数、SE、対数尤度、`tidy` の t 統計量と p 値。3 標本 |
| HC0–HC5 | `sandwich::vcovHC` | `hc_covariance` | OLS の HC0–HC5。GLM の HC0 |
| クラスタ頑健分散 | `sandwich::vcovCL` | `cluster_covariance` | OLS の 1-way と 2-way。既定は OLS が HC1、GLM が HC0 |
| Newey–West | `NeweyWest(..., prewhite=FALSE, adjust=FALSE, lag=floor(4*(n/100)^(2/9)))` | `newey_west_covariance` | そのラグと Bartlett 核の共分散 |
| 入れ子の Wald / 尤度比 | `waldtest`、`lrtest` | `wald_test`、`likelihood_ratio_test` | 統計量と p 値 |
| Breusch–Pagan、Durbin–Watson、RESET、Breusch–Godfrey | `lmtest` | 同名の `*_test` | 統計量と p 値。Durbin–Watson の p 値は n ≥ 100 の正規近似 |
| 一般化線形モデル | `glm`。リンクは identity / logit / log / inverse | `fit_glm` | binomial、poisson、gamma の係数と SE、binomial と poisson の対数尤度、HC0 |
| Kaplan–Meier | `survfit`（既定の log 区間） | `survival_curve` | 時点 0.5, 1, 1.5 の生存率と区間 |
| Nelson–Aalen | `survfit(..., stype=2)` | `survival_curve(..., kind="nelson_aalen")` | 同じ時点の生存率 |
| log-rank | `survdiff` | `log_rank` | 層なし、rho = 0 のカイ二乗 |
| Cox | `coxph(Surv(time, status) ~ x)`。既定の同順位は Efron | `cox_ph`。式は `Surv(time, status) ~ x`、列名も残す | Efron、Breslow、`strata` の係数、モデルベース SE、部分尤度 |
| 条件付きロジスティック | `clogit`。既定は exact | `conditional_logit`。式は `y ~ x + strata(set)`、列名も残す | exact の係数、モデルベース SE、条件付き対数尤度。`method="efron"` は時間を 1 にした `cox_ph` |
| 重み付き Cox | `coxph(..., weights=)`。整数でない重みは Lin–Wei 分散、整数の重みはモデルベース | `cox_ph(..., weights=)` | 係数、頑健 SE、部分尤度 |
| Cox の推論 | `residuals.coxph`（7 種）、`cox.zph(transform="km")`、`concordance`、`basehaz(centered=FALSE)`、`predict(type="expected")` | `fit.residuals(kind=)`、`proportional_hazards_test`、`fit.concordance()`、`fit.baseline_hazard()`、`fit.predict(kind="expected")` | fixture `cox_inference.json`。同順位あり（Efron / Breslow）、層と重み、計数過程（クラスタありとなし）の 6 標本。残差 rtol 1e-6、`cox.zph` は項ごとのカイ二乗と自由度、C と SE |
| クラスタ頑健分散 | `coxph(..., cluster=id)`。計数過程 `Surv(start, stop, status)` を含む | `cox_ph(..., cluster=)` または式の `cluster(id)` | 頑健共分散 rtol 1e-6 |
| 加速故障時間 | `survreg` | `accelerated_failure` | weibull、lognormal、exponential の係数、SE、対数尤度。尺度は `Log(scale)` |
| Aalen–Johansen | `survfit(Surv(time, factor(event)) ~ 1)`。左切り捨ては `Surv(entry, time, factor(event))` | `survival_curve(..., kind="aalen_johansen", entry=)` | 累積発生、Aalen 型 SE、plain 区間 |
| Fine–Gray | `finegray` のあと `coxph` | `fine_gray` のあと `cox_ph(..., entry=, weights=)` | 係数、モデルベース SE と行ごとの頑健 SE（重みが整数でないので coxph の既定）、部分尤度 |
| 最近傍（logit） | `matchit(..., distance="glm", link="logit", m.order="data")` | `match_sample(..., distance="logit", order="data")` | 組、x1 のマッチ後標準化差 |
| 最近傍（マハラノビス） | `distance="mahalanobis"` | `distance="mahalanobis"` | `order="data"` の組 |
| 完全一致、CEM | `method="exact"`、`method="cem"` | `method="exact"`、`method="cem"` | 重み |
| 最適マッチ | `method="optimal"` | `method="optimal"` | 1:1 の総距離。重み |
| full matching | `method="full"` | `method="full"` | 重みとサブクラスの分割 |
| 加法モデル | `gam(y ~ s(x, bs="cr", k=8), method="REML")` | `gam`、`smooth(..., k=8)` | 平滑化パラメータ、edf、REML、係数 |
| 加法モデルの範囲 | `ps` / `cc` / `re` / `by`、線形項つき `cr`、binomial、poisson、gamma（inverse） | `smooth(..., basis=)`、`family=` | 平滑化パラメータ、edf、REML、係数 |
| thin plate | `s(x, bs="tp")` | `smooth(..., basis="tp")` | edf と当てはめ。係数の向きは mgcv の固有ベクトルと違う |
| テンソル | `te(x, z)` | `tensor_smooth` | edf（rtol 3e-3）と当てはめ（atol 1e-3）。ヌルに近い周辺の平滑化パラメータは平坦 |
| テンソル交互作用 | `ti(x, z)` | `tensor_interaction` | edf（rtol 3e-3）、REML（atol 1e-3）、当てはめ（atol 1e-3）。ヌルに近い周辺の平滑化パラメータは平坦 |
| 重みと offset | `gam(..., weights=, offset=)` | `gam(..., weights=, offset=)`。列名 | 平滑化パラメータ、edf、REML、係数、当てはめ |
| 条件付き推論木 | `partykit::ctree`。二次形式、Šidák（`testtype="Bonferroni"`）、`minsplit=20`、`minbucket=7` | `conditional_tree` | 根の統計量と調整済み p 値、終端ノードの平均 |
| 共線性 | `performance::check_collinearity` | `check_collinearity` | VIF、SE factor、許容度と区間 |
| 線形混合 | `lmer`。`(1 \| g)` または `(1 + x \| g)`、REML または ML | `fit_mixed`（既定 `engine="lme"` = lme-python 0.2.6） | fixture `lmm.json` 3 標本: coef rel 最大約 1.9×10⁻⁴。pytest は coef 5e-4、SE 1e-3、RE 1e-2、σ² 5e-4、loglik atol 1e-5。STAR 公開スライス（2026-10-04）では lme の coef rel 最大 1.18×10⁻³（変量傾き 10,000 行）、ML 1,000 行は 7.41×10⁻⁴。mixedlm-rs extra は変量切片で旧 1e-6 級に近い |
| ポアソン GLMM | `glmmTMB(..., family=poisson)`、`glmer(..., family=poisson, nAGQ=9)`。`offset(log(years))`、`(1 \| g)` と `(1 + x \| g)` | `fit_mixed(..., family="poisson")`（`engine="laplace"`）、`n_agq=` | fixture `glmm_count.json` 3 標本。係数 rtol 1e-4 / atol 2e-5（glmmTMB は自分の許容差で止まるので、こちらの対数尤度は常に同じか高い）、SE rtol 1e-4、対数尤度 atol 1e-6、変量効果の分散 rtol 1e-3。求積は glmer と係数 rtol 1e-4 |
| 負の二項 GLMM | `glmmTMB(..., family=nbinom2)` | `fit_mixed(..., family="negative_binomial")` | 同じ 3 標本で係数、SE、対数尤度、分散、`theta`（rtol 1e-4） |
| 二項 GLMM | `glmer(..., family=binomial)`。`(1 \| g)`、Laplace | `fit_mixed(..., family="binomial")` | fixture 未固定。indo n=602（2026-10-04）: インドメタシン OR rel 3.66×10⁻³（0.466 vs 0.464）、クラスタ分散 rel 1.44×10⁻³、loglik abs 2.14×10⁻³。固定効果の最大 rel は小さい係数 `sod_yes` で 0.115 |
| Wilkinson 式 | `model.matrix`、`lm` | `model_matrix`、`fit_ols` | 設計行列、応答、`offset()`、`y ~ x * stage` と `log(y) ~ x` と `y ~ x + offset(z)` の係数と SE。fixture は `wilkinson.json` |

各行は 3 標本です。許容差はサンドイッチ共分散は rtol 1e-8、Newton 法の係数は rtol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6、p 値は atol 1e-6 です。マッチの組は完全一致です。加法モデルの範囲、thin plate、テンソル、共線性、最適マッチ、full matching は `gam_scope.json`、`collinearity.json`、`match_opt_full.json` の 1 標本です。重みと offset、テンソル交互作用は `gam_weight_ti.json`、条件付き推論木は `ctree.json` の 1 標本です。

ガンマ GLM の対数尤度は statsmodels の密度を返すため、fixture の比較には入れていません。係数、SE、HC0 は比べています。

## 呼び出せるが、fixture ではまだ固定していないもの

- ユークリッド、尺度つきユークリッド、ロバスト・マハラノビスの最近傍（実装は MatchIt の組と合わせてある。コミットした JSON はマハラノビス）
- 区間分割 `split_follow_up`
- ブートストラップ共分散（乱数生成器が R と違う）

## この版の対象外

- クラスタ頑健分散の HC2 / HC3

## 既存の名前

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival` はそのままです。`sm_summary2df` は警告付きで残しています（置き換え先は `Fit.tidy`）。マッチングは `match_sample`、回帰の係数表は `fit_glm` と `Fit.tidy` です。
