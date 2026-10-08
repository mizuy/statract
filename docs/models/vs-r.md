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
| 重み付き Cox | `coxph(..., weights=)` の Lin–Wei 分散 | `cox_ph(..., weights=)` | 係数、頑健 SE、部分尤度 |
| 加速故障時間 | `survreg` | `accelerated_failure` | weibull、lognormal、exponential の係数、SE、対数尤度。尺度は `Log(scale)` |
| Aalen–Johansen | `survfit(Surv(time, factor(event)) ~ 1)`。左切り捨ては `Surv(entry, time, factor(event))` | `survival_curve(..., kind="aalen_johansen", entry=)` | 累積発生、Aalen 型 SE、plain 区間 |
| Fine–Gray | `finegray` のあと `coxph` | `fine_gray` のあと `cox_ph(..., entry=, weights=)` | 係数、モデルベース SE、部分尤度 |
| 累積発生（cmprsk） | `cuminc(ftime, fstatus, group, strata, rho=)`、`timepoints` | `cumulative_incidence(data, time, event, by, strata=, rho=)`、`.at(times)` | 曲線の角の時刻・推定値・分散、Gray 検定の統計量と p 値（層あり、rho = 0 と 1）、時点の推定値と分散。fixture は `cmprsk.json` |
| Fine–Gray（cmprsk） | `crr(ftime, fstatus, cov1, failcode=, cengroup=)`、`predict.crr` | `fine_gray_regression`。式は `Surv(time, status) ~ x + grp`、`cause=`、`censor_group=` | 係数、Fine–Gray のサンドイッチ分散、情報行列、擬似対数尤度（null も）、基準ハザードの跳び、スコア残差、予測 CIF。rtol 1e-8 |
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
| ROC 曲線と AUC | `pROC::roc`（既定の水準、`direction="auto"`、`na.rm=TRUE`） | `roc_curve` | 閾値、感度、特異度、AUC（rtol 1e-10）。fixture は `proc.json` の 3 標本（同順位なし、同順位と欠測あり、完全分離の小標本） |
| AUC の分散と区間 | `var`、`ci.auc(method="delong")` | `RocCurve.var_auc`、`RocCurve.ci_auc` | DeLong の分散と 95% / 90% 区間（rtol 1e-8） |
| 最適な閾値 | `coords(..., "best", best.method=, transpose=FALSE)`、閾値の指定 | `RocCurve.coords` | youden と closest.topleft の閾値、特異度、感度、正確度、NPV、PPV（rtol 1e-10） |
| 2 本の AUC の比較 | `roc.test(method="delong")` | `roc_test` | 対応あり（Z）と対応なし（t と自由度）の統計量（rtol 1e-8）と p 値（atol 1e-8）。欠測の位置が違うときの共通部分での再計算も含む |
| 共線性 | `performance::check_collinearity` | `check_collinearity` | VIF、SE factor、許容度と区間 |
| 線形混合 | `lmer`。`(1 \| g)` または `(1 + x \| g)`、REML または ML | `fit_mixed`（既定 `engine="lme"` = lme-python 0.2.6） | fixture `lmm.json` 3 標本: coef rel 最大約 1.9×10⁻⁴。pytest は coef 5e-4、SE 1e-3、RE 1e-2、σ² 5e-4、loglik atol 1e-5。STAR 公開スライス（2026-10-04）では lme の coef rel 最大 1.18×10⁻³（変量傾き 10,000 行）、ML 1,000 行は 7.41×10⁻⁴。mixedlm-rs extra は変量切片で旧 1e-6 級に近い |
| 二項 GLMM | `glmer(..., family=binomial)`。`(1 \| g)`、Laplace | `fit_mixed(..., family="binomial")` | fixture 未固定。indo n=602（2026-10-04）: インドメタシン OR rel 3.66×10⁻³（0.466 vs 0.464）、クラスタ分散 rel 1.44×10⁻³、loglik abs 2.14×10⁻³。固定効果の最大 rel は小さい係数 `sod_yes` で 0.115 |
| Wilkinson 式 | `model.matrix`、`lm` | `model_matrix`、`fit_ols` | 設計行列、応答、`offset()`、`y ~ x * stage` と `log(y) ~ x` と `y ~ x + offset(z)` の係数と SE。fixture は `wilkinson.json` |

各行は 3 標本です。許容差はサンドイッチ共分散は rtol 1e-8、Newton 法の係数は rtol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6、p 値は atol 1e-6 です。マッチの組は完全一致です。加法モデルの範囲、thin plate、テンソル、共線性、最適マッチ、full matching は `gam_scope.json`、`collinearity.json`、`match_opt_full.json` の 1 標本です。重みと offset、テンソル交互作用は `gam_weight_ti.json`、条件付き推論木は `ctree.json` の 1 標本です。ROC の 4 行は `proc.json` で、再生成は `Rscript tests/r_oracle/scripts/proc.R` です。

ガンマ GLM の対数尤度は statsmodels の密度を返すため、fixture の比較には入れていません。係数、SE、HC0 は比べています。

## 呼び出せるが、fixture ではまだ固定していないもの

- Cox の比例ハザード検定、残差、一致指数、ベースライン累積ハザード
- ユークリッド、尺度つきユークリッド、ロバスト・マハラノビスの最近傍（実装は MatchIt の組と合わせてある。コミットした JSON はマハラノビス）
- 区間分割 `split_follow_up`
- ブートストラップ共分散（乱数生成器が R と違う）

## この版の対象外

- クラスタ頑健分散の HC2 / HC3
- 計数過程（`entry` がある）Cox の頑健分散。`fine_gray` のあと `cox_ph` の SE はモデルベースです。`crr` と同じサンドイッチ SE は `fine_gray_regression` です
- `crr` の時間依存項（`cov2`、`tf`）

## 既存の名前

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival` はそのままです。`sm_summary2df` は警告付きで残しています（置き換え先は `Fit.tidy`）。マッチングは `match_sample`、回帰の係数表は `fit_glm` と `Fit.tidy` です。
