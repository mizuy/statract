# models と R パッケージの対応

呼び方は Python の関数と列名です。同じ標本で R と数値が一致することを、`tests/r_oracle` の fixture で確認します。pytest の実行時に R は要りません。fixture の再生成は `Rscript tests/r_oracle/scripts/generate.R` です。

使い方とモジュールの分け方は [Models](overview.md)、式の文法は [Wilkinson 式](formula.md) です。公開データでの秒数と最大誤差の一覧は [Benchmarks](benchmarks.md) です。ガウスの `lmer` は `fit_mixed`（既定エンジン lme-python）です。二項・ポアソン・負の二項の GLMM も同じ関数で、既定は自前の Laplace エンジンです。

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
| Cox の多方向クラスタ頑健分散 | `sandwich::vcovCL(fit, cluster = ~ id_patient + e_examiner)`（HC0、`cadjust=TRUE`） | `cluster_covariance(cox_fit, ["id_patient", "e_examiner"], data=frame)` | fixture `cox_two_way.json` 8 例: 二元、HC1、`adjust=False`、一元、係数 1 個（vcovCL が落ちるので R 側は同じ包除を手で書き、他の例で vcovCL と一致を確認）、重みと層と Breslow、三元。共分散 rtol 1e-6 |
| クラスタ頑健分散 | `coxph(..., cluster=id)`。計数過程 `Surv(start, stop, status)` を含む | `cox_ph(..., cluster=)` または式の `cluster(id)` | 頑健共分散 rtol 1e-6 |
| 加速故障時間 | `survreg` | `accelerated_failure` | weibull、lognormal、exponential の係数、SE、対数尤度。尺度は `Log(scale)` |
| Aalen–Johansen | `survfit(Surv(time, factor(event)) ~ 1)`。左切り捨ては `Surv(entry, time, factor(event))` | `survival_curve(..., kind="aalen_johansen", entry=)` | 累積発生、Aalen 型 SE、plain 区間 |
| Fine–Gray | `finegray` のあと `coxph` | `fine_gray` のあと `cox_ph(..., entry=, weights=)` | 係数、モデルベース SE と行ごとの頑健 SE（重みが整数でないので coxph の既定）、部分尤度 |
| 累積発生（cmprsk） | `cuminc(ftime, fstatus, group, strata, rho=)`、`timepoints` | `cumulative_incidence(data, time, event, by, strata=, rho=)`、`.at(times)` | 曲線の角の時刻・推定値・分散、Gray 検定の統計量と p 値（層あり、rho = 0 と 1）、時点の推定値と分散。fixture は `cmprsk.json` |
| Fine–Gray（cmprsk） | `crr(ftime, fstatus, cov1, failcode=, cengroup=)`、`predict.crr` | `fine_gray_regression`。式は `Surv(time, status) ~ x + grp`、`cause=`、`censor_group=` | 係数、Fine–Gray のサンドイッチ分散、情報行列、擬似対数尤度（null も）、基準ハザードの跳び、スコア残差、予測 CIF。rtol 1e-8 |
| Cox 回帰標準化（生存関数） | `stdReg2::standardize_coxph(measure="survival")` と `tidy()`。Breslow の Cox、Sjölander (2016) のサンドイッチ | `standardize_cox(..., measure="survival")`、`.tidy(contrast=, reference=, transform=, ci=)` | fixture `stdreg_cox.json` の 4 例: 二値曝露と交互作用（打ち切りと事象が同じ時刻に重なる）、クラスタ、3 水準の連続曝露、変換。推定値、水準間の共分散、表を rtol 1e-6 |
| Cox 回帰標準化（RMST） | `standardize_coxph(measure="rmean")`。群ごとの Efron の Cox、Chen–Tsiatis (2001) | `standardize_cox(..., measure="rmean")` | 同じ fixture の 3 例（主効果、交互作用、同順位）。同順位の例は、下の「群の取り方」の 1 行だけを直した R と比べる |
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
| ポアソン GLMM | `glmmTMB(..., family=poisson)`、`glmer(..., family=poisson, nAGQ=9)`。`offset(log(years))`、`(1 \| g)` と `(1 + x \| g)` | `fit_mixed(..., family="poisson")`（`engine="laplace"`）、`n_agq=` | fixture `glmm_count.json` 3 標本。係数 rtol 1e-4 / atol 2e-5（glmmTMB は自分の許容差で止まるので、こちらの対数尤度は常に同じか高い）、SE rtol 1e-4、対数尤度 atol 1e-6、変量効果の分散 rtol 1e-3。求積は glmer と係数 rtol 1e-4 |
| 負の二項 GLMM | `glmmTMB(..., family=nbinom2)` | `fit_mixed(..., family="negative_binomial")` | 同じ 3 標本で係数、SE、対数尤度、分散、`theta`（rtol 1e-4） |
| 二項 GLMM | `glmmTMB(..., family=binomial)`、`glmer(..., family=binomial, nAGQ=9)`。`(1 \| g)` と `(1 + x \| g)` | `fit_mixed(..., family="binomial")`（`engine="laplace"`）、`n_agq=` | fixture `glmm_binary_zero.json` 3 標本（変量切片、変量傾き、2〜3 行の小さいクラスタ多数）。許容差はポアソン GLMM と同じ。求積は glmer と係数 rtol 1e-4、分散 rtol 1e-3。fixture `glmm_binary_glmer.json`: glmer の Laplace と `nAGQ=25` で係数、SE、対数尤度、分散、MOR、`predict(re.form=NA)` の参照行。glmer は既定の `tolPwrss=1e-7` だと Laplace の対数尤度が約 1e-4 ずれるので、`tolPwrss=1e-13` の glmer とは係数 rtol 2e-6、SE rtol 2e-5、既定の glmer とは係数 rtol 1e-3、SE rtol 1e-2 |
| 交差変量切片と ar1 | `glmmTMB(y ~ x + (1 \| id_patient) + (1 \| e_examiner))`、`ar1(year + 0 \| e_examiner)`。nbinom2、poisson、binomial | `fit_mixed(..., "... + (1 \| id_patient) + ar1(year + 0 \| e_examiner)")` | fixture `glmm_crossed.json` 5 例（990 行、患者 300 × 検査医 30 × 6 年）。係数 rtol 1e-4、SE rtol 1e-4、対数尤度 atol 1e-6、分散 rtol 1e-3、`rho` atol 1e-4、BLUP atol 1e-4。実測の差はどれも約 1e-6。二項の ar1 は rho = 1 の境界で、SE rtol 1e-3 |
| 平滑 + 変量切片 | `gamm4(y ~ s(x), random = ~(1 \| examiner))`、binomial / poisson、`tp` と `cr` | `gamm(frame, "y", [smooth("x", basis="tp")], random="(1 \| examiner)")` | fixture `gamm4.json` 6 例。係数 atol 3e-5、検査医の分散と MOR rtol 1e-4、平滑の値と線形予測 atol 3e-5、BLUP atol 1e-4。SE と edf は gamm4 の式を pivot なしの Cholesky で計算した値と rtol 1e-4。gamm4 0.2-6 は Matrix 1.6 以降で `chol(V, pivot = TRUE)` の置換を読めず、報告する edf と SE が約 1% ずれるため |
| ゼロ過剰 GLMM | `glmmTMB(..., ziformula = ~1 / ~z)`。poisson、nbinom2 | `fit_mixed(..., zero_inflation=True / ["z"])` | 同じ fixture の 3 例。係数、SE、対数尤度、分散、`theta`、ゼロ部分の係数と SE（rtol 1e-4） |
| ハードル GLMM | `glmmTMB(..., family=truncated_poisson / truncated_nbinom2, ziformula = ~z)` | `fit_mixed(..., hurdle=True)` | 同じ fixture の 2 例。量は上と同じ |
| 多重代入の統合 | `mice::pool`、`summary(pool(...), conf.int = TRUE)` | `pool`、`MultipleImputation.pool` | fixture `mice_pool.json`。mice（m=5、pmm / logreg / polyreg）で埋めた同じ 5 データに lm、ロジスティック glm、coxph を当て、推定値、ubar、b、t、Barnard–Rubin 自由度、riv、λ、fmi、SE、p 値、区間（rtol 1e-6）と `dfcom` |
| Wilkinson 式 | `model.matrix`、`lm` | `model_matrix`、`fit_ols` | 設計行列、応答、`offset()`、`y ~ x * stage` と `log(y) ~ x` と `y ~ x + offset(z)` の係数と SE。fixture は `wilkinson.json` |
| ロジスティックの内的妥当性 | `rms::validate.lrm`（`method="boot"`） | `validate_logistic` | Dxy、R2、Intercept、Slope、Emax、D、U、Q、B、g、gp の index.orig、training、test、optimism、index.corrected、n（rtol 1e-6）。同じ再標本を `indices=` で渡す |
| ロジスティックの較正曲線 | `rms::calibrate`（`lrm`、lowess、既定の `predy`） | `calibrate_logistic`、図は `plot_calibration_curve` | apparent と bias-corrected の曲線（atol 1e-6）、平均絶対誤差、0.9 分位 |
| Cox の内的妥当性 | `rms::validate.cph` | `validate_cox` | Dxy、R2、Slope、D、U、Q、g の各列（rtol 1e-6） |
| Cox の較正 | `rms::calibrate.cph(cmethod="KM")`。`cph(..., surv=TRUE, time.inc=u)` | `calibrate_cox(..., u=, m=)` | 群ごとの予測生存、KM、KM.corrected、std.err、optimism（atol 1e-6） |
| 一致指数 | `Hmisc::somers2`、`rms` の `dxy.cens` | `somers_dxy` | C と Dxy |
| lowess | `stats::lowess` | `statract._lowess_r.lowess_r`（内部） | 既定（iter=3）、iter=0、f=0.2 と delta=0 の当てはめ |

各行は 3 標本です。許容差はサンドイッチ共分散は rtol 1e-8、Newton 法の係数は rtol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6、p 値は atol 1e-6 です。マッチの組は完全一致です。加法モデルの範囲、thin plate、テンソル、共線性、最適マッチ、full matching は `gam_scope.json`、`collinearity.json`、`match_opt_full.json` の 1 標本です。重みと offset、テンソル交互作用は `gam_weight_ti.json`、条件付き推論木は `ctree.json` の 1 標本です。ROC の 4 行は `proc.json` で、再生成は `Rscript tests/r_oracle/scripts/proc.R` です。

rms の行は `rms.json`（rms 6.7-1、R 4.3.3）です。再生成は `Rscript tests/r_oracle/scripts/rms.R` です。R の乱数は再現しないので、スクリプトは `set.seed` のあとの `sample(n, replace=TRUE)` を B 回くり返して再標本を作り、`predab.resample(debug=TRUE)` が表示する訓練標本の行と一致することを確かめてから書き出します。rms の `lrm.fit` は −2 log L の変化が 0.025 未満で、`cph` は相対変化 1e-4 で止まるので、`statract` も同じ手順で当てはめます。係数は `fit_glm` や `cox_ph` と小さな標本で 1e-4 程度ずれることがあります。`seed=` だけで呼ぶと numpy の乱数を使うので、値は R と一致しません。

ガンマ GLM の対数尤度は statsmodels の密度を返すため、fixture の比較には入れていません。係数、SE、HC0 は比べています。

## 呼び出せるが、fixture ではまだ固定していないもの

- ユークリッド、尺度つきユークリッド、ロバスト・マハラノビスの最近傍（実装は MatchIt の組と合わせてある。コミットした JSON はマハラノビス）
- 区間分割 `split_follow_up`
- ブートストラップ共分散（乱数生成器が R と違う）
- `impute_chained` の埋めた値（乱数生成器が R と違う。テストは型、観測値の保存、MAR での推定値の回復を見る）

## この版の対象外

- クラスタ頑健分散の HC2 / HC3
- `crr` の時間依存項（`cov2`、`tf`）。`crr` と同じサンドイッチ SE は `fine_gray_regression` です

## stdReg2 と違うところ

`standardize_cox` は stdReg2 1.0.8 に合わせていますが、次の点は R と違います。

- 生存関数の分散では、事象時刻と同じ時刻で打ち切られた行にも、その時刻の基底ハザードの増分が入ります。R が時刻の値で行に割り当てるためで、これは R に合わせています。
- RMST で外す項: R は `grep(曝露名, 項名)` で外すので、曝露が `ope` なら `operation` の項も消えます。statract は曝露の列を使う項だけを外します。
- RMST の各事象の群: R は事象時刻を `match()` で同じ時刻の最初の行に当て、その行の群を使います。その行は打ち切りや別の群のこともあります。statract は事象を起こした行の群を使います。同じ時刻の事象が無ければ一致します。
- RMST で曝露のほかに共変量が無い式は拒否します。R も `reformulate()` で止まります。
- RMST の `times` が 2 個以上なら `ValueError` です。R は警告を出して最大値を使います。
- RMST の `values` を `[1, 0]` の順に渡しても、各行の値と推定値は対応します。R は推定値を 0、1 の順に並べたまま、ラベルだけを `values` の順にします。
- 文字列のクラスタ列も使えます。R は列を data.table の `by` に渡すので、文字列だと列名と読んで止まります。fixture のクラスタは整数です。
- `transform="logit"` と `"odds"` は生存関数で使えます。R の `summary_std_coxph` は作る前の `out$measure` を読むので止まります。fixture はグローバルに `out` を置いて R に計算させています。
- `ci="log"` と `contrast="difference"` の組み合わせは拒否します。参照行の推定値が 0 になり、R では区間が NaN になります。
- ケースウェイトはありません。R も Cox にウェイトを渡しません。

## 既存の名前

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival` はそのままです。`sm_summary2df` は警告付きで残しています（置き換え先は `Fit.tidy`）。マッチングは `match_sample`、回帰の係数表は `fit_glm` と `Fit.tidy` です。
