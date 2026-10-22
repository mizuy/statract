# models と R パッケージの対応

呼び方は Python の関数と列名です。同じ標本で R と数値が一致することを、`tests/r_oracle` の fixture で確認します。pytest の実行時に R は要りません。fixture の再生成は `task r-oracle` です（R と go-task が要ります）。`task r-oracle -- htest rms` のようにスクリプト名を渡すと、その分だけ作り直します。fixture を変えずに今の R の出力と比べるだけなら `task r-oracle:check` です（相対誤差 1e-4 以内なら ok）。

使い方とモジュールの分け方は [Models](overview.md)、式の文法は [Wilkinson 式](formula.md) です。公開データでの秒数と最大誤差の一覧は [Benchmarks](benchmarks.md) です。ガウスの `lmer` は `fit_mixed`（既定エンジン lme-python）です。二項・ポアソン・負の二項の GLMM も同じ関数で、既定は自前の Laplace エンジンです。

## 役割の対応

| 役割 | R | `statract` | テストが比べる量 |
|------|---|----------------|------------------|
| スプライン | `splines::ns`、`splines::bs`、`rms::rcs`。新しい行は `makepredictcall` の節点 | 式の `ns()`、`bs()`、`rcs()` | fixture `splines.json`。13 の式で列名と行列（欠損、同順位、100 行未満、端に 15% の同値）と範囲外を含む新しい行、`lm` の係数 |
| スプラインの検定と効果曲線 | `anova(lrm)`、`anova(cph)` の項と `Nonlinear` の行、`contrast(fit, list(x = at), list(x = ref))` | `spline_test`、`spline_effect` | 同じ fixture。カイ二乗、自由度、p 値、対比の推定値、SE、区間。glm と coxph のスプラインつき係数と SE |
| 最小二乗 | `lm` | `fit_ols` | 係数、SE、対数尤度、`tidy` の t 統計量と p 値。3 標本 |
| HC0–HC5 | `sandwich::vcovHC` | `hc_covariance` | OLS の HC0–HC5。GLM の HC0 |
| クラスタ頑健分散 | `sandwich::vcovCL` | `cluster_covariance` | OLS の 1-way と 2-way。既定は OLS が HC1、GLM が HC0 |
| Newey–West | `NeweyWest(..., prewhite=FALSE, adjust=FALSE, lag=floor(4*(n/100)^(2/9)))` | `newey_west_covariance` | そのラグと Bartlett 核の共分散 |
| 入れ子の Wald / 尤度比 | `waldtest`、`lrtest` | `wald_test`、`likelihood_ratio_test` | 統計量と p 値 |
| Breusch–Pagan、Durbin–Watson、RESET、Breusch–Godfrey | `lmtest` | 同名の `*_test` | 統計量と p 値。Durbin–Watson の p 値は n ≥ 100 の正規近似 |
| 一般化線形モデル | `glm`。リンクは identity / logit / log / inverse | `fit_glm` | binomial、poisson、gamma の係数と SE、binomial と poisson の対数尤度、HC0 |
| 負の二項回帰（NB2） | `MASS::glm.nb`（対数リンク） | `fit_glm(..., family="negative_binomial")` | fixture `glm_nb.json` の 4 例（offset、重み、交互作用、切片なし）。係数と SE、`theta` とその SE、対数尤度、AIC、BIC、逸脱度、当てはめ値、HC0。係数 rtol 1e-7 |
| 修正ポアソン（リスク比） | `glm(family=poisson)` と `sandwich::vcovHC(type="HC0")`、`vcovCL(cluster=, type="HC0")` | `fit_risk_ratio(data, "y ~ arm + age")`、`cluster=` | fixture `risk_ratio.json` の 3 例（重みあり、交互作用）。係数 rtol 1e-10、HC0 とクラスタ頑健共分散 rtol 1e-6、モデルベース共分散、逸脱度、対数尤度、AIC |
| log-binomial | `glm(family=binomial(link="log"))`、`start=` | `fit_risk_ratio(..., method="log-binomial", start=)` | 同じ fixture の 3 例（既定の開始値、重み、`start=`）。係数 rtol 1e-10、モデルベースと HC0 の共分散、クラスタ頑健共分散。当てはめは McCullagh と Nelder の IRLS を書き起こしたもの。0 < mu < 1 を外れる一歩は半分に戻す |
| Firth ロジスティック | `logistf::logistf`（`pl=TRUE`、`logistf.control()` の既定）。logistf が入らないので、Firth（1993）と Heinze・Schemper（2002）の式から base R で別に書いた実装と比べる | `fit_firth(data, "y ~ arm + x")`、`tidy()`、`tidy(method="wald")`、`glance()` | fixture `firth.json` の 7 例（度数の重み、交互作用、準完全分離と完全分離、切片なし、90% 区間）。係数 rtol 1e-9、共分散、罰則付き対数尤度、null の対数尤度、係数ごとの尤度比カイ二乗と p 値、プロファイル区間 rtol 1e-7、全体の尤度比検定 |
| GLM のプロファイル尤度区間 | `confint(glm)`（`MASS::confint.glm`、R 4.4 からは `stats::confint.glm`） | `profile_interval(fit, level=)` | fixture `profile_glm.json` の 7 例（二項、度数の重み、交互作用と 90%、オフセットつきポアソン、ガンマ、重みつきガウス）。`confint` とは SE の 1e-3 倍以内（スプライン補間の誤差。最大 2.5e-4 倍）、`uniroot` で解いた正確な端とは 1e-7 |
| ペナルティ付き回帰（リッジ、lasso、elastic net） | `glmnet` 4.1-8。`family=` gaussian / binomial / poisson / cox（Breslow）、`alpha`、`weights`、`offset`、`penalty.factor`、`standardize`、`intercept=FALSE`、`lambda=` | `penalized_regression(data, "y ~ x1 + grp", family=, alpha=)`。`.coef(lambda_)`、`.predict(data, lambda_, kind=)`、`.tidy()`、`.path()` | fixture `penalized.json` の 16 例（横長の標本、重みと offset、ペナルティ係数 0 と 2、標準化なし、切片なし）。自動の lambda 列（gaussian は rtol 1e-8、ほかは 1e-5）、係数、切片、df、`dev.ratio`（atol 1e-4）、`nulldev`。係数は、同じ lambda でいちばん大きい係数の 3e-3 倍まで（glmnet の `thresh=1e-7` で残る収束誤差の大きさ。横長の標本は 0.1 倍）。どの lambda でも、statract の目的関数の値が glmnet の係数での値以下。利用者の lambda 列と R の lambda の渡し直し。path の間の `coef` と `predict`（`exact=FALSE` の線形補間） |
| ペナルティ付き回帰の交差検証 | `cv.glmnet(foldid=)`。`type.measure` は deviance、mse、mae、class、auc | `cv_penalized(data, formula, foldid="fold", measure=)` | 同じ fixture の 10 例（重み、利用者の lambda 列、横長の標本、Cox と重みつき Cox）。`cvm` rtol 5e-4、`cvsd` rtol 5e-3、`nzero`、`lambda.min`、`lambda.1se`。class と auc は lambda の 4 分の 3 以上で一致し、ほかも差 0.01 以内 |
| 順序ロジスティック | `ordinal::clm`（logit、probit、cloglog、loglog）。`gradTol=1e-10` | `ordinal_regression(data, "y ~ x + stage", link=)` | fixture `ordinal.json` の 7 例（文字列の水準、整数の水準と重み、欠測と交互作用）。係数、閾値、共分散（解析的 Hessian）rtol 1e-6、対数尤度、AIC、予測確率 |
| 比例オッズ（MASS） | `MASS::polr(Hess=TRUE)`。`control=list(reltol=1e-14)` で収束させる | `ordinal_regression` | 同じ 7 例。係数と閾値 rtol 1e-6、逸脱度、AIC、予測確率。SE は rtol 2e-3（`polr` の Hessian は `optim` の差分近似で、約 1e-3 ずれる） |
| Brant 検定 | R パッケージなし。Brant (1990) から R で書いたもの（切れ目ごとの二値 `glm` と、適合間の共分散による Wald 検定。出力は `brant` パッケージと比べた） | `brant_test(fit)` | 同じ fixture の 3 例。切れ目ごとの係数、全体と係数ごとのカイ二乗と自由度（rtol 1e-6）、p 値 |
| 多項ロジスティック | `nnet::multinom`。`reltol=1e-12, abstol=1e-14, maxit=1000` で当て直す | `multinomial_regression(data, "y ~ x + sex")` | fixture `multinom.json` の 3 例（3 水準、4 水準と重みと交互作用、2 水準）。係数と SE rtol 2e-5（BFGS は reltol 1e-12 でも最適点から 1e-6 程度で止まる）、逸脱度 rtol 1e-11 で R 以下、AIC、予測確率。既定の `multinom` とは係数 rtol 1e-2 |
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
| 累積発生（cmprsk） | `cuminc(ftime, fstatus, group, strata, rho=)`、`timepoints` | `cumulative_incidence(data, time, event, by, strata=, rho=)`、`.at(times)` | 曲線の角の時刻・推定値・分散、Gray 検定の統計量と p 値（層あり、rho = 0 と 1）、時点の推定値と分散。fixture は `cmprsk.json`。推定値 rtol 1e-12、分散 rtol 1e-10、統計量 rtol 1e-8。実装は Gray (1988) の式 (2.3)、(2.5)、(2.6)、(2.10)、(2.11) から書いた。cmprsk のコードは読んでいない。R は出力の確認だけに使った。論文にない細部（同時刻の事象の扱い、小さいリスク集合）は R の出力と比べて決めた |
| Fine–Gray（cmprsk） | `crr(ftime, fstatus, cov1, failcode=, cengroup=)`、`predict.crr` | `fine_gray_regression`。式は `Surv(time, status) ~ x + grp`、`cause=`、`censor_group=` | 係数、Fine–Gray のサンドイッチ分散、情報行列、擬似対数尤度（null も）、基準ハザードの跳び、スコア残差、予測 CIF。rtol 1e-8。実装は Fine and Gray (1999) の式 (6)、(7)、(8) と 5 節から書いた。cmprsk のコードは読んでいない。R は出力の確認だけに使った。打ち切り重みの左極限、`cengroup` の分散、Newton 法の止め方は論文にないので、R の出力と比べて決めた |
| Cox 回帰標準化（生存関数） | `stdReg2::standardize_coxph(measure="survival")` と `tidy()`。Breslow の Cox、Sjölander (2016) のサンドイッチ | `standardize_cox(..., measure="survival")`、`.tidy(contrast=, reference=, transform=, ci=)` | fixture `stdreg_cox.json` の 4 例: 二値曝露と交互作用（打ち切りと事象が同じ時刻に重なる）、クラスタ、3 水準の連続曝露、変換。推定値、水準間の共分散、表を rtol 1e-6 |
| Cox 回帰標準化（RMST） | `standardize_coxph(measure="rmean")`。群ごとの Efron の Cox、Chen–Tsiatis (2001) | `standardize_cox(..., measure="rmean")` | 同じ fixture の 3 例（主効果、交互作用、同順位）。同順位の例は、下の「群の取り方」の 1 行だけを直した R と比べる |
| GLM の回帰標準化 | `marginaleffects` 0.18 の `avg_predictions` / `avg_comparisons`（`type="response"`、`comparison="difference"` / `"ratioavg"` / `"lnratioavg"` / `"lnoravg"`、`wts=`、`vcov=TRUE` / `"HC0"` / `"HC3"` / `~site`）。`stdReg2::standardize_glm` のサンドイッチ | `standardize_glm`、`.tidy(contrast=, reference=, ci=)`。`covariates="sampled"` が stdReg2 型 | fixture `standardize_glm.json` の 6 例: 二値曝露、交互作用、3 水準の因子、因子 × 共変量、offset 付きポアソン、重み。平均、共分散、差・比・オッズ比の表を rtol 1e-6。共分散は marginaleffects の数値微分の誤差（約 1e-10）があるので atol を対角の最大 × 1e-6。stdReg2 は入らないので、`covariates="sampled"` は R で手書きした同じ推定量と比べる |
| RMST の 2 群比較 | `survRM2::rmst2`（調整なし）。`summary(survfit(...), rmean=tau)`、論文の式を base R で書いたもの、survRM2 1.0-4 の出力 | `restricted_mean_survival(data, "time", "status", by="arm", tau=)` | fixture `rmst.json` の 6 例（0/1 と文字列の群、同順位、`tau` が観察時刻と同じ、短い群が事象で終わり `tau` がその先）。群ごとの RMST と SE は `survfit` と rtol 1e-12。区間、RMTL、差・RMST 比・RMTL 比の推定値、区間、p 値は式と survRM2 の両方と rtol 1e-12 |
| 最近傍（logit） | `matchit(..., distance="glm", link="logit", m.order="data")` | `match_sample(..., distance="logit", order="data")` | 組、x1 のマッチ後標準化差 |
| 最近傍（マハラノビス） | `distance="mahalanobis"` | `distance="mahalanobis"` | `order="data"` の組 |
| 完全一致、CEM | `method="exact"`、`method="cem"` | `method="exact"`、`method="cem"` | 重み |
| 最適マッチ | `method="optimal"` | `method="optimal"` | 1:1 の総距離。重み |
| full matching | `method="full"` | `method="full"` | 重みとサブクラスの分割 |
| IPTW の重み | `WeightIt::weightit(method="glm", estimand=, stabilize=, s.weights=)`、`trim(at=, lower=)` | `propensity_weights(..., estimand=, stabilize=, trim=, trim_lower=, sampling_weights=)` | 傾向スコアと重み（rtol 1e-8）。fixture `iptw.json` 15 例: ATE / ATT / ATC / ATO、安定化、分位点と個数の切り詰め、下側も、3 水準と 2 水準の因子、欠測、`s.weights`。R 側は WeightIt 2.1.0 と cobalt 5.0.0（下の「WeightIt と cobalt」） |
| 重み付けのバランス | `cobalt::bal.tab(W, un=TRUE, binary=, disp.means=TRUE, disp.v.ratio=TRUE)` | `PropensityWeights.balance(binary=, sd_denominator=)`、`effective_sample_size()` | 群の平均、差（SMD または割合の差）、分散比（rtol 1e-7）、群ごとの有効標本サイズ（rtol 1e-8）。同じ fixture、R 側は cobalt 5.0.0 |
| 重み付けのアウトカムモデル | `WeightIt::glm_weightit(y ~ ..., weightit = W, vcov = "asympt" / "HC0", cluster = )` | `weighted_glm(ipw, "y ~ ...", family=, vcov=, cluster=)` | fixture `iptw_outcome.json` 9 例: gaussian、binomial、poisson、gamma、ATE / ATT / ATC / ATO、安定化、`s.weights`、クラスタ。係数（rtol 1e-8）、傾向スコアの推定を含めた共分散と HC0（rtol 1e-7）、`sandwich::vcovHC` / `vcovCL`（rtol 1e-4）。R 側は M 推定を `glm()` で書いたもの（下の「WeightIt と cobalt」） |
| 重み付けの周辺効果 | 重み付きの群平均と差、比、オッズ比（`glm_weightit(y ~ A)` と `marginaleffects::avg_comparisons`）。デルタ法 | `PropensityWeights.effect(outcome, measure=, vcov=)` | 同じ fixture の 7 例。群の平均と推定値（rtol 1e-8）、`"asympt"` と `"HC0"` の SE（rtol 1e-7）。HC0 の差は重み付き線形モデル `y ~ treat` の HC0 と一致 |
| 加法モデル | `gam(y ~ s(x, bs="cr", k=8), method="REML")` | `gam`、`smooth(..., k=8)` | 平滑化パラメータ、edf、REML、係数 |
| 加法モデルの範囲 | `ps` / `cc` / `re` / `by`、線形項つき `cr`、binomial、poisson、gamma（inverse） | `smooth(..., basis=)`、`family=` | 平滑化パラメータ、edf、REML、係数 |
| thin plate | `s(x, bs="tp")` | `smooth(..., basis="tp")` | edf と当てはめ。係数の向きは mgcv の固有ベクトルと違う |
| テンソル | `te(x, z)` | `tensor_smooth` | edf（rtol 3e-3）と当てはめ（atol 1e-3）。ヌルに近い周辺の平滑化パラメータは平坦 |
| テンソル交互作用 | `ti(x, z)` | `tensor_interaction` | edf（rtol 3e-3）、REML（atol 1e-3）、当てはめ（atol 1e-3）。ヌルに近い周辺の平滑化パラメータは平坦 |
| 重みと offset | `gam(..., weights=, offset=)` | `gam(..., weights=, offset=)`。列名 | 平滑化パラメータ、edf、REML、係数、当てはめ |
| 説明率と入れ子の比較 | `summary.gam` の `dev.expl`、`r.sq`、`deviance()`、`null.deviance`、`anova(m1, m2, test=)` | `glance()`、`compare_gams` | fixture `gam_summary.json` 7 モデル: 二項、重み付きガウス、offset つきポアソン、ガウス、ガンマ。逸脱度、帰無逸脱度、説明率、調整済み R²、edf（rtol 1e-4、平坦な 2 例は 5e-4）。逸脱度の変化（atol 5e-3）。mgcv の `df` は平滑化パラメータの不確実性で補正した edf の差なので、`df` と p 値は合わせない |
| 平滑項の検定と補正 AIC | `summary.gam` の `s.table`、`sum(m$edf2)`、`logLik()`、`AIC()` | `smooth_table()`、`glance()` の `edf2`、`aic` | fixture `gam_summary.json` の 7 モデル。`ref_df` と統計量 rtol 1e-3（ガンマは 3e-3）。p 値は論文の帰無分布をモンテカルロで確かめ、mgcv より少し小さい。edf2 rtol 1e-2、対数尤度 rtol 1e-5、AIC は edf2 の 2% 以内 |
| GAM の最尤（ML） | `gam(..., method = "ML")` | `gam(..., method="ml")` | fixture `gam_summary.json` の `ml`: 二項、重み付きガウス、offset つきポアソン、ガウス、ガンマ。基準関数の値 rtol 1e-6、平滑化パラメータ rtol 1e-3、edf rtol 1e-4、尺度 rtol 1e-4 |
| 条件付き推論木 | `partykit::ctree`。二次形式、Šidák（`testtype="Bonferroni"`）、`minsplit=20`、`minbucket=7` | `conditional_tree` | 根の統計量と調整済み p 値、終端ノードの平均 |
| ROC 曲線と AUC | `pROC::roc`（既定の水準、`direction="auto"`、`na.rm=TRUE`） | `roc_curve` | 閾値、感度、特異度、AUC（rtol 1e-10）。fixture は `proc.json` の 3 標本（同順位なし、同順位と欠測あり、完全分離の小標本） |
| AUC の分散と区間 | `var`、`ci.auc(method="delong")` | `RocCurve.var_auc`、`RocCurve.ci_auc` | DeLong の分散と 95% / 90% 区間（rtol 1e-8） |
| 最適な閾値 | `coords(..., "best", best.method=, transpose=FALSE)`、閾値の指定 | `RocCurve.coords` | youden と closest.topleft の閾値、特異度、感度、正確度、NPV、PPV（rtol 1e-10） |
| 診断精度 | `epiR::epi.tests(method=, conf.level=)`（epiR 2.0.67） | `diagnostic_accuracy(test, disease, method=, conf_level=)`、`.tidy()` | fixture `diagnostic.json` の 4 表（0 のセルを含む）× 5 方法 × 2 水準。18 の統計量の推定値と区間 rtol 1e-10。`agresti` は区間だけ（下の「epiR と違うところ」） |
| E-value | `EValue::evalues.RR` / `evalues.OR` / `evalues.HR` / `evalues.MD` の式 | `e_value(est, lower, upper, measure=, rare=)` | R は入らないので、論文の例（RR 3.9、区間 1.8–8.7 で 7.26 と 3.0）と変換式をテストする |
| 2 本の AUC の比較 | `roc.test(method="delong")` | `roc_test` | 対応あり（Z）と対応なし（t と自由度）の統計量（rtol 1e-8）と p 値（atol 1e-8）。欠測の位置が違うときの共通部分での再計算も含む |
| t 検定 | `t.test`（1 標本、Welch、`var.equal=TRUE`、`paired=TRUE`） | `t_test` | t、自由度、p 値、区間、平均（rtol 1e-10）。fixture は `htest.json` |
| Wilcoxon 検定 | `wilcox.test`（符号付き順位、順位和、`paired`、`conf.int=TRUE`） | `wilcox_test` | V / W、p 値、Hodges–Lehmann 推定値と区間（rtol 1e-10）。正確分布（n < 50、同順位なし）と正規近似（同順位・ゼロ、連続性補正の有無）。正確分布は教科書の母関数で数える。正規近似の区間と推定値は `scipy.optimize.brentq` で根を探す。止め方は `uniroot` と同じだが、途中の点が違うので R とは `tol_root`（既定 1e-4）以内で一致する。テストは 2 × `tol_root` で比べる |
| McNemar 検定 | `mcnemar.test` | `mcnemar_test` | 2 × 2（補正あり / なし）、3 × 3 の Bowker 検定、欠測を含む 2 列の統計量と p 値 |
| 正確二項検定 | `binom.test` | `binom_test` | p 値と Clopper–Pearson 区間。0 / n の端、片側 |
| 比率の検定 | `prop.test` | `prop_test` | 1 標本の Wilson 区間（補正あり / なし）、2 標本の差の区間、k 標本と `p` 指定の χ²。片側も |
| 多重比較の補正 | `p.adjust` | `p_adjust` | holm、hochberg、hommel、bonferroni、BH、BY、fdr、none。欠測と `n` 指定 |
| 比率の区間 | `prop.test(correct=FALSE)$conf.int`、`binom.test(...)$conf.int` | `proportion_ci(..., method="wilson" / "clopper-pearson")` | 下限と上限（rtol 1e-10）。既定の `method="wald"` は今までどおり |
| 共線性 | `performance::check_collinearity` | `check_collinearity` | VIF、SE factor、許容度と区間 |
| 線形混合 | `lmer`。`(1 \| g)` または `(1 + x \| g)`、REML または ML | `fit_mixed`（既定 `engine="lme"` = lme-python 0.2.6） | fixture `lmm.json` 3 標本: coef rel 最大約 1.9×10⁻⁴。pytest は coef 5e-4、SE 1e-3、RE 1e-2、σ² 5e-4、loglik atol 1e-5。STAR 公開スライス（2026-10-04）では lme の coef rel 最大 1.18×10⁻³（変量傾き 10,000 行）、ML 1,000 行は 7.41×10⁻⁴。mixedlm-rs extra は変量切片で旧 1e-6 級に近い |
| ポアソン GLMM | `glmmTMB(..., family=poisson)`、`glmer(..., family=poisson, nAGQ=9)`。`offset(log(years))`、`(1 \| g)` と `(1 + x \| g)` | `fit_mixed(..., family="poisson")`（`engine="laplace"`）、`n_agq=` | fixture `glmm_count.json` 3 標本。係数 rtol 1e-4 / atol 2e-5（glmmTMB は自分の許容差で止まるので、こちらの対数尤度は常に同じか高い）、SE rtol 1e-4、対数尤度 atol 1e-6、変量効果の分散 rtol 1e-3。求積は glmer と係数 rtol 1e-4 |
| 負の二項 GLMM | `glmmTMB(..., family=nbinom2)` | `fit_mixed(..., family="negative_binomial")` | 同じ 3 標本で係数、SE、対数尤度、分散、`theta`（rtol 1e-4） |
| 二項 GLMM | `glmmTMB(..., family=binomial)`、`glmer(..., family=binomial, nAGQ=9)`。`(1 \| g)` と `(1 + x \| g)` | `fit_mixed(..., family="binomial")`（`engine="laplace"`）、`n_agq=` | fixture `glmm_binary_zero.json` 3 標本（変量切片、変量傾き、2〜3 行の小さいクラスタ多数）。許容差はポアソン GLMM と同じ。求積は glmer と係数 rtol 1e-4、分散 rtol 1e-3。fixture `glmm_binary_glmer.json`: glmer の Laplace と `nAGQ=25` で係数、SE、対数尤度、分散、MOR、`predict(re.form=NA)` の参照行。glmer は既定の `tolPwrss=1e-7` だと Laplace の対数尤度が約 1e-4 ずれるので、`tolPwrss=1e-13` の glmer とは係数 rtol 2e-6、SE rtol 2e-5、既定の glmer とは係数 rtol 1e-3、SE rtol 1e-2 |
| 交差変量切片と ar1 | `glmmTMB(y ~ x + (1 \| id_patient) + (1 \| e_examiner))`、`ar1(year + 0 \| e_examiner)`。nbinom2、poisson、binomial | `fit_mixed(..., "... + (1 \| id_patient) + ar1(year + 0 \| e_examiner)")` | fixture `glmm_crossed.json` 5 例（990 行、患者 300 × 検査医 30 × 6 年）。係数 rtol 1e-4、SE rtol 1e-4、対数尤度 atol 1e-6、分散 rtol 1e-3、`rho` atol 1e-4、BLUP atol 1e-4。実測の差はどれも約 1e-6。二項の ar1 は rho = 1 の境界で、SE rtol 1e-3 |
| gamm の尤度比較と分散の区間 | `gamm4` の変量効果なしとありで `logLik(g$mer)`、`AIC`、`BIC`、`anova(g0$mer, g1$mer)`。`confint(profile(g$mer))` | `gamm(..., random=None)`、`glance()`、`compare_gamms`、`variance_table()` | fixture `gamm4_compare.json`。対数尤度、AIC、BIC（atol 1e-4）、パラメータ数、χ² 統計量（atol 2e-4）。`method="profile"` の区間は `confint(profile(g$mer))` と SD で rtol 1e-4。分散がほぼ 0 の例（下限 0）を含む。既定の Wald 区間は profile 区間と SD で rtol 2% の範囲 |
| 平滑 + 変量切片 | `gamm4(y ~ s(x), random = ~(1 \| examiner))`、binomial / poisson、`tp` と `cr` | `gamm(frame, "y", [smooth("x", basis="tp")], random="(1 \| examiner)")` | fixture `gamm4.json` 6 例。係数 atol 3e-5、検査医の分散と MOR rtol 1e-4、平滑の値と線形予測 atol 3e-5、BLUP atol 1e-4。SE と edf は fixture スクリプトが R で計算し直した値と rtol 1e-4。gamm4 0.2-6 を Matrix 1.6 以降で使うと、報告される edf と SE が約 1% ずれるため |
| ゼロ過剰 GLMM | `glmmTMB(..., ziformula = ~1 / ~z)`。poisson、nbinom2 | `fit_mixed(..., zero_inflation=True / ["z"])` | 同じ fixture の 3 例。係数、SE、対数尤度、分散、`theta`、ゼロ部分の係数と SE（rtol 1e-4） |
| ハードル GLMM | `glmmTMB(..., family=truncated_poisson / truncated_nbinom2, ziformula = ~z)` | `fit_mixed(..., hurdle=True)` | 同じ fixture の 2 例。量は上と同じ |
| 多重代入の統合 | `mice::pool`、`summary(pool(...), conf.int = TRUE)` | `pool`、`MultipleImputation.pool` | fixture `mice_pool.json`。mice（m=5、pmm / logreg / polyreg）で埋めた同じ 5 データに lm、ロジスティック glm、coxph を当て、推定値、ubar、b、t、Barnard–Rubin 自由度、riv、λ、fmi、SE、p 値、区間（rtol 1e-6）と `dfcom` |
| Wilkinson 式 | `model.matrix`、`lm` | `model_matrix`、`fit_ols` | 設計行列、応答、`offset()`、`y ~ x * stage` と `log(y) ~ x` と `y ~ x + offset(z)` の係数と SE。fixture は `wilkinson.json` |
| ロジスティックの内的妥当性 | `rms::validate.lrm`（`method="boot"`） | `validate_logistic` | Dxy、R2、Intercept、Slope、Emax、D、U、Q、B、g、gp の index.orig、training、test、optimism、index.corrected、n（rtol 1e-10）。同じ再標本を `indices=` で渡す |
| ロジスティックの較正曲線 | `rms::calibrate`（`lrm`、lowess、既定の `predy`） | `calibrate_logistic`、図は `plot_calibration_curve` | apparent と bias-corrected の曲線（atol 1e-9）、平均絶対誤差、0.9 分位 |
| Cox の内的妥当性 | `rms::validate.cph` | `validate_cox` | Dxy、R2、Slope、D、U、Q、g の各列（rtol 1e-10） |
| Cox の較正 | `rms::calibrate.cph(cmethod="KM")`。`cph(..., surv=TRUE, time.inc=u)` | `calibrate_cox(..., u=, m=)` | 群ごとの予測生存、KM、KM.corrected、std.err、optimism（atol 1e-12） |
| 一致指数 | `Hmisc::somers2`、`rms` の `dxy.cens` | `somers_dxy` | C と Dxy |
| lowess | `stats::lowess` | `statract._core.lowess.lowess`（内部） | 既定（iter=3、atol 1e-10）、iter=0（atol 1e-13）、f=0.2 と delta=0（atol 2e-8）の当てはめ。statsmodels の lowess（BSD-3）にもとづく。頑健化の重みの細部が R と違うので、頑健化ありの狭い窓では 1e-8 程度ずれる |

各行は 3 標本です。許容差はサンドイッチ共分散は rtol 1e-8、Newton 法の係数は rtol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6、p 値は atol 1e-6 です。マッチの組は完全一致です。加法モデルの範囲、thin plate、テンソル、共線性、最適マッチ、full matching は `gam_scope.json`、`collinearity.json`、`match_opt_full.json` の 1 標本です。重みと offset、テンソル交互作用は `gam_weight_ti.json`、条件付き推論木は `ctree.json` の 1 標本です。ROC の 4 行は `proc.json` で、再生成は `Rscript tests/r_oracle/scripts/proc.R` です。

`htest.json` は R 4.3.3 の stats です。再生成は `Rscript tests/r_oracle/scripts/htest.R` です。R の `t.test` は等分散の方法名を `" Two Sample t-test"` と先頭に空白つきで返しますが、`t_test` は空白を付けません。

rms の行は `rms.json`（rms 6.7-1、R 4.3.3）です。再生成は `Rscript tests/r_oracle/scripts/rms.R` です。R の乱数は再現しないので、スクリプトは `set.seed` のあとの `sample(n, replace=TRUE)` を B 回くり返して再標本を作り、`predab.resample(debug=TRUE)` が表示する訓練標本の行と一致することを確かめてから書き出します。rms の `lrm.fit` は −2 log L の変化が 0.025 未満で、`cph` は相対変化 1e-4 で止まるので、`statract` も同じところで止めます。ロジスティックと Cox は、どちらも素の Newton–Raphson です。ロジスティックは、変化が 0.025 未満になったあと、もう 1 歩進めると rms の係数と一致します（rtol 1e-12）。Cox は、対数部分尤度の相対変化が 1e-4 未満になった点で止めると一致します（rtol 1e-12）。係数は `fit_glm` や `cox_ph` と小さな標本で 1e-4 程度ずれることがあります。`seed=` だけで呼ぶと numpy の乱数を使うので、値は R と一致しません。

ガンマ GLM の対数尤度は statsmodels の密度を返すため、fixture の比較には入れていません。係数、SE、HC0 は比べています。

## 呼び出せるが、fixture ではまだ固定していないもの

- ユークリッド、尺度つきユークリッド、ロバスト・マハラノビスの最近傍（実装は MatchIt の組と合わせてある。コミットした JSON はマハラノビス）
- 区間分割 `split_follow_up`
- ブートストラップ共分散（乱数生成器が R と違う）
- `impute_chained` の埋めた値（乱数生成器が R と違う。テストは型、観測値の保存、MAR での推定値の回復を見る）

## この版の対象外

- クラスタ頑健分散の HC2 / HC3 の R との照合（一元クラスタだけ実装。クラスタごとの (I - H_gg) で調整し、1 行 1 クラスタでは `hc_covariance` の HC2 / HC3 と一致。多元クラスタと Cox では受け付けない）
- `crr` の時間依存項（`cov2`、`tf`）。`crr` と同じサンドイッチ SE は `fine_gray_regression` です

## stdReg2 と違うところ

`standardize_cox` は stdReg2 1.0.8 に合わせていますが、次の点は R と違います。

- 生存関数の分散では、事象時刻と同じ時刻で打ち切られた行にも、その時刻の基底ハザードの増分が入ります。これは R の出力に合わせています。
- RMST で外す項: R の出力では、名前に曝露名を含む項も外れます（曝露が `ope` なら `operation` の項も）。statract は曝露の列を使う項だけを外します。
- RMST の各事象の群: 同じ時刻に打ち切りや別の群の行があると、R の値は statract と違います。statract は事象を起こした行の群を使います。同じ時刻の事象が無ければ一致します。
- RMST で曝露のほかに共変量が無い式は拒否します。R もエラーになります。
- RMST の `times` が 2 個以上なら `ValueError` です。R は警告を出して最大値を使います。
- RMST の `values` を `[1, 0]` の順に渡しても、各行の値と推定値は対応します。R は推定値を 0、1 の順に並べたまま、ラベルだけを `values` の順にします。
- 文字列のクラスタ列も使えます。R は文字列のクラスタ列でエラーになります。fixture のクラスタは整数です。
- `transform="logit"` と `"odds"` は生存関数で使えます。R の `summary_std_coxph` はこの指定でエラーになります。fixture はそのエラーを避ける手順で R に計算させています。
- `ci="log"` と `contrast="difference"` の組み合わせは拒否します。参照行の推定値が 0 になり、R では区間が NaN になります。
- ケースウェイトはありません。R も Cox にウェイトを渡しません。

## WeightIt と cobalt

`iptw.json` は `tests/r_oracle/scripts/iptw_weightit.R` が WeightIt 2.1.0 と cobalt 5.0.0 で書きます（`weightit()`、`trim()`、`bal.tab()` を直接呼ぶ）。`tests/r_oracle/scripts/iptw.R` は同じ計算を base R で書いたもので、パッケージのない環境で作り直すためのものです。15 例すべてでパッケージと一致します。

パッケージと照合して分かった点:

- `trim` の分位点は `quantile(type = 3)` です。
- ATO のバランス表の SMD の分母は、cobalt の `"weighted"`（全標本を重み付きで計算した標準偏差）です。
- statract の既定の SMD は cobalt と違います（2 値も標準化し、分母は推定対象によらず pooled SD）。照合では `binary=` と `sd_denominator=`（ATE `"pooled"`、ATT `"treated"`、ATC `"control"`、ATO `"weighted"`）を cobalt に合わせて渡します。[実装メモ](../api/models/weighting.md#smd) を見てください。

確かめていない点:

- 2 水準の文字列の列は 2 番目の水準だけ、論理値の列は `列名_TRUE` で出します。cobalt の名前の付け方とずれることがあります。
- 交互作用や `I(age^2)` の項は設計行列の列名のまま 1 行にします。cobalt は名前の付け方が違います。

WeightIt と違うところ:

- `stabilize=True` を ATE 以外で使うと `ValueError` です。
- `trim=(low, high)` の固定の上下限は WeightIt にありません。
- 欠測のある行は落とします。WeightIt の `missing="ind"` はありません。

アウトカムのモデル（`weighted_glm`、`PropensityWeights.effect`）:

- WeightIt 1.x の `glm_weightit()` はこの環境に入らないので、`iptw_outcome.json` は `tests/r_oracle/scripts/iptw_outcome.R` が M 推定を `glm()` と base R で書いて作ります。ヤコビアンは解析的な式で、スクリプトの中で中心差分と比べます。パッケージとは照合していません。
- 交差項（重みの傾向スコアの係数による微分）は、statract と `iptw_outcome.R` では解析的な式です。WeightIt の出力とは、数値微分の誤差の程度の差があります。
- `vcov="HC0"` は返した係数での解析的な値です。`sandwich::vcovHC` の値とは、`glm` の収束誤差の分（ここでは最大 1e-5 程度）ずれます。
- `stabilize=True` の処置群の割合も推定値として重ねます。WeightIt がこれを含めるかは確かめていません。安定化の係数は群ごとに一定なので、`y ~ treat` だけのモデルでは結果が変わりません。
- 切り詰めた重みで `vcov="asympt"` はエラーです。WeightIt の扱いは確かめていません。
- `vcov="bootstrap"` はパーセンタイル区間です。WeightIt の `vcov="BS"` は複製の共分散から Wald 型の区間を作ります。乱数も違うので値は一致しません。
- 周辺効果は飽和モデル（群ごとの重み付き平均）だけです。共変量を入れたアウトカムモデルからの g 計算（`avg_comparisons`）はありません。

## polr、multinom、survRM2 と違うところ

- `ordinal_regression` は正確な Hessian の Newton 法です。共分散は観測情報行列の逆行列で、`clm` と同じです。`polr` は `optim` の差分近似の Hessian なので、SE が約 1e-3 ずれます。
- `polr` は BFGS を `reltol = 1e-8` で止めます。fixture は `reltol = 1e-14` で当て直しています。
- cauchit リンクは R と比べていません。`polr` と `clm` は cauchit で本当の最適点を返さないことがあります。`ordinal_regression` は切らずに最適化します。
- `ordinal_regression` の `n_obs` は行数です。重みがあると `clm` の `nobs` は重みの和です。`polr` は 3 水準以上を求めますが、`ordinal_regression` は 2 水準でも当てます。
- 空の水準は警告を出して落とします（`multinom` と同じ）。`polr` は空の水準で閾値が発散します。
- Brant 検定の重みは度数の重みとして扱います。R の `brant` パッケージは重みを扱いません。
- `multinomial_regression` は Newton 法で、`multinom` の既定（BFGS、`reltol = 1e-8`）より最適点に近い値を返します。差は係数で 1e-4 程度です。`decay`（荷重減衰）はありません。
- `fit_risk_ratio(method="log-binomial")` の対数尤度は、R と同じく重みを試行数とみなし、`round()` した二項分布で計算します。整数の重みなら度数の重みと同じです。`fit_glm` の二項の対数尤度とは扱いが違います。
- `fit_firth` の fixture は、Firth（1993）と Heinze・Schemper（2002）の式から base R で書いた別の実装です（logistf は CRAN に届かず入らないため）。statract とはコードを共有しません。logistf のコードも使っていません。既定の値と 1e-5 の収束の基準は、`logistf.control()` のヘルプの説明に合わせています。`logistf` は区間の端を Venzon–Moolgavkar 法で 1e-5 まで求めるので、`logistf` 本体とは 1e-5 程度まで一致するはずです。`logistf` 1.24 以降の既定 `fit = "IRLS"` も同じ点に収束します。`flic` / `flac` と `firth = FALSE` はありません。
- `profile_interval` は区間の端を直接解きます。`confint.glm` はプロファイルの格子点にスプラインを引いて読むので、差は SE の 1e-4 倍程度です。ガウスではプロファイルが直線なので一致します。
- `restricted_mean_survival` の R スクリプトは survRM2 のコードを使いません。RMST と分散は Klein と Moeschberger の式、対比は Uno ら（2014）の式で、base R と survival だけで書いています。survRM2 1.0-4 は出力を見るだけです。入っている環境では、スクリプトが 6 例すべてで survRM2 と一致することを確かめます（rtol 1e-12）。その出力も fixture に入れています。`tau` の既定と上限は `rmst2` と同じです。短い群の最終時刻の行がすべて事象なら 2 群の最終観察時刻の大きいほう、そうでなければ小さいほうです。共変量で調整する `rmst2(covariates=)` はありません。

## epiR と違うところ

- `epi.tests(method="agresti")` の `est` は、Agresti–Coull の補正をした割合です。尤度比と Youden 指数の点推定と、尤度比の区間にもその値が入ります。statract はどの方法でも、推定値と尤度比には補正しない割合を使います。割合の区間と Youden 指数の区間は R と同じです。

## glmnet と違うところ

- `penalized_regression` は、論文（Friedman ら 2010、Simon ら 2011、Tibshirani ら 2012）から書いた座標降下法です。glmnet のコードは使っていません。glmnet は結果の確認にだけ使います。
- 各問題は、glmnet の既定の `thresh` より 1e-4 倍きつく解きます。そのため R との差は、glmnet 側の収束誤差です。係数で 1e-4 から 1e-3 程度です。横長の標本ではもっと大きくなります。テストでは、どの lambda でも statract の目的関数の値が glmnet の係数での値以下であることも確かめます。
- 自動の lambda 列は、説明できた逸脱度の割合の伸びが、その割合の 1e-5 倍を下回ると止まります。gaussian では R と同じ長さです（差は 1 個まで）。二項、ポアソン、Cox では R より長くなることがあります。R の止め方は family ごとに違い、ヘルプに詳しい説明がないためです。
- リッジ（`alpha=0`）の自動の経路の最初の点は、係数がすべて 0 の null モデルです。R は 1e-37 ほどの係数を返し、df に数えます。
- ペナルティ係数は、除いた列（`inf`）を除いて、列数の和にそろえます。
- 二項は 0/1 の結果だけです。2 列の度数行列と多項（`multinomial`）、`type.logistic="modified.Newton"` はありません。
- `lower.limits`、`upper.limits`、`dfmax`、`pmax`、`relax`、疎行列、`mgaussian` はありません。`exclude` の代わりに `penalty_factor` に `inf` を渡します。
- Cox は `Surv(time, status)` だけです。`strata()` と `Surv(start, stop, status)`（glmnet の `cox.path`）はありません。`cv_penalized` の Cox は deviance だけで、`type.measure="C"` はありません。
- 切片のない式（`y ~ x - 1`）では、`coef()` に切片の行がありません。R は 0 の `(Intercept)` 行を残します。
- `predict` と `coef` は `exact=FALSE` と同じ線形補間です。`exact=TRUE`（当て直し）はありません。
- `foldid` を渡さないと、分割は `seed` から作る並べ替えです。R の `sample` とは一致しません。
- gaussian 以外と、罰則のない列があるときは、`lambda_max` が null モデルの当てはめから決まります。R はこれを反復で解くので、lambda 列が rtol 1e-5 程度ずれることがあります。
- 交差検証の `class` と `auc` は、予測値の階段関数です。係数の小さな差で 1 行が入れ替わり、いくつかの lambda で値が変わることがあります。
- `lambda_max` そのもので当てるとき（R の自動の lambda 列を `lambdas=` に渡し直すとき）は、先頭の係数が丸め誤差の 0 です。R でも Python でも 1e-16 の値が出ることがあり、df が 1 違うことがあります。

## 既存の名前

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival` はそのままです。係数表は `Fit.tidy` です（`sm_summary2df` は削除しました）。マッチングは `match_sample`、回帰の係数表は `fit_glm` と `Fit.tidy` です。
