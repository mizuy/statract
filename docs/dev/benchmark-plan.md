# 公開データでの数値と速度

新たに入れた推定器について、主要な関数の主要なオプションごとに、R と同じ数値になるかと、どちらがどれだけかかるかを測る。行数は 1,000 と 10,000 の桁、説明変数は元の列で 10 前後。データは公開されているものだけを使い、合成標本は使わない。

`tests/r_oracle` の小さな fixture はそのまま残す。あちらは pytest が R なしで読む数値の固定で、こちらは手元で R と Python を同じ機械で走らせるベンチマークである。速度に合格ラインは置かない。数値の許容差は [R パッケージとの対応](../models/vs-r.md) と同じにする。

計測スクリプトは `bench/stat/` にある。`uv run python bench/stat/run_all.py` が取得、計測、比較まで行う。比較表は `bench/stat/results/comparison.csv`（サイト掲載は [benchmarks](../models/benchmarks.md)）。データ本体は `/tmp/statract-bench` に置き、リポジトリには入れない。R 側は `Rscript bench/stat/run_r.R`（jsonlite, sandwich, lmtest, survival, MatchIt, mgcv, lme4。後から足したタスクには pROC, rms, stdReg2, cmprsk, gamm4, glmmTMB, partykit, mice も要る。これらは `::` で呼ぶので、入っていないパッケージはそのタスクだけが失敗する）。

## 測り方

両方に同じ表を渡す。Python は Polars、R は `data.frame`。ダウンロードと型変換は計測に含めない。計測対象は、表がメモリに載ったあと、その関数を呼んで結果を取り出すまで。

1. ウォームアップを 1 回走らせる。
2. 続けて 5 回測り、壁時計の中央値を秒で残す。
3. `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` は 1。スレッド数で比が揺れないようにする。
4. 結果の表には、タスク、行数、比べた量、最大相対誤差、許容差を満たしたか、Python の秒、R の秒、Python / R を書く。
5. 因子の水準は両方でソート順に固定する。参照水準は先頭、ダミー名は `列名 + 水準`、切片は `(Intercept)`。R 側も `factor(..., levels=)` でこの順にする。
6. 欠損は、そのタスクが使う列について完全ケースにしてから行数を切る。切ったあとの実際の行数を結果に残す。
7. 行の順はデータの契約に含める。`order="data"` のマッチングと Newey–West は順に依存する。
8. データ本体はリポジトリに入れない。取得元と引用をこの文書に書き、キャッシュは作業ディレクトリの外に置く。

断面データ（Adult、SUPPORT2、STAR の生徒抽出）の 1,000 行と 10,000 行は、シード `20260927` の乱数で引く。二値の結果を持つタスクは、その割合が元データから大きくずれないよう層別する。時系列（Bike Sharing）は時刻順の連続区間を使い、行をばらばらに引かない。

10,000 行に届かない完全ケースは、大きい側を全件とする。行を水増ししない。

## データ

4 つとも公開データで、役割が違う。1 つの表で回帰、生存、クラスター、時系列を兼ねると、オプションの意味が崩れる。

| 名前 | 行の桁 | 用途 | 取得 |
|------|--------|------|------|
| Adult（Census Income） | 48,842 から 1,000 と 10,000 | 線形、GLM、頑健分散、線形検定、マッチング | UCI Machine Learning Repository, CC BY 4.0。Kohavi (1996) |
| Bike Sharing（hour） | 17,379 時間から、先頭 1,000 時間と先頭 10,000 時間 | Newey–West、Durbin–Watson、Breusch–Godfrey、ポアソン、GAM | UCI, CC BY 4.0。Fanaee-T and Gama (2013) |
| SUPPORT2 | 9,105 人。完全ケースが 10,000 未満なら大きい側は全件、小さい側は 1,000 | 生存曲線、log-rank、Cox、加速故障時間 | Vanderbilt Department of Biostatistics。利用は原論文の引用と、取得元が `https://hbiostat.org/data` である旨の記載が条件。UCI にも同一データがある（id 880） |
| Project STAR | 公開生徒ファイル 11,601 人。幼稚園から 3 年の縦持ち | 線形混合、クラスター頑健分散 | Achilles et al., Harvard Dataverse, doi:10.7910/DVN/SIWH9F |

STAR の行を単独で引くとクラスが壊れる。生徒を単位に引き、その生徒の学年行をすべて残す。行数が目標以上になったところで止め、実際の行数を記録する。群は「学校 × 学年 × クラス」とし、学校番号だけが同じ別クラスを混ぜない。

### Adult の列

OLS の結果は `hours_per_week`。説明変数は次の 10 列。

`age`、`education_num`、`capital_gain`、`capital_loss`、`sex`、`race`、`marital3`、`workclass4`、`relationship3`、`us_native`

`marital3` は married / never_married / other、`workclass4` は private / self_employed / government / other、`relationship3` は husband / wife / other、`us_native` は出生国が United-States かどうか。`occupation` と `native-country` の細かい水準は展開すると係数が膨らむので使わない。

二項 GLM の結果は年収が 5 万ドルを超えるか。説明変数は上の 10 列のうち `relationship3` を `hours_per_week` に替えた 10 列。重み付き OLS の重みは `fnlwgt` で、説明変数には入れない。

マッチングの処置は `sex`。共変量は処置と結果を除いた 9 列に `hours_per_week` を足して 10 列。

### Bike Sharing の列

結果は `cnt`。説明変数は `season`、`mnth`、`hr`、`weekday`、`workingday`、`weathersit`、`temp`、`hum`、`windspeed`、`yr`。時刻順を保つ。先頭 1,000 時間は `season` と `yr` が一定なので、その切片の回帰からは外す。月と時刻はゼロ埋めの文字列（`01`）にする。R の `read.csv` がこれを整数にすると、ダミー名が `mnth02` と `mnth2` に分かれる。GAM はこの版が平滑 1 本だけなので、`cnt ~ s(temp)` だけで他の列は入れない。edf は hat 行列のトレースで、切片を含む。R は `sum(fit$edf)` を使う。`summary` の平滑項 edf は切片の 1 を含まない。mgcv の外側の Newton 法は既定の収束判定だと log sp の勾配が 1e-3 ほど残ったまま止まり、10,000 行では係数が最大 1e-3 動く。R の秒は既定の呼び出しで測り、比べる数値は `gam.control(newton=list(conv.tol=1e-12), epsilon=1e-12)` で収束させた当てはめ直しから取る。

### SUPPORT2 の列

時間は `d.time`、イベントは `death`。説明変数は `age`、`sex`、`num.co`、`scoma`、`meanbp`、`hrt`、`temp`、`resp`、`diabetes`、`ca`。`dzgroup` は層の指定にだけ使い、説明変数には数えない。時間 0 以下は加速故障時間の前に落とす。

### STAR の列

結果は読解の得点。説明変数は性別、人種、給食補助、生まれ年、学年（0 から 3 の数値）、クラス種別、都市度、教員経験年数、教員の性別、クラスサイズの 10 列。特別支援は幼稚園と 1 年にしか無いので入れてない。群は「学校 × 学年 × 教員」、クラスターの第 1 キーは学校、第 2 キーは学年。

## タスク

各タスクを 1,000 行と 10,000 行（または上の規則で全件になった大きい側）の両方で走らせる。許容差は vs-r.md のまま。係数の Newton 法は rtol 1e-6、サンドイッチ共分散は rtol 1e-8（各要素の差を √(V_ii V_jj) で割る。対角では相対誤差と同じで、相関がほぼ 0 の非対角で比が膨らまない）、p 値は atol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6。混合モデルは `fit_mixed`（既定エンジン lme-python、収束の許容差 1e-8）で、固定効果 rtol 3e-3、標準誤差 rtol 1e-4、変量共分散 rtol 3e-3、残差分散 rtol 1e-4、対数尤度 atol 1e-4。もとの目標（固定効果 1e-6、変量共分散 1e-4、残差分散 1e-5、対数尤度 1e-8）から緩めたのは、STAR の生まれ年を中心化しないと計画行列の条件数が 7e6 ほどになり、lme-python がそこで精度を落とすため。lme4 の deviance 関数で評価すると `lmer` の θ の方が低く、生まれ年を中心化すると lme-python の θ と対数尤度は `lmer` に一致する。中心化しないままの差は、固定効果で最大 1.2e-3（変量傾きの 10,000 行）、対数尤度で 4e-5。lme-python を上げたときは、この差が縮むか、広がっていないかを測り直す。

| ID | データ | 関数とオプション | R | 比べる量 |
|----|--------|------------------|---|---------|
| ols | Adult | `fit_ols` | `lm` | 係数、SE、対数尤度、t、p |
| ols-w | Adult | `fit_ols(weights=fnlwgt)` | `lm(..., weights=fnlwgt)` | 同じ |
| glm-bin | Adult | `fit_glm(family="binomial")` | `glm(..., family=binomial)` | 係数、SE、対数尤度 |
| glm-gamma | Adult | `fit_glm(family="gamma")`、結果は `hours_per_week` | `glm(..., family=Gamma(link="inverse"))` | 係数、SE。対数尤度は比べない |
| glm-pois | Bike | `fit_glm(family="poisson")`、結果は `cnt` | `glm(..., family=poisson)` | 係数、SE、対数尤度 |
| hc | Adult | OLS のあと `hc_covariance` を HC0, HC1, HC2, HC3, HC4, HC5 | `vcovHC` の同じ type | 共分散 |
| cl1 | STAR | OLS の 1-way `cluster_covariance`、学校、既定の HC1 | `vcovCL` | 共分散 |
| cl2 | STAR | 学校と学年の 2-way | `vcovCL` の 2 列 | 共分散 |
| nw | Bike | `newey_west_covariance`。ラグは `floor(4*(n/100)^(2/9))` | `NeweyWest(..., lag=その値, prewhite=FALSE, adjust=FALSE)` | 共分散 |
| wald | Adult | 入れ子の `wald_test`（`us_native` を落とす） | `lmtest::waldtest` | 統計量、p |
| lr | Adult | `likelihood_ratio_test`、同じ入れ子 | `lmtest::lrtest` | 統計量、p |
| bp | Adult | `breusch_pagan_test` の `studentize=True` と `False` | `bptest(..., studentize=)` | 統計量、p |
| reset | Adult | `ramsey_reset_test`、power `(2, 3)` | `resettest` | 統計量、p |
| dw | Bike | `durbin_watson_test`、片側 greater | `dwtest` | 統計量。p は n ≥ 100 の正規近似 |
| bg | Bike | `breusch_godfrey_test`、`order=1` と `order=4`、`distribution="chi2"` | `bgtest(..., order=, type="Chisq")` | 統計量、p |
| km | SUPPORT2 | `survival_curve`、`confidence="log"`。時点は大きい側の時間の 25、50、75 パーセンタイル | `survfit` の `summary` | その時点の生存率と区間。全体と `by=sex` |
| na | SUPPORT2 | `kind="nelson_aalen"` | `survfit(..., stype=2)` | 同じ時点の生存率 |
| lrk | SUPPORT2 | `log_rank`、`by=sex`、`rho=0` と `rho=1`。層ありは `strata=ca` で `rho=0` | `survdiff` | カイ二乗、p |
| cox-e | SUPPORT2 | `cox_ph(ties="efron")` | `coxph` | 係数、モデルベース SE、部分尤度 |
| cox-b | SUPPORT2 | `ties="breslow"` | `coxph(..., ties="breslow")` | 同じ |
| cox-s | SUPPORT2 | `strata=dzgroup`、Efron | `coxph` の `strata` | 同じ |
| aft-w | SUPPORT2 | `accelerated_failure(distribution="weibull")` | `survreg(..., dist="weibull")` | 係数、SE、対数尤度。尺度は `Log(scale)`。`coef()` には無く、`log(fit$scale)` と `vcov()` の最後の対角を足す |
| aft-ln | SUPPORT2 | `distribution="lognormal"` | `dist="lognormal"` | 同じ |
| aft-ex | SUPPORT2 | `distribution="exponential"` | `dist="exponential"` | 同じ |
| m-logit | Adult | `match_sample(distance="logit", order="data", ratio=1)` | `matchit(..., distance="glm", link="logit", m.order="data")` | 組の集合、`age` のマッチ後標準化差 |
| m-mah | Adult | `distance="mahalanobis"`、`order="data"` | `distance="mahalanobis"` | 組の集合 |
| m-exact | Adult | `method="exact"`、`exact` は `race` | `method="exact"` | 重み |
| m-cem | Adult | `method="cem"`、`cutpoints="sturges"` | `method="cem"` | 重み |
| gam-8 | Bike | `gam`、`smooth("temp", k=8)`、ガウス、REML | `gam(cnt ~ s(temp, bs="cr", k=8), method="REML")` | 平滑化パラメータ、edf、REML、係数 |
| gam-10 | Bike | `k=10` | `k=10` | 同じ |
| lmm-ri | STAR | `fit_mixed(..., groups=クラス, method="reml")` | `lmer(... + (1 \| クラス), REML=TRUE)` | 固定効果、SE、変量分散、残差分散、対数尤度 |
| lmm-ml | STAR | `method="ml"` | `REML=FALSE` | 同じ |
| lmm-rs | STAR | `slopes` に学年、`method="reml"`。群は生徒 | `lmer(... + (1 + 学年 \| 生徒))` | 同じ。学年が生徒内で 2 点以上ある行だけを残す |

ガンマの対数尤度は statsmodels の密度が R と違うので、fixture と同様に外す。完全一致と CEM の計測は `match_sample` が重みを返すまでで、サブクラス内の直積は `pairs()` を呼んだときに作る。

クラスター頑健分散（cl1、cl2）だけは許容差を 1e-6 にする。STAR は生まれ年を中心化しないので計画行列の条件数が 7e6 ほどあり、高精度で組んだ基準値から `vcovCL` も statract も 1e-8 から 3e-7 離れる。1e-8 はどちらの実装も満たせない。

最近傍の組は、同じ距離の対照のどれを先に取るかで変わる。食い違う組の対照どうしが共変量まで同じなら、同じ組とみなす。MatchIt 4.6 以降の書き直しで、ロジットの 10,000 行ではこの入れ替わりが 16 組ある。マハラノビスは全水準の指標を使うので、プールした共分散が特異になる。MatchIt はその一般化逆行列をピボット付き Cholesky で分解し、末尾の塊は LAPACK のビルドで変わる。x86-64 Linux の参照 LAPACK では組が一致する。macOS（arm64）では一致せず、失敗のまま注記する。

## 追加したタスク（SUPPORT2）

最初の版のあとに入れた関数を、SUPPORT2 の同じ 2 つの切片（1,000 行と全件）で測る。多重代入だけは欠測を残した別の切片 `support-mi-1000` / `support-mi-large` を使う。どれも、R の呼び出しは `tests/r_oracle/scripts/` の fixture スクリプトと同じにしてある。許容差は vs-r.md のまま。

SUPPORT2 の切片には次の列を足した。行は変わらない。

- `hospdead`（入院中の死亡）。二値の結果に使う。
- `slos`（入院日数）。負の二項 GLMM の結果。
- `cause`（競合リスクの原因コード）。SUPPORT2 には原因コードが無いので、既存の列から作った。`death = 0` なら 0（打ち切り）、`death = 1` で `hospdead = 1` なら 1（入院中の死亡）、`death = 1` で `hospdead = 0` なら 2（退院後の死亡）。時間は `d.time`。臨床的な意味よりも、互いに排他な 2 つの事象を公開データで作ることを優先した。
- `support-mi` は `hospdead`、`age`、`sex`、`num.co`、`meanbp`、`hrt` が揃った行で、`alb`、`bili`、`pafi`、`wblc`、`income` の欠測を残す。`income` は `inc1`〜`inc4`（under $11k、$11–25k、$25–50k、>$50k）に置き換え、ロケールで並びが変わらないようにした。

| ID | 関数とオプション | R | 比べる量 |
|----|------------------|---|---------|
| ttest | `t_test`。`meanbp` を `hospdead` で 2 群、Welch と `var_equal=True` | `t.test` | t、自由度、p、区間、平均（rtol 1e-10、p は atol 1e-10） |
| wilcox | `wilcox_test(conf_int=True)`。同じ 2 群。同順位ありの正規近似 | `wilcox.test(conf.int=TRUE)` | W、p、Hodges–Lehmann 推定値と区間（両方 `uniroot` の tol 1e-4 なので rtol 1e-6） |
| prop | `prop_test`。性別ごとの入院中死亡の割合 | `prop.test` | χ²、p、差の区間、割合 |
| padj | `p_adjust`。7 列の Welch t 検定の p を holm、hochberg、hommel、BH、BY | `p.adjust` | 補正後の p（rtol 1e-8） |
| roc | 2 つの二項 GLM（10 列と 4 列）の線形予測子で `roc_curve`、`ci_auc`、対応ありの `roc_test` | `pROC::roc`、`ci.auc(method="delong")`、`var`、`roc.test(method="delong")` | AUC、DeLong の分散と区間、Z と p。秒は GLM の当てはめを含まない |
| val-lrm | `validate_logistic`、B = 20。説明変数は Cox の 10 列から `temp` を外した 9 列 | `rms::validate(lrm(...), B=20)` | `index.orig` だけ（atol 1e-6） |
| cal-lrm | `calibrate_logistic`、B = 20。同じ 9 列 | `rms::calibrate(lrm(...), B=20)` | `predy` と見かけの曲線 `calibrated.orig` |
| val-cph | `validate_cox`、B = 20 | `rms::validate(cph(...), B=20)` | `index.orig` だけ |
| cal-cph | `calibrate_cox(u=180, m=150)`、B = 20 | `rms::calibrate(cph(..., surv=TRUE, time.inc=180), cmethod="KM", u=180, m=150, B=20)` | 群ごとの平均予測、KM、std.err、`index.orig` |
| std-cox | `standardize_cox`。曝露は `diabetes`（0 と 1）、時点は km と同じ 3 点 | `stdReg2::standardize_coxph(measure="survival")` | 標準化生存率と水準間の共分散（rtol 1e-6） |
| cuminc | `cumulative_incidence(by=sex)`。層なしと `strata=ca` | `cmprsk::cuminc`、`timepoints` | km と同じ 3 時点の推定値と分散、原因ごとの Gray 検定（rtol 1e-8、p は atol 1e-8）。層ありは検定だけ |
| crr | `fine_gray_regression(cause=1)`。説明変数は Cox と同じ 10 列 | `cmprsk::crr(..., failcode=1)` と `model.matrix` | 係数、SE、擬似対数尤度（rtol 1e-8） |
| gamm | `gamm(hospdead ~ age + s(meanbp, cr, k=8), random="(1 \| dzgroup)", binomial)` | `gamm4::gamm4` | 係数（atol 3e-5）、`dzgroup` の分散（rtol 1e-4）、対数尤度（atol 1e-5）、平滑の値（atol 3e-5） |
| glmm-pois | `fit_mixed("num_co ~ age + sex + meanbp + ca + (1 \| dzgroup)", family="poisson", engine="laplace")` | `glmmTMB(..., family=poisson)` | 係数と SE（rtol 1e-4）、対数尤度（atol 1e-6）、分散（rtol 1e-3） |
| glmm-nb | `slos ~ age + sex + num_co + meanbp + ca + (1 \| dzgroup)`、`family="negative_binomial"` | `glmmTMB(..., family=nbinom2)` | 同じに `theta`（rtol 1e-4） |
| ctree | `conditional_tree(hospdead, Cox と同じ 10 列)` | `partykit::ctree`（quadratic、Bonferroni、minsplit 20、minbucket 7） | 分割の変数と分割点（深さ優先、完全一致）、終端ノード数、根の統計量と調整済み p（`sctest.constparty`）、各行の予測 |
| mice | `impute_chained(m=5, n_iter=5)` のあと二項 GLM を `pool` | `mice(m=5, maxit=5)`、`with` 相当の `glm`、`pool` | 代入する列とその方法、欠測数だけ。秒は代入から統合まで |

乱数を使う 3 つは比べ方を変えた。

- validate と calibrate。ブートストラップの再標本は R の `sample` と numpy で違う。比べるのは再標本に依らない見かけの値（`index.orig`、見かけの較正曲線、群ごとの KM）だけで、秒は同じ B で測る。statract は `indices=` を受け取るので、R の `set.seed` と `sample(n, replace=TRUE)` で作った行を渡せば全列を比べられる（fixture `rms.json` はそうしている）。ベンチマークではそこまでしない。
- mice。代入値は乱数生成器が違うので一致しない。決定的な部分（どの列をどの方法で埋めるか、欠測の数）だけを比べる。統合した推定値は JSON に残すが判定には使わない。`impute_chained` は 1 回に 1,000 行で約 40 秒、全件で約 150 秒かかる（`income4` の polyreg が L-BFGS で数千回まわる）。このタスクだけはウォームアップのあと 1 回を測る（manifest の `repeats=1`）。
- gamm。R の秒は既定の `gamm4` 呼び出しで測り、比べる数値は `tolPwrss=1e-13` まで詰めた glmer の当てはめ直しから取る（GAM と同じ扱い）。edf と SE は比べない。gamm4 0.2-6 は Matrix 1.6 以降で `chol(V, pivot=TRUE)` の置換を読めず、報告する edf と SE が約 1% ずれる。statract はピボットを正しく扱った式に合わせている（fixture `gamm4.json` のスクリプトがその補正をしている）。Python の edf は JSON に残すだけ。

ロジスティックの 2 つで `temp` を外すのは、statract の `lrm.fit` の移植が、ほぼ一定の `temp`（約 37）と切片が並ぶと情報行列の最小特異値が「1e-7 × 最大要素」を下回るとして特異と判定し、「did not converge」で止まるため（2026-10 時点）。rms 6.7 の `lrm.fit` は同じデータで当てはまるはずで、R で確かめたら直すべき差である。

ctree の調整済み p は `partykit:::sctest.constparty` から取る。rms の `lrm` は版によって収束判定が違う（fixture は rms 6.7-1）。新しい rms で見かけの値が 1e-6 を超えてずれたら、まずそこを疑う。

手元で SUPPORT2 だけを用意するときは `uv run python bench/stat/prepare.py --only support` とする。取得できない環境では `--support-parquet <support2.csv と同じ表の Parquet>` で読み込める。`run_python.py --task ID ...` は指定したタスクだけを走らせる（そのときの `python.json` はその分だけになる）。

## 入れないオプション

実装が数値を R に合わせていないもの、またはこの 4 つのデータでそのオプションが空になるものは、速度だけを並べても数値の判定ができない。

- ブートストラップ共分散。乱数生成器が R と違う。
- クラスター頑健分散の HC2 と HC3。
- Cox のケース重み。SUPPORT2 に調査重みがない。
- `survival::finegray` 経由の Fine–Gray と Aalen–Johansen。cmprsk 版（`cuminc`、`crr`）は下の追加タスクで測る。
- ユークリッド距離の最近傍。マハラノビスで距離のオプションは代表する。
- ベンチマークの GAM は cubic regression のガウス 1 本。他の基底と分布は fixture で比べる。
- 二項 GLMM。カウントの GLMM（ポアソン、負の二項）は下の追加タスクで測る。
- 図。`plot_survival` など既存の描画関数。

## スクリプト

pytest の既定には入れない。ネットワークが要るのは取得の 1 回だけで、計測のたびに R を起動する。

- `bench/stat/prepare.py` が取得と整形をする。正本は Parquet で、R 用に同じ内容の CSV も書く。
- タスク表は `manifest.json` の 1 か所。Python と R が同じ ID を読む。
- `run_python.py` と `run_r.R` を分け、`compare.py` が同じ列の表に結合する。
- 結合した表は `bench/stat/results/comparison.csv` に、走らせた機械のログとして残す。一覧用の読みは [Benchmarks（速度と R との差）](../models/benchmarks.md) が CSV を集約する。

最近傍は n = 10,000 で距離の全組を持つとメモリを食いやすい。計測がメモリで止まったら、そのタスクは失敗として秒を空欄にし、アルゴリズム側は変えない。

残っている差をどの順で潰すかは [次の速度改善](speed-plan.md) に書いた。
