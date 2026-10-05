# 公開データでの数値と速度

新たに入れた推定器について、主要な関数の主要なオプションごとに、R と同じ数値になるかと、どちらがどれだけかかるかを測る。行数は 1,000 と 10,000 の桁、説明変数は元の列で 10 前後。データは公開されているものだけを使い、合成標本は使わない。

`tests/r_oracle` の小さな fixture はそのまま残す。あちらは pytest が R なしで読む数値の固定で、こちらは手元で R と Python を同じ機械で走らせるベンチマークである。速度に合格ラインは置かない。数値の許容差は [R パッケージとの対応](vs-r.md) と同じにする。

計測スクリプトは `bench/stat/` にある。`uv run python bench/stat/run_all.py` が取得、計測、比較まで行う。結果の CSV は `bench/stat/results/comparison.csv`。データ本体は `/tmp/endolab-bench` に置き、リポジトリには入れない。

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

結果は `cnt`。説明変数は `season`、`mnth`、`hr`、`weekday`、`workingday`、`weathersit`、`temp`、`hum`、`windspeed`、`yr`。時刻順を保つ。先頭 1,000 時間は `season` と `yr` が一定なので、その切片の回帰からは外す。月と時刻はゼロ埋めの文字列（`01`）にする。R の `read.csv` がこれを整数にすると、ダミー名が `mnth02` と `mnth2` に分かれる。GAM はこの版が平滑 1 本だけなので、`cnt ~ s(temp)` だけで他の列は入れない。edf は hat 行列のトレースで、切片を含む。R は `sum(fit$edf)` を使う。`summary` の平滑項 edf は切片の 1 を含まない。

### SUPPORT2 の列

時間は `d.time`、イベントは `death`。説明変数は `age`、`sex`、`num.co`、`scoma`、`meanbp`、`hrt`、`temp`、`resp`、`diabetes`、`ca`。`dzgroup` は層の指定にだけ使い、説明変数には数えない。時間 0 以下は加速故障時間の前に落とす。

### STAR の列

結果は読解の得点。説明変数は性別、人種、給食補助、生まれ年、学年（0 から 3 の数値）、クラス種別、都市度、教員経験年数、教員の性別、クラスサイズの 10 列。特別支援は幼稚園と 1 年にしか無いので入れてない。群は「学校 × 学年 × 教員」、クラスターの第 1 キーは学校、第 2 キーは学年。

## タスク

各タスクを 1,000 行と 10,000 行（または上の規則で全件になった大きい側）の両方で走らせる。許容差は vs-r.md のまま。係数の Newton 法は rtol 1e-6、サンドイッチ共分散は rtol 1e-8、p 値は atol 1e-6、平滑化パラメータは rtol 1e-3、edf は rtol 1e-4、REML は atol 1e-6。混合モデルは `fit_mixed`（既定エンジン lme-python）の許容差で、固定効果 rtol 1e-6、標準誤差 rtol 1e-4、変量共分散 rtol 1e-4、残差分散 rtol 1e-5、対数尤度 atol 1e-8（mixedlm-rs 時代の目標。lme-python が外す場合は vs-r に実測を書く）。

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

## 入れないオプション

実装が数値を R に合わせていないもの、またはこの 4 つのデータでそのオプションが空になるものは、速度だけを並べても数値の判定ができない。

- ブートストラップ共分散。乱数生成器が R と違う。
- クラスター頑健分散の HC2 と HC3。
- Cox のケース重み。SUPPORT2 に調査重みがない。
- Fine–Gray と Aalen–Johansen。この 4 つに競合リスクの原因コードがない。
- ユークリッド距離の最近傍。マハラノビスで距離のオプションは代表する。
- ベンチマークの GAM は cubic regression のガウス 1 本。他の基底と分布は fixture で比べる。
- 二項・ポアソンの GLMM と glmmTMB。
- 図。`plot_survival` など既存の描画関数。

## スクリプト

pytest の既定には入れない。ネットワークが要るのは取得の 1 回だけで、計測のたびに R を起動する。

- `bench/stat/prepare.py` が取得と整形をする。正本は Parquet で、R 用に同じ内容の CSV も書く。
- タスク表は `manifest.json` の 1 か所。Python と R が同じ ID を読む。
- `run_python.py` と `run_r.R` を分け、`compare.py` が同じ列の表に結合する。
- 結合した表は `bench/stat/results/comparison.csv` に、走らせた機械のログとして残す。一覧用の読みは [Benchmarks（速度と R との差）](benchmarks.md) が CSV を集約する。

最近傍は n = 10,000 で距離の全組を持つとメモリを食いやすい。計測がメモリで止まったら、そのタスクは失敗として秒を空欄にし、アルゴリズム側は変えない。

残っている差をどの順で潰すかは [次の速度改善](speed-plan.md) に書いた。
