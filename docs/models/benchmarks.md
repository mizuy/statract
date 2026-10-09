# 速度と R との差

公開データで **同じ表** を Python（このライブラリ）と R に渡し、壁時計と数値差を測った結果。設計・許容差・データの引用は [リポジトリの benchmark-plan](https://github.com/mizuy/statract/blob/main/docs/dev/benchmark-plan.md)。R 関数の対応は [vs R](vs-r.md)。

**このページの表は 2026-10-09 の全タスク再計測。** 出典は [`comparison.csv`](benchmarks/comparison.csv)（`bench/stat/run_all.py` と同じ比較。344 行、102 タスク）。表は `bench/stat/render_tables.py`、図は `bench/stat/plot_charts.py` が CSV から作る。2026-10-07 の 68 タスクに、SUPPORT2 の 17 種（両スライスで 34 タスク）を足した。足したのは検定（`t.test` ほか）、pROC、rms の validate / calibrate、stdReg2、cmprsk、gamm4、glmmTMB（ポアソン / 負の二項）、partykit の ctree、mice。設計と比べる量は [benchmark-plan](https://github.com/mizuy/statract/blob/main/docs/dev/benchmark-plan.md)。Python は statract 0.1.0。R の版は前回の表と同じ機械で、足したパッケージの版はこの CSV に記録していない。ウォームアップ 1 回のあと 5 回の中央値（mice は 1 回）。BLAS / OMP スレッドは 1。混合モデルの Python 側は lme-python（`fit_mixed`）。

秒は Apple M4 Max（macOS）の壁時計で、前回（2026-10-05）の VM とは機械が違う。前回の表とは秒を比べない。R の秒はおおむね 1 ms 刻み。比が空なのは R 側が 0 s と記録されたため。

前回からの数値の変化は次のとおり。OLS は悪条件の計画行列で QR に回し、(X'WX)⁻¹ を列スケールした QR から作るようにした。これで HC とクラスタが R にそろった。共分散は |ΔV_ij| / √(V_ii V_jj) で比べる。GAM は mgcv の外側の Newton 法を収束させた当てはめ直しと比べる（R の秒は既定の呼び出し）。混合モデルは lme-python に収束の許容差 1e-8 を渡す。クラスタ頑健分散と混合モデルは、STAR の生まれ年を中心化しないことによる悪条件のため許容差を緩めた（表の「緩めた許容差」）。最近傍の組は、共変量が同じ対照への入れ替わりをタイとして同じ組とみなす。理由と数値は [benchmark-plan](https://github.com/mizuy/statract/blob/main/docs/dev/benchmark-plan.md) にある。

データは実行時に取得し、リポジトリには入れない。UCI Adult、Bike Sharing、Vanderbilt SUPPORT2、Harvard Dataverse STAR（file 666716）。STAR の取得は User-Agent 付きの HTTPS が必要だった。

## 読み方

| 列 | 意味 |
|----|------|
| 手法 | `bench/stat` のタスク ID |
| n | 切ったあとの行数 |
| Python / R | そのタスクの壁時計 |
| 比 | Python ÷ R。1 未満なら Python が速い |
| vs R | CSV の全チェック。括弧の数は quantity の行数。pass は [vs-r.md](vs-r.md) の許容差。「緩めた許容差」は benchmark-plan で緩めたタスク |

`tableone` はこのスイートに入っていない。Fine–Gray は SUPPORT2 に作った競合原因（院内死亡と退院後死亡）で `crr` と比べる。解析例は [解析例](../examples/index.md)。

## スピードの並び

タスクごとの壁時計。棒は [`comparison.csv`](benchmarks/comparison.csv) の `python_s` と `r_s`（ミリ秒、対数軸）。青が Python、青緑が R。どちらもロゴと同じ青から青緑で、数値の成否を色では分けていない。

ラベルの「tol外」は、そのタスクの quantity が許容差の外だったという印である。棒の長さは速度であり、数値比較が成功したという意味ではない。R のタイマーが 0 s のタスク（両側の Wald と尤度比、1,000 行側の KM（群））は対数軸に載せない。秒は下の表にある。

indo の二項 GLMM は再計測していないので、この図には入れていない。indo の記録はページ末尾。

![大きい側の壁時計。Python と R をタスクごとに並べた対数軸の棒グラフ](benchmarks/speed-large.png)

![1,000 行側の壁時計。Python と R をタスクごとに並べた対数軸の棒グラフ](benchmarks/speed-small.png)

### 同じ速さからの倍率

中心は Python と R が同じ壁時計。軸は log2(R の壁時計 / Python の壁時計) で、0 が等速。右へ行くほど R が遅い（Python が速い）。左へ行くほど Python が遅い。目盛りは 2 倍、4 倍のように、どちらが何倍遅いかを示す。棒の色と向きは速さだけである。斜線と「tol外」は数値比較が許容差の外であることであり、速さの成否ではない。

タスク分けは上の壁時計の図と同じ。R のタイマーが 0 s のタスクは比が定義できないので載せていない。indo の二項 GLMM は再計測していないので、この図にも入れていない。

![大きい側。等速を中心に、R と Python のどちらが何倍遅いか](benchmarks/ratio-large.png)

![1,000 行側。等速を中心に、R と Python のどちらが何倍遅いか](benchmarks/ratio-small.png)

## 大きい側

SUPPORT2 の生存は完全ケース 9,103 行。STAR の変量傾きは生徒単位の別スライス（n=10,000）。

| 手法 | API | n | Python | R | 比 | vs R |
|------|-----|--:|------:|--:|---:|------|
| OLS | `fit_ols` / `lm` | 10000 | 2.13 ms | 4.00 ms | 0.532 | 一致（5） |
| OLS（重み） | `fit_ols(weights=)` / `lm` | 10000 | 2.37 ms | 5.00 ms | 0.474 | 一致（5） |
| GLM 二項 | `fit_glm(binomial)` / `glm` | 10000 | 8.89 ms | 39.00 ms | 0.228 | 一致（3） |
| GLM ガンマ | `fit_glm(gamma)` / `glm` | 10000 | 5.79 ms | 20.00 ms | 0.290 | 一致（2） |
| sandwich HC0–HC5 | `hc_covariance` / `vcovHC` | 10000 | 2.10 ms | 12.00 ms | 0.175 | 一致（6） |
| Wald | `wald_test` / `waldtest` | 10000 | 0.023 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| 尤度比 | `likelihood_ratio_test` / `lrtest` | 10000 | 0.017 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| Breusch–Pagan | `breusch_pagan_test` / `bptest` | 10000 | 1.15 ms | 7.00 ms | 0.164 | 一致（4） |
| RESET | `ramsey_reset_test` / `resettest` | 10000 | 3.36 ms | 10.00 ms | 0.336 | 一致（2） |
| 最近傍 logit | `match_sample` / `matchit` | 10000 | 23.76 ms | 81.00 ms | 0.293 | 一致（2）。16 組は共変量が同じ別の対照（タイの破り方） |
| 最近傍 マハラノビス | `distance="mahalanobis"` | 10000 | 261.7 ms | 2.133 s | 0.123 | tol 外（1/1）。組が一致しない（3247 組中 2278 組が同じ）。LAPACK のビルドに依存 |
| 完全一致 | `method="exact"` | 10000 | 2.05 ms | 2.00 ms | 1.027 | 一致（1） |
| CEM | `method="cem"` | 10000 | 8.02 ms | 13.00 ms | 0.617 | 一致（1） |
| GLM ポアソン | `fit_glm(poisson)` / `glm` | 10000 | 13.65 ms | 83.00 ms | 0.164 | 一致（3） |
| Newey–West | `newey_west_covariance` / `NeweyWest` | 10000 | 4.91 ms | 271.0 ms | 0.018 | 一致（1） |
| Durbin–Watson | `durbin_watson_test` / `dwtest` | 10000 | 1.03 ms | 51.00 ms | 0.020 | 一致（2） |
| Breusch–Godfrey | `breusch_godfrey_test` / `bgtest` | 10000 | 4.74 ms | 30.00 ms | 0.158 | 一致（4） |
| GAM k=8 | `gam` / `mgcv::gam` | 10000 | 2.48 ms | 83.00 ms | 0.030 | 一致（4） |
| GAM k=10 | 同上 | 10000 | 2.38 ms | 110.0 ms | 0.022 | 一致（4） |
| Kaplan–Meier | `survival_curve` / `survfit` | 9103 | 0.519 ms | 3.00 ms | 0.173 | 一致（3） |
| KM（群） | `survival_curve(by=)` | 9103 | 1.68 ms | 3.00 ms | 0.561 | 一致（3） |
| Nelson–Aalen | `kind="nelson_aalen"` | 9103 | 0.471 ms | 4.00 ms | 0.118 | 一致（3） |
| log-rank | `log_rank` / `survdiff` | 9103 | 1.42 ms | 4.00 ms | 0.354 | 一致（6） |
| Cox Efron | `cox_ph` / `coxph` | 9103 | 7.69 ms | 19.00 ms | 0.405 | 一致（3） |
| Cox Breslow | `ties="breslow"` | 9103 | 5.30 ms | 20.00 ms | 0.265 | 一致（3） |
| Cox 層 | `strata=` | 9103 | 8.05 ms | 21.00 ms | 0.383 | 一致（3） |
| AFT Weibull | `accelerated_failure` / `survreg` | 9103 | 8.22 ms | 19.00 ms | 0.432 | 一致（3） |
| AFT lognormal | 同上 | 9103 | 8.47 ms | 19.00 ms | 0.446 | 一致（3） |
| AFT exponential | 同上 | 9103 | 6.49 ms | 19.00 ms | 0.341 | 一致（3） |
| クラスタ 1-way | `cluster_covariance` / `vcovCL` | 10000 | 3.21 ms | 4.00 ms | 0.802 | 一致（1）。緩めた許容差 |
| クラスタ 2-way | 同上 2 列 | 10000 | 10.17 ms | 18.00 ms | 0.565 | 一致（1）。緩めた許容差 |
| LMM 変量切片 REML | `fit_mixed` / `lmer` | 10000 | 26.52 ms | 45.00 ms | 0.589 | 一致（5）。緩めた許容差 |
| LMM ML | `method="ml"` | 10000 | 26.08 ms | 45.00 ms | 0.580 | 一致（5）。緩めた許容差 |
| LMM 変量傾き | `slopes=` | 10000 | 609.1 ms | 102.0 ms | 5.972 | 一致（5）。緩めた許容差 |
| t 検定 | `t_test` / `t.test` | 9103 | 0.062 ms | 0 s | — | 一致（10）。R のタイマーが 0 |
| Wilcoxon 順位和 | `wilcox_test` / `wilcox.test` | 9103 | 20.80 ms | 196.0 ms | 0.106 | 一致（4） |
| 比率の検定 | `prop_test` / `prop.test` | 9103 | 0.068 ms | 0 s | — | 一致（5）。R のタイマーが 0 |
| p 値の補正 | `p_adjust` / `p.adjust` | 9103 | 0.005 ms | 0 s | — | 一致（5）。R のタイマーが 0 |
| ROC と DeLong | `roc_curve`, `roc_test` / pROC | 9103 | 7.89 ms | 13.00 ms | 0.607 | 一致（5） |
| validate（ロジスティック） | `validate_logistic` / `rms::validate` | 9103 | 106.5 ms | 543.0 ms | 0.196 | 一致（1） |
| calibrate（ロジスティック） | `calibrate_logistic` / `rms::calibrate` | 9103 | 148.1 ms | 412.0 ms | 0.359 | 一致（2） |
| validate（Cox） | `validate_cox` / `rms::validate` | 9103 | 210.3 ms | 385.0 ms | 0.546 | 一致（1） |
| calibrate（Cox） | `calibrate_cox` / `rms::calibrate` | 9103 | 252.3 ms | 1.006 s | 0.251 | 一致（4） |
| Cox 標準化 | `standardize_cox` / `stdReg2` | 9103 | 40.21 ms | 94.00 ms | 0.428 | 一致（2） |
| 累積発生と Gray 検定 | `cumulative_incidence` / `cuminc` | 9103 | 6.40 ms | 3.00 ms | 2.134 | 一致（6） |
| Fine–Gray（crr） | `fine_gray_regression` / `crr` | 9103 | 1.312 s | 1.578 s | 0.831 | 一致（3） |
| GAMM | `gamm` / `gamm4` | 9103 | 22.959 s | 6.166 s | 3.723 | 一致（4） |
| GLMM ポアソン | `fit_mixed(poisson)` / `glmmTMB` | 9103 | 3.822 s | 389.0 ms | 9.824 | 一致（4） |
| GLMM 負の二項 | `fit_mixed(negative_binomial)` / `glmmTMB` | 9103 | 11.810 s | 4.011 s | 2.944 | 一致（5） |
| 条件付き推論木 | `conditional_tree` / `ctree` | 9103 | 41.07 ms | 39.00 ms | 1.053 | tol 外（2/5）。splits equal 1.000e+00、pred rel 8.054e-01 |
| 多重代入と統合 | `impute_chained`, `pool` / `mice` | 9104 | 47.831 s | 3.699 s | 12.931 | 一致（2） |

許容差の外に残るのはマハラノビスの最近傍と、9,103 行の ctree である。ctree は根の検定統計量と p 値、終端ノードの数は R と一致するが、深い分岐の 1 つ以上が R と違い、予測が最大 0.81 ずれる。1,000 行では分岐も予測も一致する。原因はまだ調べていない。

マハラノビスは、全水準の指標でプールした共分散が特異になり、MatchIt はその一般化逆行列をピボット付き Cholesky で分解する。末尾の塊が LAPACK のビルドで変わるので、x86-64 Linux の参照 LAPACK では組が一致し、この macOS では一致しない。ロジットの 10,000 行では 16 組が、共変量の同じ別の対照を取っている（タイの破り方）。

速さでは、変量傾きの LMM（比 6.0）に加え、新しく足した GLMM（ポアソン 9.8、負の二項 2.9）、GAMM（3.7）、mice（12.9）で Python が遅い。mice は 4 水準の因子の多項ロジット代入に時間の大半を使う。累積発生（cuminc）も 2.1 だが、どちらも 10 ms 未満である。lme-python の反復が 150 回を超える。変量切片は収束の許容差を 1e-8 に締めたぶん前回より秒が延びたが、`lmer` より速い。R の `lmer` は変量傾きで `boundary (singular) fit` を出した。

## 1,000 行側

同じタスクの小さいスライス。STAR の小さい側は生徒を切った結果 n=1,002 と n=1,001。

| 手法 | n | Python | R | 比 | vs R |
|------|--:|------:|--:|---:|------|
| OLS | 1000 | 1.17 ms | 1.00 ms | 1.167 | 一致（5） |
| OLS（重み） | 1000 | 1.05 ms | 1.000 ms | 1.049 | 一致（5） |
| GLM 二項 | 1000 | 1.64 ms | 4.00 ms | 0.409 | 一致（3） |
| GLM ガンマ | 1000 | 1.65 ms | 2.00 ms | 0.824 | 一致（2） |
| sandwich HC0–HC5 | 1000 | 0.260 ms | 2.00 ms | 0.130 | 一致（6） |
| Wald | 1000 | 0.027 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| 尤度比 | 1000 | 0.017 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| Breusch–Pagan | 1000 | 0.182 ms | 1.00 ms | 0.182 | 一致（4） |
| RESET | 1000 | 0.464 ms | 1.00 ms | 0.464 | 一致（2） |
| 最近傍 logit | 1000 | 4.90 ms | 7.00 ms | 0.700 | 一致（2） |
| 最近傍 マハラノビス | 1000 | 4.74 ms | 27.00 ms | 0.175 | tol 外（1/1）。組が一致しない（330 組中 165 組が同じ）。LAPACK のビルドに依存 |
| 完全一致 | 1000 | 0.213 ms | 1.000 ms | 0.213 | 一致（1） |
| CEM | 1000 | 0.803 ms | 3.00 ms | 0.268 | 一致（1） |
| GLM ポアソン | 1000 | 2.01 ms | 5.00 ms | 0.402 | 一致（3） |
| Newey–West | 1000 | 0.421 ms | 9.00 ms | 0.047 | 一致（1） |
| Durbin–Watson | 1000 | 0.231 ms | 3.00 ms | 0.077 | 一致（2） |
| Breusch–Godfrey | 1000 | 0.477 ms | 2.00 ms | 0.238 | 一致（4） |
| GAM k=8 | 1000 | 1.55 ms | 6.00 ms | 0.259 | 一致（4） |
| GAM k=10 | 1000 | 1.25 ms | 7.00 ms | 0.178 | 一致（4） |
| Kaplan–Meier | 1000 | 0.142 ms | 1.000 ms | 0.142 | 一致（3） |
| KM（群） | 1000 | 0.339 ms | 1.000 ms | 0.339 | 一致（3） |
| Nelson–Aalen | 1000 | 0.122 ms | 1.000 ms | 0.122 | 一致（3） |
| log-rank | 1000 | 0.904 ms | 1.000 ms | 0.904 | 一致（6） |
| Cox Efron | 1000 | 1.52 ms | 3.00 ms | 0.505 | 一致（3） |
| Cox Breslow | 1000 | 1.30 ms | 2.00 ms | 0.651 | 一致（3） |
| Cox 層 | 1000 | 1.55 ms | 3.00 ms | 0.517 | 一致（3） |
| AFT Weibull | 1000 | 1.99 ms | 2.00 ms | 0.997 | 一致（3） |
| AFT lognormal | 1000 | 2.22 ms | 2.00 ms | 1.110 | 一致（3） |
| AFT exponential | 1000 | 1.69 ms | 3.00 ms | 0.564 | 一致（3） |
| クラスタ 1-way | 1002 | 0.345 ms | 1.00 ms | 0.345 | 一致（1）。緩めた許容差 |
| クラスタ 2-way | 1002 | 0.968 ms | 5.00 ms | 0.194 | 一致（1）。緩めた許容差 |
| LMM 変量切片 REML | 1002 | 9.33 ms | 10.00 ms | 0.933 | 一致（5）。緩めた許容差 |
| LMM ML | 1002 | 9.49 ms | 10.00 ms | 0.949 | 一致（5）。緩めた許容差 |
| LMM 変量傾き | 1001 | 49.04 ms | 16.00 ms | 3.065 | 一致（5）。緩めた許容差 |
| t 検定 | 1000 | 0.056 ms | 0 s | — | 一致（10）。R のタイマーが 0 |
| Wilcoxon 順位和 | 1000 | 5.06 ms | 26.00 ms | 0.194 | 一致（4） |
| 比率の検定 | 1000 | 0.058 ms | 0 s | — | 一致（5）。R のタイマーが 0 |
| p 値の補正 | 1000 | 0.005 ms | 0 s | — | 一致（5）。R のタイマーが 0 |
| ROC と DeLong | 1000 | 1.42 ms | 2.00 ms | 0.712 | 一致（5） |
| validate（ロジスティック） | 1000 | 20.23 ms | 69.00 ms | 0.293 | 一致（1） |
| calibrate（ロジスティック） | 1000 | 21.79 ms | 53.00 ms | 0.411 | 一致（2） |
| validate（Cox） | 1000 | 23.23 ms | 44.00 ms | 0.528 | 一致（1） |
| calibrate（Cox） | 1000 | 26.81 ms | 137.0 ms | 0.196 | 一致（4） |
| Cox 標準化 | 1000 | 6.03 ms | 11.00 ms | 0.548 | 一致（2） |
| 累積発生と Gray 検定 | 1000 | 1.50 ms | 1.00 ms | 1.498 | 一致（6） |
| Fine–Gray（crr） | 1000 | 27.32 ms | 30.00 ms | 0.911 | 一致（3） |
| GAMM | 1000 | 846.9 ms | 738.0 ms | 1.148 | 一致（4） |
| GLMM ポアソン | 1000 | 638.4 ms | 68.00 ms | 9.388 | 一致（4） |
| GLMM 負の二項 | 1000 | 2.454 s | 360.0 ms | 6.817 | 一致（5） |
| 条件付き推論木 | 1000 | 4.89 ms | 7.00 ms | 0.699 | 一致（5） |
| 多重代入と統合 | 1000 | 11.332 s | 489.0 ms | 23.174 | 一致（2） |

1,000 行側では LMM の 3 件と OLS が R とほぼ同じか遅い。どれも 1 ms から 50 ms の範囲である。新しいタスクでは GLMM（比 6.8 から 9.4）と mice（23）が遅く、GAMM と cuminc がほぼ同じ。

## 未計測

| 手法 | 理由 |
|------|------|
| `tableone` / `write_tableone_artifacts` | `bench/stat` のタスク表に無い |
| `glmm_gpboost` | experimental extra。この実行では入れてない |
| `plot_survival` など図 | 数値ベンチ対象外 |
| indo の二項 GLMM | 下の 2026-10-04 記録。再計測していない |

行ごとの quantity は [comparison.csv](benchmarks/comparison.csv)。

## 2026-10-04 の indo 二項 GLMM（再計測していない）

`bench/stat` のタスク表に indo は無い。次の数は 2026-10-04 の記録で、出典は [`indo_glmm_compare.json`](benchmarks/indo_glmm_compare.json)。今回の機械では再計算していない。

式 `pep ~ indomethacin + age + risk + sod_yes + pdstent_yes + (1 | site)`。Laplace。n=602。

| | lme-python | R `glmer` | 差 |
|--|----------:|----------:|---|
| 壁時計 | 12.85 ms | 113 ms | 比 0.114 |
| インドメタシン OR | 0.4661 | 0.4644 | rel 3.66×10⁻³ |
| 切片 OR | 0.0806 | 0.0785 | rel 2.71×10⁻² |
| クラスタ分散 | 0.2954 | 0.2958 | rel 1.44×10⁻³ |
| 対数尤度 | −217.633 | −217.631 | abs 2.14×10⁻³ |

処置 OR の相対差は約 0.4%。固定効果の最大相対差は `sod_yes`（係数が約 0.01）。
