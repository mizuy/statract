# 速度と R との差

公開データで **同じ表** を Python（このライブラリ）と R に渡し、壁時計と数値差を測った結果。設計・許容差・データの引用は [リポジトリの benchmark-plan](https://github.com/mizuy/statract/blob/main/docs/dev/benchmark-plan.md)。R 関数の対応は [vs R](vs-r.md)。

**このページの表は 2026-10-07 の全タスク再計測。** 出典は [`comparison.csv`](benchmarks/comparison.csv)（`bench/stat/run_all.py` と同じ比較。208 行、68 タスク）。表は `bench/stat/render_tables.py`、図は `bench/stat/plot_charts.py` が CSV から作る。Python は statract 0.1.0（Python 3.13.1、NumPy 2.5.3、SciPy 1.18.1、lme-python 0.2.6）。R は 4.6.1 + lme4 2.0.1 + survival 3.8.6 + mgcv 1.9.4 + MatchIt 4.7.2 + sandwich 3.1.1 + lmtest 0.9.40。ウォームアップ 1 回のあと 5 回の中央値。BLAS / OMP スレッドは 1。混合モデルの Python 側は lme-python（`fit_mixed`）。

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

`tableone` はこのスイートに入っていない。Fine–Gray も未計測。解析例は [解析例](../examples/index.md)。

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
| OLS | `fit_ols` / `lm` | 10000 | 2.17 ms | 4.00 ms | 0.541 | 一致（5） |
| OLS（重み） | `fit_ols(weights=)` / `lm` | 10000 | 2.23 ms | 5.00 ms | 0.447 | 一致（5） |
| GLM 二項 | `fit_glm(binomial)` / `glm` | 10000 | 8.17 ms | 39.00 ms | 0.209 | 一致（3） |
| GLM ガンマ | `fit_glm(gamma)` / `glm` | 10000 | 6.13 ms | 20.00 ms | 0.307 | 一致（2） |
| sandwich HC0–HC5 | `hc_covariance` / `vcovHC` | 10000 | 2.15 ms | 12.00 ms | 0.179 | 一致（6） |
| Wald | `wald_test` / `waldtest` | 10000 | 0.024 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| 尤度比 | `likelihood_ratio_test` / `lrtest` | 10000 | 0.014 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| Breusch–Pagan | `breusch_pagan_test` / `bptest` | 10000 | 1.09 ms | 6.00 ms | 0.182 | 一致（4） |
| RESET | `ramsey_reset_test` / `resettest` | 10000 | 3.39 ms | 11.00 ms | 0.308 | 一致（2） |
| 最近傍 logit | `match_sample` / `matchit` | 10000 | 25.09 ms | 83.00 ms | 0.302 | 一致（2）。16 組は共変量が同じ別の対照（タイの破り方） |
| 最近傍 マハラノビス | `distance="mahalanobis"` | 10000 | 267.0 ms | 2.206 s | 0.121 | tol 外（1/1）。組が一致しない（3247 組中 2278 組が同じ）。LAPACK のビルドに依存 |
| 完全一致 | `method="exact"` | 10000 | 1.93 ms | 2.00 ms | 0.964 | 一致（1） |
| CEM | `method="cem"` | 10000 | 8.07 ms | 13.00 ms | 0.620 | 一致（1） |
| GLM ポアソン | `fit_glm(poisson)` / `glm` | 10000 | 13.75 ms | 84.00 ms | 0.164 | 一致（3） |
| Newey–West | `newey_west_covariance` / `NeweyWest` | 10000 | 4.94 ms | 274.0 ms | 0.018 | 一致（1） |
| Durbin–Watson | `durbin_watson_test` / `dwtest` | 10000 | 1.05 ms | 51.00 ms | 0.021 | 一致（2） |
| Breusch–Godfrey | `breusch_godfrey_test` / `bgtest` | 10000 | 4.61 ms | 31.00 ms | 0.149 | 一致（4） |
| GAM k=8 | `gam` / `mgcv::gam` | 10000 | 2.49 ms | 82.00 ms | 0.030 | 一致（4） |
| GAM k=10 | 同上 | 10000 | 2.36 ms | 111.0 ms | 0.021 | 一致（4） |
| Kaplan–Meier | `survival_curve` / `survfit` | 9103 | 0.461 ms | 3.00 ms | 0.154 | 一致（3） |
| KM（群） | `survival_curve(by=)` | 9103 | 1.57 ms | 2.00 ms | 0.787 | 一致（3） |
| Nelson–Aalen | `kind="nelson_aalen"` | 9103 | 0.399 ms | 3.00 ms | 0.133 | 一致（3） |
| log-rank | `log_rank` / `survdiff` | 9103 | 1.12 ms | 4.00 ms | 0.281 | 一致（6） |
| Cox Efron | `cox_ph` / `coxph` | 9103 | 7.47 ms | 18.00 ms | 0.415 | 一致（3） |
| Cox Breslow | `ties="breslow"` | 9103 | 5.19 ms | 16.00 ms | 0.324 | 一致（3） |
| Cox 層 | `strata=` | 9103 | 8.37 ms | 19.00 ms | 0.441 | 一致（3） |
| AFT Weibull | `accelerated_failure` / `survreg` | 9103 | 8.11 ms | 18.00 ms | 0.451 | 一致（3） |
| AFT lognormal | 同上 | 9103 | 7.58 ms | 17.00 ms | 0.446 | 一致（3） |
| AFT exponential | 同上 | 9103 | 6.01 ms | 18.00 ms | 0.334 | 一致（3） |
| クラスタ 1-way | `cluster_covariance` / `vcovCL` | 10000 | 3.07 ms | 4.00 ms | 0.768 | 一致（1）。緩めた許容差 |
| クラスタ 2-way | 同上 2 列 | 10000 | 9.98 ms | 15.00 ms | 0.666 | 一致（1）。緩めた許容差 |
| LMM 変量切片 REML | `fit_mixed` / `lmer` | 10000 | 26.97 ms | 43.00 ms | 0.627 | 一致（5）。緩めた許容差 |
| LMM ML | `method="ml"` | 10000 | 28.66 ms | 42.00 ms | 0.682 | 一致（5）。緩めた許容差 |
| LMM 変量傾き | `slopes=` | 10000 | 605.4 ms | 98.00 ms | 6.178 | 一致（5）。緩めた許容差 |

許容差の外に残るのはマハラノビスの最近傍だけである。全水準の指標でプールした共分散が特異になり、MatchIt はその一般化逆行列をピボット付き Cholesky で分解する。末尾の塊が LAPACK のビルドで変わるので、x86-64 Linux の参照 LAPACK では組が一致し、この macOS では一致しない。ロジットの 10,000 行では 16 組が、共変量の同じ別の対照を取っている（タイの破り方）。

速さでは、変量傾きの LMM だけ Python が遅い（比 6.2）。lme-python の反復が 150 回を超える。変量切片は収束の許容差を 1e-8 に締めたぶん前回より秒が延びたが、`lmer` より速い。R の `lmer` は変量傾きで `boundary (singular) fit` を出した。

## 1,000 行側

同じタスクの小さいスライス。STAR の小さい側は生徒を切った結果 n=1,002 と n=1,001。

| 手法 | n | Python | R | 比 | vs R |
|------|--:|------:|--:|---:|------|
| OLS | 1000 | 1.01 ms | 1.00 ms | 1.014 | 一致（5） |
| OLS（重み） | 1000 | 0.893 ms | 1.00 ms | 0.893 | 一致（5） |
| GLM 二項 | 1000 | 1.57 ms | 4.00 ms | 0.392 | 一致（3） |
| GLM ガンマ | 1000 | 1.48 ms | 2.00 ms | 0.739 | 一致（2） |
| sandwich HC0–HC5 | 1000 | 0.249 ms | 2.00 ms | 0.125 | 一致（6） |
| Wald | 1000 | 0.029 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| 尤度比 | 1000 | 0.016 ms | 0 s | — | 一致（2）。R のタイマーが 0 |
| Breusch–Pagan | 1000 | 0.174 ms | 1.00 ms | 0.174 | 一致（4） |
| RESET | 1000 | 0.451 ms | 1.00 ms | 0.451 | 一致（2） |
| 最近傍 logit | 1000 | 4.18 ms | 8.00 ms | 0.522 | 一致（2） |
| 最近傍 マハラノビス | 1000 | 4.38 ms | 29.00 ms | 0.151 | tol 外（1/1）。組が一致しない（330 組中 165 組が同じ）。LAPACK のビルドに依存 |
| 完全一致 | 1000 | 0.211 ms | 1.000 ms | 0.211 | 一致（1） |
| CEM | 1000 | 0.832 ms | 2.00 ms | 0.416 | 一致（1） |
| GLM ポアソン | 1000 | 1.86 ms | 5.00 ms | 0.372 | 一致（3） |
| Newey–West | 1000 | 0.417 ms | 9.00 ms | 0.046 | 一致（1） |
| Durbin–Watson | 1000 | 0.230 ms | 3.00 ms | 0.077 | 一致（2） |
| Breusch–Godfrey | 1000 | 0.449 ms | 2.00 ms | 0.225 | 一致（4） |
| GAM k=8 | 1000 | 1.56 ms | 6.00 ms | 0.260 | 一致（4） |
| GAM k=10 | 1000 | 1.24 ms | 8.00 ms | 0.155 | 一致（4） |
| Kaplan–Meier | 1000 | 0.145 ms | 1.000 ms | 0.145 | 一致（3） |
| KM（群） | 1000 | 0.334 ms | 0 s | — | 一致（3）。R のタイマーが 0 |
| Nelson–Aalen | 1000 | 0.126 ms | 1.000 ms | 0.126 | 一致（3） |
| log-rank | 1000 | 0.829 ms | 1.000 ms | 0.829 | 一致（6） |
| Cox Efron | 1000 | 1.48 ms | 3.00 ms | 0.495 | 一致（3） |
| Cox Breslow | 1000 | 1.34 ms | 2.00 ms | 0.670 | 一致（3） |
| Cox 層 | 1000 | 1.56 ms | 3.00 ms | 0.519 | 一致（3） |
| AFT Weibull | 1000 | 1.76 ms | 2.00 ms | 0.879 | 一致（3） |
| AFT lognormal | 1000 | 1.91 ms | 2.00 ms | 0.953 | 一致（3） |
| AFT exponential | 1000 | 1.45 ms | 3.00 ms | 0.483 | 一致（3） |
| クラスタ 1-way | 1002 | 0.374 ms | 1.00 ms | 0.374 | 一致（1）。緩めた許容差 |
| クラスタ 2-way | 1002 | 0.961 ms | 3.00 ms | 0.320 | 一致（1）。緩めた許容差 |
| LMM 変量切片 REML | 1002 | 10.43 ms | 10.00 ms | 1.043 | 一致（5）。緩めた許容差 |
| LMM ML | 1002 | 10.16 ms | 9.00 ms | 1.129 | 一致（5）。緩めた許容差 |
| LMM 変量傾き | 1001 | 50.78 ms | 16.00 ms | 3.174 | 一致（5）。緩めた許容差 |

1,000 行側では LMM の 3 件と OLS が R とほぼ同じか遅い。どれも 1 ms から 50 ms の範囲である。

## 未計測

| 手法 | 理由 |
|------|------|
| `tableone` / `write_tableone_artifacts` | `bench/stat` のタスク表に無い |
| `glmm_gpboost` | experimental extra。この実行では入れてない |
| Fine–Gray / Aalen–Johansen | この 4 データに競合原因コードが無い |
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
