# statract — 速度と R との差

公開データで **同じ表** を Python（このライブラリ）と R に渡し、壁時計と数値差を測った結果の一覧。設計・許容差・データの引用は [ベンチマークの設計](benchmark-plan.md)。R 関数の対応は [vs R](vs-r.md)。残課題の優先は [速度改善](speed-plan.md)。

**数値の出典:** リポジトリの [`bench/stat/results/comparison.csv`](../../bench/stat/results/comparison.csv)（コミット `e85af13`、2026-09-27）。測り方はウォームアップ 1 回のあと 5 回の中央値、BLAS/OMP スレッド 1。このページの秒と誤差は CSV を集約しただけで、手で作っていない。

この作業環境で **全タスクは再計測していない**。2026-10-04 に混合モデルだけ R 4.3.3 + lme4 で測った結果は下の [lme-python 既定の LMM / 二項 GLMM](#2026-10-04-lme-python-既定の-lmm--二項-glmm) 節。全スイートの再実行は `uv run python bench/stat/run_all.py`。

R の秒はおおむね 1 ms 刻み。比 Python / R が空なのは R 側が 0 s と記録されたため（LR 検定など、サブミリ秒）。

## 読み方

| 列 | 意味 |
|----|------|
| 手法 | `bench/stat` のタスク ID |
| n | 切ったあとの行数（大きい側。1,000 行側は下表） |
| Python / R | そのタスクの壁時計 |
| 比 | Python ÷ R。1 未満なら Python が速い |
| vs R | 係数・SE・HR 相当・p・組など、CSV の全チェック。pass は [vs-r.md](vs-r.md) の許容差 |

`tableone` はこのスイートに **入っていない**（R の `tableone` / `gtsummary` との速度・数値ベンチは未計測）。Fine–Gray も未計測。解析例の実データフローは [Examples / Gallery](examples.md)。

**2026-10-04（PR #57）:** 既定エンジン lme-python の混合モデルだけを、同じ STAR スライスと indo 全例で再計測した。出典 [`bench/stat/results/lmm-lme-python-2026-10-04.csv`](../../bench/stat/results/lmm-lme-python-2026-10-04.csv) と [`indo_glmm_compare.json`](../../bench/stat/results/indo_glmm_compare.json)。測り方はウォームアップ 1 回のあと 5 回の中央値、スレッド 1。R は `Rscript` 4.3.3 + lme4 1.1.35.1。上表の LMM 行は **mixedlm-rs 時代（2026-09-27）のまま**で、新しい秒は次節。

## 大きい側（約 10,000 行）

SUPPORT2 の生存は完全ケース 9,103 行。STAR 混合は 10,000 行（変量傾きは生徒単位の別スライス）。

| 手法 | API | n | Python | R | 比 | vs R |
|------|-----|--:|------:|--:|---:|------|
| OLS | `fit_ols` / `lm` | 10000 | 9.94 ms | 6.00 ms | 1.66 | 一致（5/5）。max rel `t` = 2.15×10⁻¹²（tol 10⁻⁶） |
| OLS（重み） | `fit_ols(weights=)` / `lm` | 10000 | 9.77 ms | 8.00 ms | 1.22 | 一致（5/5）。max rel `t` = 1.84×10⁻¹² |
| GLM 二項 | `fit_glm(binomial)` / `glm` | 10000 | 21.13 ms | 54.00 ms | 0.391 | 一致（coef / SE / loglik）。max rel SE = 2.82×10⁻¹¹ |
| GLM ガンマ | `fit_glm(gamma)` / `glm` | 10000 | 10.39 ms | 28.00 ms | 0.371 | 一致（coef / SE）。max rel coef = 1.01×10⁻¹⁰ |
| GLM ポアソン | `fit_glm(poisson)` / `glm` | 10000 | 32.80 ms | 90.00 ms | 0.364 | 一致。max rel coef = 3.67×10⁻¹² |
| sandwich HC0–HC5 | `hc_covariance` / `vcovHC` | 10000 | 3.00 ms | 21.00 ms | 0.143 | **tol 外**（6/6）。相対差は HC0–HC3 で約 2–4×10⁻⁷、HC4 で 5.13×10⁻⁶、HC5 で 1.04×10⁻⁷（許容 rtol 10⁻⁸） |
| クラスタ 1-way | `cluster_covariance` / `vcovCL` | 10000 | 4.52 ms | 7.00 ms | 0.646 | **tol 外**。cov rel = 9.41×10⁻⁵（rtol 10⁻⁸） |
| クラスタ 2-way | 同上 2 列 | 10000 | 27.79 ms | 27.00 ms | 1.03 | **tol 外**。cov rel = 4.29×10⁻⁵ |
| Newey–West | `newey_west_covariance` / `NeweyWest` | 10000 | 14.58 ms | 205.00 ms | 0.071 | 一致。cov rel = 4.24×10⁻¹⁰（rtol 10⁻⁸） |
| Wald | `wald_test` / `waldtest` | 10000 | 0.047 ms | 1.00 ms | 0.047 | 一致。max rel stat = 3.05×10⁻¹² |
| 尤度比 | `likelihood_ratio_test` / `lrtest` | 10000 | 0.027 ms | 0 s | — | 一致。max rel stat = 2.51×10⁻¹⁰。R のタイマーが 0 |
| Breusch–Pagan | `breusch_pagan_test` / `bptest` | 10000 | 1.67 ms | 8.00 ms | 0.209 | 一致 |
| RESET | `ramsey_reset_test` / `resettest` | 10000 | 5.05 ms | 15.00 ms | 0.337 | 一致 |
| Durbin–Watson | `durbin_watson_test` / `dwtest` | 10000 | 4.71 ms | 55.00 ms | 0.086 | 一致（stat rel 4×10⁻¹⁶、p abs 0） |
| Breusch–Godfrey | `breusch_godfrey_test` / `bgtest` | 10000 | 9.69 ms | 31.00 ms | 0.313 | 一致 |
| Kaplan–Meier | `survival_curve` / `survfit` | 9103 | 1.15 ms | 6.00 ms | 0.191 | 一致（生存率と区間）。max rel ≲ 3×10⁻¹⁶ |
| KM（群） | `survival_curve(by=)` | 9103 | 2.78 ms | 4.00 ms | 0.696 | 一致 |
| Nelson–Aalen | `kind="nelson_aalen"` | 9103 | 1.18 ms | 6.00 ms | 0.197 | 一致（estimate 差 0） |
| log-rank | `log_rank` / `survdiff` | 9103 | 1.72 ms | 7.00 ms | 0.246 | 一致（ρ=0/1 と層）。max rel stat = 1.44×10⁻¹² |
| Cox Efron | `cox_ph` / `coxph` | 9103 | 30.53 ms | 31.00 ms | 0.985 | 一致（coef / SE / 部分尤度）。max rel SE = 3.6×10⁻¹² |
| Cox Breslow | `ties="breslow"` | 9103 | 18.53 ms | 26.00 ms | 0.713 | 一致。max rel coef = 2.23×10⁻¹² |
| Cox 層 | `strata=` | 9103 | 42.38 ms | 31.00 ms | 1.37 | 一致。max rel coef = 9.33×10⁻¹³ |
| AFT Weibull | `accelerated_failure` / `survreg` | 9103 | 17.96 ms | 26.00 ms | 0.691 | 一致 |
| AFT lognormal | 同上 | 9103 | 17.66 ms | 26.00 ms | 0.679 | 一致 |
| AFT exponential | 同上 | 9103 | 13.60 ms | 25.00 ms | 0.544 | 一致。max rel coef = 2.1×10⁻¹⁰ |
| 最近傍 logit | `match_sample` / `matchit` | 10000 | 171.86 ms | 668.00 ms | 0.257 | **組が不一致**（件数はどちらも 3247）。`age` のマッチ後 SMD の相対差 0.466 |
| 最近傍 マハラノビス | `distance="mahalanobis"` | 10000 | 172.13 ms | 1.386 s | 0.124 | **組が不一致**（件数 3247 対 3247） |
| 完全一致 | `method="exact"` | 10000 | 2.85 ms | 5.00 ms | 0.570 | 一致。weights rel = 1.62×10⁻¹⁶ |
| CEM | `method="cem"` | 10000 | 10.85 ms | 66.00 ms | 0.164 | 一致。weights rel = 2.21×10⁻¹⁶ |
| GAM k=8 | `gam` / `mgcv::gam` | 10000 | 4.29 ms | 116.00 ms | 0.037 | sp / edf / REML は tol 内。**coef rel = 6.70×10⁻⁴**（coef の rtol 10⁻⁶ は未達） |
| GAM k=10 | 同上 | 10000 | 7.35 ms | 149.00 ms | 0.049 | sp / edf / REML は tol 内。**coef rel = 1.61×10⁻⁵** |
| LMM 変量切片 REML | `fit_mixed` / `lmer` | 10000 | 17.10 ms | 72.00 ms | 0.237 | SE / RE / σ² は tol 内。**coef rel = 2.72×10⁻⁶**、**loglik abs = 6.23×10⁻⁸**（coef rtol 10⁻⁶、loglik atol 10⁻⁸） |
| LMM ML | `method="ml"` | 10000 | 17.03 ms | 73.00 ms | 0.233 | coef / SE / RE / σ² は一致。**loglik abs = 4.78×10⁻⁸** |
| LMM 変量傾き | `slopes=` | 10000 | 25.14 ms | 190.00 ms | 0.132 | SE / RE / σ² は tol 内。**coef rel = 6.45×10⁻⁵**、**loglik abs = 9.53×10⁻⁷** |

HC とクラスタの「失敗」は、差がゼロではないが、多くの場合 10⁻⁷–10⁻⁵ 相対で、計画のサンドイッチ rtol 10⁻⁸ より緩い。Cox / GLM / KM の主結果（HR・OR に相当する係数と SE、生存率）は記録上すべて許容差内。

## 1,000 行側

同じタスクの小さいスライス。数値判定の方向は大きい側と同じものが多い。

| 手法 | n | Python | R | 比 | vs R |
|------|--:|------:|--:|---:|------|
| OLS | 1000 | 2.24 ms | 2.00 ms | 1.12 | 一致。max rel coef = 7.27×10⁻¹³ |
| OLS（重み） | 1000 | 2.33 ms | 2.00 ms | 1.16 | 一致 |
| GLM 二項 | 1000 | 3.03 ms | 6.00 ms | 0.505 | 一致 |
| GLM ガンマ | 1000 | 2.11 ms | 4.00 ms | 0.527 | 一致 |
| GLM ポアソン | 1000 | 2.90 ms | 7.00 ms | 0.414 | 一致 |
| HC0–HC5 | 1000 | 0.32 ms | 3.00 ms | 0.107 | **tol 外**。cov rel 約 2×10⁻⁸–4×10⁻⁷ |
| Wald | 1000 | 0.048 ms | 1.00 ms | 0.049 | 一致 |
| 尤度比 | 1000 | 0.029 ms | 0 s | — | 一致。R タイマー 0 |
| Breusch–Pagan | 1000 | 0.207 ms | 1.00 ms | 0.207 | 一致 |
| RESET | 1000 | 0.562 ms | 2.00 ms | 0.281 | 一致 |
| Durbin–Watson | 1000 | 0.497 ms | 5.00 ms | 0.099 | 一致 |
| Breusch–Godfrey | 1000 | 0.609 ms | 2.00 ms | 0.304 | 一致 |
| KM | 1000 | 0.480 ms | 1.00 ms | 0.480 | 一致 |
| KM（群） | 1000 | 1.72 ms | 2.00 ms | 0.858 | 一致 |
| Nelson–Aalen | 1000 | 0.512 ms | 1.00 ms | 0.512 | 一致 |
| log-rank | 1000 | 0.973 ms | 1.00 ms | 0.973 | 一致 |
| Cox Efron | 1000 | 4.49 ms | 5.00 ms | 0.898 | 一致 |
| Cox Breslow | 1000 | 3.42 ms | 5.00 ms | 0.684 | 一致 |
| Cox 層 | 1000 | 8.15 ms | 6.00 ms | 1.36 | 一致 |
| AFT Weibull | 1000 | 2.67 ms | 4.00 ms | 0.667 | 一致 |
| AFT lognormal | 1000 | 3.23 ms | 4.00 ms | 0.807 | 一致 |
| AFT exponential | 1000 | 2.23 ms | 4.00 ms | 0.559 | 一致 |
| 最近傍 logit | 1000 | 23.39 ms | 15.00 ms | 1.56 | **組不一致**（330 対 330）。SMD rel 0.150 |
| 最近傍 マハラノビス | 1000 | 3.41 ms | 17.00 ms | 0.201 | **組不一致**（330 対 330） |
| 完全一致 | 1000 | 0.365 ms | 2.00 ms | 0.183 | 一致（weights 差 0） |
| CEM | 1000 | 1.29 ms | 6.00 ms | 0.214 | 一致（weights 差 0） |
| GAM k=8 | 1000 | 1.75 ms | 12.00 ms | 0.146 | 一致（4/4） |
| GAM k=10 | 1000 | 2.25 ms | 12.00 ms | 0.187 | 一致（4/4） |
| クラスタ 1-way | 1002 | 0.604 ms | 3.00 ms | 0.201 | **tol 外**。cov rel = 1.50×10⁻⁵ |
| クラスタ 2-way | 1002 | 2.98 ms | 7.00 ms | 0.426 | **tol 外**。cov rel = 2.30×10⁻⁶ |
| LMM 変量切片 REML | 1002 | 3.51 ms | 19.00 ms | 0.184 | coef/SE/RE/σ² は tol 内。**loglik abs = 2.17×10⁻⁸** |
| LMM ML | 1002 | 3.05 ms | 19.00 ms | 0.160 | 一致（5/5） |
| LMM 変量傾き | 1001 | 9.65 ms | 30.00 ms | 0.322 | **coef rel = 4.99×10⁻⁵**、**RE rel = 2.18×10⁻⁴**、**loglik abs = 2.02×10⁻⁷** |

## 未計測

| 手法 | 理由 |
|------|------|
| `tableone` / `write_tableone_artifacts` | `bench/stat` のタスク表に無い。R 対照も無い |
| `glmm_gpboost` | experimental extra。二項 GLMM の本体は `fit_mixed(..., family="binomial")` |
| Fine–Gray / Aalen–Johansen | 4 データに競合原因コードが無い |
| `plot_survival` など図 | 数値ベンチ対象外 |

行ごとの quantity は CSV を直接見る。

## 2026-10-04: lme-python 既定の LMM / 二項 GLMM

全スイートは回していない。STAR の `lmm-ri` / `lmm-ml` / `lmm-rs`（既存 `bench/stat` タスク）と、indo 全例の `fit_mixed(..., family="binomial")` vs R `glmer`。Python は `engine="lme"`（既定）と、入っていた extra の `engine="mixedlm_rs"`。

### STAR LMM（対 `lme4::lmer`）

大きい側 n=10,000。R のタイマーは 1 ms 刻み。

| 手法 | n | lme-python | mixedlm-rs | R `lmer` | lme / R | mix / R | vs R（lme、旧 tol） |
|------|--:|----------:|-----------:|--------:|--------:|--------:|------|
| LMM 変量切片 REML | 10000 | 15.31 ms | 11.11 ms | 83.00 ms | 0.184 | 0.134 | **tol 外**。coef rel = 4.27×10⁻⁴、SE 2.64×10⁻⁴、RE 7.60×10⁻⁴、σ² 7.53×10⁻⁵、loglik abs = 9.70×10⁻⁵ |
| LMM ML | 10000 | 14.20 ms | 11.03 ms | 78.00 ms | 0.182 | 0.141 | **tol 外**。coef rel = 4.46×10⁻⁴、SE 2.72×10⁻⁴、RE 7.87×10⁻⁴、σ² 7.83×10⁻⁵、loglik abs = 1.04×10⁻⁴ |
| LMM 変量傾き | 10000 | 806.54 ms | 19.50 ms | 265.00 ms | 3.04 | 0.074 | **tol 外**。coef rel = 1.18×10⁻³、SE 1.29×10⁻⁵（SE は tol 内）、RE 8.81×10⁻⁴、σ² 7.44×10⁻⁶（σ² は tol 内）、loglik abs = 8.64×10⁻⁸ |

1,000 行側:

| 手法 | n | lme-python | mixedlm-rs | R `lmer` | lme / R | mix / R | vs R（lme） |
|------|--:|----------:|-----------:|--------:|--------:|--------:|------|
| LMM 変量切片 REML | 1002 | 5.57 ms | 3.75 ms | 19.00 ms | 0.293 | 0.198 | coef rel = 1.12×10⁻⁴（旧 coef tol 外）。SE は tol 内 |
| LMM ML | 1002 | 5.23 ms | 3.36 ms | 19.00 ms | 0.275 | 0.177 | coef rel = 7.41×10⁻⁴、RE 9.52×10⁻⁴、loglik abs = 1.99×10⁻⁵ |
| LMM 変量傾き | 1001 | 27.14 ms | 10.32 ms | 34.00 ms | 0.798 | 0.304 | coef rel = 1.42×10⁻⁴。R `lmer` は singular |

mixedlm-rs は変量切片で旧 tol にほぼ入る（loglik だけ 10⁻⁸ をわずかに超える行あり）。変量傾き 10,000 行の coef rel は 8.77×10⁻⁵（旧 10⁻⁶ は未達、SE/RE/σ² は tol 内）。

### indo 二項 GLMM（対 `lme4::glmer`、n=602）

式 `pep ~ indomethacin + age + risk + sod_yes + pdstent_yes + (1 | site)`。Laplace。出典 [`indo_glmm_compare.json`](../../bench/stat/results/indo_glmm_compare.json)。

| | lme-python | R `glmer` | 差 |
|--|----------:|----------:|---|
| 壁時計 | 12.85 ms | 113 ms | 比 0.114 |
| インドメタシン OR | 0.4661 | 0.4644 | rel 3.66×10⁻³ |
| 切片 OR | 0.0806 | 0.0785 | rel 2.71×10⁻² |
| クラスタ分散 | 0.2954 | 0.2958 | rel 1.44×10⁻³ |
| 対数尤度 | −217.633 | −217.631 | abs 2.14×10⁻³ |
| 固定効果 最大 rel | | | 0.115（`sod_yes`、係数 ≈ 0.01 の相対）。処置の coef rel は 4.76×10⁻³ |

fixture 未固定。処置 OR は 0.4% 相対。R より速い。
