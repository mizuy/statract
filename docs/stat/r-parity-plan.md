# statract — 統計機能の実装計画

survival、sandwich、lmtest、MatchIt、mgcv が担う機能と、ガウスの線形混合（`lmer`）を `statract` に足す。二項・ポアソンの GLMM と glmmTMB は §9 に残す。数値の目標は R と同じ出力で、呼び出し方は Python 側の流儀にする。

この文書はスコープ、公開インターフェース、順序、受け入れ基準を固定する。各フェーズの着手時に該当節を protocol として参照する。

**追記:** 公開入口は `fit_mixed`。既定エンジンは lme-python（lme-rs）。ガウスは `lmer`、二項は `family="binomial"`（`glmer`）。mixedlm-rs は `engine="mixedlm_rs"`（optional extra）。`glmm_gpboost` は本体にしない。

## 0. 方針

1. **R を実行経路に置かない。** `stat/r.py` は利用コードが明示的に呼ぶブリッジで、optional extra `statract[r]` のときだけ rpy2 を読む。オラクル生成は `Rscript` と JSON のままにする。新しい計算は R を起動しない。
2. **テストは複数標本で R と数値を合わせる。** 呼び出しの形は比べない（§4）。
3. **既定の統計手法は R の既定に合わせる。** Cox の同順位は Efron、生存曲線の信頼区間は log、マッチングの estimand は ATT、GAM の選択基準は REML。
4. **公開 API は Python の流儀にする。** `data` を第一引数にし、列は名前または `pl.Expr` で渡す。オプションはキーワード専用。`fit_ols`、`fit_glm`、`fit_mixed` は Wilkinson 式も受ける。`cox_ph`、`accelerated_failure`、`fine_gray` は `Surv(time, status) ~ x + strata(site)` も受ける。`s(x, k=10)` は式にせず `smooth` で渡す。
5. **入出力は Polars。** 推定結果は結果オブジェクトで、`tidy()` と `glance()` が Polars を返す。`pl.Enum` の先頭カテゴリを参照水準にする。欠損のある行は落とし、残った行の元インデックスを結果に持たせる。
6. **今回やらないものを書く。** ガウスの `lmer` は `fit_mixed` で入れる。二項・ポアソンの GLMM と glmmTMB は §9 に判断だけ残す。各パッケージの「入れない」も守る。
7. **図は論文に必要なものだけ。** 生存曲線、love plot、平滑項の部分効果。R のグラフィックス再現は目標外。
8. **既存の Python 実装は、複数標本で R と一致する範囲だけエンジンにする。** 一致しない量は自前で書く。R のラッパは使わない。推定器が違うライブラリは、名前が似ていても本体にしない（§6）。

## 1. 呼び出し方

既存の `statract` に合わせる。`plot_survival` や `log_rank_pvalue` と同じく、データフレームと列参照で構成する。

```python
fit = cox_ph(
    data,
    time="years",
    event="death",
    predictors=["age", "stage"],
    strata="site",
    cluster="hospital",
    ties="efron",
)
fit.tidy(exponentiate=True)
fit.predict(new_data, kind="survival")

curve = survival_curve(data, time="years", event="death", by="arm")
curve.at([1, 3, 5])

matched = match_sample(
    data,
    treatment="tx",
    covariates=["age", "stage"],
    method="nearest",
    estimand="att",
)
matched.balance()

fit = gam(
    data,
    outcome="y",
    smooths=[smooth("age", k=10, basis="cr")],
    predictors=["sex"],
    method="reml",
)
```

| 規則 | 内容 |
|------|------|
| 第一引数 | 推定・照合・曲線は `data: pl.DataFrame` |
| 列 | `str \| pl.Expr`。複数列は `list[str]` |
| グループ | `by`。既存 `plot_survival` の `hue` は残す |
| カテゴリ | `pl.Enum` の定義順。先頭が参照水準。上書きは `levels=` |
| 重み・オフセット | 列名 `weights` / `offset` |
| 結果 | `tidy()` の列は `term, estimate, std_error, statistic, p_value, conf_low, conf_high`。`exponentiate=True` で `exp_estimate, exp_conf_low, exp_conf_high` を足す |
| 予測の種類 | `kind=`（`"link"`, `"response"`, `"survival"` など）。R の `type=` は公開名にしない |
| 式 | `fit_ols`、`fit_glm`、`fit_mixed` の結果引数、および `model_matrix`。列名リストも残す。演算子は `+` `-` `*` `:` `/` `%in%` `^`、`0`/`1`、`.`、`I()`、`offset()`、`log` など。変量効果は `(1 \| g)` と `(1 + x \| g)` |

`model_matrix` は R の `terms` と `model.matrix`（treatment contrast）に合わせる。項は次数順、因子の並びは式の中で最初に現れた順、切片を外したときの最初の因子は全水準です。文法は `docs/stat/formula.md` にまとめた。実験的な `glmm_gpboost` も同じ展開を使います。

### 結果オブジェクト

sandwich 系の共分散と、係数の検定は、このオブジェクトに対して掛ける。

```python
class Fit:
    coefficients: np.ndarray
    covariance: np.ndarray          # モデルベース
    names: list[str]
    n_obs: int
    log_likelihood: float | None
    residual_df: int | None

    def tidy(self, *, level: float = 0.95, exponentiate: bool = False) -> pl.DataFrame: ...
    def glance(self) -> pl.DataFrame: ...
    def predict(self, data: pl.DataFrame, *, kind: str) -> np.ndarray: ...
    def score_contributions(self) -> np.ndarray: ...   # n × p。sandwich の meat の材料
    def bread(self) -> np.ndarray: ...
```

statsmodels の OLS / GLM を `Fit` に載せる `fit_glm(data, outcome, predictors, family="binomial")` を Phase 0 で用意し、Phase 1 の共分散と検定はまずここで検証する。

## 2. 現状

| 既存 | 実装 | 今回 |
|------|------|------|
| KM / log-rank / `cumulative_survival_ci` / `plot_survival` | lifelines | 公開名は維持。中身は fixture に通る範囲で lifelines を使い、通らない量だけ自前 |
| `psmatch` / `GLMHelper` | 互換 shim は削除済み | 公開名は `match_sample` と `fit_glm` / `Fit.tidy` |
| `glmm_gpboost` | gpboost（optional extra） | 本体にしない。GLMM は `fit_mixed(..., family="binomial")` |
| `sm_summary2df` | statsmodels 要約 | `Fit.tidy` に統合 |
| `stat/r.py` | rpy2（`statract[r]`） | 解析ブリッジ。オラクル生成は `Rscript` |

## 3. スコープ

R の名前は「どの出力と合わせるか」のラベルである。公開名は Python 側。

### 3.1 共分散（sandwich）

| 合わせる R | 公開名 | 今回 |
|------------|--------|------|
| `vcovHC` | `hc_covariance(fit, kind="HC3")` | ○ HC0–HC3、HC4、HC4m、HC5、`"const"` |
| `vcovCL` | `cluster_covariance(fit, cluster=, kind="HC1")` | ○ 1-way と多方向（包除）。lm/glm の R 既定は HC1、それ以外は HC0 |
| `vcovBS` | `bootstrap_covariance(fit, cluster=, n=250, kind="xy")` | ○ case / cluster / wild（Rademacher, Webb, Mammen） |
| `NeweyWest` | `newey_west_covariance(fit, lags=None)` | ○ Bartlett カーネルと自動ラグだけ |
| `estfun` / `bread` / `meat` | `score_contributions` / `bread` / `meat` | ○ |
| `vcovHAC` の一般カーネル、`vcovPL`、`vcovPC` | — | 入れない |

サンドイッチは `(1/n) · bread · meat · bread`。Cox のクラスタ頑健分散は、スコア寄与を dfbeta 由来のスコア残差、bread を情報行列の逆にする（Lin–Wei）。

### 3.2 検定（lmtest）

| 合わせる R | 公開名 | 今回 |
|------------|--------|------|
| `coeftest` | `coefficient_test(fit, covariance=None, df=None)` | ○ `covariance` は行列か、`Fit` を受け取る関数 |
| `coefci` | `coefficient_interval` | ○ |
| `waldtest` | `wald_test(full, reduced, covariance=None, distribution="f")` | ○ `"f"` または `"chi2"` |
| `lrtest` | `likelihood_ratio_test(full, reduced)` | ○ `log_likelihood` を持つ結果 |
| `bptest` | `breusch_pagan_test` | ○ OLS |
| `dwtest` | `durbin_watson_test` | ○ OLS |
| `resettest` | `ramsey_reset_test` | ○ OLS |
| `bgtest` | `breusch_godfrey_test` | ○ OLS |
| `harvtest`, `raintest`, `gqtest`, `hmctest`, `grangertest`, `encomptest`, `jtest`, `coxtest`, `petest` | — | 入れない |

### 3.3 生存（survival）

内視鏡の解析でいちばん使う。lifelines は KM・log-rank・`ties="efron"` の Cox について fixture が通る範囲をエンジンにする。信頼区間の型、層別 log-rank、Aalen–Johansen、survival 3.x の比例ハザード検定、counting process、Fine–Gray、加速故障時間は自前にする。

| 合わせる R | 公開名 | 今回 |
|------------|--------|------|
| `Surv` + `survfit`（KM / Nelson–Aalen） | `survival_curve(data, time, event, by=None, *, kind="kaplan_meier", confidence="log", weights=None)` | ○ `kind` は `"kaplan_meier"` または `"nelson_aalen"`。`confidence` は log / log-log / plain / logit / arcsin |
| `summary.survfit` / `quantile` | `curve.at(times)` / `curve.quantile(p)` | ○ 中央値の区間は Brookmeyer–Crowley |
| Aalen–Johansen | `survival_curve(..., kind="aalen_johansen", entry=)` | ○ 累積発生と SE。`entry` は左切り捨て（`entry < t <= time`） |
| `survdiff` | `log_rank(data, time, event, by, strata=None, rho=0)` | ○ k 群。`rho=1` は Peto–Peto。層別可。既存 `log_rank_pvalue` は 2 群の式 API として残す |
| `coxph` | `cox_ph(data, "Surv(time, status) ~ x + strata(site)")`。列名は `cox_ph(data, time, event, predictors, ...)` | ○ Efron / Breslow、層別、重み、オフセット、counting process（`Surv(start, stop, status)` または `entry`） |
| `clogit` | `conditional_logit(data, "y ~ x + strata(set)")`。列名は `conditional_logit(data, y, predictors, strata=)` | ○ 既定 exact はマッチセットの離散尤度。`efron` / `breslow`（`approximate`）は時間を 1 にした `cox_ph`。一般の `cox_ph(ties="exact")` は入れない |
| `cox.zph` | `proportional_hazards_test(fit, time_transform="kaplan_meier")` | ○ survival ≥ 3.0 のスコア検定。項ごとに集約 |
| 残差 | `fit.residuals(kind=)` | ○ martingale / deviance / score / schoenfeld / scaled_schoenfeld / dfbeta / dfbetas |
| 予測、基底ハザード、新規データでの曲線 | `fit.predict(kind=)` / `fit.baseline_hazard()` / `fit.survival_curve(data)` | ○ `kind` は `"linear_predictor"` / `"risk"` / `"expected"` / `"survival"` |
| `concordance` | `fit.concordance()` | ○ Harrell の C と SE |
| `survreg` | `accelerated_failure(data, "Surv(time, status) ~ x", distribution="weibull")`。列名も残す | ○ weibull / exponential / lognormal / loglogistic / gaussian / logistic。パラメータ化は R と同じ（intercept + log(scale)） |
| `finegray` | `fine_gray(data, "Surv(time, status) ~ x", cause)` のあと重み付き `cox_ph`。列名も残す | ○ |
| `survSplit` | `split_follow_up(data, time, event, cuts)` | ○ |
| `tt()`, `frailty()`, `ridge()`, `coxph(..., ties="exact")` | — | 入れない。マッチセットの exact は `conditional_logit` |
| `tmerge`, `pyears`, `survexp`, `coxme`, 区間打切り | — | 入れない |

Cox は事象時刻でソートし、同順位ブロックで Efron 補正を掛ける。Newton–Raphson に step-halving。収束は `eps=1e-9`、`max_iter=20`。`entry` があるときはソート後にリスク集合への加入と離脱を扱う。`proportional_hazards_test` はスケール化 Schoenfeld 残差と時間変換のスコア検定で、lifelines の `proportional_hazard_test` とは統計量が違う。

### 3.4 マッチング（MatchIt）

公開名は `match_sample` です。距離のロジスティック回帰は statsmodels、最適マッチングは `scipy.optimize.linear_sum_assignment`。

| 合わせる R | 公開名 | 今回 |
|------------|--------|------|
| `matchit` | `match_sample(data, treatment, covariates, *, method, distance, link, estimand, exact, caliper, std_caliper, ratio, replace, order, discard, reestimate)` | ○ |
| `method="nearest"` | `method="nearest"` | ○ `order` は `"largest"` / `"smallest"` / `"random"` / `"data"`。caliper、ratio、replace |
| `"exact"` / `"subclass"` / `"cem"` | 同 | ○ |
| `"optimal"` | 同 | ○ ratio > 1 は対照を複製 |
| `"full"` | 同 | △ 最小費用流。networkx を optional にするか、自前にするかは実装時に決める |
| `"genetic"` / `"cardinality"` / `"quick"` | — | 入れない |
| `distance="glm"`（logit / probit）, mahalanobis, robust mahalanobis, euclidean, scaled euclidean, 数値ベクトル | `distance=` | ○ `"logit"` を glm+logit の公開名にする |
| gam / gbm / random forest / lasso 距離 | — | 入れない |
| `summary` | `matched.balance()` | ○ SMD（ATT の分母は処置群 SD）、分散比、eCDF の平均と最大、ペア距離、標本数（All / Matched / Matched ESS / Unmatched / Discarded） |
| `match.data` / `get_matches` | `matched.frame()` / `matched.pairs()` | ○ `weights`, `subclass`, `distance` 列 |
| love plot | `matched.love_plot()` | ○ |
| jitter / qq / ecdf / density | `matched.balance_plot(kind="density")` | △ density だけ |

重みの定義は MatchIt 4.x に合わせる（ATT の非復元 1:1 は 1）。`order="random"` 以外は、マッチした組が R と完全一致することを受け入れ基準にする。

### 3.5 加法モデル（mgcv）

今回の最後に置く。平滑項は式ではなくオブジェクトで渡す。

```python
smooth("age", k=10, basis="cr")
smooth("age", by="sex", basis="tp")
tensor_smooth("age", "year", k=(10, 8))
```

| 合わせる R | 公開名 | 今回 |
|------------|--------|------|
| `gam` | `gam(data, outcome, predictors, smooths, *, family, method, weights, offset, select, gamma)` | ○ `method` は `"reml"` / `"ml"` / `"gcv"`。family は gaussian / binomial / poisson / gamma |
| `s(bs="tp"\|"cr"\|"cc"\|"ps"\|"re")` | `smooth(..., basis=)` | ○ |
| `te` / `ti`、`by` が因子 | `tensor_smooth`、`tensor_interaction`、`smooth(..., by=)` | ○ |
| `bs="fs"`, `"gp"`, `"so"`, `"ad"`, `"mrf"`, `"sos"`, `t2` | — | 入れない |
| `summary.gam` | `fit.smooth_table()` と `fit.tidy()` | ○ 平滑項は edf、参照自由度、統計量、p 値（Wood 2013）。全体は調整済み R²、逸脱度の説明率、REML |
| `predict.gam` | `fit.predict(data, kind=, se=False, exclude=None)` | ○ `kind` は `"link"` / `"response"` / `"terms"` |
| `plot.gam` | `fit.partial_effect(term)` | ○ 部分効果と帯 |
| `gam.check` / `k.check` | `fit.check()` | ○ |
| `anova.gam`, `concurvity` | `compare_gams` / `fit.concurvity()` | ○ |
| `nb`, `tw`, `ocat`, `betar`, `scat`, `cox.ph`, `bam`, `gamm`, `jagam` | — | 入れない。変量効果の平滑は `basis="re"` で足りる範囲だけ |

P-IRLS と複数の二次ペナルティ。識別制約（和が 0）は QR で吸収する。`tp` は Wood (2003) の打ち切り固有分解、`cr` は分位点ノットの cubic regression spline、`ps` は B スプラインと差分ペナルティ、テンソルは周辺基底の row-Kronecker。pygam と statsmodels `GLMGam` は基底と選択基準が違うので本体にしない。

## 4. テスト

```
tests/r_oracle/
  data/         # seed 固定の合成データ。R は読むだけ
  scripts/      # R: 読んで fit し、推定量を JSON に書く
  fixtures/     # コミットする JSON
  test_*.py     # fixture と Python を比較。実行時に R は不要
```

- 合成データにする理由は、ライセンスを避け、同順位・ゼロ事象・完全分離・マッチが落ちる caliper を意図的に作れること。
- 推定器ごとに最低 3 標本。(1) 均衡で教科書通り。(2) 不均衡、同順位、層別など分岐がある形。(3) 境界。
- 比べるのは論文に出る量（係数、SE、曲線、マッチした組）と目的関数（対数尤度、REML 基準、GCV）。同じ極なら最適化の経路が違っても通す。
- 生成は `Rscript`。Cloud Agent と Ubuntu 24.04 は `scripts/cloud-install-r.sh`。fixture に R とパッケージの版を入れる。

| 量 | 基準 |
|----|------|
| Newton で収束する係数（glm、Cox、log-rank 統計量、KM） | rtol 1e-6 |
| サンドイッチ共分散 | rtol 1e-8 |
| 平滑化パラメータ | rtol 1e-3。edf は rtol 1e-4。目的関数 atol 1e-6 |
| p 値 | atol 1e-6 |
| マッチング（乱数なし） | 組の完全一致。バランス統計 rtol 1e-8 |
| ブートストラップ | seed 固定の回帰テストと、R の標準誤差 rtol 0.1 |

## 5. 順序と受け入れ基準

`0 → 1 → 2 → 3 → 4`。共分散は Cox の `cluster` が使うので先に置く。生存が本業。マッチングは `match_sample`。加法モデルは最後。

| Phase | 成果物 | 受け入れ基準 |
|-------|--------|--------------|
| **0 基盤** | `design_matrix`, `Fit`, `fit_glm`, `tidy` / `glance`, `tests/r_oracle/` | `fit_glm(...).tidy()` が `broom::tidy` と一致。Enum の参照水準と欠損行の落とし方が R と同じ |
| **1 共分散と検定** | `hc_covariance`, `cluster_covariance`, `bootstrap_covariance`, `newey_west_covariance`, `coefficient_test`, `wald_test`, `likelihood_ratio_test`, OLS 診断 4 つ | lm/glm で HC0–HC3 と 1-way / 2-way クラスタが rtol 1e-8。Wald と尤度比の統計量と p が一致 |
| **2a 生存の核** | `survival_curve`（KM / Nelson–Aalen / Aalen–Johansen）, `log_rank`, `cox_ph`（Efron / Breslow、層、重み、オフセット） | 既存の生存テストが通る。R fixture と rtol 1e-6 |
| **2b 生存の推論** | `proportional_hazards_test`, 残差, `predict`, `baseline_hazard`, `survival_curve` on a Cox fit, `concordance` | 残差、比例ハザード統計量、C 統計量が一致 |
| **2c 生存の拡張** | counting process（`entry`）, クラスタ頑健分散, `fine_gray`, `accelerated_failure`, `split_follow_up` | 時間依存共変量と Fine–Gray の係数が一致 |
| **3a マッチングの核** | `match_sample` の nearest / exact / subclass、logit と mahalanobis、`balance`, `frame`, `love_plot` | `order="data"` で組が完全一致 |
| **3b マッチングの拡張** | optimal, cem, full, caliper 併用, `pairs` | optimal は総距離が一致。full は重みと層が一致 |
| **4a 加法モデルの核** | `smooth` の tp / cr / cc / ps / re、`gam`（gaussian / binomial / poisson、REML / ML / GCV）, `predict`, `tidy`, `partial_effect` | 平滑化パラメータ rtol 1e-3、edf rtol 1e-4、係数と予測 rtol 1e-4 |
| **4b 加法モデルの拡張** | `tensor_smooth`, `by`, `select=True`, gamma, `check`, `compare_gams`, `concurvity` | 同上 |

各フェーズで `docs/stat/vs-r.md` と API ページ、`mkdocs.yml` の nav を更新する。

### 配置

既存の `survival.py` をモジュールのまま残すため、新しい生存時間のコードは `surv/` に置く。`tests_linear.py` は pytest がテストとして集めるので、線形モデルの検定は `linear_tests.py` にする。

```
src/endolab/stat/
  design.py
  fit.py
  covariance.py
  linear_tests.py
  surv/
    curve.py  logrank.py  cox.py  aft.py  fine_gray.py
  matching/__init__.py
  gam/__init__.py
```

ファイル名も Python のモジュール名として読む。`coxph.py` や `matchit/` にはしない。fixture が数値を固定している範囲は `docs/stat/vs-r.md` に書く。

## 6. 依存

- **必須のまま**: numpy, scipy, polars, statsmodels（GLM の family、`distance="logit"`、OLS/GLM の HC と 1-way cluster が fixture に通る範囲）。設計行列は自前。formulaic は直接依存から外した。statsmodels と lifelines が推移依存として残し、どちらも pandas を直接依存している。
- **エンジンとして残す**: lifelines。fixture に通った KM、log-rank、Efron の Cox。
- **ガウス混合のエンジン**: mixedlm-rs（`fit_mixed`）。`lme4::lmer` と fixture で比べる。import は `fit_mixed` の中で、pandas もそのとき読む。二項・ポアソンの GLMM のエンジンにはしない。
- **本体にしない**: gpboost、pygam、statsmodels `GLMGam`、MatchIt 互換を謳う傾向スコア包装。
- **optional**: `statract[r]`（rpy2 と rpy2-arrow。解析ブリッジ `stat/r.py`。システム R も要る）。`endolab[match]`（networkx。full matching を自前にしない場合）。
- 新しいモジュールの import で rpy2 / R を起動しない。

## 7. リスク

| リスク | 対処 |
|--------|------|
| 最適化の差で数値がずれる | 決定的な量は厳しい許容差。平滑化パラメータは目的関数も比べる |
| R の引数名が公開 API に戻る | レビュー時に §1 の表と突き合わせる。fixture は数値だけを持つ |
| スコープが混ざる | 二項・ポアソンの GLMM と glmmTMB、各節の「入れない」は別 PR にしない限り触らない |
| 既存 API が壊れる | `cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival` の引数は維持する |

## 8. 今回の決定

1. **実装するのは** 共分散、線形モデルの検定、生存、マッチング、加法モデル、ガウスの線形混合（`fit_mixed`）。混合は §5 のあと。
2. **公開名は Python の snake_case。** データフレームと列名で呼べる。Wilkinson 式は `fit_ols`、`fit_glm`、`fit_mixed`、`model_matrix` の共通インターフェースである。
3. **テストは複数標本の数値一致。** 呼び出し互換は要求しない。
4. **lifelines** は fixture に通る関数のエンジンとして残す。
5. **`psmatch` と `GLMHelper` は削除した。** `sm_summary2df` は `DeprecationWarning` 付きで残す（置き換え先は `Fit.tidy`）。`glmm_gpboost` は今回そのまま。

## 9. 混合モデル

ガウスの変量切片と変量傾きは `fit_mixed` が mixedlm-rs を呼ぶ。比べる相手は `lme4::lmer` の REML / ML で、fixture は `tests/r_oracle/fixtures/lmm.json` です。mixedlm-rs と pandas は関数の中で import する。二項・ポアソンの GLMM と glmmTMB は、積分と分散モデルが別問題なので、まだ入れていません。

lme4 と glmmTMB を自前で書くときの判断は残しています。

- 線形混合（lme4 の `lmer`）はプロファイル REML / ML。θ を固定すると β と σ は罰則付き最小二乗で閉じる。Laplace 近似ではない。
- 指数型分布族の GLMM（`glmer`）は PIRLS + Laplace。変量効果がスカラー 1 個のときだけ適応的ガウス・エルミート求積。
- ゼロ過剰、分散の回帰、負の二項、`ar1` や compound symmetry（glmmTMB）は、Laplace 周辺尤度の自動微分が要る。lme4 の手続きの延長ではない。ガウス最尤は両者で突き合わせてよい。REML の定義は違うので、REML の一致対象は lme4 側だけにする。
- ガウスの `lmer` のエンジンは mixedlm-rs である。固定効果、分散、REML / ML の対数尤度、固定効果の標準誤差が fixture に通る範囲が対象である。gpboost は傾きと切片の相関を推定しないので、その本体にはしない。

## 参考

- `docs/cea/vs-r.md`
- Wood (2011) *Fast stable REML and ML estimation of semiparametric GLMs*. JRSS-B 73(1); Wood (2013) *On p-values for smooth components of an extended GAM*. Biometrika 100(1)
- Therneau & Grambsch (2000) *Modeling Survival Data*; survival 3.x の `cox.zph`
- Zeileis, Köll, Graham (2020) *Various Versatile Variances*. JSS 95(1)
- Ho, Imai, King, Stuart (2011) *MatchIt*. JSS 42(8); MatchIt 4.x *Matching Methods* / *Assessing Balance*
- Bates et al. (2015) JSS 67(1); Brooks et al. (2017) The R Journal 9(2) — §9 用。ガウスの `lmer` は mixedlm-rs。自前の Laplace / TMB はまだ書いていない
