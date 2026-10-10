# Models

推定です。データは Polars の `DataFrame`、結果の表も Polars です。依存は Python 側の数値ライブラリで、`import statract` は R を起動しません。

## 何ができるか

| 層 | モジュール | R での近いもの |
|----|------------|----------------|
| 設計行列 | `statract.models.design`、`model_matrix` | `model.matrix` の treatment contrast |
| 線形モデル | `statract.models.fit` | `lm` / `glm` |
| 共分散 | `statract.models.covariance` | sandwich |
| 線形の検定 | `statract.models.linear_tests` | lmtest |
| 基本の検定と区間 | `statract.models.htest` | `t.test`、`wilcox.test`、`mcnemar.test`、`binom.test`、`prop.test`、`p.adjust` |
| 生存時間 | `statract.surv` | survival |
| マッチング | `statract.models.matching` | MatchIt |
| 加法モデル | `statract.models.gam` | mgcv。gaussian / binomial / poisson / gamma、`cr` / `tp` / `cc` / `ps` / `re`、テンソル、`ti`、`by`、重み、offset |
| 条件付き推論木 | `statract.models.tree` | `partykit::ctree`。数値の応答、二次形式、Šidák 調整 |
| 線形・一般化線形混合 | `statract.models.mixed` | `lmer` / `glmer` / `glmmTMB`。ガウスとガンマは lme-python（lme-rs）、二項・ポアソン・負の二項は自前の Laplace。ゼロ過剰とハードルも |
| 加法混合 | `statract.models.gamm` | `gamm4`。平滑と変量切片を一つの GLMM で |
| 多重代入 | `statract.models.impute` | `mice`（pmm / logreg / polyreg）と `pool` |
| 予測の評価 | `statract.models.binary`、`statract.models.probability` | 較正、Brier、決定曲線、閾値。図は [Figures](../viz/overview.md) |

公開名は `statract` からまとめて import できます。数値の対応は [R パッケージとの対応](vs-r.md)、速度と差の一覧は [Benchmarks](benchmarks.md)、実データの一連の流れは [解析例](../examples/index.md) です。

## 共通の呼び方

- 第 1 引数はデータ。説明変数は列名のリストか、結果を左辺に置いた Wilkinson 式です。
- `fit_ols`、`fit_glm`、`fit_mixed` は `y ~ x * stage`、`y ~ x + (1 | g)` を受けます。`cox_ph`、`accelerated_failure`、`fine_gray` は `Surv(time, status) ~ age + sex`、`conditional_logit` は `y ~ x + strata(set)` を受けます。文法は [Wilkinson 式](formula.md) です。
- カテゴリの最初の水準が参照です。文字列はソート順、`pl.Enum` は定義順です。論理値は `FALSE` / `TRUE` です。ダミー名は `列名 + 水準` で、切片は `(Intercept)` です。交互作用は `:` でつなぎます。
- 欠損がある行は、そのモデルが使う列についてまとめて落とします。
- 係数表 `tidy()` の列は `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` です。
- 予測の種類は `kind` です。グループは `by` です。

## Quickstart

```python
import polars as pl
from statract import (
    cox_ph,
    fit_glm,
    fit_ols,
    gam,
    hc_covariance,
    match_sample,
    smooth,
    survival_curve,
)
```

### 線形モデル

```python
frame = pl.DataFrame(
    {
        "y": [1.2, 0.4, 2.1, 1.7, 0.9, 2.4],
        "x": [0.0, 0.2, 0.5, 0.7, 1.0, 1.2],
        "stage": ["I", "II", "I", "III", "II", "I"],
    }
)
fit = fit_ols(frame, "y", ["x", "stage"])
print(fit.tidy())
print(hc_covariance(fit, kind="HC1"))

# 同じモデルを Wilkinson 式でも書けます。* は主効果と交互作用です。
same = fit_ols(frame, "y ~ x * stage")
print(same.tidy())
```

`stage` は文字列なので水準は I, II, III の順になり、参照は I です。係数名は `stageII` と `stageIII` です。式の展開は R の `model.matrix` に合わせています。演算子、切片、対比、`offset()`、変量効果の列名は [Wilkinson 式](formula.md) です。`model_matrix` が設計行列そのものを返します。

二値の結果は `fit_glm(..., family="binomial")` です。`predict(kind="response")` は成功確率、`kind="link"` はロジットです。ポアソンは `family="poisson"`、ガンマは `family="gamma"`（リンクは逆数）です。

### 生存時間

```python
frame = pl.DataFrame(
    {
        "time": [0.4, 1.2, 0.8, 2.0, 1.5],
        "event": [1, 0, 1, 1, 0],
        "arm": [0, 0, 1, 1, 0],
        "x": [0.2, -0.4, 0.1, 0.5, -0.2],
    }
)
curve = survival_curve(frame, "time", "event", by="arm")
print(curve.at([1.0]))
model = cox_ph(frame, "Surv(time, event) ~ x + arm")
print(model.tidy())
# 列名でも同じモデルです。
# cox_ph(frame, "time", "event", ["x", "arm"])
```

患者と検査医のような交差する二元クラスタは `cluster_covariance(fit, ["id_patient", "e_examiner"], data=frame)` です。`sandwich::vcovCL(fit, cluster = ~ id_patient + e_examiner)` と同じ値で、係数が 1 個でも使えます。係数表は `coefficient_test(fit, cov)` です。`cox_ph(cluster=)` は 1 列の Lin–Wei（`coxph` と同じ）で、`cluster_covariance` の 1 列とは `G / (G - 1)` だけ違います。

`survival_curve` の既定は Kaplan–Meier で、信頼区間は log です。Nelson–Aalen は `kind="nelson_aalen"` です。Cox の同順位は既定で Efron、`ties="breslow"` も選べます。層は `strata(arm)` か `strata=`、重みは `weights=` です。加速故障時間は `accelerated_failure(frame, "Surv(time, event) ~ x")`、Fine–Gray の展開は `fine_gray(frame, "Surv(time, status) ~ x", cause=1)` です。

競合リスクの累積発生と Gray 検定は `cumulative_incidence(frame, "time", "status", by="arm")` です（`cmprsk::cuminc`）。`.tests` が原因ごとの検定、`.at([1, 3])` が時点の値です。`crr` と同じ Fine–Gray 回帰は `fine_gray_regression(frame, "Surv(time, status) ~ x + arm", cause=1)` で、SE は打ち切り分布の推定を含むサンドイッチです。`.predict(new)` が共変量ごとの CIF を返します。

連続変数の非線形は、式に `rcs(age, 4)`（`rms::rcs`）、`ns(age, df = 3)`、`bs(age, df = 5)` と書きます。`spline_test(fit, "rcs(age, 4)")` が非線形の検定、`spline_effect(fit, data, "age", at=..., reference=60, exponentiate=True)` が基準値に対する OR や HR の曲線、`plot_spline_effect` がその図です。詳しくは [Wilkinson 式](formula.md#スプライン) です。

Cox モデルでの回帰標準化は `standardize_cox(frame, "Surv(time, status) ~ ope * age + sex", values={"ope": [0, 1]}, times=[1, 3, 5])` です（`stdReg2::standardize_coxph`）。曝露を各値に置き換えた生存の標本平均で、分散は共変量のばらつきを含むサンドイッチです。`.tidy()` が曲線、`.tidy(contrast="difference", reference=0)` が差です。`measure="rmean"` は 0/1 の曝露での制限付き平均生存時間です。R との違いは [vs R](vs-r.md) の「stdReg2 と違うところ」 にあります。

### マッチング

```python
frame = pl.DataFrame(
    {
        "treat": [1, 1, 1, 0, 0, 0],
        "x1": [0.2, 0.4, 1.0, 0.1, 0.5, 0.8],
        "x2": [1.0, 0.2, -0.4, 0.9, 0.1, -0.2],
    }
)
matched = match_sample(frame, "treat", ["x1", "x2"], method="nearest", distance="logit", order="data")
print(matched.pairs())
print(matched.balance())  # 行は distance と各共変量
```

距離は `logit` のほか `mahalanobis`、`euclidean`、`scaled_euclidean`、`robust_mahalanobis` です。完全一致は `method="exact"`、粗化完全一致は `method="cem"` です。

### 加法モデル

```python
frame = pl.DataFrame(
    {
        "x": [0.0, 0.25, 0.5, 0.75, 1.0],
        "y": [0.1, 0.8, 0.2, -0.4, 0.0],
    }
)
# k は一意な x の個数以下にする
fitted = gam(frame, "y", [smooth("x", k=4)])
print(fitted.smooth_table())
print(fitted.predict())
```

この版が合わせるのは、ガウス分布、`basis="cr"`、`method="reml"`、平滑 1 本です。

平滑と変量切片を一つのモデルにするときは `gamm` です。`gamm4(y ~ s(pre_size_mm), random = ~(1 | examiner), family = binomial)` に当たります。

```python
from statract import gamm, median_odds_ratio, smooth

fit = gamm(frame, "y", [smooth("pre_size_mm", basis="tp")], random="(1 | examiner)", family="binomial")
print(fit.tidy(), fit.smooth_table())
print(median_odds_ratio(fit.variance_table()["variance"][0]))
print(fit.partial_effect("pre_size_mm", [10, 20, 30]))
```

平滑を固定の零空間と iid の変量効果に分け（`mgcv::smooth2random`）、ほかの変量効果と一緒に Laplace の最尤で解きます。`gam(..., smooth("examiner", basis="re"))` は mgcv の `bs = "re"`（REML）と一致しますが、これは gamm4 とは別の近似で、合成データでは MOR が 1.8% ずれました。混合モデル形は基底の座標に依存するので、thin plate は mgcv と同じ座標（共変量の中心化、固有値で割った基底、列の二乗平均を 1）で作ります。基底は `tp`（一意な値 2000 個まで）と `cr`、分布は binomial と poisson です。

### 線形混合モデル

`fit_mixed` は変量切片（と数値の変量傾き）の混合モデルの公開入口です。ガウスは `lmer(y ~ x + (1 | g))` に相当し、推定は lme-python（lme-rs）です。ガンマは `family=` で `glmer` 相当です。lme-python は `fit_mixed` を呼んだときに読みます。

二項（logit リンク）、ポアソン、負の二項（`family="negative_binomial"`、分散 `mu + mu^2 / theta`）は自前の Laplace 近似で推定します（`engine="laplace"`、既定）。`glmmTMB(..., family=binomial / poisson / nbinom2)` と `glmer(..., nAGQ=)` に合わせています。二項の probit などは lme-python に回ります。率のモデルは `y ~ arm + offset(log(years)) + (1 | site)`、列で書くときは `offset="log_years"` です。変量切片 1 本なら `n_agq=9` などで適応的 Gauss–Hermite 求積になります。固定効果の分散は、全パラメータの Hessian の逆行列から取ります（glmmTMB と同じ）。負の二項の `theta` は `fit.theta` です。

交差する変量切片は `y ~ x + (1 | id_patient) + (1 | e_examiner)` です。検査医ごとの年次変動は `ar1(year + 0 | e_examiner)` で、分散 `sigma^2 rho^|i - j|` の AR(1) になります（glmmTMB と同じ書き方）。複数の項では条件付きモードを疎な Newton 法でまとめて解き、一番大きい項（患者など）を消去してから残りを密に分解します。`fit.variance_table()` は項ごとの分散と ar1 の `rho`、`fit.random_effects()` は `group`、`level`、`term`、`blup` の縦長の表です。`fit.group_covariance` は最初の項の分散です。

ゼロ過剰は `zero_inflation=True`（ゼロの確率が定数）か `zero_inflation=["z"]`（列の logit モデル）です。`glmmTMB(..., ziformula = ~z)` に当たります。`hurdle=True` にするとハードル（ゼロかどうかを logit、正の値を切断ポアソンか切断負の二項）になり、`family=truncated_poisson` / `truncated_nbinom2` と `ziformula` の組に当たります。ゼロ部分の係数は `fit.zero_table()` です。`predict()` はゼロ部分を込めた応答の平均を返します。ゼロ部分は固定効果だけで、変量効果は付けられません。

```python
from statract import fit_mixed

frame = pl.DataFrame(
    {
        "y": [1.2, 1.5, 0.9, 1.1, 2.4, 2.1, 2.8, 2.0, 0.3, 0.6, 0.1, 0.8],
        "x": [0.1, 0.4, 0.2, 0.5, 0.2, 0.6, 0.3, 0.7, 0.1, 0.5, 0.2, 0.4],
        "g": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    }
)
mixed = fit_mixed(frame, "y ~ x + (1 | g)")
print(mixed.tidy())
print(mixed.variance_table())
logit = fit_mixed(frame.with_columns((pl.col("y") > 1).cast(pl.Int8).alias("z")), "z ~ x + (1 | g)", family="binomial")
print(logit.tidy(exponentiate=True))

counts = frame.with_columns((pl.col("y") * 3).round().cast(pl.Int64).alias("n"), pl.lit(2.0).alias("years"))
rate = fit_mixed(counts, "n ~ x + offset(log(years)) + (1 | g)", family="negative_binomial")
print(rate.tidy(exponentiate=True), rate.theta)
```

列で書くときは `fit_mixed(frame, "y", ["x"], groups="g")` です。変量傾きは `y ~ x + (1 + x | g)`、または `slopes=["x"]` です。`x` は固定効果にも入ります。ガウスを最尤にするときは `method="ml"`（既定は `"reml"`）です。`(x || g)` の無相関な傾きは受け付けません。`glmm_gpboost` は optional extra の実験実装で、GLMM の本体ではありません。glmmTMB の分散モデル、`nbinom1`、ゼロ部分の変量効果は対象外です。

### 多重代入

`impute_chained` は連鎖方程式（MICE）で欠損を `m` 回埋めます。既定の方法は `mice` と同じで、数値は予測平均マッチング（`pmm`、ドナー 5）、2 水準は Bayes ロジスティック（`logreg`）、3 水準以上は多項ロジスティック（`polyreg`）です。各列を他の全列から `n_iter` 回まわして埋めます。論理値と `pl.Enum` は型を保ちます。`pool` は各データで当てたモデルを Rubin のルールでまとめ、自由度は Barnard–Rubin です。`mice::pool` と `summary(pool(...), conf.int = TRUE)` に合わせています。

```python
from statract import fit_ols, impute_chained

mi = impute_chained(frame_with_nulls, m=20, n_iter=10, seed=1)
table = mi.pool(lambda d: fit_ols(d, "y ~ x1 + x2 + g"))
print(table.select("term", "estimate", "std_error", "df", "fmi", "conf_low", "conf_high"))
```

`pool` は `names`、`coefficients`、`covariance` を持つどのモデルでも使えます（`fit_glm`、`cox_ph`、`fit_mixed` など）。完全データの自由度 `dfcom` は、線形モデルと GLM が残差自由度、Cox がイベント数 − 係数の数、それ以外が n − 係数の数です（`mice` と同じ）。乱数生成器が R と違うので、埋めた値は `mice` と分布として同じで、桁までは一致しません。

### 既存の関数

`cumulative_survival_ci`、`log_rank_pvalue`、`plot_survival`、`tableone` はそのまま使えます。`sm_summary2df` は削除しました（置き換え先は `Fit.tidy`）。マッチングは `match_sample`、係数表は `fit_glm` / `Fit.tidy` です。

## 次に読む

- [Wilkinson 式](formula.md) — 式の文法
- [解析例ギャラリー](../examples/index.md) — Table 1 → モデル → 図の一連の流れ。リポジトリ側の目次は [`examples/`](https://github.com/mizuy/statract/tree/main/examples)
- [R パッケージとの対応](vs-r.md)、[Benchmarks](benchmarks.md)
