# statract — Quickstart

依存は Python 側の数値ライブラリです。このページの例は R を呼びません。

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

## 線形モデル

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

## 生存時間

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

`survival_curve` の既定は Kaplan–Meier で、信頼区間は log です。Nelson–Aalen は `kind="nelson_aalen"` です。Cox の同順位は既定で Efron、`ties="breslow"` も選べます。層は `strata(arm)` か `strata=`、重みは `weights=` です。加速故障時間は `accelerated_failure(frame, "Surv(time, event) ~ x")`、Fine–Gray の展開は `fine_gray(frame, "Surv(time, status) ~ x", cause=1)` です。

## マッチング

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

## 加法モデル

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

公開臨床データでの一連の流れ（Table 1 → モデル → 図）は合成データではなく [解析例ギャラリー](examples.md) を見る。リポジトリ側の目次は [`examples/`](https://github.com/mizuy/statract/tree/main/examples)。

## 線形混合モデル

`fit_mixed` は変量切片（と数値の変量傾き）の混合モデルです。ガウスは `lmer(y ~ x + (1 | g))` に相当し、推定は lme-python（lme-rs）です。二項は `family="binomial"` で `glmer` 相当です。

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
```

列で書くときは `fit_mixed(frame, "y", ["x"], groups="g")` です。変量傾きは `y ~ x + (1 + x | g)`、または `slopes=["x"]` です。`x` は固定効果にも入ります。ガウスを最尤にするときは `method="ml"` です。`(x || g)` の無相関な傾きは受け付けません。ガウスだけ mixedlm-rs に戻すときは `engine="mixedlm_rs"` です。

公開臨床データでの一連の流れ（Table 1 → モデル → 図）は合成データではなく [解析例ギャラリー](examples.md) を見る。リポジトリ側の目次は [`examples/`](https://github.com/mizuy/statract/tree/main/examples)。Python 対 R の秒数と係数差は [Benchmarks](benchmarks.md)。
