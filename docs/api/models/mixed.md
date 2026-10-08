# mixed

線形混合モデル（ガウス LMM）と一般化線形混合モデル（二項などの GLMM）です。`y ~ x + (1 | g)` のように Wilkinson 式でも書けます。ガウスとガンマは lme-python（lme-rs）を呼びます。`family="binomial"`（logit）、`"poisson"`、`"negative_binomial"` は自前の Laplace / 適応的 Gauss–Hermite で、`offset` を受けます。ポアソンと負の二項は `zero_inflation=` でゼロ過剰、`hurdle=True` でハードルになります。この 3 族は `(1 | a) + (1 | b)` の交差と `ar1(time + 0 | g)` も受けます。lme-python は呼び出されたときに import します。使い方は [Models](../../models/overview.md) です。

::: statract.mixed
