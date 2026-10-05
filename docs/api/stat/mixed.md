# mixed

線形混合モデル（ガウス LMM）と一般化線形混合モデル（二項などの GLMM）です。`fit_mixed` は既定で lme-python（lme-rs）を呼びます。`y ~ x + (1 | g)` のように Wilkinson 式でも書けます。`family="binomial"` が indo 型の `glmer` 経路です。ガウスだけ mixedlm-rs に戻すときは `engine="mixedlm_rs"`（optional extra）。lme-python は呼び出されたときに import します。使い方は [Quickstart](../../stat/quickstart.md) です。

::: statract.mixed
