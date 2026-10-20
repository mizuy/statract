# mixed

Linear mixed models (Gaussian LMM) and generalized linear mixed models (binomial and other GLMMs). You can also use Wilkinson formulas such as `y ~ x + (1 | g)`. Gaussian and gamma call lme-python (lme-rs). `family="binomial"` (logit), `"poisson"`, and `"negative_binomial"` use our own Laplace / adaptive Gauss–Hermite code and accept `offset`. Poisson and negative binomial become zero-inflated with `zero_inflation=` and hurdle models with `hurdle=True`. These 3 families also accept crossed effects `(1 | a) + (1 | b)` and `ar1(time + 0 | g)`. lme-python is imported when first called. For usage, see [Models](../../models/overview.md).

::: statract.models.mixed
