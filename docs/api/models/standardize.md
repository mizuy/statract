# standardize

Regression standardization (g-computation) after a GLM. It sets the exposure to each value, averages the predictions over everyone, and gives the adjusted risk difference, risk ratio, and odds ratio. The variance is either the delta method with covariates held fixed (`marginaleffects::avg_comparisons`), or a sandwich that also includes covariate variability (the estimator of `stdReg2::standardize_glm`). For usage, see [Models](../../models/overview.md). For numerical agreement with R, see [vs R](../../models/vs-r.md). Standardization for Cox models is `standardize_cox` in [surv](../surv/surv.md).

::: statract.models.standardize
