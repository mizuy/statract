# surv

Survival curves, log-rank, Cox, conditional logistic, accelerated failure time, Fine–Gray, and cumulative incidence for competing risks. `cox_ph`, `accelerated_failure`, `fine_gray`, and `fine_gray_regression` take `Surv(time, status) ~ age + sex`. `cumulative_incidence` and `fine_gray_regression` match `cuminc` and `crr` in `cmprsk`. `conditional_logit` takes `y ~ x + strata(set)`. `standardize_cox` is Cox regression standardization (survival function and RMST) matching `stdReg2::standardize_coxph`. The unadjusted two-group RMST comparison is `restricted_mean_survival`, and its API is in [rmst](rmst.md). For the syntax, see [Wilkinson formula](../../models/formula.md). Kaplan–Meier plots are in `statract.viz.km`, and the API is in [km](../viz/km.md).

::: statract.surv
