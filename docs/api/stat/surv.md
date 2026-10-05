# surv

生存曲線、log-rank、Cox、条件付きロジスティック、加速故障時間、Fine–Gray です。`cox_ph`、`accelerated_failure`、`fine_gray` は `Surv(time, status) ~ age + sex` を受けます。`conditional_logit` は `y ~ x + strata(set)` を受けます。文法は [Wilkinson 式](../../stat/formula.md) です。既存の `statract.survival`（Kaplan–Meier の図）とは別モジュールです。

::: statract.surv
