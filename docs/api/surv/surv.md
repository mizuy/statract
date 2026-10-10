# surv

生存曲線、log-rank、Cox、条件付きロジスティック、加速故障時間、Fine–Gray、競合リスクの累積発生です。`cox_ph`、`accelerated_failure`、`fine_gray`、`fine_gray_regression` は `Surv(time, status) ~ age + sex` を受けます。`cumulative_incidence` と `fine_gray_regression` は `cmprsk` の `cuminc` と `crr` に合わせています。`conditional_logit` は `y ~ x + strata(set)` を受けます。`standardize_cox` は `stdReg2::standardize_coxph` に合わせた Cox 回帰標準化（生存関数と RMST）です。調整なしの RMST の 2 群比較は `restricted_mean_survival` で、API は [rmst](rmst.md) です。文法は [Wilkinson 式](../../models/formula.md) です。Kaplan–Meier の図は `statract.viz.km` で、API は [km](../viz/km.md) です。

::: statract.surv
