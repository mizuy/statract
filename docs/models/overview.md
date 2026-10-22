# Models

This part covers estimation. Data is a Polars `DataFrame`, and result tables are Polars too. The dependencies are Python numerical libraries. `import statract` does not start R.

## What it covers

| Layer | Module | Closest in R |
|----|------------|----------------|
| Design matrix | `statract.models.design`, `model_matrix` | treatment contrasts in `model.matrix` |
| Linear models | `statract.models.fit` | `lm` / `glm` |
| Risk ratio | `statract.models.risk` | modified Poisson (`glm` + `sandwich`), log-binomial |
| Ordinal logistic | `statract.models.ordinal` | `MASS::polr`, `ordinal::clm`, Brant test |
| Multinomial logistic | `statract.models.multinom` | `nnet::multinom` |
| Covariance | `statract.models.covariance` | sandwich |
| Regression standardization for GLMs | `statract.models.standardize` | `marginaleffects::avg_comparisons`, `stdReg2::standardize_glm` |
| Linear tests | `statract.models.linear_tests` | lmtest |
| Basic tests and intervals | `statract.models.htest` | `t.test`, `wilcox.test`, `mcnemar.test`, `binom.test`, `prop.test`, `p.adjust` |
| Survival | `statract.surv` | survival. Two-group RMST comparison is `survRM2::rmst2` |
| Matching | `statract.models.matching` | MatchIt |
| Inverse probability weighting | `statract.models.weighting`, `statract.models.weighting_outcome` | WeightIt (`method="glm"`, `glm_weightit`), `bal.tab` and `love.plot` from cobalt |
| Additive models | `statract.models.gam` | mgcv. gaussian / binomial / poisson / gamma, `cr` / `tp` / `cc` / `ps` / `re`, tensors, `ti`, `by`, weights, offset |
| Conditional inference trees | `statract.models.tree` | `partykit::ctree`. Numeric response, quadratic form, Šidák adjustment |
| Linear and generalized linear mixed models | `statract.models.mixed` | `lmer` / `glmer` / `glmmTMB`. Gaussian and gamma use lme-python (lme-rs). Binomial, Poisson, and negative binomial use our own Laplace. Zero-inflated and hurdle models too |
| Additive mixed models | `statract.models.gamm` | `gamm4`. Smooths and random intercepts in one GLMM |
| Multiple imputation | `statract.models.impute` | `mice` (pmm / logreg / polyreg) and `pool` |
| Prediction performance | `statract.models.binary`, `statract.models.probability` | Calibration, Brier, decision curves, thresholds. Figures are in [Figures](../viz/overview.md) |

You can import all public names from `statract`. For numerical agreement with R, see [Mapping to R packages](vs-r.md). For speed and differences, see [Benchmarks](benchmarks.md). For full workflows on real data, see [Examples](../examples/index.md).

## Common conventions

- The first argument is the data. Predictors are a list of column names, or a Wilkinson formula with the outcome on the left side.
- `fit_ols`, `fit_glm`, and `fit_mixed` accept `y ~ x * stage` and `y ~ x + (1 | g)`. `cox_ph`, `accelerated_failure`, and `fine_gray` accept `Surv(time, status) ~ age + sex`. `conditional_logit` accepts `y ~ x + strata(set)`. The grammar is in [Wilkinson formula](formula.md).
- The first level of a categorical variable is the reference. Strings use sorted order. `pl.Enum` uses its defined order. Booleans are `FALSE` / `TRUE`. Dummy names are `column name + level`, and the intercept is `(Intercept)`. Interactions are joined with `:`.
- Rows with missing values in any column the model uses are dropped together.
- The columns of the coefficient table `tidy()` are `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high`.
- The type of prediction is `kind`. Groups are `by`.

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

### Linear models

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

# You can write the same model as a Wilkinson formula. * means main effects plus interaction.
same = fit_ols(frame, "y ~ x * stage")
print(same.tidy())
```

`stage` is a string, so the levels are in the order I, II, III, and the reference is I. The coefficient names are `stageII` and `stageIII`. Formula expansion matches R's `model.matrix`. Operators, the intercept, contrasts, `offset()`, and random-effect column names are in [Wilkinson formula](formula.md). `model_matrix` returns the design matrix itself.

For a binary outcome, use `fit_glm(..., family="binomial")`. `predict(kind="response")` gives the probability of success, and `kind="link"` gives the logit. Poisson is `family="poisson"`. Gamma is `family="gamma"` (inverse link). For overdispersed counts, use `family="negative_binomial"` (NB2, variance `mu + mu^2 / theta`, log link). It also estimates `theta` with the same procedure as `MASS::glm.nb`. The estimate is `fit.theta`, and its SE is `fit.theta_std_error`. `offset(log(years))` also works.

Regression standardization with a GLM is `standardize_glm(frame, "y ~ trt * age + sex", values={"trt": [0, 1]})`. It sets the exposure to each value and averages the predicted probabilities over everyone. `.tidy()` gives the adjusted risks, and `.tidy(contrast="difference", reference=0)` gives the risk difference. `contrast="ratio"` gives the risk ratio and `"odds_ratio"` gives the odds ratio. `ci="log"` gives intervals on the log scale. The default variance is the delta method with covariates held fixed. It matches `marginaleffects::avg_comparisons`. `vcov="HC3"` or `cluster="site"` uses a sandwich for the coefficient variance. `covariates="sampled"` is a sandwich in the style of `stdReg2::standardize_glm` that also includes the variability of the covariates. `weights=` are GLM weights and are also used in the average. Poisson is `family="poisson"`, and you can include `offset(log(years))`.

The risk ratio is `fit_risk_ratio(frame, "event ~ arm + age")`. The default is modified Poisson (Zou 2004), with an HC0 sandwich SE. For clusters such as sites, use `cluster="site"`. Log-binomial is `method="log-binomial"`. If it does not converge, pass `start=`. The estimates and intervals in `tidy()` are risk ratios.

For binary outcomes with few events or with separation, use `fit_firth(frame, "event ~ arm + age")` (`logistf`). The default `tidy()` gives profile penalized likelihood intervals and likelihood ratio p-values. Profile likelihood intervals for `fit_glm` coefficients are `profile_interval(fit)` (`confint.glm`).

For ordered outcomes (such as severity), use `ordinal_regression(frame, "grade ~ age + stage", levels=["none", "mild", "severe"])`. It is the proportional odds model of `polr` and `clm`. `tidy(exponentiate=True)` gives cumulative odds ratios. Check the proportional odds assumption with `brant_test(fit)`. For unordered outcomes with 3 or more levels, use `multinomial_regression(frame, "subtype ~ age + sex")` (`nnet::multinom`). The first level is the baseline, and `tidy(exponentiate=True)` gives relative risk ratios. In both, `probabilities()` returns the predicted probability of each level.

### Survival

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
# The same model with column names.
# cox_ph(frame, "time", "event", ["x", "arm"])
```

For crossed two-way clusters, such as patient and examiner, use `cluster_covariance(fit, ["id_patient", "e_examiner"], data=frame)`. It gives the same values as `sandwich::vcovCL(fit, cluster = ~ id_patient + e_examiner)`, and it works even with a single coefficient. The coefficient table is `coefficient_test(fit, cov)`. `cox_ph(cluster=)` is the one-column Lin–Wei estimator (same as `coxph`). It differs from one-column `cluster_covariance` by a factor of `G / (G - 1)`.

The default for `survival_curve` is Kaplan–Meier with log confidence intervals. Nelson–Aalen is `kind="nelson_aalen"`. Cox ties default to Efron, and `ties="breslow"` is also available. Strata are `strata(arm)` or `strata=`, and weights are `weights=`. Accelerated failure time is `accelerated_failure(frame, "Surv(time, event) ~ x")`. The Fine–Gray expansion is `fine_gray(frame, "Surv(time, status) ~ x", cause=1)`.

Cumulative incidence for competing risks and Gray's test are `cumulative_incidence(frame, "time", "status", by="arm")` (`cmprsk::cuminc`). `.tests` gives the test for each cause, and `.at([1, 3])` gives values at time points. Fine–Gray regression matching `crr` is `fine_gray_regression(frame, "Surv(time, status) ~ x + arm", cause=1)`. Its SE is a sandwich that includes the estimation of the censoring distribution. `.predict(new)` returns the CIF for each covariate pattern.

For nonlinear effects of continuous variables, write `rcs(age, 4)` (`rms::rcs`), `ns(age, df = 3)`, or `bs(age, df = 5)` in the formula. `spline_test(fit, "rcs(age, 4)")` tests nonlinearity. `spline_effect(fit, data, "age", at=..., reference=60, exponentiate=True)` gives the OR or HR curve against a reference value, and `plot_spline_effect` draws it. See [Wilkinson formula](formula.md#splines) for details.

Two-group comparison of restricted mean survival time (RMST) is `restricted_mean_survival(frame, "time", "event", by="arm", tau=5)` (`survRM2::rmst2`). `.arms` gives RMST and RMTL for each group. `.contrasts` gives the difference, the RMST ratio, and the RMTL ratio.

Regression standardization with a Cox model is `standardize_cox(frame, "Surv(time, status) ~ ope * age + sex", values={"ope": [0, 1]}, times=[1, 3, 5])` (`stdReg2::standardize_coxph`). It is the sample average of survival with the exposure set to each value. The variance is a sandwich that includes the variability of the covariates. `.tidy()` gives the curves, and `.tidy(contrast="difference", reference=0)` gives the differences. `measure="rmean"` gives the restricted mean survival time for a 0/1 exposure. Differences from stdReg2 are covered in [vs R](vs-r.md) (in Japanese).

### Matching

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
print(matched.balance())  # rows are distance and each covariate
```

Besides `logit`, the distances are `mahalanobis`, `euclidean`, `scaled_euclidean`, and `robust_mahalanobis`. Exact matching is `method="exact"`, and coarsened exact matching is `method="cem"`.

### Inverse probability weighting

`propensity_weights` gives IPTW for a binary treatment (`WeightIt::weightit(method = "glm")`). The propensity score comes from logistic regression with `fit_glm`. `estimand` is `ATE`, `ATT`, `ATC`, or `ATO`.

```python
from statract import cox_ph, propensity_weights, weighted_glm

ipw = propensity_weights(frame, "treat ~ age + sex + stage", estimand="ATE", stabilize=True, trim=0.99)
print(ipw.balance(threshold=0.1))  # same rows and columns as cobalt::bal.tab
print(ipw.effective_sample_size())  # Kish's effective sample size
ipw.love_plot("love.png")  # cobalt::love.plot
```

- Weights: ATE is `1/e` for treated and `1/(1-e)` for controls. ATT is 1 for treated and `e/(1-e)` for controls. ATC is `(1-e)/e` for treated and 1 for controls. ATO is `1-e` for treated and `e` for controls.
- `stabilize=True` multiplies by the proportion of the unit's own group. As in WeightIt, it applies to ATE only.
- `trim` is `WeightIt::trim`. A value below 1 is a quantile, and only the upper tail is trimmed. An integer of 1 or more is the number of weights to trim. `trim_lower=True` also trims the lower tail. ATT leaves treated weights unchanged, and ATC leaves control weights unchanged. `(low, high)` sets fixed lower and upper bounds. WeightIt does not have this.
- The differences in `balance()` are SMDs. The denominator is the pooled SD before weighting, `sqrt((s1² + s0²) / 2)`. It does not depend on the estimand and is the same before and after weighting. Binary variables are also standardized, with variance `p(1-p)`. You can reproduce cobalt's defaults (raw difference in proportions for binary variables, treated-group SD for ATT, and so on) with `binary="raw"` and `sd_denominator=` ([implementation notes](../api/models/weighting.md#smd)). Factors with 3 or more levels show all levels. `matched.balance()` for matching uses the same definition.
- For a table with arbitrary weights (such as matching weights), use `balance_table(frame, "treat ~ ...", weights="w")`.

Outcome models use `weighted_glm` (`WeightIt::glm_weightit`) and `effect()`. The default `vcov="asympt"` includes the propensity score estimation in the variance (M-estimation). `"HC0"` treats the weights as known and is conservative for ATE. `"bootstrap"` re-estimates the propensity score each time. You cannot use `"asympt"` with trimmed weights. See the [weighting API](../api/models/weighting.md) for details.

```python
ipw = propensity_weights(frame, "treat ~ age + sex + stage", estimand="ATE")
ipw.effect("death", measure=["difference", "ratio"])  # weighted group means, difference, and ratio
fit = weighted_glm(ipw, "death ~ treat + age", family="binomial", cluster="site")
fit.tidy()
frame = ipw.frame()  # ps and weights columns. Rows dropped by the formula are null
cox = cox_ph(frame, "Surv(time, status) ~ treat", weights="weights")  # non-integer weights give a robust variance
```

`sampling_weights=` is `s.weights` in WeightIt. They enter the propensity score model and the tables. They are not included in the `weights` column. `weighted_glm` and `effect()` use the product. Pass the product to `cox_ph`.

### Additive models {#additive-models}

```python
frame = pl.DataFrame(
    {
        "x": [0.0, 0.25, 0.5, 0.75, 1.0],
        "y": [0.1, 0.8, 0.2, -0.4, 0.0],
    }
)
# k must be at most the number of unique x values
fitted = gam(frame, "y", [smooth("x", k=4)])
print(fitted.smooth_table())
print(fitted.predict())
```

This version matches the Gaussian family, `basis="cr"`, `method="reml"`, and a single smooth.

`fitted.glance()` gives the deviance, the null deviance, the deviance explained, and the adjusted R², as `summary.gam`'s `dev.expl` and `r.sq`. `compare_gams(small, large)` compares nested models, as `anova(m1, m2)`: the deviance change, the change in edf (`df`), and a chi-squared test (binomial, Poisson) or F test (Gaussian, gamma). The p-value treats the smoothing parameters as known. mgcv uses an edf corrected for their uncertainty, so its `df` is larger and its p-value is larger too.

To put smooths and random intercepts in one model, use `gamm`. It corresponds to `gamm4(y ~ s(pre_size_mm), random = ~(1 | examiner), family = binomial)`.

```python
from statract import gamm, median_odds_ratio, smooth

fit = gamm(frame, "y", [smooth("pre_size_mm", basis="tp")], random="(1 | examiner)", family="binomial")
print(fit.tidy(), fit.smooth_table())
print(median_odds_ratio(fit.variance_table()["variance"][0]))
print(fit.partial_effect("pre_size_mm", [10, 20, 30]))
eta, se = fit.predict(frame, se=True)  # population level, random effects at zero
```

It splits each smooth into a fixed null space and iid random effects (`mgcv::smooth2random`), then solves it together with the other random effects by Laplace maximum likelihood. `gam(..., smooth("examiner", basis="re"))` matches mgcv's `bs = "re"` (REML). That is a different approximation from gamm4, and on synthetic data the MOR differed by 1.8%. The mixed model form depends on the basis coordinates, so thin plate splines use the same coordinates as mgcv (centered covariates, basis divided by eigenvalues, column mean squares of 1). The bases are `tp` (up to 2000 unique values) and `cr`. The families are binomial and poisson.

### Linear mixed models

`fit_mixed` is the public entry point for mixed models with random intercepts (and numeric random slopes). Gaussian corresponds to `lmer(y ~ x + (1 | g))` and is estimated with lme-python (lme-rs). Gamma via `family=` corresponds to `glmer`. lme-python is loaded when you call `fit_mixed`.

Binomial (logit link), Poisson, and negative binomial (`family="negative_binomial"`, variance `mu + mu^2 / theta`) are estimated with our own Laplace approximation (`engine="laplace"`, the default). They match `glmmTMB(..., family=binomial / poisson / nbinom2)` and `glmer(..., nAGQ=)`. Binomial with probit and other links goes to lme-python. A rate model is `y ~ arm + offset(log(years)) + (1 | site)`. With column names, use `offset="log_years"`. With a single random intercept, `n_agq=9` and similar values give adaptive Gauss–Hermite quadrature. The variance of the fixed effects comes from the inverse Hessian of all parameters (same as glmmTMB). The negative binomial `theta` is `fit.theta`.

Crossed random intercepts are `y ~ x + (1 | id_patient) + (1 | e_examiner)`. Year-to-year variation within each examiner is `ar1(year + 0 | e_examiner)`, an AR(1) with variance `sigma^2 rho^|i - j|` (same syntax as glmmTMB). With multiple terms, the conditional modes are solved together by sparse Newton. The largest term (such as patients) is eliminated first, and the rest is factorized densely. `fit.variance_table()` gives the variance of each term and the ar1 `rho`. `fit.random_effects()` is a long table with `group`, `level`, `term`, and `blup`. `fit.group_covariance` is the variance of the first term.

Zero inflation is `zero_inflation=True` (constant zero probability) or `zero_inflation=["z"]` (a logit model on columns). It corresponds to `glmmTMB(..., ziformula = ~z)`. `hurdle=True` gives a hurdle model (logit for zero or not, and truncated Poisson or truncated negative binomial for positive values). It corresponds to `family=truncated_poisson` / `truncated_nbinom2` with `ziformula`. The zero-part coefficients are `fit.zero_table()`. `predict()` returns the mean response including the zero part. The zero part has fixed effects only. It cannot have random effects.

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

With column names, use `fit_mixed(frame, "y", ["x"], groups="g")`. Random slopes are `y ~ x + (1 + x | g)` or `slopes=["x"]`. `x` also enters the fixed effects. For maximum likelihood with Gaussian, use `method="ml"` (the default is `"reml"`). Uncorrelated slopes `(x || g)` are not accepted. `glmm_gpboost` is an experimental implementation in an optional extra. It is not the main GLMM implementation. glmmTMB dispersion models, `nbinom1`, and random effects in the zero part are out of scope.

### Multiple imputation {#multiple-imputation}

`impute_chained` fills missing values `m` times with chained equations (MICE). The default methods are the same as `mice`: predictive mean matching for numeric columns (`pmm`, 5 donors), Bayesian logistic regression for 2 levels (`logreg`), and multinomial logistic regression for 3 or more levels (`polyreg`). Each column is imputed from all other columns, cycling `n_iter` times. Booleans and `pl.Enum` keep their types. `pool` combines the models fitted to each dataset with Rubin's rules, with Barnard–Rubin degrees of freedom. It matches `mice::pool` and `summary(pool(...), conf.int = TRUE)`.

```python
from statract import fit_ols, impute_chained

mi = impute_chained(frame_with_nulls, m=20, n_iter=10, seed=1)
table = mi.pool(lambda d: fit_ols(d, "y ~ x1 + x2 + g"))
print(table.select("term", "estimate", "std_error", "df", "fmi", "conf_low", "conf_high"))
```

`pool` works with any model that has `names`, `coefficients`, and `covariance` (`fit_glm`, `cox_ph`, `fit_mixed`, and so on). The complete-data degrees of freedom `dfcom` are the residual degrees of freedom for linear models and GLMs, the number of events − the number of coefficients for Cox, and n − the number of coefficients otherwise (same as `mice`). The random number generator differs from R, so the imputed values have the same distribution as `mice` but do not match digit for digit.

### Existing functions

`cumulative_survival_ci`, `log_rank_pvalue`, `plot_survival`, and `tableone` work as before. `sm_summary2df` has been removed (replaced by `Fit.tidy`). Matching is `match_sample`, inverse probability weighting is `propensity_weights`, and coefficient tables are `fit_glm` / `Fit.tidy`.

## What to read next

- [Wilkinson formula](formula.md) — formula grammar
- [Examples gallery](../examples/index.md) — full workflows from Table 1 to models to figures. The table of contents in the repository is [`examples/`](https://github.com/mizuy/statract/tree/main/examples)
- [Mapping to R packages](vs-r.md), [Benchmarks](benchmarks.md)
