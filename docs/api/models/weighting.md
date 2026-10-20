# weighting

Inverse probability of treatment weighting (IPTW) for a binary treatment. The public names are `propensity_weights`, `PropensityWeights`, `balance_table`, `weighted_glm`, `WeightedGLM`, and `weighted_effect`.

- The propensity score is a logistic regression from `fit_glm`. The weight formulas, `stabilize`, and `trim` match `WeightIt::weightit(method = "glm")` and `WeightIt::trim`.
- `balance()` and `balance_table` give a table with the same rows and columns as `cobalt::bal.tab`. Factors are split by level. Differences are SMDs for both binary and continuous variables, and continuous variables also get variance ratios. For the SMD definition, see "Implementation notes: SMD" below.
- `effective_sample_size()` is Kish's effective sample size `(Σw)² / Σw²` for each group.
- The plot is [`plot_love`](../viz/balance.md).

- Outcome models are `weighted_glm` (`WeightIt::glm_weightit`) and `PropensityWeights.effect`. The default variance accounts for estimating the propensity score (see "Outcome models and variance" below).
- For Cox models, pass the `weights` column of `frame()` to `cox_ph(..., weights="weights")`. For usage, see "Inverse probability weighting" in [Models](../../models/overview.md).

## Outcome models and variance

```python
from statract import propensity_weights, weighted_glm

ipw = propensity_weights(frame, "clip ~ age + size_mm + anticoag", estimand="ATE")
ipw.effect("bleed", measure=["difference", "ratio", "odds_ratio"])  # marginal effects
ipw.effect("bleed", vcov="bootstrap", n_boot=2000, seed=1)  # re-estimates the PS each time
fit = weighted_glm(ipw, "bleed ~ clip + age", family="binomial", cluster="hospital")
fit.tidy()
```

- `weighted_glm(weights, formula, family=, vcov=, cluster=)` is a GLM with weight × sampling weight as prior weights. The data are the rows of `weights.data` that have weights. If the outcome formula drops rows, it raises an error.
- `PropensityWeights.effect(outcome, measure=)` (the function is `weighted_effect`) gives the weighted means by group, `mean_treated` and `mean_control`, and their difference (`"difference"`), ratio (`"ratio"`), and odds ratio (`"odds_ratio"`). The estimand is the estimand of the weights. It is the same as the saturated model `y ~ treat`, so for a binary outcome the difference is the risk difference from a weighted linear model. Tests and intervals for the ratio and odds ratio are on the log scale, and `std_error` is also the SE on the log scale.

`vcov` has 3 options.

| `vcov` | What it does | R |
| --- | --- | --- |
| `"asympt"` (default) | M-estimation. Stacks the outcome estimating equations with the scores of the propensity score logistic regression, and takes the upper-left block of the full sandwich $A^{-1} B A^{-\top}$ | `glm_weightit(vcov = "asympt")` |
| `"HC0"` | Outcome part only. Treats the weights as known | `glm_weightit(vcov = "HC0")`, `sandwich::vcovHC(type = "HC0")` |
| `"bootstrap"` | Resamples rows (clusters if `cluster` is given) with replacement, and re-estimates the propensity score, weights, and outcome each time. SE is the SD of the replicates, and the interval is percentile | Close to `glm_weightit(vcov = "BS")` (intervals are built differently) |

- The cross term in $A$ is the derivative of the weights with respect to the propensity score coefficients. $\partial w / \partial \gamma = (\partial w / \partial e)\, e(1-e)\, z$, where $\partial w / \partial e$ is $-1/e^2$ for treated and $1/(1-e)^2$ for controls under ATE, $1/(1-e)^2$ for controls under ATT (0 for treated), $-1/e^2$ for treated under ATC (0 for controls), and $-1$ for treated and $1$ for controls under ATO (Lunceford and Davidian 2004).
- With `stabilize=True`, the proportion treated $p_1$ is also stacked as an estimate (estimating equation $s(a - p_1)$).
- `cluster` sums the estimating equations within clusters and multiplies by $G/(G-1)$ (`cadjust` in `sandwich::vcovCL`).
- For ATE, the SE that accounts for propensity score estimation is smaller than HC0. For ATT and others it can be larger.
- Trimmed weights (`trim=`) do not change smoothly with the propensity score coefficients, so `vcov="asympt"` raises an error. `"HC0"` and `"bootstrap"` (which re-trims each time) work.
- The bootstrap uses `numpy.random.default_rng(seed)`. Replicates missing one of the groups are redrawn.

## Implementation notes: SMD { #smd }

statract balance tables (`PropensityWeights.balance()`, `balance_table`, `MatchedSample.balance()`) use the same SMD by default.

$$
\mathrm{SMD} = \frac{\bar x_1 - \bar x_0}{\sqrt{(s_1^2 + s_0^2)/2}}
$$

- The variances $s_1^2, s_0^2$ are computed once on the sample before adjustment (before weighting or matching). The same denominator is used for the rows before and after adjustment. Sampling weights are kept.
- Binary variables are also standardized, with variance $p(1-p)$ (Austin 2009). This is the same as tableone's formula for 2-level categorical variables.
- The default does not change with the estimand (ATE, ATT, ATC, ATO).

The denominator is fixed so that a change in SMD before and after adjustment reads as a change in the mean difference. If the variance is recomputed on the adjusted sample, the SMD moves just because the variance changed.

Differences from R packages, and why:

- **MatchIt** uses the SD of the treated group as the denominator for ATT (the control group for ATC). In ATT matching the treated group stays almost intact, so this gives a fixed scale measured in the units of the target population.
- **cobalt** reports binary variables as raw differences in proportions. The SD of a binary variable is determined by the proportion, so the raw difference is easier to read (Greifer). For ATE the denominator is the pooled SD before adjustment. For ATT it is the treated group, for ATC the control group, and for ATO the SD of the full weighted sample.
- **tableone** recomputes the pooled SD within the table you pass. The SMD in a Table 1 built from the matched sample has a different denominator, so it may not match `smd_matched` / `diff_adjusted` in the balance table.

You can reproduce the R conventions with options.

| To reproduce | statract setting |
| --- | --- |
| cobalt `bal.tab` (ATE) | `balance(binary="raw", sd_denominator="pooled")` |
| cobalt `bal.tab` (ATT / ATC) | `balance(binary="raw", sd_denominator="treated")` / `"control"` |
| cobalt `bal.tab` (ATO) | `balance(binary="raw", sd_denominator="weighted")` |
| MatchIt `summary` (ATT / ATC) | `MatchedSample.balance(sd_denominator="treated")` / `"control"` |
| tableone `ExtractSmd` (before adjustment) | Default (the rows before adjustment match) |

`sd_denominator` also accepts `"all"` (the full sample without splitting by group). `continuous="raw"` leaves continuous variables unstandardized.

::: statract.models.weighting

::: statract.models.weighting_outcome
