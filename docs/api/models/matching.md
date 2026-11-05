# matching

Matching on propensity scores and distances. The public name is `match_sample`. `total_distance` is the sum of the distances of the matched edges. Weights for full matching follow MatchIt's `normalize=TRUE`.

## Estimands, weights and missing values { #weights }

- `estimand="ATE"` works with `full`, `subclass`, `exact` and `cem`. Within subclass $s$, a treated unit weighs $n_s / n_{1s}$ and a control $n_s / n_{0s}$ (Hansen 2004; Stuart 2010). Only subclasses that hold both groups are kept. Pair matching (`nearest`, `optimal`) has no ATE and raises `ValueError`; use full matching, subclassification or IPTW instead.
- With `ratio` above 1, each matched treated unit weighs 1. A treated unit matched to $k_i$ controls gives each of them $1/k_i$, and a control's weight is the sum over the treated units it is matched to. The controls are then rescaled so that their weights sum to the number of matched controls. This matters when treated units get different numbers of controls, for example with a caliper.
- A missing treatment value raises `ValueError`. Drop those rows first.
- A row with a missing value in an `exact` variable, or in a CEM variable, is never matched (weight 0, subclass 0). CEM cutpoints are computed from the rows that are complete on the CEM variables.
- The same holds for the matrix distances: with `scaled_euclidean`, `robust_mahalanobis`, `mahalanobis` or `euclidean`, a row with a missing covariate is never matched, and the scaling (pooled SDs, ranks, covariance) uses the complete rows only. With `reestimate=True`, the refit propensity score is missing for such rows.
- Optimal matching with `ratio=k` needs at least `k` controls per treated unit. With fewer, it warns and falls back to 1:1 matching.

## Subclass and CEM bins { #bins }

- `method="subclass"` cuts the distance at quantiles of the focal group: the treated units for ATT, the controls for ATC, all units for ATE (Stuart 2010). The outer cut points are widened so every unit falls in a subclass.
- Subclass and CEM bins are right-closed, as R's `cut(..., include.lowest = TRUE)`: a value on a cut point falls in the lower bin.

## Implementation notes: SMD { #smd }

`smd_all` and `smd_matched` in `balance()` are the same SMD as in the weighting balance table. The default denominator is the pooled SD of the full sample before matching, $\sqrt{(s_1^2 + s_0^2)/2}$, and the same value is used before and after matching. Since the denominator is fixed, a change in SMD reflects a change in the mean difference. Binary variables are standardized with the variance $p(1-p)$ (Austin 2009). `pair_distance` is divided by the same denominator. The `distance` row is always standardized.

The columns always describe the actual groups: `mean_treated_*` is the treated mean and `smd_*` is treated minus control, also for ATC. A Boolean covariate is one row coded 0/1. A factor (Enum, Categorical, String) is coded as level indicators, as in `balance_table`: a two-level factor shows its second level as `name_level`, a factor with three or more levels shows every level. Indicators are binary variables, so they follow the `binary=` setting. `variance_ratio_matched` uses the matching weights (weighted variance with reliability weights, as cobalt). `n_all` counts every unit and `n_matched` the units with a positive matching weight.

MatchIt uses the SD of the treated group as the denominator for ATT. In ATT matching the treated group stays almost intact, so this gives a fixed scale in the units of the target population. You can reproduce it with `balance(sd_denominator="treated")` (`"control"` for ATC). To report binary variables as raw differences in proportions, as cobalt does, use `binary="raw"`. tableone recomputes the pooled SD within the matched table, so the SMD in a Table 1 after matching may not match `smd_matched`. The reasons and a mapping table are in the [weighting implementation notes](weighting.md#smd).

::: statract.models.matching
