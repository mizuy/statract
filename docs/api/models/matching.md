# matching

Matching on propensity scores and distances. The public name is `match_sample`. `total_distance` is the sum of the distances of the matched edges. Weights for full matching follow MatchIt's `normalize=TRUE`.

## Implementation notes: SMD { #smd }

`smd_all` and `smd_matched` in `balance()` are the same SMD as in the weighting balance table. The default denominator is the pooled SD of the full sample before matching, $\sqrt{(s_1^2 + s_0^2)/2}$, and the same value is used before and after matching. Since the denominator is fixed, a change in SMD reflects a change in the mean difference. Binary variables are standardized with the variance $p(1-p)$ (Austin 2009). `pair_distance` is divided by the same denominator. The `distance` row is always standardized.

MatchIt uses the SD of the treated group as the denominator for ATT. In ATT matching the treated group stays almost intact, so this gives a fixed scale in the units of the target population. You can reproduce it with `balance(sd_denominator="treated")` (`"control"` for ATC). To report binary variables as raw differences in proportions, as cobalt does, use `binary="raw"`. tableone recomputes the pooled SD within the matched table, so the SMD in a Table 1 after matching may not match `smd_matched`. The reasons and a mapping table are in the [weighting implementation notes](weighting.md#smd).

::: statract.models.matching
