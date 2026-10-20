# diagnostic

Diagnostic accuracy of a binary test against a reference standard. It gives sensitivity, specificity, positive and negative predictive values, likelihood ratios, the diagnostic odds ratio, the Youden index, and more, each with a confidence interval. It matches `epiR::epi.tests`. Choose the interval for proportions with `method=`. The default is the Wilson score interval. `"exact"` (Clopper–Pearson), `"agresti"`, and `"jeffreys"` are also available. Likelihood ratios and the diagnostic odds ratio use normal-approximation intervals on the log scale.

```python
from statract import diagnostic_accuracy

acc = diagnostic_accuracy(df["risk"] >= 0.05, df["bleed"])  # test, reference standard (0/1 or bool)
acc.table()                                                  # 2×2 table
acc.tidy()                                                   # estimate and interval for each statistic
acc.predictive_values([0.01, 0.05, 0.20])                    # PPV and NPV at other prevalences
diagnostic_accuracy(tp=23, fp=259, fn=12, tn=706, method="exact")
```

How the `statistic` names in `tidy()` map to epiR names:

| statract | epiR | Meaning |
|----------|------|------|
| `apparent_prevalence` | `ap` | Proportion test positive |
| `prevalence` | `tp` | Prevalence |
| `sensitivity` / `specificity` | `se` / `sp` | Sensitivity / specificity |
| `accuracy` | `diag.ac` | Diagnostic accuracy |
| `diagnostic_or` | `diag.or` | Diagnostic odds ratio |
| `nnd` | `nndx` | Number needed to diagnose (1 / Youden index) |
| `youden` | `youden` | Youden index |
| `ppv` / `npv` | `pv.pos` / `pv.neg` | Positive predictive value / negative predictive value |
| `lr_pos` / `lr_neg` | `lr.pos` / `lr.neg` | Positive likelihood ratio / negative likelihood ratio |
| `rule_out` / `rule_in` | `p.rout` / `p.rin` | Proportion test negative / positive |
| `false_positive_rate` / `false_negative_rate` | `p.tpdn` / `p.tndp` | False positive rate / false negative rate |
| `false_discovery_rate` / `false_omission_rate` | `p.dntp` / `p.dptn` | Proportion without disease among test positives / proportion with disease among test negatives |

::: statract.models.diagnostic
