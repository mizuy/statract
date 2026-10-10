# validation

ブートストラップによる内的妥当性と較正です。`validate_logistic` は `rms::validate.lrm`、`calibrate_logistic` は `rms::calibrate`（`lrm`）、`validate_cox` は `rms::validate.cph`、`calibrate_cox` は `rms::calibrate.cph(cmethod="KM")` に対応します。式は `fit_glm` と `cox_ph` と同じく `"y ~ age + sex"` や `"Surv(time, status) ~ age + sex"` を受けます。R の再標本を `indices=`（B 行 n 列、欠測を除いた行の 0 始まり番号）で渡すと rms と同じ値になります。R との数値対応は [vs R](../../models/vs-r.md) です。

```python
from statract import calibrate_logistic, plot_calibration_curve, validate_logistic

validate_logistic(df, "y ~ age + sex", B=200, seed=1)
cal = calibrate_logistic(df, "y ~ age + sex", B=200, seed=1)
plot_calibration_curve(cal, "fig_calibration.png")
```

::: statract.models.validation
