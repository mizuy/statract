# validation

Bootstrap internal validation and calibration. `validate_logistic` corresponds to `rms::validate.lrm`, `calibrate_logistic` to `rms::calibrate` (`lrm`), `validate_cox` to `rms::validate.cph`, and `calibrate_cox` to `rms::calibrate.cph(cmethod="KM")`. Formulas take `"y ~ age + sex"` or `"Surv(time, status) ~ age + sex"`, the same as `fit_glm` and `cox_ph`. If you pass R's resamples as `indices=` (B rows by n columns, 0-based row numbers after dropping missing values), you get the same values as rms. For numerical agreement with R, see [vs R](../../models/vs-r.md).

```python
from statract import calibrate_logistic, plot_calibration_curve, validate_logistic

validate_logistic(df, "y ~ age + sex", B=200, seed=1)
cal = calibrate_logistic(df, "y ~ age + sex", B=200, seed=1)
plot_calibration_curve(cal, "fig_calibration.png")
```

::: statract.models.validation
