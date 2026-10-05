<p align="center">
  <img src="docs/assets/statract-logo.png" alt="Statract" width="420">
</p>

# statract

推定、生存時間、回帰、記述統計、統計図、費用効果分析（CEA）。データフレームは Polars。

核は pandas に依存しない。Kaplan–Meier は同梱の `survival_curve`（lifelines は使わない）。statsmodels は optional extra `sm`。

```bash
pip install statract
pip install "statract[sm]"   # OLS、割合の信頼区間、probit
```

```python
import polars as pl
from statract import fit_glm, fit_ols, plot_forest, survival_curve
from statract.surv import cox_ph
from statract.cea import calculate_icers, simulate_cohort_markov
from statract.figure import funnel_plot
from statract.task_io import prepare_task_output
```

R ブリッジは extra `r`（`statract.r`）。`import statract` では rpy2 を読まない。
