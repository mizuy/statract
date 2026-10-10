<p align="center">
  <img src="docs/assets/statract-logo.png" alt="Statract" width="420">
</p>

# statract

推定、生存時間、回帰、記述統計、統計図、費用効果分析（CEA）。データフレームは Polars。CSV companion や `out/` 準備は `statract.report.artifacts` / `statract.report.task_io`。

核は pandas に依存しない。Kaplan–Meier は同梱の `survival_curve`（lifelines は使わない）。statsmodels にも依存しない。

ドキュメントは [https://mizuy.github.io/statract/](https://mizuy.github.io/statract/) にある。入口は tableone、models、viz、cea、解析例です。

```bash
pip install statract
```

```python
import polars as pl
from statract import fit_glm, fit_ols, plot_forest, survival_curve
from statract.surv import cox_ph
from statract.cea import calculate_icers, simulate_cohort_markov
from statract.viz.misc import funnel_plot
```

R ブリッジは extra `r`（`statract.r`）。`import statract` では rpy2 を読まない。
