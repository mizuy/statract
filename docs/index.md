<p align="center">
  <img src="assets/statract-logo.png" alt="Statract" width="360">
</p>

# statract

## 概要

推定、生存時間、回帰、統計図、費用効果分析を、Polars の表を渡して結果の表と図を受け取る一つのライブラリとしてつなぎます。

## 核

核は pandas を使いません。データと結果の表は Polars です。Kaplan–Meier は自前の `survival_curve` です。statsmodels は optional extra `sm`（`statract[sm]`）です。

## 最初の import

```python
import polars as pl
from statract import cox_ph, fit_ols, survival_curve
from statract.cea import simulate_cohort_markov

frame = pl.DataFrame(
    {
        "y": [1.2, 0.4, 2.1],
        "x": [0.0, 0.5, 1.0],
        "time": [0.4, 1.2, 0.8],
        "event": [1, 0, 1],
    }
)
fit_ols(frame, "y", ["x"])
survival_curve(frame, "time", "event")
cox_ph(frame, "Surv(time, event) ~ x")
```

`simulate_cohort_markov` は状態名と推移行列から、割引した費用と QALY を返します。引数の並びは [CEA の Quickstart](cea/quickstart.md) です。

解析パイプラインの出力ディレクトリと CSV companion は [studyloop](https://github.com/mizuy/studyloop) です（`statract.task_io` は shim）。

```python
from studyloop.task_io import prepare_task_output, save_frames
```

## 次に読む

| 節 | 中身 |
|----|------|
| [Table One](tableone/overview.md) | ベースライン表、集計、群間の検定 |
| [Models](models/overview.md) | 回帰、生存時間、混合、GAM、木、マッチング。[Quickstart](models/quickstart.md)、[R との対応](models/vs-r.md)、[速度](models/benchmarks.md) |
| [Figures](viz/overview.md) | forest、Kaplan–Meier、較正、funnel |
| [CEA](cea/overview.md) | cohort Markov、ICER、DSA / PSA |
| [解析例](examples/index.md) | 公開データで、入力から図まで |

出力ディレクトリと CSV companion の型は studyloop です。`statract.reporting` と `statract.task_io` はそこへの shim です。
