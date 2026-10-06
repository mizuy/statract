<p align="center">
  <img src="assets/statract-logo.png" alt="Statract" width="360">
</p>

# statract

Polars の表を渡して、結果の表と図を受け取る。ベースライン表、回帰、生存時間、統計図、費用効果分析を、一つのライブラリでつなぎます。

```mermaid
flowchart LR
  D[("Polars DataFrame")] --> T["Table One<br/>ベースライン表"]
  D --> M["Models<br/>回帰・生存時間・混合・GAM・マッチング"]
  D --> C["Cost-effectiveness<br/>Markov・ICER・DSA / PSA"]
  M --> F["Figures<br/>forest・KM・較正・DCA"]
  C --> F
  T --> O[("表 (Polars / CSV)・図")]
  M --> O
  F --> O
  C --> O
```

## 何ができるか

| 領域 | できること | 主な入口 |
|------|------------|----------|
| [Table One](tableone/overview.md) | ベースライン表、集計、群間の検定 | `tableone` |
| [Models](models/overview.md) | 線形・一般化線形、Cox / AFT / Fine–Gray、混合、GAM、木、マッチング、予測の評価 | `fit_ols`、`fit_glm`、`cox_ph`、`fit_mixed`、`match_sample` |
| [Figures](viz/overview.md) | forest、Kaplan–Meier、較正、決定曲線、funnel | `plot_forest`、`plot_survival` |
| [Cost-effectiveness](cea/overview.md) | cohort Markov、ICER、DSA / PSA | `simulate_cohort_markov`、`calculate_icers` |

## 動作環境と依存ライブラリ

**Python 3.13 以上**が必要です。

```bash
pip install statract
```

`pip install statract` で次が入ります。

| ライブラリ | 用途 |
|------------|------|
| [Polars](https://pola.rs/) | 入力のデータと結果の表 |
| [NumPy](https://numpy.org/) | 数値計算の核 |
| [SciPy](https://scipy.org/) | 分布、検定、最適化 |
| [Numba](https://numba.pydata.org/) | 生存時間・マッチング・回帰の内側のループの高速化 |
| [lme-python](https://pypi.org/project/lme-python/) | 線形混合・一般化線形混合（`fit_mixed` の既定エンジン） |
| [Matplotlib](https://matplotlib.org/) | forest、Kaplan–Meier、較正などの図 |
| [japanize-matplotlib](https://pypi.org/project/japanize-matplotlib/) | 図の日本語フォント（Python 3.12 以降は `setuptools` も使う） |
| [Plotly](https://plotly.com/python/) | 対話的な補助の図 |
| [Pillow](https://python-pillow.org/) | Plotly の図をつなげて 1 枚の画像にする |
| [great_tables](https://posit-dev.github.io/great-tables/) | Table One の表示 |

必要なときだけ入れる extra です。

| extra | 入るもの | 用途 |
|-------|----------|------|
| `statract[r]` | rpy2、rpy2-arrow | R ブリッジ `statract.r`。R 本体と使う R パッケージは別に入れる |
| `statract[gpboost]` | gpboost、pandas | 実験的な `glmm_gpboost` |

```bash
pip install "statract[r]"
```

pandas、statsmodels、lifelines、R は既定では使いません。`import statract` だけでは R も rpy2 も読み込みません。

## 最小の例

```python
import polars as pl
from statract import cox_ph, fit_ols, survival_curve

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

## どこから読むか

| 目的 | ページ |
|------|--------|
| 回帰や生存時間を動かしたい | [Models](models/overview.md)（Quickstart つき）、式の書き方は [Wilkinson 式](models/formula.md) |
| 論文用の図を作りたい | [Figures](viz/overview.md) |
| 費用効果分析をしたい | [Cost-effectiveness](cea/overview.md)（Quickstart つき） |
| 入力から図までの通しの例を見たい | [解析例ギャラリー](examples/index.md) |
| R との数値の対応や速度を確認したい | [Models vs R](models/vs-r.md)、[Benchmarks](models/benchmarks.md) |
| 関数の引数を調べたい | [API](api/models/fit.md) |

## 設計の前提

- 核は pandas を使いません。データと結果の表は Polars です。Kaplan–Meier は自前の `survival_curve` で、lifelines は使いません。
- 解析パイプラインの出力ディレクトリと CSV companion は `statract.reporting` と `statract.task_io` が担当します。

```python
from statract.task_io import prepare_task_output, save_frames
```
