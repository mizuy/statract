<p align="center">
  <img src="docs/assets/statract-logo.png" alt="Statract" width="420">
</p>

# statract

Polars の表を渡して、結果の表と図を受け取る。ベースライン表、回帰、生存時間、統計図、費用効果分析（CEA）を、一つのライブラリでつなぎます。

- 核は pandas、statsmodels、lifelines に依存しません。Kaplan–Meier は同梱の `survival_curve` です。
- R ブリッジ `statract.r` は extra `r` で入ります。`import statract` だけでは R も rpy2 も読み込みません。

## インストール

Python 3.13 以上が必要です。PyPI にはまだ公開していないので、GitHub から入れます。

```bash
pip install "git+https://github.com/mizuy/statract"
# R ブリッジも使う場合
pip install "statract[r] @ git+https://github.com/mizuy/statract"
```

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

## ドキュメント

[https://mizuy.github.io/statract/](https://mizuy.github.io/statract/)

| セクション | 内容 |
|------------|------|
| [Guide](https://mizuy.github.io/statract/) | 手法の選び方、Table One、Models、Figures、Cost-effectiveness |
| [Learn](https://mizuy.github.io/statract/learn/) | 臨床の問いから始める統計（因果推論、予測、統計モデリング、生存時間） |
| [Examples](https://mizuy.github.io/statract/examples/) | 公開データを使った通しの解析例 |
| [Benchmarks](https://mizuy.github.io/statract/models/benchmarks/) | 速度と、R との数値の一致 |
| [Reference](https://mizuy.github.io/statract/api/models/fit/) | API |

## ライセンス

MIT License（[LICENSE](LICENSE)）。ドキュメントのビルドに使う第三者のファイルは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
