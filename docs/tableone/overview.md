# tableone — 概要

ベースライン表です。`tableone` が Polars の表から great-tables の Table 1 を返し、`tableone_raw` が同じ中身の DataFrame を返します。集計は `statract.agg`、群間の検定は `statract.stat` です。

| 層 | モジュール | 役割 |
|----|------------|------|
| 表 | `statract.tableone` | `tableone`、`tableone_raw`、`tableone_gt_from_frame` |
| 集計 | `statract.agg` | 平均、中央値、割合、カテゴリ度数 |
| 検定 | `statract.stat` | カイ二乗、Fisher、ANOVA、Kruskal–Wallis、SMD、オッズ比。`add_pvalue=True` と `add_smd=True` が使う |

`hue` が群の列です。カテゴリの子行は名前の先頭が半角スペース 2 つで、表示では全角スペースに置き換わります。`pl.Enum` の水準は度数が 0 でも行に残します。

```python
import polars as pl
from statract import agg_category_n, tableone

frame = pl.DataFrame(
    {
        "group": ["A", "A", "B"],
        "category": pl.Series(["X", "Y", "X"], dtype=pl.Enum(["X", "Y", "Z"])),
    }
)
tableone(
    frame,
    {"Category": ("category", agg_category_n)},
    hue="group",
    add_all=True,
)
```

`add_pvalue=True` は群間の P 値の列を足します。`statfunc` を省くと、カテゴリは Fisher、中央値の集計（`agg_median_iqr` など）は Kruskal–Wallis、それ以外の数値は ANOVA です。`statfunc="kruskal"` のように名前でも選べます。`hue` に列を複数渡すと、表の列と同じ組み合わせの群で検定します。2×2 より大きい表の Fisher はモンテカルロで、seed を固定しているので同じ表は同じ P 値になります。

`add_smd=True` は R の tableone と同じ標準化差（SMD）の列を足します。数値は 2 群の平均差を標本分散の平均の平方根で割ったもの、カテゴリは Yang と Dalton の多項版です。3 群以上では全ペアの平均です。単体の関数は `standardized_difference` です。

公開データでの表は [解析例](../examples/index.md) の各ページにあります。API は [tableone](../api/tableone/tableone.md)、[agg](../api/tableone/agg.md)、[stat](../api/tableone/stat.md) です。推定は [models](../models/overview.md) です。
