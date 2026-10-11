# stat

## 実装メモ: SMD

`standardized_difference`（`tableone(..., add_smd=True)` の SMD 列）は R の tableone と同じ式です。数値は $|\bar x_1 - \bar x_0| / \sqrt{(s_1^2 + s_0^2)/2}$ で、分散は標本分散です。カテゴリは Yang と Dalton (2012) の多項版で、2 水準では割合 $p$ の分散 $p(1-p)$ を使う式になります（Austin 2009）。

分母は渡された表の中で毎回計算します。調整前の標本なら、`match_sample(...).balance()` の `smd_all` や `propensity_weights(...).balance()` の `diff_unadjusted` と一致します（符号は除きます）。マッチ後の標本で作った Table 1 は分母をマッチ後の分散で計算し直すので、分母を調整前に固定するバランス表の `smd_matched` / `diff_adjusted` と一致しないことがあります。バランスの確認にはバランス表を使ってください。理由は [weighting の実装メモ](../models/weighting.md#smd) にあります。

0/1 の数値の列は集計関数で扱いが決まります。平均などの数値の集計では連続変数として標本分散（`ddof=1`）を使います。R の tableone で `factorVars` に入れない場合と同じです。カテゴリの集計（`agg_category_n` など）か `categorical=True` では $p(1-p)$ を使い、バランス表の 2 値と一致します。2 つの分散の違いは各群の $n/(n-1)$ の因子だけです。

::: statract.tableone.stat

