# 条件付き推論木の変数選択

対象: `src/statract/models/tree.py` の `_select_variables`（呼び出し元 `_grow`）。

## 1. 目的

条件付き推論木（conditional inference tree）の各ノードで、共変量ごとの無調整 p 値を多重比較調整し、分割を試す共変量の順番を決める。調整 p 値は `node.p_value` に入り、`ConditionalTree.tests()` に出る。R の `partykit::ctree()`（`ctree_control(testtype = "Bonferroni")`）と出力を比べる。

## 2. 出典

- Hothorn, T., Hornik, K. & Zeileis, A. (2006). Unbiased recursive partitioning: a conditional inference framework. *J Comput Graph Stat* 15:651–674, Algorithm 1 の step 1、sec. 3（変数選択）。
- Šidák, Z. (1967). Rectangular confidence regions for the means of multivariate normal distributions. *JASA* 62:626–633（調整式 $1-(1-p)^m$）。

## 3. モデルと式

ノードで検定した共変量を $j=1,\dots,m$、検定統計量（線形統計量の二次形式）を $c_j$、無調整 p 値を $p_j$ とする。調整 p 値は

$$\tilde p_j = 1-(1-p_j)^m.$$

これは Šidák 調整。ヘルプページでは "Bonferroni" と呼ぶ。選ぶ共変量は $\tilde p_j$ が最小のもの（Hothorn ら 2006, Algorithm 1 step 1）。$\min_j \tilde p_j \ge \alpha$ なら、そのノードは葉。

## 4. 計算の手順

1. $\tilde p_j = -\mathrm{expm1}(m\,\mathrm{log1p}(-p_j))$ で計算する。$p_j \ll 10^{-17}$ でも $\tilde p_j \approx m p_j > 0$ が保たれる。
2. 順位づけの基準は $\log \tilde p_j$。小さい順に並べる。
3. 基準がほぼ等しい組（相対差 $10^{-10}$ 以下、または両方 $-\infty$）は同順位とみなし、統計量 $c_j$ が大きい順に並べる。
4. $\tilde p_j < \alpha$ のものだけ残し、先頭から最大 `splittry` 個を返す。呼び出し元はこの順に分割を試し、分割できた最初の共変量を使う。

## 5. 数値上の注意

- $1-(1-p)^m$ をそのまま計算すると、$p < 10^{-17}$ で $(1-p)^m$ が 1 に丸まり、調整値が 0 になる。複数の共変量が 0 になると最初の列が選ばれてしまう。expm1/log1p を使う理由はこれ。
- 無調整 p 値そのものが 0 に下溢した場合は $\log\tilde p=-\infty$ の同順位になり、統計量で決まる。
- 同順位の許容誤差 $10^{-10}$（相対）は自分で決めた値。テストで決まる値ではない。

## 6. R と出力を合わせるための公開仕様

`ctree_control` のヘルプページの既定値: `alpha = 0.05`、`minsplit = 20`、`minbucket = 7`、`minprob = 0.01`、`splittry = 2`、`testtype = "Bonferroni"`（中身は Šidák 型の調整）。

## 7. 検証

- `tests/models/test_tree_format.py::test_format_and_terminal_count`: 数値と因子の両方で分割すること。
- `::test_tiny_p_values_pick_the_smallest`: 両方の p 値が下溢域に近いとき、調整 p 値が正で、効果の強い共変量が選ばれること。
- R の fixture との数値比較テストはない。

## 8. R と違うところ、未確認の点

- 同順位の扱い（相対 $10^{-10}$ で統計量に移る）は R と同じか確かめていない。
