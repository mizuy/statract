# マッチングの距離（Mahalanobis 距離と尺度つき Euclid 距離）

対象: `src/statract/models/matching/__init__.py` の `_binary_pooled_sd`、`_generalized_inverse`、`_pivoted_chol_coordinates`、`_mahalanobis_coordinates`（呼び出し元 `_distance_vector` → `match_sample(..., distance=...)`）。

## 1. 目的

最近傍マッチングの距離行列を、座標の Euclid 距離として表す。

- `distance="mahalanobis"`: 座標の Euclid 距離が Mahalanobis 距離になるように変換する。
- `distance="scaled_euclidean"`: 各列を群内プールの標準偏差（standard deviation, SD）で割る。0/1 の列は割合の形の分散を使う。

R の `MatchIt::matchit(distance = "mahalanobis")`（MatchIt 4.5.5）のペアと比べる。

## 2. 出典

- Rubin, D. B. (1980). Bias reduction using Mahalanobis-metric matching. *Biometrics* 36:293–298。
- Stuart, E. A. (2010). Matching methods for causal inference: a review and a look forward. *Stat Sci* 25:1–21, sec. 2.2（距離の定義、プールした分散）。
- Penrose, R. (1955). A generalized inverse for matrices. *Proc Camb Phil Soc* 51:406–413。
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed., sec. 4.2.8–4.2.9（ピボットつき Cholesky、Alg. 4.2.4）、sec. 5.5.4（数値ランク）。

## 3. モデルと式

共変量ベクトルを $x_i\in\mathbb R^k$、処置群を $g_i\in\{0,1\}$ とする。

- Mahalanobis 距離: $d(i,j)^2 = (x_i-x_j)'S^{-}(x_i-x_j)$。$S$ は群内プール共分散
  $$S = \frac{1}{n-G}\sum_{g}\sum_{i\in g}(x_i-\bar x_g)(x_i-\bar x_g)'$$
  （$G$ は群の数）。$S$ が特異なら $S^{-}$ は Moore–Penrose 逆行列。
- 半正定値行列 $A$ のピボットつき Cholesky: 上三角 $R$ と置換 $P$ で $R'R = P'AP$。列を元の順に戻した $U = RP'$ は $U'U=A$。座標 $z_i = U x_i$ なら $\|z_i-z_j\|^2 = (x_i-x_j)'A(x_i-x_j)$。
- 0/1 の列の群内平方和: 群 $g$ の大きさ $n_g$、割合 $p_g$ で $n_g p_g(1-p_g)$。プール SD は $\sqrt{\sum_g n_g p_g(1-p_g)/(n-G)}$。

## 4. 計算の手順

Mahalanobis:

1. 欠測のない行だけで計算する（欠測のある行は座標も欠測）。
2. 各列を全体の平均で中心化し、SD（除数 $n-1$）で割る（MatchIt の `distance` ヘルプページの記述どおり、R の `scale` と同じ）。
3. 群ごとに中心化した交差積を足し、$n-G$ で割って $S$ を作る。和は拡張精度（`np.longdouble`）で取る。
4. $S$ の Moore–Penrose 逆行列を SVD で求める。最大特異値の $\sqrt{\varepsilon}$ 倍以下の特異値は 0 とみなす。正則なら通常の逆行列と（丸め誤差の範囲で）同じ。
5. $S^{-}$ をピボットつき Cholesky（LAPACK `dpstrf`）で分解する。ランク $r$ より後ろの行は 0 にする。列を元の順に戻して $U$ を作り、座標 $X U'$ を返す。

0/1 列のプール SD:

1. 群ごとに $n_g p_g(1-p_g)$ を足し、$n-G$ で割って平方根を取る。連続の列と同じ除数なので、0/1 列でも連続列の式と同じ値になる。この除数は自分で決めた（テストはない）。
2. 0 や非有限の値は呼び出し元 `_pooled_sd` が 1 に置き換える。

## 5. 数値上の注意

- 因子は、距離が MatchIt の出力と合うように、すべての水準の指示変数を $1/\sqrt2$ 倍して表す（この符号化は出力に合わせて選んだ）。列の和が一定になるので $S$ は特異になる。一般化逆行列はこのため。
- 最近傍のペアは $S$ の最後の桁の違いで変わることがある。R の `sum` のヘルプページは、可能なら拡張精度で累積すると書いている。それに合わせて和を `longdouble` で取る。
- 一般化逆行列の切り捨て閾値 $\sqrt{\varepsilon}$ は、R の `MASS::ginv()` のヘルプページに書かれた既定値 `tol = sqrt(.Machine$double.eps)` と同じ。
- `dpstrf` はランク落ちのとき後ろのブロックを未定義のまま返すので、0 に置き換える。

## 6. R と出力を合わせるための公開仕様

- MatchIt の `distance` ヘルプページ: Mahalanobis 距離は共変量を標準化し、群内プール共分散を使う。
- R の `?chol`: `chol(A, pivot = TRUE)` の結果の列を `order(attr(., "pivot"))` で並べ直すと、交差積が $A$ になる因子が得られる。

## 7. 検証

- `tests/models/test_estimators.py::test_mahalanobis_accepts_a_factor`。
- `::test_mahalanobis_pairs_match_matchit_455`: 24 行のデータ、特異な共分散。MatchIt 4.5.5（R 4.3.3）の 1 回の実行で得たペアと完全一致。
- `::test_matrix_distance_rejects_subclass_matching`: 行列の距離で subclass マッチングは ValueError。

## 8. R と違うところ、未確認の点

- 0/1 列のプール SD の除数はテストがなく、R と同じかは確かめていない。
- ペアの一致は 1 つのデータで確かめただけ。ほかのデータで最後の桁の違いからペアが変わる可能性は残る。
