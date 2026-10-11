# 多重比較の p 値調整

対象: `src/statract/models/htest.py` の `p_adjust` と補助関数 `_hommel_sorted`。

## 1. 目的

$m$ 個の p 値を、多重比較の手法ごとに調整した p 値に変える。入力と同じ長さと順序で返し、NaN は NaN のまま。R の `p.adjust()` と出力を比べる。

## 2. 出典

- Bonferroni: Dunn, O. J. (1961). Multiple comparisons among means. *JASA* 56:52–64（Bonferroni 不等式の適用）。
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand J Stat* 6:65–70.
- Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple tests of significance. *Biometrika* 75:800–802.
- Hommel, G. (1988). A stagewise rejective multiple test procedure based on a modified Bonferroni test. *Biometrika* 75:383–386.
- Simes, R. J. (1986). An improved Bonferroni procedure for multiple tests of significance. *Biometrika* 73:751–754.
- Wright, S. P. (1992). Adjusted p-values for simultaneous inference. *Biometrics* 48:1005–1013（調整 p 値の定義と Hommel の調整 p 値）。
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57:289–300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Ann Stat* 29:1165–1188, Theorem 1.1.

## 3. モデルと式

昇順に並べた p 値を $p_{(1)}\le\dots\le p_{(m)}$、仮説の総数を $n\ (\ge m)$ とする。調整 p 値 $\tilde p$ は「その仮説を棄却できる最小の有意水準」（Wright 1992）。

- Bonferroni: $\tilde p_{(i)}=\min(1,\,n\,p_{(i)})$。
- Holm（ステップダウン）: $\tilde p_{(i)}=\min\bigl(1,\ \max_{j\le i}(n-j+1)p_{(j)}\bigr)$。
- Hochberg（ステップアップ）: $\tilde p_{(i)}=\min\bigl(1,\ \min_{j\ge i}(n-j+1)p_{(j)}\bigr)$。
- BH（"BH"、別名 "fdr"）: $\tilde p_{(i)}=\min\bigl(1,\ \min_{j\ge i}\tfrac{n}{j}p_{(j)}\bigr)$。
- BY: $q=\sum_{k=1}^{n}1/k$ として $\tilde p_{(i)}=\min\bigl(1,\ \min_{j\ge i}\tfrac{q\,n}{j}p_{(j)}\bigr)$。
- Hommel: 閉検定手順で、各交差仮説を Simes 検定で検定する。大きさ $s$ の集合 $I$ の Simes p 値は $p_I=\min_{k}\, s\,p_{(k:I)}/k$（$p_{(k:I)}$ は $I$ の中で $k$ 番目に小さい値）。仮説 $H_i$ の調整 p 値は $\tilde p_i=\max_{I\ni i}p_I$。

## 4. 計算の手順

1. method を確かめ、"fdr" を "BH" に読み替える。
2. NaN でない値を取り出し、数 $m$ を数える。$n$ の既定は $m$。
3. $n\le1$ なら入力の float コピーをそのまま返す。$n=2$ の Hommel は Hochberg と同じなので Hochberg で計算する。
4. 安定ソート（同じ値は入力順を保つ）で昇順に並べる。
5. 3 節の式を、累積最大（Holm）または後ろからの累積最小（Hochberg、BH、BY）で一度に計算する。
6. Hommel は、欠けた $n-m$ 個の仮説を $p=1$ として末尾に足し、長さ $n$ で計算してから先頭の $m$ 個を使う。
7. 元の順序に戻し、NaN の位置に NaN を戻す。

### Hommel の計算（自分で導いた形）

Simes p 値は各成分について単調非減少。だから $H_{(i)}$ を含む大きさ $s$ の集合のうち $p_I$ が最大になるのは、次のどちらか。

- $H_{(i)}$ が大きい方から $s$ 個に入るとき: その $s$ 個そのもの。値は $c_s=\min_{k=1}^{s} s\,p_{(n-s+k)}/k$。
- 入らないとき: $H_{(i)}$ と大きい方から $s-1$ 個。$p_{(i)}\le p_{(n-s+1)}$ なので、値は $\min(s\,p_{(i)},\,c_s)$ に等しい。

そこで $\tilde p_{(i)}$ を $p_{(i)}$（$s=1$）で初期化し、$s=n,n-1,\dots,2$ について上の値との最大をとる。計算量は $O(n^2)$。

## 5. 数値上の注意

- すべての手順で 1 を上限とする。
- 同じ値の p は安定ソートで入力順に並べる。累積最大・最小をとるので、結果は同じ値どうしで等しくなる。
- $n>m$ で Holm・Hochberg・BH・BY の乗数には $n$ を使う。足した $p=1$ の仮説は上限 1 に吸収されるので、実際に足す必要はない（Hommel だけ足す）。
- Hommel の $O(n^2)$ は数千個まで実用的。

## 6. R と出力を合わせるための公開仕様

`?p.adjust` より。

- `method` は "holm"（既定）、"hochberg"、"hommel"、"bonferroni"、"BH"、"BY"、"fdr"、"none"。
- `n` の既定は NA でない p 値の数。それより小さい `n` は許さない。
- `n` が NA でない値の数より大きいとき、欠けた仮説は p 値が 1 だったものとして扱う。
- エラー文: `f"method must be one of {P_ADJUST_METHODS}"`、"n must be at least the number of non-missing p values"。

## 7. 検証

- `tests/r_oracle/test_htest.py::test_p_adjust[<method>-n<k>]`（18 件）。fixture `tests/r_oracle/fixtures/htest.json` の `p_adjust`（R 4.3）。
- 入力の長さより大きい $n$、2 個と 3 個の短い入力（`[0.03, 0.04]`、`[0.04, 0.01, 0.04]`）を含む。
- 許容誤差 rtol $10^{-10}$。すべて通過。

## 8. R と違うところ、未確認の点

- 知られている違いはない。
- 大きい $n$ での Hommel の速度は測っていない。
