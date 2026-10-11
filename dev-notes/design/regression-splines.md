# 回帰スプライン基底（bs、ns、rcs のノット）

## 1. 目的

式の中の `bs()`、`ns()`、`rcs()` の計画行列を作る。比較相手は R の公開関数 `splines::bs()`、`splines::ns()`、`rms::rcs()`。

- `_bs_basis`: B スプライン基底。境界ノットの外は外挿する。
- `_ns_basis`: 自然 3 次スプライン基底。R と列ごとに一致させる（列空間が同じだけでは足りない）。
- `_rcs_knots`: 制限付き 3 次スプライン（restricted cubic spline、RCS）の既定ノット位置。基底そのもの（`_rcs_basis`）は既存。

実装: `src/statract/models/splines.py`。

## 2. 出典

- de Boor C (2001). *A Practical Guide to Splines*, rev. ed. Springer. ch. IX（B スプラインの漸化式）、ch. X（微分）。
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, 2nd ed. sec. 5.2.1（自然 3 次スプライン）、ch. 5 付録（B スプライン）。
- Green PJ, Silverman BW (1994). *Nonparametric Regression and Generalized Linear Models*. ch. 2.
- Harrell FE (2015). *Regression Modeling Strategies*, 2nd ed. sec. 2.4.5–2.4.6, Table 2.3.
- Stone CJ, Koo CY (1985). Additive splines in statistics. *Proc. Stat. Comp. Sect. ASA*, 45–48.

## 3. モデルと式

### B スプライン

次数 $d$、階数 $m = d + 1$。境界ノット $a < b$ を $m$ 回ずつ繰り返し、内部ノットを挟んだノット列 $t$ を作る。基底 $B_1, \dots, B_K$ は de Boor の漸化式で定まる。$K = (\text{内部ノット数}) + m$。

$[a, b]$ の外では、近い方の境界 $c \in \{a, b\}$ でのテイラー多項式で延長する。

$$B_j(x) = \sum_{r=0}^{d} \frac{B_j^{(r)}(c)}{r!} (x - c)^r .$$

切片なし（`intercept = FALSE`）のときは第 1 列を落とす。

### 自然 3 次スプライン

3 次 B スプライン $B(x) \in \mathbb{R}^{K}$（切片なしなら第 1 列を落とした $K-1$ 本）を使う。境界の外では値と 1 階微分で線形に延長する。

$$B_j(x) = B_j(c) + B_j'(c)(x - c).$$

制約行列 $C$（$K \times 2$）の列は境界 $a$、$b$ での 2 階微分 $B''(a)$、$B''(b)$。自然スプラインは $C^\top \beta = 0$ を満たす係数の空間。$C$ の完全な Householder QR 分解を $C = QR$ とする。$Q = [Q_1, Q_2]$（$Q_1$ は最初の 2 列）。基底は

$$N(x) = B(x)^\top Q_2 .$$

$\beta = Q_2 \gamma$ は制約を満たす。$Q_2$ は $C$ の列の直交補空間の正規直交基底。

### RCS の既定ノット

ノット数 $k$ に対して、外側の裾確率 $p_0$ を $k = 3$ で 0.10、$4 \le k \le 6$ で 0.05、$k \ge 7$ で 0.025 とする。確率 $p_0, \dots, 1 - p_0$ を等間隔に $k$ 個とり、標本分位点（R の type 7、線形補間）をノットにする（Harrell 2015, Table 2.3）。

## 4. 計算の手順

### `_bs_basis`

1. ノット列を作る。
2. $[a, b]$ 内の点は B スプラインの値をそのまま使う（SciPy の `BSpline`）。
3. 外の点は境界での 0–$d$ 階微分を求め、上のテイラー多項式で計算する。
4. 切片なしなら第 1 列を落とす。

### `_ns_basis`

1. 3 次のノット列で B スプライン基底を作る。外の点は線形延長する。
2. 境界での 2 階微分から $C$ を作る。切片なしなら $B$ と $C$ の第 1 行（基底の第 1 本）を落とす。
3. `numpy.linalg.qr(C, mode="complete")` で $Q$ を得て、$B Q_2$ を返す。

### `_rcs_knots`

入力は欠測を除いた値。$n$ は個数。

1. $k < 3$ ならエラー。$n < 6$ ならエラー。
2. 相異なる値が少ない場合（相異なる値の数 $- 2 \le k$）は、最小値と最大値を除いた相異なる値をすべてノットにする。
3. 端の同順位（割合 `fractied` = 0.05）。内部のどの値も 5 % に達せず、最小値が 5 % 以上を占めるとき、最初のノットは内側の次の相異なる値にする。最大値側も同じ。
4. 残りのノットは、固定した端ノットより厳密に内側の観測で置く。数は $k$ から固定した端の数を引いたもの。確率は等間隔で、裾確率 $p_0$ は元の $k$ で決める。残りが 1 個なら中央値。
5. $n < 100$ のとき、同順位で固定していない側の外側ノットを、小さい方から 5 番目または大きい方から 5 番目の観測に置き換える（Harrell 2015, sec. 2.4.6）。
6. 相異なるノットが $\min(k, 3)$ 個未満ならエラー（ノットを直接渡すよう促す）。

## 5. 数値上の注意

- 境界ちょうどの点は区間内として扱う。右端 $b$ でも B スプラインの値が正しく出る（SciPy は最後の非空区間を使う）。
- 境界での微分は区間の内側から評価する。境界ノットの重複があるので外側の区間は空。
- QR の符号: Householder の標準規約（$R$ の対角は $-\mathrm{sign}(a_{jj})\lVert a_j \rVert$）。LAPACK と LINPACK はこの規約で、NumPy も同じ。符号が違うと列の符号が変わり、係数の比較で落ちる。
- 分位点は既存の `_quantile7`（R の type 7）を使う。

## 6. R と出力を合わせるための公開仕様

- `?splines::bs`: 境界ノットの外は外挿する。既定の境界ノットはデータの範囲。`df` から内部ノットを分位点で置く。
- `?splines::ns`: 境界で 2 階微分 0、外では線形。
- `?rms::rcs`、`?Hmisc::rcspline.eval`: 既定のノット数 5、裾確率の表、$n < 100$ で外側ノットを 5 番目の観測に置く規則、`fractied` の既定 0.05。
- 4 の手順 4 の裾確率をどの $k$ で決めるかはヘルプに書かれていない。fixture（splines_c の `rcs(x, 4)`）と比べて、元の $k$ を使う方だけが一致した。

## 7. 検証

- `tests/r_oracle/test_splines.py`（fixture `fixtures/splines.json`、スクリプト `scripts/splines.R`）
  - `test_model_matrix`: 計画行列 rtol 1e-10、atol 1e-12。係数 rtol 1e-6。
  - `test_new_rows_keep_knots`: 新しい行（境界の外を含む）rtol 1e-10、atol 1e-12。
  - `test_glm_with_splines`、`test_cox_with_splines`、`test_anova_rows`、`test_contrast`、`test_spline_errors`。
- `tests/viz/test_effect.py::test_plot_spline_effect`。
- 全 34 件が通る。許容誤差は変えていない。

## 8. R と違うところ、未確認の点

- $n < 100$ で同順位の規則も働く場合（片側だけ 5 番目の観測に置き換える）を確かめる fixture はない。
- 相異なる値がとても少ないときの細かい挙動は、仕様の文言どおりに実装しただけで R と比べていない。
