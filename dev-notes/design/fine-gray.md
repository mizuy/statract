# Fine–Gray の部分分布ハザード用データ展開

## 1. 目的

競合リスクのデータを、重み付き計数過程の行に展開する。展開後に通常の重み付き Cox モデル `cox_ph(out, "Surv(fgstart, fgstop, fgstatus) ~ ...", weights="fgwt")` を当てはめると、Fine–Gray の部分分布ハザードモデルの推定になる。出力を比べる相手は R の `survival::finegray()`。

実装: `src/statract/surv/fine_gray.py` の `fine_gray`。

## 2. 出典

- Fine, J. P. & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA* 94, 496–509. 3 節（打ち切りの逆確率重み付きスコア、式 (6)–(7)）。
- Geskus, R. B. (2011). Cause-specific cumulative incidence estimation and the Fine and Gray model under both left truncation and right censoring. *Biometrics* 67, 39–49. 2 節（重み付きデータセットとしての書き換え）。

## 3. モデルと式

注目する原因を $c$ とする。部分分布ハザード $\lambda_c(t \mid Z) = \lambda_{c0}(t) e^{\beta^\top Z}$。他の原因で $T_i$ に失敗した行は $T_i$ 以降もリスク集合に残り、重み

$$w_i(t) = \frac{\hat G(t-)}{\hat G(T_i-)}, \quad t > T_i$$

を持つ（Fine & Gray 1999, 式 (6)）。$\hat G$ は打ち切り分布の Kaplan–Meier 推定。打ち切りの行を「事象」とし、同時刻では失敗が打ち切りより先（その時刻に失敗した行は打ち切りのリスク集合にいない）。

## 4. 推定・計算の手順

1. 事象コードを整数にする（null/NaN は 0、真偽値は 0/1、数値文字列は数値）。`cause` が文字列なら数値に直して比べる。
2. $\hat G$ を全行で計算する（遅延エントリがあれば $s_i < u$ の行だけがリスク集合）。$\hat G$ の折れ点は打ち切り時刻と最大観察時刻。
3. 打ち切りの行と原因 $c$ で失敗した行は 1 行のまま（重み 1）。
4. 他の原因で $T$ に失敗した行は、自分の区間を $T$ 以上の最初の折れ点まで延ばし（重み 1、$\hat G$ はその間一定）、その後は隣り合う折れ点の間 $(b_j, b_{j+1}]$ を続きの行にする。重みは $\hat G(b_j)/\hat G(T-)$。区間内に原因 $c$ の失敗時刻がない続きの行は部分尤度に寄与しないので出力しない。
5. 出力列: 入力の列から時刻・事象・エントリ列を除いたもの（元の順）、次に `fgstart`、`fgstop`、`fgstatus`（原因 $c$ なら 1）、`fgwt`。行順は入力行ごとに自分の行、続きの行の順。

## 5. 数値上の注意

- エントリがないとき開始は 0。最小時刻が 0 以下なら共通の開始を $2\min - 1$ にする（`?finegray` に合わせた）。
- $\hat G(T-)$ は $T$ より前の打ち切り時刻までの積。$T$ ちょうどに打ち切りがあれば、その打ち切りは分母に入らない。

## 6. R と出力を合わせるための公開仕様

- `finegray` のヘルプと survival の "compete" ビネット: 重みは打ち切り分布の KM、続きの行は打ち切り分布の変化点で切る。
- 引数の扱いとエラーメッセージは仕様書どおり（式の形と列名の形）。

## 7. 検証

- `tests/r_oracle/test_cox_inference.py::test_fine_gray_uses_the_row_robust_variance`（fixture の `fine_gray`、データ `fg_a.csv`、300 行・同順位多数）: 展開後の行数が R と一致、係数 rtol 1e-6、対数尤度 rtol 1e-9、素朴・頑健共分散 rtol 1e-6。通る。
- `tests/models/test_estimators.py::test_survival_accepts_a_surv_formula`。

## 8. R と違うところ、未確認の点

- 左切り捨て（`entry`）では $\hat G$ を遅延エントリのリスク集合で計算するだけで、切り捨て分布の重みは入れていない。fixture にない。
