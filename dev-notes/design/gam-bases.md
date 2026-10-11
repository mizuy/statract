# GAM の平滑化基底（cr、cc、ps）

## 1. 目的

一般化加法モデル（generalized additive model、GAM）の 1 変数平滑化項の基底行列と罰則行列を作る。比較相手は R の `mgcv::s()` の `bs = "cr"`、`"cc"`、`"ps"`。

- `_cr_basis_rows`（`src/statract/models/gam/__init__.py`）: 3 次回帰スプライン（cubic regression spline、cr）の基底の行。罰則と 2 階微分写像は既存の `_cr_penalty` が作る。
- `_cc`（`src/statract/models/gam/general.py`）: 周期 3 次回帰スプライン（cyclic cubic、cc）。
- `_ps`（同）: P スプライン（penalized B-spline、ps）。

`_cc` と `_ps` は `(基底, 罰則, 罰則の零空間の次元, rebuild)` を返す。和ゼロ制約は後で既存の `_absorb` がかける。

## 2. 出典

- Wood SN (2017). *Generalized Additive Models: An Introduction with R*, 2nd ed. CRC. sec. 5.3.1（cr）、5.3.2（cc）、5.3.3（ps）。
- Lancaster P, Šalkauskas K (1986). *Curve and Surface Fitting*. Academic Press. ch. 4.
- Eilers PHC, Marx BD (1996). Flexible smoothing with B-splines and penalties. *Statistical Science* 11(2):89–121.
- Green PJ, Silverman BW (1994). ch. 2（ノット値と 2 階微分による 3 次スプラインの表現）。

## 3. モデルと式

### cr（Wood 2017, sec. 5.3.1）

ノット $t_1 < \dots < t_K$。係数 $\beta_j = f(t_j)$、2 階微分 $\delta_j = f''(t_j)$。自然スプラインなので $\delta_1 = \delta_K = 0$。$h_j = t_{j+1} - t_j$。1 階微分の連続性から $B\delta^{-} = D\beta$（$\delta^{-}$ は内部ノットの 2 階微分）。

$x \in [t_j, t_{j+1}]$ では

$$f(x) = a^-_j \beta_j + a^+_j \beta_{j+1} + c^-_j \delta_j + c^+_j \delta_{j+1},$$

$$a^-_j = \frac{t_{j+1} - x}{h_j},\quad a^+_j = \frac{x - t_j}{h_j},$$

$$c^-_j = \frac{1}{6}\left[\frac{(t_{j+1} - x)^3}{h_j} - h_j (t_{j+1} - x)\right],\quad c^+_j = \frac{1}{6}\left[\frac{(x - t_j)^3}{h_j} - h_j (x - t_j)\right].$$

$\delta = F\beta$ と書くと、基底の行は $a^-_j e_j + a^+_j e_{j+1} + c^-_j F_{j\cdot} + c^+_j F_{j+1,\cdot}$。罰則は $\int f''^2 = \beta^\top D^\top B^{-1} D \beta$。

外では端の接線で延長する。左端の傾きは

$$f'(t_1) = \frac{\beta_2 - \beta_1}{h_1} - \frac{h_1}{3}\delta_1 - \frac{h_1}{6}\delta_2,$$

右端は

$$f'(t_K) = \frac{\beta_K - \beta_{K-1}}{h_{K-1}} + \frac{h_{K-1}}{6}\delta_{K-1} + \frac{h_{K-1}}{3}\delta_K .$$

### cc（Wood 2017, sec. 5.3.2）

ノット $t_0 < \dots < t_m$。周期 $L = t_m - t_0$。両端は円の上で同じ点なので、自由な係数は $\beta_0, \dots, \beta_{m-1}$（$\beta_m = \beta_0$）。基底の列数はノット数より 1 少ない。添字を $m$ を法として回すと、各ノットでの 1 階微分の連続性は

$$\frac{h_{i-1}}{6}\delta_{i-1} + \frac{h_{i-1} + h_i}{3}\delta_i + \frac{h_i}{6}\delta_{i+1} = \frac{\beta_{i-1}}{h_{i-1}} - \left(\frac{1}{h_{i-1}} + \frac{1}{h_i}\right)\beta_i + \frac{\beta_{i+1}}{h_i}.$$

これを $\tilde B \delta = \tilde D \beta$ と書く（どちらも巡回 3 重対角）。$\delta = \tilde B^{-1}\tilde D\beta$。罰則 $\int f''^2 = \beta^\top \tilde D^\top \tilde B^{-1} \tilde D \beta$。零空間は定数（次元 1）。区間内の式は cr と同じで、右隣の添字を $m$ を法として回す。

### ps（Eilers & Marx 1996; Wood 2017, sec. 5.3.3）

$k$ 本の 3 次 B スプラインを等間隔ノットの上に置く。罰則は 2 階差分

$$P = D_2^\top D_2,\quad (D_2\beta)_i = \beta_i - 2\beta_{i+1} + \beta_{i+2},$$

$D_2$ は $(k-2) \times k$。零空間は定数列と 1 次列（次元 2）。

## 4. 計算の手順

### `_cr_basis_rows`

1. 区間番号 $j$ を「$x$ 以下の最大のノット」で決め、$[1, K-1]$ に丸める。内部ノットちょうどは右の区間。
2. 区間内の点は上の式で行を作る。NumPy でベクトル化する。
3. 外の点は端の値 $e_1$ または $e_K$ に傾きの行 × 距離を足す。

### `_cc`

1. ノット: $x$ の相異なる値を並べ、確率 $0, 1/(k-1), \dots, 1$ の分位点（線形補間）。$k$ が相異なる値の数以上なら $\max(\text{相異なる値の数}, 4)$ 個。
2. $\tilde B$、$\tilde D$ を作り、$\tilde B^{-1}\tilde D$ と罰則を求める。
3. $[t_0, t_m]$ の外の $x$ は $t_0 + ((x - t_0) \bmod L)$ に移す。
4. 区間ごとの式で基底を作る。

### `_ps`

1. データの範囲 $[x_{\min}, x_{\max}]$ を両側に幅の 0.1 % 広げる。
2. それを $k - 3$ 等分し、両外側に 3 個ずつノットを足す。ノットは合計 $k + 4$ 個。
3. SciPy の `BSpline` で基底を作る。罰則は $D_2^\top D_2$。

## 5. 数値上の注意

- 既存の `_cr_penalty` は写像 $F$ を転置して保持する（第 1 列と最後の列が 0）。`_cr_basis_rows` では `f_map.T` を使う。仕様書の「行」の読み方では gamm4 のテストが落ちた。
- cc で $m$ が小さいとき、両隣が同じ添字になることがある。行列への加算は上書きでなく足し込む。
- cc の右端 $t_m$ ちょうどは最後の区間に入れ、その右隣を添字 0 に回す。
- ps の外側（最も外のノットの外）は端の多項式片をそのまま延ばす。

## 6. R と出力を合わせるための公開仕様

- `?mgcv::smooth.construct.cr.smooth.spec`: cr の既定ノットは相異なる値の等間隔分位点。
- `?mgcv::smooth.construct.cc.smooth.spec`: 端で値、傾き、曲率が一致。基底の次元はノット数 − 1。
- `?mgcv::smooth.construct.ps.smooth.spec`: 既定 `m = c(2, 2)`（3 次基底、2 階差分罰則）。範囲をわずかに広げて等間隔ノット。
- 罰則の尺度合わせと和ゼロ制約は既存の `_absorb` が行う。

## 7. 検証

- cr: `tests/r_oracle/test_gamm4.py::test_gamm_matches_gamm4[binomial_cr|binomial_age_two_smooths]`（係数、リンク、平滑化関数 atol 3e-5、SE rtol 1e-4、対数尤度 atol 1e-5）、`tests/models/test_estimators.py::test_gam_accepts_the_planned_scope`、`tests/models/test_laplace_gradient.py::test_gamm_fit_unchanged`。すべて通る。
- `tests/models/test_gam_bases.py`（新規、R fixture なし）:
  - cr: ノットで単位行列、$\beta^\top S\beta$ が数値積分した $\int f''^2$ と一致（rtol 1e-3）、外で線形。
  - ps: 1 の分割、罰則の階数 $k-2$、定数列と 1 次列が零空間、$\beta^\top P\beta = \sum (\Delta^2\beta)^2$、内部ノットでの値 1/6、2/3、1/6、1 次係数が 1 次関数を再現。
  - cc: 列数 $k-1$、ノットで単位行列、つなぎ目で値・傾き・曲率が一致、周期的な折り返し、罰則の階数 $k-2$、$\int f''^2$ と一致。
  - cc と ps の平滑化を含む `gam` が真の関数を復元（RMSE < 0.08）。

## 8. R と違うところ、未確認の点

- ps と cc は mgcv の出力と比べていない（fixture がない。作業環境にネットワークがなくヘルプページも確認できなかった）。
- ps で最も外のノットより外の予測は、mgcv と違うかもしれない。
- cc で相異なる値が 4 個未満のときのノットは仕様の文言どおり。
