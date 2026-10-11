# Cox モデルの残差と比例ハザード検定

## 1. 目的

当てはめた Cox 比例ハザードモデル（`CoxFit`）から次を計算する。

- マルチンゲール残差、逸脱度残差（R の `residuals(coxph, type = "martingale" | "deviance")`）
- スコア残差、dfbeta、dfbetas（`type = "score" | "dfbeta" | "dfbetas"`）
- Schoenfeld 残差とスケール化 Schoenfeld 残差（`type = "schoenfeld" | "scaledsch"`）
- 比例ハザード（PH）のスコア検定（`cox.zph(transform = "km", terms = TRUE)`）

スコア残差は、クラスタ頑健分散（Lin–Wei）、`cluster_covariance`、Cox の回帰標準化でも使う。

実装: `src/statract/surv/cox.py` の `_martingale_residuals`、`_score_residuals`、`_schoenfeld`、`_zph_score`。

## 2. 出典

- Therneau, T. M., Grambsch, P. M. & Fleming, T. R. (1990). Martingale-based residuals for survival models. *Biometrika* 77, 147–160.
- Therneau, T. M. & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer. 3.3 節（同順位）、4.2 節（マルチンゲール・逸脱度残差）、4.4 節（スコア残差）、4.6 節と 6.2 節（Schoenfeld 残差、PH 検定）、7.3 節（dfbeta、頑健分散）。
- Schoenfeld, D. (1982). Partial residuals for the proportional hazards regression model. *Biometrika* 69, 239–241.
- Grambsch, P. M. & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika* 81, 515–526.
- Lin, D. Y. & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA* 84, 1074–1078.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA* 72, 557–565.

## 3. モデルと式

行 $i$ の区間を $(s_i, t_i]$、事象指示を $\delta_i$、ケース重みを $w_i$、線形予測子を $\eta_i = x_i^\top\hat\beta + \text{offset}_i$、$r_i = \exp(\eta_i)$ とする。時点 $u$ で行 $i$ がリスク集合にいるのは $s_i < u \le t_i$（同じ層の中）のとき。

リスク集合の和: $S_0(u) = \sum_i Y_i(u) w_i r_i$、$S_1(u) = \sum_i Y_i(u) w_i r_i x_i$、$S_2(u) = \sum_i Y_i(u) w_i r_i x_i x_i^\top$。
平均 $\bar x(u) = S_1/S_0$、分散 $V(u) = S_2/S_0 - \bar x \bar x^\top$。

**Breslow 型の基底ハザードの増分**: 死亡時刻 $u$ の死亡重み和を $W(u)$ として $d\hat\Lambda_0(u) = W(u)/S_0(u)$。

**Efron の近似**（Efron 1977; Therneau & Grambsch 2000, 3.3 節）: 時刻 $u$ に $d$ 個の同順位死亡があるとき、$d$ 個の「擬似事象」$k = 0, \dots, d-1$ に分ける。死亡した行のリスク和 $D_0, D_1, D_2$ を割合 $k/d$ だけ減らし、各擬似事象は平均死亡重み $W/d$ を持つ。

$$d\hat\Lambda_k = \frac{W/d}{S_0 - (k/d) D_0}, \qquad \bar x_k = \frac{S_1 - (k/d) D_1}{S_0 - (k/d) D_0}.$$

Breslow のときは、1 時刻に擬似事象が 1 個（$k = 0$、重み $W$）とみなせば同じ式で書ける。

**マルチンゲール残差**（Therneau ら 1990, 式 (2.1)）: $M_i = N_i(\infty) - \int Y_i(u) r_i\, d\hat\Lambda_0(u)$。Efron では、死亡しない行は各擬似事象の増分を全部受け、時刻 $u$ で死亡した行は $k$ 番目の増分を $(1 - k/d)$ 倍で受ける。

**逸脱度残差**: $\operatorname{sign}(M_i)\sqrt{-2[M_i + \delta_i \log(\delta_i - M_i)]}$。

**スコア残差**（Therneau ら 1990, 3 節; Lin & Wei 1989）: $L_{ij} = \int (x_{ij} - \bar x_j(u))\, d\hat M_i(u)$。
$dN$ の部分は、死亡した行で $x_i - \frac1d\sum_k \bar x_k$。補償子の部分は $r_i \sum_k c_{ik}\, d\hat\Lambda_k (x_i - \bar x_k)$（$c_{ik}$ は上と同じ $1$ または $1-k/d$）。$\sum_i w_i L_i$ は部分尤度スコアで、MLE では 0。

dfbeta $= (w \cdot L) I^{-1}$、dfbetas は dfbeta を $\sqrt{\operatorname{diag}(I^{-1})}$ で割る。頑健分散は $I^{-1}(\sum_g U_g U_g^\top) I^{-1}$、$U_g$ はクラスタ内の $w_i L_i$ の和（Lin & Wei 1989; Therneau & Grambsch 2000, 7.3 節）。

**Schoenfeld 残差**（Schoenfeld 1982）: 死亡行ごとに $x_i - \bar x(t_i)$。Efron では $\bar x(t_i)$ を $\frac1d\sum_k \bar x_k$ に置き換える。スケール化: $r^*_i = d_{\text{tot}}\, I^{-1} r_i + \hat\beta$（Grambsch & Therneau 1994, 式 (2)）。

**PH スコア検定**（Grambsch & Therneau 1994, 3 節）: 共変量 $x\,g(t)$ を加えた $2p$ パラメータの部分尤度のスコアと情報量を $(\hat\beta, 0)$ で評価する。時刻 $u$ のリスク集合では追加共変量の平均が $g(u)\bar x(u)$、分散が $g V$、$g^2 V$ になるので、

$$U = \Big(\sum_{\text{死亡}} w_i (x_i - \bar x),\ \sum_{\text{死亡}} w_i g(t_i)(x_i - \bar x)\Big),\quad
\mathcal I = \sum_{\text{擬似事象}} \frac{W}{d}\begin{pmatrix} V_k & g V_k \\ g V_k & g^2 V_k\end{pmatrix}.$$

検定統計量は第 1 ブロックを 0 にしたスコアで $s^\top \mathcal I_{\text{sub}}^{-1} s$。項ごとの検定は、第 1 ブロック全部と該当項の第 2 ブロック列を使う（呼び出し側 `proportional_hazards_test`）。

## 4. 推定・計算の手順

1. 層ごとに、$x$ を列平均で中心化し、$\eta$ から層内最大値を引く（どちらも残差を変えない）。
2. 死亡時刻 $\tau_1 < \dots < \tau_K$ を求める。$S_0, S_1$（PH 検定では $S_2$ も）を「$t_i \ge \tau$ の行の和 − $s_i \ge \tau$ の行の和」で計算する（$s_i < t_i$ なので差がリスク集合になる）。累積和と二分探索でベクトル化する。
3. 擬似事象の表（時刻番号、$k/d$、$d\hat\Lambda_k$、$\bar x_k$、重み）を作る。
4. 時刻ごとに $A = \sum_k d\hat\Lambda_k$、$B = \sum_k d\hat\Lambda_k \bar x_k$ と、死亡行用の $A' = \sum_k (1-k/d) d\hat\Lambda_k$、$B'$ を作り、累積和にする。行の区間 $(s_i, t_i]$ の寄与は累積和の差。死亡行は自分の時刻の $A, B$ を $A', B'$ に差し替える。
5. マルチンゲール: $\delta_i - r_i \cdot (\text{累積 } A)$。スコア: $\delta_i(x_i - \bar{\bar x}) - r_i (x_i \cdot \text{累積 }A - \text{累積 }B)$。

## 5. 数値上の注意

- リスク和が 0 以下の死亡時刻は寄与なし（重み 0 の行だけのとき）。
- 区間内に死亡時刻がない行は $M_i = \delta_i$。
- 累積和の差を使うので、極端に長い追跡でも相対誤差は 1e-12 程度。中心化と $\eta$ のシフトでオーバーフローを避ける。
- `ties="exact"`（条件付きロジット）では呼び出し側が拒否する。

## 6. R と出力を合わせるための公開仕様

- 残差はケース重みを掛けない（`residuals.coxph` のヘルプ: weighted = FALSE が既定）。dfbeta は重みを掛ける。
- Schoenfeld 残差の行順: 層（`np.unique` 順）、層内は時刻の昇順（同時刻は元の順）。`_schoenfeld_times` も同じ順に並べ直した（以前は元の行順で、時刻と残差の組が未ソート入力でずれていた）。
- `cox.zph` は `ties = "breslow"` 以外を Efron で扱う。

## 7. 検証

- `tests/r_oracle/test_cox_inference.py`（fixture `cox_inference.json`、6 例: Efron/Breslow、層と非整数重み、計数過程、クラスタ）: 残差 rtol 1e-6 / atol 1e-9、期待値 rtol 1e-6、頑健分散 rtol 1e-6、PH 検定 rtol 1e-6。すべて通る。
- `tests/r_oracle/test_cox_two_way.py`（多元クラスタ分散）。共分散側のスタブが未実装の間は失敗する。
- `tests/surv/test_conditional_logit.py`、`tests/surv/test_diagnostics.py`。

## 8. R と違うところ、未確認の点

- 既知の差はない。
