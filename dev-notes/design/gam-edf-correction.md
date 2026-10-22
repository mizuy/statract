# 平滑化パラメータの不確実性で補正した edf（edf2）

## 1. 目的

`gam` の有効自由度（effective degrees of freedom、edf）を、平滑化パラメータの推定の不確実性で補正する。使い道は二つある。

- 条件付き AIC: $-2\ell(\hat\beta) + 2\tau$。補正前の edf を使うと、大きいモデルを選びすぎる（Greven & Kneib 2010）。
- 入れ子の比較（`compare_gams`）の自由度。いまは補正前の edf の差を使っており、mgcv の `anova` より df と p 値が小さい。

参照実装の mgcv は GPL なので、論文と公開ヘルプだけからこのノートを書き、コードはこのノートから書く。

## 2. 出典

- Wood SN, Pya N, Säfken B (2016). Smoothing parameter and model selection for general smooth models. *JASA* 111:1548–1563. 補正した共分散と AIC の節。
- Greven S, Kneib T (2010). On the behaviour of marginal and conditional AIC in linear mixed models. *Biometrika* 97:773–789（補正がない場合の偏り）。
- Wood SN (2017). *Generalized Additive Models*, 2nd ed. sec. 6.11（AIC と edf）。
- mgcv のヘルプ `?logLik.gam`、`?gamObject`（`edf1`、`edf2` の意味）。

## 3. モデルと式

$\rho_j = \log\lambda_j$。罰則つき対数尤度 $\ell(\beta) - \tfrac12\beta^\top S_\lambda\beta$ の最大点 $\hat\beta_\rho$ は $\rho$ に依存する。

- $\mathcal{I}$: 罰則なしの負の対数尤度の Hessian（正準リンクなら $X^\top W X/\phi$）。
- $V_\beta = (\mathcal{I} + S_\lambda)^{-1}$: ベイズ事後共分散（既存の `covariance`）。
- $F = V_\beta\,\mathcal{I}$。補正前の edf は $\tau_0 = \mathrm{tr}(F)$、上限の目安は $\tau_1 = \mathrm{tr}(2F - F^2)$（`edf1`）。

**$\hat\beta$ の $\rho$ による微分。** 罰則つきスコア $g(\hat\beta) - S_\lambda\hat\beta = 0$ を $\rho_j$ で微分すると（陰関数定理）

$$\frac{\partial\hat\beta}{\partial\rho_j} = -V_\beta\,\lambda_j S_j\hat\beta .$$

これを並べた $p \times m$ 行列を $J$ とする。

**$\hat\rho$ の共分散。** $V_\rho$ は、REML（または ML）基準を $\rho$ の関数とみたときの Hessian の逆行列。尺度 $\phi$ を推定する族では、$(\rho, \log\phi)$ の同時 Hessian の逆行列の $\rho$ ブロックを使う。$\phi$ をプロファイルした基準の Hessian の逆行列でも同じになる（Schur 補元）。

**補正した共分散（Wood, Pya & Säfken 2016, sec. 4 と付録 D）。**

$$V'_\beta = V_\beta + J V_\rho J^\top + V'' ,\qquad V''_{jm} = \sum_{i,k,l}\frac{\partial R_{ij}}{\partial\rho_k}V_{\rho,kl}\frac{\partial R_{im}}{\partial\rho_l} .$$

$R$ は $R^\top R = V_\beta$ となる因子で、付録 D の作り方に合わせて $R = L^{-1}$（$LL^\top = V_\beta^{-1}$ の Cholesky 因子）とする。$V''$ は因子の選び方で変わる。

**補正した edf。**

$$\tau_2 = \mathrm{tr}(V'_\beta\,\mathcal{I}) .$$

$\tau_2$ が $\tau_1$ を超えるときは $\tau_1$ を使う（`?gamObject`: 「`edf1` は `edf2` の経験的な上限」）。AIC は $-2\ell(\hat\beta) + 2\tau_2$（式 10）。

## 4. 計算の手順

1. `refit(ρ)`: $\rho$ を固定して当てはめ直し、基準関数の値、$\hat\beta$、$V_\beta$ を返す（`_fit_once`、`_covariance`）。
2. $V_\rho$: 基準関数の中心差分 Hessian（刻み $10^{-3}$）の擬似逆行列。
3. $J$ と $\partial R/\partial\rho_k$: 当てはめ直しの中心差分（刻み $10^{-4}$）。作業重みの変化も入る。
4. $\tau_2$ を計算し、$\min(\tau_2, \tau_1)$ を `edf2` とする（`models/gam/edf_correction.py`）。
5. `glance()` に `edf2`、`log_likelihood`、`df`、`aic` を足す。`df` は edf2 に、ガウスとガンマでは尺度の分の 1 を足す。ガウスの $\ell$ は尺度の MLE（重み付き残差平方和 / $n$）、ガンマの $\ell$ は当てはめの尺度（REML 推定）を使う。ガンマで `logLik.gam` と合うのは後者だった（Pearson 推定や deviance/$n$ では合わない）。
6. `compare_gams` の `aic` は `glance()` の補正 AIC。`df` は補正前の edf の差のまま（§8）。

## 5. 数値上の注意

- 平滑化パラメータが上限（`log_sp = 18`）や下限に張りつくと、その方向の基準関数は平坦で、Hessian が特異に近い。論文は擬似逆行列を使うとだけ書く。こちらは Hessian の対角が $10^{-3}$ 未満（$\log\lambda$ の SD が 30 超）の $\rho_j$ を $V_\rho$ から外す。重み付きガウスの 2 平滑の例（z が直線）では、外さないと $\tau_2$ が上限 9.94 に張りつき、外すと 9.165 で mgcv の 9.163 と合った。
- $V_\rho$ の差分の刻みは、基準関数の最適化の許容誤差より十分大きくとる（例 $10^{-3}$）。
- `method="gcv"` では $V_\rho$ に意味がないので補正しない（mgcv も ML と REML のときだけ補正する）。

## 6. R と出力を合わせるための公開仕様

- `?logLik.gam`: `logLik(m)` の `df` 属性は補正した自由度。ガウスは尺度の MLE を使う。
- `?gamObject`: `edf`、`edf1`、`edf2`。
- `?anova.gam`: 複数モデルでは `anova.glm` を使い、自由度の差は補正した edf の差（可能な場合）。

## 7. 検証

- `gam_summary.R` に `sum(m$edf2)`、`logLik(m)` とその `df`、`AIC(m)` を足した。二項 2、重み付きガウス 2、ポアソン、ガウス、ガンマの 7 モデルで、edf2 は rtol 1e-2、対数尤度は rtol 1e-5、AIC は edf2 の 2% 以内で一致する。
- 2026-10-22 の値（statract / mgcv）: 二項 4.3305 / 4.3046、重み付きガウス 8.3176 / 8.3147、2 平滑の重み付きガウス 9.1646 / 9.1633、ガウス 7.9046 / 7.9011。ポアソン、ガンマ、2 平滑の二項は上限 $\tau_1$ に張りつき、一致する。

## 8. R と違うところ、未確認の点

- 2026-10-22: 論文 PDF で式を確認し、二次の項 $V''$ を入れて実装した。一次の補正だけでは二項の例で 4.088 だったが、$V''$ を入れて 4.33 になった。残りの差（約 0.6%）は $R$ の選び方と差分の誤差による。
- `compare_gams` の `df` は補正前の edf の差のまま。mgcv の `anova` の残差自由度（二項の例で $800 - 4.99$）は edf、edf1、edf2 のどれとも一致せず、公開ヘルプからは定義を特定できない。
- `method="gcv"` では補正しない（`edf2 = edf`）。
- `method="ml"` の二項とポアソンは、基準関数に対数行列式の項がないため平滑化パラメータが 0 に向かう（edf が上限になる）。edf2 もその影響を受ける。別に直す。
