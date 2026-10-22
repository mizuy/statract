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

**一次の補正。**

$$V'_\beta = V_\beta + J V_\rho J^\top .$$

論文にはさらに、$V_\beta$ の Cholesky 因子の $\rho$ 微分から作る二次の項 $V''$ がある。まず一次の補正だけを実装し、二次の項は §8 の確認項目とする。

**補正した edf。**

$$\tau_2 = \mathrm{tr}(V'_\beta\,\mathcal{I}) .$$

$\tau_2$ が $\tau_1$ を超えるときは $\tau_1$ を使う（`?gamObject`: 「`edf1` は `edf2` の経験的な上限」）。

## 4. 計算の手順

1. 当てはめの最後に $V_\beta$、$\mathcal{I}$、$\hat\beta$、$\lambda$、各 $S_j$ はすでに手元にある（`_fit_once`、`_covariance`）。
2. $J$ の列 $-V_\beta\lambda_j S_j\hat\beta$ を作る。$S_j$ は該当する列のブロックだけに入れる。
3. $V_\rho$: 最初は既存の基準関数（`_fit_once` の score）の $\log\lambda$ についての中心差分 Hessian で作り、その逆行列をとる。解析的な微分は後で速度が問題になったときに加える。
4. $V'_\beta$、$\tau_2 = \mathrm{tr}(V'_\beta\mathcal{I})$、$\tau_1$ を計算し、$\min(\tau_2, \tau_1)$ を `edf2` として持つ。
5. AIC: $-2\ell(\hat\beta) + 2\,\mathrm{edf2}$。ガウスでは尺度の MLE を使い、尺度パラメータの分として 1 を足す（`?logLik.gam`）。
6. `compare_gams` の `df` を `edf2` の差に替える（mgcv の `anova` の説明: 「可能なときは平滑化パラメータの不確実性で補正した edf の差」）。

## 5. 数値上の注意

- 平滑化パラメータが上限（`log_sp = 18`）や下限に張りつくと、その方向の基準関数は平坦で、Hessian が特異に近い。その $\rho_j$ は $V_\rho$ から外す（固定とみなす）。
- $V_\rho$ の差分の刻みは、基準関数の最適化の許容誤差より十分大きくとる（例 $10^{-3}$）。
- `method="gcv"` では $V_\rho$ に意味がないので補正しない（mgcv も ML と REML のときだけ補正する）。

## 6. R と出力を合わせるための公開仕様

- `?logLik.gam`: `logLik(m)` の `df` 属性は補正した自由度。ガウスは尺度の MLE を使う。
- `?gamObject`: `edf`、`edf1`、`edf2`。
- `?anova.gam`: 複数モデルでは `anova.glm` を使い、自由度の差は補正した edf の差（可能な場合）。

## 7. 検証

- `gam_summary.R` に `sum(m$edf2)`、`attr(logLik(m), "df")`、`AIC(m)` を足し、二項・重み付きガウス・ポアソン・ガンマで比べる。
- 平滑化パラメータを固定した当てはめ（`sp=` 指定）では $V_\rho$ が 0 なので、`edf2 = edf` になることを確かめる。

## 8. R と違うところ、未確認の点

- 二次の項 $V''$ を含めるかどうか。論文の式を確認してから決める。
- 2026-10-22 の調べでは、二項の例（gamm_binary_a、`s(pre_size_mm, k=10, bs="cr")`）で `sum(edf)=3.83`、`sum(edf1)=4.52`、`sum(edf2)=4.30`、`logLik` の df は 4.30 だった。一方 `anova` の残差自由度は $800 - 4.99$ で、どれとも一致しなかった。`anova` の自由度の定義は公開ヘルプからは特定できていない。まず `logLik` と `AIC` に合わせ、`anova` は差を記録する。
