# 変量効果の分散の profile 区間（Laplace 当てはめ）

## 1. 目的

`gamm` と `fit_mixed`（Laplace エンジン）の変量効果の SD（分散）について、profile 尤度による信頼区間を出す。`confint(merMod, method = "profile")` の `.sig01` に当たる。MOR の区間は SD の区間の両端から作る（MOR は SD の単調増加関数）。

いまの `variance_table()` は log SD の Wald 区間を返す。0 の近くでは Wald 区間が不正確なので、profile 区間を加える。lme4 は GPL なので、論文と公開ヘルプからこのノートを書く。

## 2. 出典

- Bates D, Mächler M, Bolker B, Walker S (2015). Fitting linear mixed-effects models using lme4. *J Stat Softw* 67(1). profile の節（signed square root の尤度比 $\zeta$ と、それを使う区間）。
- Venzon DJ, Moolgavkar SH (1988). A method for computing profile-likelihood-based confidence intervals. *Appl Stat* 37:87–94。
- Self SG, Liang KY (1987). *JASA* 82:605–610（境界での尤度比の分布）。
- `?lme4::profile-methods`、`?lme4::confint.merMod`。

## 3. モデルと式

$\ell(\psi)$: Laplace 近似の対数尤度。$\psi = (\beta, \text{各項の共分散因子パラメータ})$。注目する SD を $\sigma$ とし、ほかのすべてのパラメータ $\psi_{-}$ について最大化した値を $\ell_p(\sigma) = \max_{\psi_-}\ell(\sigma, \psi_-)$ とする。

$$\zeta(\sigma) = \mathrm{sign}(\sigma - \hat\sigma)\sqrt{2\{\ell(\hat\psi) - \ell_p(\sigma)\}} .$$

水準 $1-\alpha$ の区間は $\{\sigma : |\zeta(\sigma)| \le z_{1-\alpha/2}\}$。

- gamm では、平滑の分散も $\psi_-$ に入り、$\sigma$ ごとに最大化し直す。
- 下限: $\sigma \to 0$ で $|\zeta| < z$ のままなら、下限は 0。
- 変量傾きを持つ項（`us`）では、Cholesky のパラメータから $\sigma_i = \sqrt{(LL^\top)_{ii}}$ を作る。$\sigma_i$ を固定する制約つき最大化が要るので、最初は対角の第 1 成分（切片の SD、$\sigma_1 = L_{11}$）だけを対象にする。

## 4. 計算の手順

1. 当てはめ済みの $\hat\psi$ と $\ell(\hat\psi)$ を持つ。
2. `_maximize` に「固定するパラメータの添字と値」を渡せるようにする。いまの `free` マスクの仕組みを一般化する。$\log\sigma$ を固定し、残りを最大化して $\ell_p$ を得る。開始値は直前の $\sigma$ での最適点にする（ウォームスタート）。
3. `models/_profile.py` の `profile_limit` を $\log\sigma$ の尺度で使う。`drop(b) = 2(\ell(\hat\psi) - \ell_p(e^b))`、`scale` は Wald の SE（`log_sd_std_error`）。
4. 下限側で $\log\sigma$ が `_LOG_SD_FLOOR` に達しても $\zeta$ が $-z$ に届かなければ、下限は 0 とする。
5. `variance_table(method="profile")` で分散の区間（$\sigma^2$）を返す。既定は `"wald"` のままにし、profile は時間がかかることを docstring に書く。

## 5. 数値上の注意

- $\ell_p$ の誤差がそのまま $\zeta$ に入る。内側の最大化の許容誤差は、外側の根探索の許容誤差より小さくする。
- 1 点あたり、当てはめ 1 回分に近い時間がかかる。根探索は両側で約 10–20 点なので、当てはめ 20–40 回分になる。遅ければ、ウォームスタートと内側の Newton 研磨だけで済ませる。それでも足りないときに numba を考える（速度の方針: アルゴリズム → numba → Rust）。
- $\hat\sigma$ が境界（特異当てはめ）なら、上限だけを求める。

## 6. R と出力を合わせるための公開仕様

- `?lme4::confint.merMod`: `method = "profile"` の既定は `level = 0.95`。区間は SD の尺度で返る（`oldNames = FALSE` なら `sd_(Intercept)|group`）。
- `?lme4::profile-methods`: `which = "theta_"` で分散成分だけを profile する。

## 7. 検証

- fixture `gamm4_compare.json` の `profile_sd`（gamm_binary_a、検査医の SD の profile 区間 0.4991–1.0759）と rtol 1e-3 で比べる。lme4 は spline 補間、こちらは根探索なので、完全には一致しない。
- `glmm_binary_*.csv` で `glmer` の profile 区間を fixture に足す。分散が小さく、下限が 0 になる例を 1 つ入れる。

## 8. R と違うところ、未確認の点

- lme4 は $\zeta$ を数点で計算して spline で補間する。こちらは交点を直接解く。差は補間誤差の分だけ出る。
- 変量傾きの SD と相関の profile は後回し。
