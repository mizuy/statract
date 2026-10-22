# GAM の最尤（ML）基準

## 1. 目的

`gam(..., method="ml")` の平滑化パラメータを mgcv の `method = "ML"` と同じ基準で選ぶ。2026-10-22 までは、二項とポアソンの ML 基準に対数行列式の項がなく、平滑化パラメータが 0 に向かって edf が上限に張りついていた。ガウスとガンマは $\log|H|$ を全係数でとっていたので、REML に近い値になり、少しずれていた（ガウスで sp 15.50、mgcv は 15.73）。mgcv は GPL なので、論文からこのノートを書く。

## 2. 出典

- Wood SN (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS B* 73:3–36. sec. 2（REML と ML の Laplace 近似）。
- Wood SN (2017). *Generalized Additive Models*, 2nd ed. sec. 6.10。

## 3. 式

罰則 $S_\lambda = \sum_j\lambda_j S_j$ の値域（range space）の正規直交基底を $U$（$p \times r$、$r = \mathrm{rank}\,S_\lambda$）とする。ML は罰則のない係数（零空間）を固定効果として最大化し、罰則のある係数だけを Laplace 近似で積分する。$H = X^\top W X + S_\lambda$ とすると、尺度既知の族で

$$\mathcal{V}_{\mathrm{ML}}(\rho) = -\ell(\hat\beta) + \tfrac12\hat\beta^\top S_\lambda\hat\beta + \tfrac12\{\log|U^\top H U| - \log|S_\lambda|_+\} .$$

REML はすべての係数を積分するので $\log|H|$ と $-\tfrac{M_p}{2}\log 2\pi$ を使う。尺度を推定する族では、$U^\top H U$ と $S_\lambda$ の $\phi$ の因子が打ち消し合い、ガウスでは $\hat\phi = (\mathrm{RSS} + \hat\beta^\top S_\lambda\hat\beta)/n$。

$U$ は $S_\lambda$ の上位 $r$ 個の固有ベクトルからとる。$r$ は単位尺度の罰則の和から決め、$\lambda$ によらない（既存の `_penalty_rank`）。

## 4. 検証

`gam_summary.R` の `ml` に、二項、重み付きガウス、offset つきポアソン、ガウス、ガンマの `sp`、`sum(edf)`、`gcv.ubre`（最小化した基準）、`sig2` を入れた。基準関数の値は rtol 1e-6、sp は rtol 1e-3 で一致する（2026-10-22）。

## 5. ガンマの尺度

mgcv の `sig2` はガンマで Pearson 推定 $\sum w_i(y_i - \hat\mu_i)^2/\hat\mu_i^2/(n - \mathrm{edf})$ で、REML でも ML でも同じ。statract もこれを `sig2` にし、共分散、平滑項の検定、`compare_gams` の F に使う。REML と ML の基準の中では尺度を別に最大化し、その値は対数尤度（AIC）にだけ使う。
