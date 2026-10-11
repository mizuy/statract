# 負の二項回帰（NB2）

## 1. 目的

計数データを負の二項分布（negative binomial、NB2 型）で回帰する。リンクは対数。分散は $\mu + \mu^2/\theta$。形状パラメータ $\theta$ は最尤法（maximum likelihood、ML）で推定する。出力を比べる相手は R の `MASS::glm.nb()` と `MASS::theta.ml()`。

実装: `src/statract/models/fit.py` の `_negbin_loglik`、`_theta_score_information`、`_theta_ml`、`_fit_negbin`。

## 2. 出典

- Lawless, J. F. (1987). Negative binomial and mixed Poisson regression. *Canadian Journal of Statistics* 15(3), 209–225. 式 (2.1)（密度）、3 節（$\theta$ の最尤推定、スコアと情報量）。
- Hilbe, J. M. (2011). *Negative Binomial Regression*, 2nd ed. Cambridge University Press. 8.2 節（NB2 の対数尤度 式 8.6、$\theta$ の推定）。
- Venables, W. N. & Ripley, B. D. (2002). *Modern Applied Statistics with S*, 4th ed. Springer. 7.4 節（係数と $\theta$ を交互に推定する方法）。
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. 2.5 節（反復重み付き最小二乗法、iteratively reweighted least squares、IRLS）。

## 3. モデルと式

観測 $i = 1, \dots, n$ について、応答 $y_i \in \{0, 1, 2, \dots\}$、計画行列の行 $x_i$、オフセット $o_i$、事前重み $w_i \ge 0$ とする。平均は $\mu_i = \exp(x_i^\top \beta + o_i)$。

密度（Lawless 1987, 式 2.1）:

$$f(y; \mu, \theta) = \frac{\Gamma(\theta + y)}{\Gamma(\theta)\, y!} \left(\frac{\theta}{\theta + \mu}\right)^{\theta} \left(\frac{\mu}{\theta + \mu}\right)^{y}$$

重み付き対数尤度（Hilbe 2011, 式 8.6 を展開）:

$$\ell(\beta, \theta) = \sum_i w_i \left[\log\Gamma(\theta + y_i) - \log\Gamma(\theta) - \log y_i! + \theta \log\theta + y_i \log\mu_i - (\theta + y_i)\log(\theta + \mu_i)\right]$$

$\mu$ を固定したときの $\theta$ のスコア $U$ と観測情報量 $I$（Lawless 1987, 3 節）。$\psi$ はディガンマ関数、$\psi'$ はトリガンマ関数:

$$U(\theta) = \sum_i w_i \left[\psi(\theta + y_i) - \psi(\theta) + \log\theta + 1 - \log(\theta + \mu_i) - \frac{\theta + y_i}{\theta + \mu_i}\right]$$

$$I(\theta) = -\frac{\partial U}{\partial \theta} = \sum_i w_i \left[\psi'(\theta) - \psi'(\theta + y_i) - \frac{1}{\theta} + \frac{2}{\theta + \mu_i} - \frac{\theta + y_i}{(\theta + \mu_i)^2}\right]$$

$\theta$ を固定すると、$\beta$ の推定は分散関数 $V(\mu) = \mu + \mu^2/\theta$ の一般化線形モデル（generalized linear model、GLM）になる。IRLS の作業重みは $w_i \mu_i^2 / V(\mu_i)$。

## 4. 推定・計算の手順

### $\theta$ の最尤推定（`_theta_ml`）

$\mu$ を固定し、$\theta$ のスコア $U$ と情報量 $I$ でニュートン法を回す（Lawless 1987, 3 節）。

1. 初期値はモーメント推定 $\theta_0 = \sum w_i / \sum w_i (y_i/\mu_i - 1)^2$（Hilbe 2011, 8.2 節）。
2. $\theta$ は正の範囲に保つ。
3. 更新幅が十分小さくなるか、反復が上限 `limit` に達したら止める。許容幅と上限は R の `theta.ml` のヘルプにある既定値と同じにする。
4. 標準誤差は $1/\sqrt{I}$。$I \le 0$ なら NaN。

### 係数と $\theta$ の交互推定（`_fit_negbin`）

Venables & Ripley (2002, 7.4 節) の交互最大化を使う。

1. ポアソン GLM で $\mu$ を得る。$\theta$ の初期値はその $\mu$ でのモーメント推定。
2. $\theta$ を最尤推定し、その $\theta$ を固定して IRLS で $\beta$ を求める。これを交互に繰り返す。
3. 対数尤度の変化と $\theta$ の変化がともに小さくなったら止める。対数尤度の変化は、残差自由度の $\chi^2$ 分布の標準偏差 $\sqrt{2\max(1, \mathrm{df})}$ を単位に測る。

返す係数・作業重み・逸脱度は、返す $\theta$ で回した IRLS のもの。係数の共分散は $\theta$ を既知として $(X^\top W X)^{-1}$（分散パラメータ 1）。

情報量規準: $\mathrm{AIC} = -2\ell + 2(r + 1)$、$\mathrm{BIC} = -2\ell + (r + 1)\log n_{\text{ok}}$。$r$ は計画行列のランク、$n_{\text{ok}}$ は重みが 0 でない行数。$+1$ は $\theta$ の分。

## 5. 数値上の注意

- $y_i = 0$ の行は $y_i \log\mu_i$ を 0 とする（$\mu_i$ がアンダーフローしても NaN にしない）。`scipy.special.xlogy` を使う。
- $\theta$ が負になったら 0 にして `RuntimeWarning` を出す。
- $\theta$ のニュートン法が上限まで収束しないとき、交互推定が上限まで落ち着かないときも `RuntimeWarning` を出す。
- 負の応答は `ValueError` にする。
- ポアソンに近いデータでは $\theta$ が非常に大きくなる。止めずにそのまま返す。

## 6. R と出力を合わせるための公開仕様

- `?glm.nb`: 初期 $\theta$ を与えないとき、ポアソン GLM の後のモーメント推定を使う。収束判定は `glm.control` の `epsilon`（既定 1e-8）と `maxit`（既定 25）。
- `?theta.ml`: 初期値はモーメント推定、上限 `limit`（既定 25）、許容 `eps = .Machine$double.eps^0.25`。戻り値の属性 `SE` が標準誤差。
- `?glm.control`: IRLS の収束は $|D - D_{\text{old}}|/(|D| + 0.1) < \varepsilon$。
- `summary.negbin` は $\theta$ を既知として係数の標準誤差を出す。

## 7. 検証

- `tests/r_oracle/test_glm_nb.py`、fixture `tests/r_oracle/fixtures/glm_nb.json`（4 例: 切片なし、オフセット 2 例、重み付き交互作用）。
- 許容誤差: 係数 rtol 1e-7（atol 1e-9）、標準誤差 rtol 1e-7、$\theta$ rtol 1e-7、$\theta$ の標準誤差 rtol 1e-6、対数尤度・AIC・BIC rtol 1e-10、逸脱度 rtol 1e-8、当てはめ値 rtol 1e-7、HC0 rtol 1e-6。
- 実測の最大相対誤差: 係数 1.4e-7（許容幅の 0.36 倍）、$\theta$ 1e-9、$\theta$ の標準誤差 3e-9、対数尤度 1e-15。
- ほかに `test_column_offset_equals_formula_offset`、`test_rejects_negative_counts`、`test_refit_reestimates_theta`。

## 8. R と違うところ、未確認の点

- $\theta$ の標準誤差は、最後のニュートン反復の情報量を使う。返した $\theta$ で計算し直すと、R との差が相対 1e-4 程度になった（最後の更新幅は $\varepsilon^{1/4} \approx 1.2 \times 10^{-4}$ 以下なので、この程度ずれる）。
- 1 回目の反復で $\theta$ を再推定するので、初期 $\theta$（モーメント推定か最尤推定か）は fixture の結果を変えない。どちらが R の既定かは出力では区別できない。
- IRLS を先に回して $\theta$ を後で更新する順序も試した。係数が最大 4.5e-7 ずれて許容誤差を外れたため、上の順序にした。
