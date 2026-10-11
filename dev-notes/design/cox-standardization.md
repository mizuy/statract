# Cox モデルの回帰標準化（生存関数と制限平均生存時間）

## 1. 目的

曝露を各値に固定したときの標本平均の生存関数、または制限平均生存時間（RMST）と、その共分散を計算する。出力を比べる相手は R の `stdReg2::standardize_coxph(measure = "survival" | "rmean")`。

実装: `src/statract/surv/standardize.py` の `_survival`、`_rmean`。

## 2. 出典

- Sjölander, A. (2016). Regression standardization with the R package stdReg. *Eur J Epidemiol* 31, 563–574. 3 節（Cox モデルの標準化、推定方程式の積み重ね）。
- Sjölander, A. (2018). Estimation of causal effect measures with the R-package stdReg. *Eur J Epidemiol* 33, 847–858.
- Stefanski, L. A. & Boos, D. D. (2002). The calculus of M-estimation. *Am Stat* 56, 29–38.
- Chen, P.-Y. & Tsiatis, A. A. (2001). Causal inference on the difference of the restricted mean lifetime between two groups. *Biometrics* 57, 1030–1038.
- Zhang, M. & Schaubel, D. E. (2011). Estimating differences in restricted mean lifetime using observational data subject to dependent censoring. *Biometrics* 67, 740–749（Chen–Tsiatis の影響関数の書き直し、4 節）。

## 3. モデルと式

### 生存関数

Breslow 同順位の Cox モデル $S(t \mid X) = \exp\{-\Lambda_0(t) e^{\beta^\top X}\}$。曝露を $a$ に固定した計画行列を $X_i(a)$ として

$$\theta_a(t) = \frac1n \sum_i \exp\{-\hat\Lambda_0(t) e^{\hat\beta^\top X_i(a)}\}.$$

パラメータ $(\theta, \beta, \Lambda_0(t))$ の推定関数を積み重ねる（Sjölander 2016）:

- $\psi_{\theta_a, i} = S_i(t; a) - \theta_a$
- $\psi_{\beta, i}$ = Cox のスコア残差（`cox-residuals.md`）
- $\psi_{\Lambda, i} = n\big[\mathbf 1\{T_i \le t\}\, dN_i(T_i)/S_0(T_i) - r_i \sum_{u \le \min(T_i, t)} dN(u)/S_0(u)^2\big]$、すなわち $n\int_0^t dM_i/S_0$

共分散はサンドイッチ $A^{-1} B A^{-\top}/n$（Stefanski & Boos 2002）。$A$ は推定方程式の平均の微分: $\partial\psi_\theta/\partial\theta = -1$、$\partial\psi_\theta/\partial\beta = -\frac1n\sum S_i \Lambda_0 r_i(a) X_i(a)$、$\partial\psi_\theta/\partial\Lambda_0 = -\frac1n\sum S_i r_i(a)$、$\partial\psi_\beta/\partial\beta = -I/n$、$\partial\psi_\Lambda/\partial\Lambda_0 = -1$、$\partial\psi_\Lambda/\partial\beta = -\sum_i \mathbf 1_i\, S_1(T_i)/S_0(T_i)^2$。$B$ は行（クラスタがあればクラスタ和）の標本共分散（分母 $m-1$）。尺度は $1/n$、クラスタありは $m/n^2$。

### RMST（Chen & Tsiatis 2001）

群 $a \in \{0, 1\}$ ごとに Efron 同順位の Cox モデルを当てはめ、全 $n$ 人で平均する（式 (4)）:

$$\hat\mu_a = \frac1n \sum_i \int_0^{L} \hat S_a(t \mid Z_i)\, dt.$$

影響関数は付録の (A.1)–(A.3)（群 1 では (A.4)–(A.6)）。(A.1)・(A.2) は群 $a$ の計数過程マルチンゲールの確率積分、(A.3) は被験者項で、互いに無相関。分散の推定量は付録の $\widehat{\mathrm{var}}(\varphi_{0,i})$ をそのまま使い、$n$ で割る:

1. 被験者項（A.3）: $\frac{1}{n^2}\sum_i (\hat\mu_{a,i} - \hat\mu_a)(\hat\mu_{b,i} - \hat\mu_b)$。2 群で共有し、非対角はこの項だけ（(A.6) が (A.4)・(A.5) と無相関、$A_i(1-A_i)=0$）。
2. マルチンゲール項（A.2）: $\int_0^L \hat h_a(t)^2\, d\hat\Lambda_a(t) / S^{(0)}_a(t)$。$\hat h_a(t) = \frac1n\sum_i \int_t^L \hat S_a(u\mid Z_i) e^{\hat\beta_a^\top Z_i}\,du$、$S^{(0)}_a$ は群 $a$ のリスク集合の $e^{\hat\beta^\top Z}$ の和。
3. 係数項（A.1）: $\hat g_a^\top \hat\Sigma_a \hat g_a$ を $n_a/n$ 倍したもの。付録の定義を代入すると $\frac1n$ 倍の分散では
   $\bar g^\top \big(n_a\hat\Sigma_a\big)^{-1} \bar g$、
   $\bar g = \frac1n\sum_i \int_0^L \int_0^u \hat S_a(u\mid Z_i) e^{\hat\beta^\top Z_i}\{\bar Z_a(t) - Z_i\}\, d\hat\Lambda_a(t)\, du$、
   $n_a\hat\Sigma_a = \sum_{\text{群 } a \text{ の死亡},\ t \le L} \big[S^{(2)}/S^{(0)} - (S^{(1)}/S^{(0)})^{\otimes 2}\big]$。

注意点（fixture との一致に効いた点）:

- $\hat\Sigma_a$ は付録の定義どおり $[0, L]$ の死亡だけで積分する。全期間の情報行列（Efron）を使うと対角が 2–5% 小さくなった。
- 分散の中の $d\hat\Lambda_a$ は付録の Breslow 推定（$dN/S^{(0)}$）。点推定の $\hat S_a$ は Efron の当てはめから作る。同順位の例で効く。

## 4. 推定・計算の手順

- 生存関数: 使う行で `cox_ph(..., ties="breslow")`。曝露を定数に置いた計画行列を `build_design` で作る。各時刻で $\psi$（$n \times (k+p+1)$）と $A$ を組んでサンドイッチ。$t \le 0$ は推定 1、共分散 0。$\min(\text{times})$ 以下に事象がなければ `ValueError`（文面は statract 独自）。
- RMST: 曝露を使わない計画列（`_columns_without`）を共変量にし、群ごとに `_newton` と `_baseline`（Efron）。生存は右連続の階段関数で厳密に積分する。$\hat h$ と $\bar g$ の $du$ 積分も同じ階段関数で厳密に計算する（$\bar g$ は $\sum_e d\hat\Lambda(e)\,\frac1n\sum_i e^{\beta^\top Z_i}\{\bar Z(e) - Z_i\}\int_e^L \hat S_i$ と書ける）。

## 5. 数値上の注意

- 生存関数の $\Lambda_0$ の推定関数は、R と合わせると標本平均が 0 にならない（下の 6.）。このためパラメータ化が結果に効く。共変量は coxph と同じ規則（値がすべて $\{-1, 0, 1\}$ の列以外を平均で中心化）で中心化する。
- RMST の分散の各項は共変量の中心化に依らない。

## 6. R と出力を合わせるための公開仕様（出力比較で決めた点）

- 生存関数: 打ち切られた行でも、追跡時刻が事象時刻と同じなら、その時刻の $1/S_0$ の項を受ける（`docs/models/vs-r.md`「stdReg2 と違うところ」に記載の挙動）。$\partial\psi_\Lambda/\partial\beta$ もその形の微分にすると fixture と一致する。
- クラスタ: クラスタ和の標本共分散（和の平均を引く）。
- RMST の事象の群は事象を起こした行の群（R は同時刻の最初の行。意図的な差）。値の順は `values` に従う。

## 7. 検証

`tests/r_oracle/test_stdreg_cox.py`（fixture `stdreg_cox.json`）。

- 生存関数 4 例（二値と交互作用、クラスタ、3 水準、変換）: 推定値 rtol 1e-6（実測 1e-15）、共分散 rtol 1e-6（実測 1e-13）、表 rtol 1e-6。通る。
- RMST 3 例（主効果、交互作用、同順位）: 推定値、共分散、表とも rtol 1e-6（実測 2e-15）。

## 8. R と違うところ、未確認の点

- 既知の差は 6. の意図的な差だけ。
