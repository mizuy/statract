# GLM の反復重みつき最小二乗法（IRLS）

## 1. 目的

一般化線形モデル（generalized linear model, GLM）の最尤係数を、反復重みつき最小二乗法（iteratively reweighted least squares, IRLS）で求める。族は gaussian（恒等リンク）、binomial（logit）、poisson（log）、gamma（逆数）、負の二項（log、$\theta$ 既知）。出力を比べる相手は R の公開関数 `glm()`（`glm.control()` の既定）と `MASS::glm.nb()`。
実装は `src/statract/models/fit.py` の `_glm_irls_python`、`_weighted_ls_step`、`_binomial_irls_numba`。

## 2. 出典

- Nelder, J. A. & Wedderburn, R. W. M. (1972). Generalized linear models. *JRSS-A* 135(3):370–384.
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. Chapman & Hall. 2.5 節（式 2.12–2.13、IRLS）、4.4 節（二項の逸脱度）。
- Green, P. J. (1984). Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives. *JRSS-B* 46(2):149–192.
- R の公開ヘルプ `?glm`、`?glm.control`、`?family`。

## 3. モデルと式

観測 $i=1,\dots,n$、応答 $y_i$、事前重み $a_i \ge 0$、説明変数行 $x_i$、offset $o_i$。線形予測子 $\eta_i = x_i^\top\beta + o_i$、平均 $\mu_i = g^{-1}(\eta_i)$、分散関数 $V(\mu)$。

作業応答と作業重み（McCullagh & Nelder 1989, 式 2.12–2.13）:

$$ z_i = \eta_i - o_i + (y_i - \mu_i)\,\frac{d\eta_i}{d\mu_i}, \qquad w_i = \frac{a_i}{V(\mu_i)}\left(\frac{d\mu_i}{d\eta_i}\right)^2 $$

更新は重みつき最小二乗:

$$ \beta^{(t+1)} = (X^\top W X)^{-1} X^\top W z $$

これは Fisher スコア法と同じ（Green 1984）。逸脱度 $D(y;\mu) = \sum_i a_i\, d(y_i,\mu_i)$。二項（logit）の単位逸脱度は

$$ d(y,\mu) = 2\left[ y\log\frac{y}{\mu} + (1-y)\log\frac{1-y}{1-\mu} \right] $$

（$0\log 0 = 0$）。正準リンク logit では $d\mu/d\eta = V(\mu) = \mu(1-\mu)$、$w_i = a_i\mu_i(1-\mu_i)$。

## 4. 推定・計算の手順

1. 初期値は族ごとの既定の $\eta$（`?family` の `mustart` 相当。二項は $\mu = (a y + 0.5)/(a+1)$）。`etastart` があればそれで置き換える。初期の逸脱度 $D_0$ を求める。
2. 各反復で、$a_i > 0$ かつ $d\mu/d\eta$ が有限で 0 でない行だけを使い、$z$ と $w$ を作る。
3. 重みつき最小二乗を正規方程式で解く。$X^\top W X$ を Cholesky 分解し、対角の相対ピボット $L_{jj}^2 / (X^\top W X)_{jj}$ が $10^{-12}$ 以下、または結果が有限でなければ、特異値分解の最小二乗（ランク落ちでは最小ノルム解）に切り替える。
4. 新しい $\eta,\mu,D$ を求める。$D$ が有限でなければ、係数を前の反復との中点へ戻す（ステップ半減）。前の反復が無い、または半減が上限回数に達したらエラー。
5. 収束判定 $|D - D_{\text{old}}| / (|D| + 0.1) < \varepsilon$。最大 `maxit` 回。収束しなくても警告しない。
6. 返す値: 係数、最後の係数を作った最小二乗の作業重み（使わなかった行は 0）、最終の $\mu$、最終の $\mu$ での逸脱度、最終の $\eta$ での作業残差 $(y-\mu)/(d\mu/d\eta)$。
7. 二項（logit）は呼び出しが多いので、同じ手順を numba でコンパイルした専用ループで行う。特異または非有限のステップでは係数を NaN で返し、呼び出し側が汎用版（最小二乗に切り替えられる）でやり直す。

## 5. 数値上の注意

- 二項の逸脱度では $\mu$ を $[10^{-12},\,1-10^{-12}]$ に切る。分離があっても有限に保つ。
- logistic 関数は $\eta$ の符号で分けて計算し、指数のオーバーフローを避ける。
- 正規方程式は条件数を二乗する。強い共線性があるがフルランクの設計では、QR 分解より精度が落ちうる。
- gamma（逆数リンク）で $\mu \le 0$ になると逸脱度が NaN になり、ステップ半減で戻る。
- 返す重みは最終平均で計算し直さない。サンドイッチ分散の推定関数がこの組（ステップの重みと最終の作業残差）を使うため。

## 6. R と出力を合わせるための公開仕様

- `?glm.control`: `epsilon = 1e-8`、`maxit = 25`。収束は $|dev - dev_{old}|/(|dev| + 0.1) < \epsilon$。
- `?family`: 各族の既定リンクと初期値。二項の初期平均 $(w y + 0.5)/(w + 1)$。
- `?glm`: `weights` は事前重み、`offset` は線形予測子に足す。`etastart` は線形予測子の初期値。

## 7. 検証

- `tests/r_oracle/test_glm_nb.py`、`test_risk_ratio.py`、`test_standardize_glm.py`、`test_iptw.py`、`test_iptw_outcome.py`、`test_profile_glm.py`、`test_rms.py`、`test_ordinal.py`（eps 1e-12、maxit 100）、`test_mice_pool.py`、`test_glmm_count.py`。
- `tests/models/test_statsmodels_parity.py`（ランク落ちで最小ノルム解、rel 1e-12）、`test_weighting_outcome.py`、`test_smd_consistency.py`。
- 許容誤差の目安: 係数 rtol 1e-8–1e-10、分散 rtol 1e-6–1e-7。許容誤差は変えていない。
- 全体: 1064 passed、31 skipped（初期値と逸脱度を実装した後）。

## 8. R と違うところ、未確認の点

- R は最小二乗を QR 分解で解く。こちらは正規方程式を先に使う。強い共線性のある設計では小さな差が出うる。現在のテストでは見えていない。
- 最初のステップから逸脱度が有限でないときは ValueError（メッセージは独自）。
- 二項の $d\mu/d\eta$ は $\mu(1-\mu)$ をそのまま使い、下限で切らない。$|\eta|$ が非常に大きい行は 0 になってその反復から外れる。

## 付録: 初期値と逸脱度残差（`_glm_start`、`_negbin_dev_resids`）

### 目的と出典

`_glm_start(y, family, prior, theta)` は族ごとに逆リンク $g^{-1}(\eta)$、その導関数 $d\mu/d\eta$、分散関数 $V(\mu)$、逸脱度 $D(y,\mu,w)=\sum w_i d(y_i,\mu_i)$、初期の線形予測子 $\eta^{(0)}$ を返す。`_negbin_dev_resids` は負の二項（NB2）の観測ごとの逸脱度。

- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. 2.5 節（初期値）、表 2.1（単位逸脱度）、6–8 章（ポアソン、ガンマ）。
- Lawless, J. F. (1987). Negative binomial and mixed Poisson regression. *Can J Stat* 15:209–225, 2 節。
- Hilbe, J. M. (2011). *Negative Binomial Regression*, 2nd ed. 8.2 節。

### 式

| 族 | リンク | $V(\mu)$ | 単位逸脱度 $d(y,\mu)$ |
|---|---|---|---|
| gaussian | 恒等 | 1 | $(y-\mu)^2$ |
| binomial | logit | $\mu(1-\mu)$ | $2[y\log(y/\mu)+(1-y)\log\{(1-y)/(1-\mu)\}]$ |
| poisson | log | $\mu$ | $2[y\log(y/\mu)-(y-\mu)]$ |
| gamma | 逆数 | $\mu^2$ | $-2[\log(y/\mu)-(y-\mu)/\mu]$ |
| 負の二項 | log | $\mu+\mu^2/\theta$ | $2[y\log(y/\mu)-(y+\theta)\log\{(y+\theta)/(\mu+\theta)\}]$ |

$y=0$ では $y\log(y/\mu)=0$（`scipy.special.xlogy`）。負の二項の式は、飽和モデル（$\mu=y$）と $\mu$ の対数尤度比の 2 倍（$\theta$ 固定）。

### 手順

1. 初期値は McCullagh & Nelder 2.5 節の考え方で、$\mu^{(0)}=y$ をリンクが有限になるよう境界から少しずらす。
   - binomial: $\mu^{(0)}=(a y+1/2)/(a+1)$（経験 logit）。
   - poisson、負の二項: $\mu^{(0)}=y+0.1$、$\eta^{(0)}=\log\mu^{(0)}$。
   - gaussian: $\eta^{(0)}=y$。gamma: $\eta^{(0)}=1/y$。
2. 呼び出し側（`_glm_irls_python`）が `etastart` を渡せば $\eta^{(0)}$ は使われない。`profile_interval` は $\eta^{(0)}$ を使わない。
3. 入力の検査: 二項は $y\in[0,1]$、ポアソンと負の二項は $y\ge0$、ガンマは $y>0$。負の二項で $\theta$ がない、または未知の族なら `ValueError`。

### 数値上の注意

- 二項の逆リンクは $\mu$ を $[\varepsilon,1-\varepsilon]$（$\varepsilon$ は機械イプシロン）に切る。分離があっても逸脱度が有限で、プロファイル区間の側では逸脱度が頭打ちになる。$d\mu/d\eta$ も $\varepsilon$ 以上にする。
- log リンクは $\exp(\eta)$ の $\eta$ をオーバーフローの手前で切り、$\mu\ge\varepsilon$ にする。
- ガンマの逆数リンクで $\eta\le0$ になると逸脱度が NaN になり、IRLS のステップ半減で戻る。

### R と合わせる点と検証

- IRLS の停止規則は逸脱度の相対変化なので、止まる点は初期値に少しよる。ポアソンの $\mu^{(0)}=y+c$ を $c=0.05,0.1,0.25,0.5$ で試し、$c=0.1$ のときだけ `test_risk_ratio.py`（係数 rtol $10^{-10}$）と `test_standardize_glm.py`（rtol $10^{-6}$）の fixture がすべて合った。他の $c$ では最適点からの距離は同程度（相対 $10^{-8}$ 程度）だが、R の止まった点との差が最大 $4\times10^{-6}$（派生量）になる。初期値はこの fixture に合わせて選んだ。
- 負の二項の逸脱度: `test_glm_nb.py`（逸脱度 rtol $10^{-8}$）。他は 7 節のテスト。許容誤差は変えていない。
- 全体: 1064 passed、31 skipped。

### 未確認の点

- 二項の $\mu$ を機械イプシロンで切るので、極端に小さい確率を持つ行の逸脱度は R と少し違いうる（テストでは見えていない）。
