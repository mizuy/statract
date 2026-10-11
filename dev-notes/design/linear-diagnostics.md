# 線形回帰の診断: Durbin–Watson、Breusch–Pagan、Breusch–Godfrey、VIF の区間

対象: `src/statract/models/linear_tests.py` の `durbin_watson_test`、`breusch_pagan_test`、`breusch_godfrey_test`、`_vif_r_interval`（補助関数 `_durbin_watson_moments`）。

## 1. 目的

- `durbin_watson_test`: 最小二乗（OLS, ordinary least squares）残差の 1 次の系列相関の検定。R の `lmtest::dwtest(exact = FALSE)` と比べる想定。
- `breusch_pagan_test`: 説明変数に関する分散不均一の検定。既定は Koenker のスチューデント化版。R の `lmtest::bptest()` と比べる想定。
- `breusch_godfrey_test`: `order` 次までの系列相関のラグランジュ乗数（LM, Lagrange multiplier）検定。R の `lmtest::bgtest()` と比べる想定。
- `_vif_r_interval`: 分散拡大係数（VIF, variance inflation factor）を $R_j^2=1-1/\mathrm{VIF}_j$ の尺度に直したときの信頼区間。`check_collinearity` が VIF と許容度（tolerance）の区間に戻す。`performance::check_collinearity()` と一致させる意図。

いずれも R の fixture はない。

## 2. 出典

- Durbin, J. & Watson, G. S. (1950). Testing for serial correlation in least squares regression. I. *Biometrika* 37:409–428, 3 節（$d$ の平均と分散）。
- Durbin, J. & Watson, G. S. (1951). Testing for serial correlation in least squares regression. II. *Biometrika* 38:159–178.
- Durbin, J. & Watson, G. S. (1971). Testing for serial correlation in least squares regression. III. *Biometrika* 58:1–19.
- Breusch, T. S. & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica* 47:1287–1294.
- Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. *J Econometrics* 17:107–112.
- Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. *Australian Economic Papers* 17:334–355.
- Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. *Econometrica* 46:1293–1301.
- Olkin, I. & Finn, J. D. (1995). Correlations redux. *Psychol Bull* 118:155–164.
- Cohen, J., Cohen, P., West, S. G. & Aiken, L. S. (2003). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*, 3rd ed., 3.6 節（$R^2$ の標準誤差）。
- Marcoulides, K. M. & Raykov, T. (2019). Evaluation of variance inflation factors in regression models using latent variable modeling methods. *Educ Psychol Meas* 79:874–882.

## 3. モデルと式

$y=X\beta+\varepsilon$、$X$ は $n\times k$（階数 $k$）、OLS 残差 $e=My$、$M=I-X(X^\top X)^{-1}X^\top$。

### Durbin–Watson

$D$ を $(n-1)\times n$ の 1 階差分行列、$A=D^\top D$（対角 $1,2,\dots,2,1$、隣は $-1$）とする。

$$d=\frac{\sum_{t=2}^{n}(e_t-e_{t-1})^2}{\sum_t e_t^2}=\frac{e^\top Ae}{e^\top e}$$

$\nu=n-k$ として、帰無仮説（誤差が独立な正規）での正確な平均と分散（Durbin & Watson 1950, 3 節）:

$$E[d]=\frac{\mathrm{tr}(MA)}{\nu},\qquad \mathrm{Var}[d]=\frac{2\{\mathrm{tr}((MA)^2)-\mathrm{tr}(MA)\,E[d]\}}{\nu(\nu+2)}$$

p 値は $z=(d-E[d])/\sqrt{\mathrm{Var}[d]}$ の正規近似。

### Breusch–Pagan

$\hat\sigma^2=\sum w_ie_i^2/n$（$n$ は重み正の行数）、$g_i=e_i^2/\hat\sigma^2$。

- 元の版（Breusch & Pagan 1979）: $g$ を $X$ に回帰した説明平方和 ESS の半分。$\mathrm{BP}=\mathrm{ESS}/2\sim\chi^2_{k-1}$。
- スチューデント化版（Koenker 1981）: $e^2$ を $X$ に回帰した決定係数 $R^2$ を使い $\mathrm{BP}_K=nR^2\sim\chi^2_{k-1}$。
- 両者の関係: $\mathrm{BP}=\mathrm{BP}_K\cdot\widehat{\mathrm{Var}}(e^2)/(2\hat\sigma^4)$（分散の分母は $n$）。正規誤差なら比はほぼ 1。

### Breusch–Godfrey

補助回帰 $e_t=x_t^\top\gamma+\sum_{l=1}^{p}\rho_le_{t-l}+u_t$。標本より前のラグは 0 とする。

- $\chi^2$ 版: $nR^2\sim\chi^2_p$。残差の平均は 0（切片を含む場合）なので $R^2=1-\mathrm{RSS}_\text{aux}/\sum e_t^2$。
- F 版: $F=\{(\sum e_t^2-\mathrm{RSS}_\text{aux})/p\}/\{\mathrm{RSS}_\text{aux}/(n-k-p)\}\sim F_{p,\,n-k-p}$。

### VIF の区間

$R^2=1-1/\mathrm{VIF}$、$p$ は係数の数。Olkin & Finn (1995)、Cohen ら (2003) の大標本標準誤差:

$$SE(R^2)=\sqrt{\frac{(1-R^2)^2(n-p-1)^2}{(n^2-1)(n+3)}}$$

ロジット尺度でデルタ法を使う。$\ell=\log\{R^2/(1-R^2)\}$、$SE(\ell)=SE(R^2)/\{R^2(1-R^2)\}$。区間 $\ell\pm z\,SE(\ell)$（$z$ は $(1+\text{ci})/2$ の正規分位点）を逆ロジットで戻す。

## 4. 計算の手順

### Durbin–Watson

1. $d$ を残差から直接計算する。
2. $X$ の正規直交基底 $Q$（QR 分解。階数落ちなら特異値分解で列を選ぶ）を作る。$MA=A-QQ^\top A$。
3. トレースは $n\times n$ 行列を作らずに計算する。
   - $\mathrm{tr}(A)=2(n-1)$、$\mathrm{tr}(A^2)=\|A\|_F^2=6n-8$（端の 2 行が 2、内側の $n-2$ 行が 6）。
   - $AQ$ は差分を 2 回とるだけで作れる。
   - $\mathrm{tr}(MA)=\mathrm{tr}(A)-\mathrm{tr}(Q^\top AQ)$。
   - $\mathrm{tr}((MA)^2)=\mathrm{tr}(A^2)-2\|AQ\|_F^2+\mathrm{tr}((Q^\top AQ)^2)$。
4. alternative: "greater"（正の相関、$d$ が小さい）は $\Phi(z)$、"less" は $1-\Phi(z)$、"two.sided" は $2\{1-\Phi(|z|)\}$。
5. 返り値 `TestResult(d, p, (n-k,), "normal", "durbin_watson")`。

### Breusch–Pagan

1. 重み正の行だけ残す。
2. $\hat\sigma^2$ と $g=e^2/\hat\sigma^2$ を作る。
3. $\sqrt w$ をかけて $g$ を $X$ に重み付き回帰し、重み付き平均のまわりの ESS と全平方和 TSS を求める。
4. 元の版は $\mathrm{ESS}/2$、Koenker 版は $n\cdot\mathrm{ESS}/\mathrm{TSS}$（$g$ と $e^2$ は定数倍なので $R^2$ は同じ）。
5. 自由度 $k-1$ の $\chi^2$ 上側確率。

### Breusch–Godfrey

1. ラグ行列を作る（先頭の欠けた部分は 0）。
2. $[X,\ \text{ラグ}]$ に $e$ を最小二乗で回帰し、$\mathrm{RSS}_\text{aux}$ を得る。
3. $\chi^2$ 版または F 版を 3 節の式で計算する。

### VIF の区間

3 節の式をベクトルで計算する。無限大や NaN は numpy の規則に任せ、例外を出さない。

## 5. 数値上の注意

- DW: $\nu=n-k$ と $\nu+2$ で割る。$n\le k$ では定義されない（検査はしていない）。
- DW: 階数落ちの $X$ では列数でなく階数を $k$ に使う。
- BP: $\hat\sigma^2$ の定義は Koenker 版に影響しない（定数倍）。元の版には効く。
- BG: 標本より前のラグを 0 にするので、行は落とさず $n$ はそのまま。
- VIF 区間: $R^2$ が 0 に近い（VIF がほぼ 1）と $1/\{R^2(1-R^2)\}$ が大きく、区間は非常に広くなる。例えば VIF 1.026 で VIF の区間が (1.000002, 317.6)。指定された方法の性質で、バグではない。

## 6. R と出力を合わせるための公開仕様

- `?lmtest::dwtest`: alternative の既定は "greater"。`exact` の既定は $n<100$ で正確法、それ以上で正規近似。本実装は正規近似だけ。
- `?lmtest::bptest`: `studentize=TRUE` が既定（Koenker 版）。
- `?lmtest::bgtest`: `order=1`、`type="Chisq"` が既定。標本より前の値は `fill=0`。
- エラー文: "alternative must be greater, less, or two.sided"、"distribution must be 'chi2' or 'f'"、各関数の "... expects an ordinary least-squares fit"。

## 7. 検証

`tests/models/test_linear_diagnostics.py`（新規、5 件、すべて通過）。手計算または論文の式と比べる。

- DW の手計算: 切片だけの回帰で $y=(1,3,2,4)$、残差 $(-1.5,0.5,-0.5,1.5)$、$d=9/5$。定数だけの回帰では $E[d]=2$、$\mathrm{Var}[d]=4(n-2)/(n^2-1)=8/15$。3 つの alternative の p 値を rtol $10^{-12}$ で確かめる。
- DW のモーメント: 乱数の $X$ で $M$、$A$ を明示的に作ったトレースと rtol $10^{-12}$ で一致。
- BP の手計算: 2 群 3 例ずつ、残差 $(-1,0,1,-3,0,3)$。元の版 1.92、Koenker 版 $6\cdot384/876$、Koenker の比の関係。
- BG の手計算: DW と同じデータで $\chi^2=1.12$、$F=1.4/1.8$（自由度 1, 2）。
- BG の別経路: 1 次の F が補助回帰（`fit_ols`）のラグ係数の $t^2$ と一致（rtol $10^{-10}$）。

## 8. R と違うところ、未確認の点

- R の fixture がないので、`lmtest` の出力とは比べていない。
- DW と BG は fit の重みを使わず、重みなしの $X$ と生の残差で計算する。重み付き fit での扱いは仕様にない。
- DW の正確な p 値（Imhof 1961 などによる $d$ の正確分布）は実装していない。$n<100$ では R の既定（正確法）と p 値が違う。
- VIF 区間は `performance::check_collinearity()` の出力と比べていない。
