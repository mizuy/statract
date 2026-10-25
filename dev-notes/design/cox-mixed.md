# 混合効果 Cox モデル（ランダム効果つき Cox、frailty）

## 1. 目的

施設や患者などのグループごとに対数ハザードがずれる Cox モデル（共有 frailty、ランダム切片とランダム傾き）を当てはめる。固定効果、ランダム効果の分散（共分散）、条件付き最頻値（BLUP）、対数尤度（積分、罰則つき）を返す。出力を比べる相手は R の `coxme::coxme()`（LGPL-2）。R は出力の比較だけに使い、ソースは読まない。

実装: `src/statract/surv/cox_mixed.py` の `cox_mixed`。

## 2. 出典

- Ripatti, S. & Palmgren, J. (2000). Estimation of multivariate frailty models using penalized partial likelihood. *Biometrics* 56, 1016–1022（対数正規 frailty、罰則つき部分尤度、Laplace 近似による周辺尤度）。
- Therneau, T. M., Grambsch, P. M. & Pankratz, V. S. (2003). Penalized survival models and frailty. *J Comput Graph Stat* 12, 156–175（罰則つき Cox モデルとしての frailty、自由度）。
- Therneau, T. M. & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer, 9 章（frailty モデル、罰則つき部分尤度、有効自由度）。
- McGilchrist, C. A. & Aisbett, C. W. (1991). Regression with frailty in survival analysis. *Biometrics* 47, 461–466（BLUP 型の推定）。
- Bates, D., Mächler, M., Bolker, B. & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *J Stat Softw* 67(1)（球面化したランダム効果 $b = \Lambda u$、Cholesky 因子のパラメータ化）。
- Pinheiro, J. C. & Bates, D. M. (1996). Unconstrained parametrizations for variance-covariance matrices. *Stat Comput* 6, 289–296。
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA* 72, 557–565（同順位の補正）。
- `?coxme`、`?coxme.control`、`?VarCorr` などのヘルプページ（公開仕様として、引数と出力の意味だけを参照）。

## 3. モデルと式

被験者 $i$ のハザード

$$\lambda_i(t) = \lambda_0(t)\exp(x_i^\top\beta + z_i^\top b),\qquad b \sim N(0, \Sigma(\theta)).$$

$z_i$ はランダム効果の計画行列 $Z$ の行。項 $t$（`(1 + x | g)` など）が $k_t$ 列と $L_t$ 水準をもつとき、$Z$ の列は「水準ごとに $k_t$ 列」の順に並べ、$\Sigma = \mathrm{blockdiag}_t (I_{L_t} \otimes G_t)$。$G_t$ は $k_t \times k_t$ の共分散。ランダム切片だけなら $G_t = \sigma_t^2$。

### 罰則つき部分尤度（PPL）

$\ell(\beta, b)$ を $\eta = X\beta + Zb + \text{offset}$ の Cox 部分尤度（Efron または Breslow）とする。

$$\mathrm{PPL}(\beta, b; \theta) = \ell(\beta, b) - \tfrac12 b^\top \Sigma^{-1} b.$$

### Laplace 近似による積分部分尤度

部分尤度を通常の尤度とみなして $b$ を積分する（Ripatti & Palmgren 2000）:

$$L(\beta, \theta) = \int \exp\{\ell(\beta, b)\}\,\phi(b; 0, \Sigma)\,db.$$

$b$ の 2 次近似で

$$\log L(\beta, \theta) \approx \mathrm{PPL}(\beta, \hat b; \theta) - \tfrac12 \log|\Sigma| - \tfrac12 \log|H_{bb}|,\qquad H_{bb} = -\frac{\partial^2 \ell}{\partial b\,\partial b^\top} + \Sigma^{-1}.$$

推定は 2 段:

1. $\theta$ を固定して、$(\beta, b)$ について PPL を同時に最大化する（$\hat\beta_\theta, \hat b_\theta$）。
2. $\theta$ について $\mathrm{IL}(\theta) = \mathrm{PPL}(\hat\beta_\theta, \hat b_\theta) - \tfrac12\log|\Sigma| - \tfrac12\log|H_{bb}(\hat\beta_\theta, \hat b_\theta)|$ を最大化する。

1. で $\beta$ も PPL で決めるので、$\log|H_{bb}|$ の $\beta$ 依存は無視している（Ripatti & Palmgren の推定法、Therneau et al. 2003）。これで `coxme` の「Integrated loglik」と一致した（6 節）。

### 球面化したパラメータ

$G_t = \Lambda_t\Lambda_t^\top$（下三角）、$\Lambda = \mathrm{blockdiag}_t(I_{L_t}\otimes\Lambda_t)$、$b = \Lambda u$、$u \sim N(0, I)$ とすると（Bates et al. 2015）

$$\mathrm{PPL} = \ell(\beta, \Lambda u) - \tfrac12 u^\top u,$$

$$\tfrac12\log|\Sigma| + \tfrac12\log|H_{bb}| = \tfrac12\log\left|\Lambda^\top I_{bb}\Lambda + I\right|,\qquad I_{bb} = -\partial^2\ell/\partial b\,\partial b^\top.$$

この形は $\Sigma^{-1}$ を使わないので、分散が 0 に近い境界でも計算できる。

### 部分尤度の 1 次・2 次微分

$W = [Z, X]$、$r_i = w_i e^{\eta_i}$（$w_i$ は重み）。層ごとに事象時刻 $\tau_k$、その時刻の死亡集合 $D_k$（$m_k$ 人、重み和 $w^d_k$）、リスク集合 $R_k = \{i : e_i < \tau_k \le t_i\}$（$e_i$ は登録時刻）。

$$S^{(0)}_k = \sum_{R_k} r_i,\quad S^{(1)}_k = \sum_{R_k} r_i W_i,\quad S^{(2)}_k = \sum_{R_k} r_i W_iW_i^\top .$$

Efron の補正では $j = 0, \dots, m_k-1$、$f_j = j/m_k$ ごとに、死亡集合の和を $f_j$ 倍引いた $S_{kj}$ と重み $\omega_k = w^d_k/m_k$ を使う。Breslow は $j = 0$ だけで $\omega_k = w^d_k$。

$$\ell = \sum_i \delta_i w_i\eta_i - \sum_{k,j}\omega_k\log S^{(0)}_{kj},\qquad
U = \sum_i \delta_i w_iW_i - \sum_{k,j}\omega_k a_{kj},\quad a_{kj} = S^{(1)}_{kj}/S^{(0)}_{kj},$$

$$I = \sum_{k,j}\omega_k\left[S^{(2)}_{kj}/S^{(0)}_{kj} - a_{kj}a_{kj}^\top\right].$$

$S^{(2)}$ の項は被験者ごとの係数にまとめられる:

$$\sum_{k,j}\omega_k S^{(2)}_{kj}/S^{(0)}_{kj} = W^\top \mathrm{diag}(r_i c_i) W,\qquad
c_i = \sum_{k: i \in R_k} h_k - \delta_i\, g_{k(i)},$$

$h_k = \sum_j \omega_k/S^{(0)}_{kj}$、$g_k = \sum_j \omega_k f_j/S^{(0)}_{kj}$、$k(i)$ は $i$ の事象時刻。$c_i$ は $h$ の累積和の差（$t_i$ までと $e_i$ まで）で求まる。$r_i c_i$ は被験者の期待事象数。

したがって $I = W^\top\mathrm{diag}(rc)W - A^\top\mathrm{diag}(\omega)A$。$A$ は死亡ごと（Efron のステップごと）に $a_{kj}$ を並べた行列。$Z$ は疎なので第 1 項は疎行列の積、第 2 項は $E \times P$（$E$: 事象数、$P$: 列数）の密行列の積になる。$S^{(1)}_k$ は「被験者 $i$ を $\tau_k \le t_i$ を満たす最後の $k$ に足し、$\tau_k \le e_i$ を満たす最後の $k$ から引く」バケットを作り、逆向きの累積和で求める。

### 固定効果の共分散と有効自由度

$(b, \beta)$ の罰則つきヘッセ行列 $H = I + \mathrm{blockdiag}(\Sigma^{-1}, 0)$ の逆行列の $\beta$ ブロックを固定効果の共分散とする（Therneau et al. 2003; `coxme` の `vcov` と一致）。$u$ のパラメータでは $J = \mathrm{blockdiag}(\Lambda, I)$、$H_u = J^\top H J$ で

$$\mathrm{Var}(\hat\beta) = \left[I_{\beta\beta} - I_{\beta b}\Lambda(\Lambda^\top I_{bb}\Lambda + I)^{-1}\Lambda^\top I_{b\beta}\right]^{-1}.$$

有効自由度（Therneau & Grambsch 2000, 5.8 節と 9 章）は $\mathrm{df} = \mathrm{tr}(H^{-1} I)$。$u$ のパラメータでは

$$\mathrm{df} = P - \mathrm{tr}\left[(H_u^{-1})_{uu}\right].$$

### 要約の検定

- 帰無の対数尤度: $\ell(0, 0)$（offset のみ）。
- Integrated: $\chi^2 = 2(\mathrm{IL} - \ell_0)$、自由度 = 固定効果の数 + 分散パラメータの数。
- Penalized: $\chi^2 = 2(\mathrm{PPL} - \ell_0)$、自由度 = df。
- AIC $= \chi^2 - 2\,\mathrm{df}$、BIC $= \chi^2 - \log(\text{事象数})\,\mathrm{df}$（`print.coxme` の表と同じ定義。事象数で割る）。

## 4. 推定・計算の手順

1. 公式を `model_matrix` で展開し、固定効果の $X$、ランダム効果の項ごとの列（切片と傾き）と水準を作る。`(1 | a/b)` は `(1 | a) + (1 | a:b)` に展開し、`a:b` は水準の組み合わせ。
2. 固定効果だけの Cox モデル（`cox_ph` と同じ同順位）を初期値 $\beta$ に、$u = 0$。
3. 外側: $\theta$ は各項の Cholesky 因子 $\Lambda_t$ の要素（対角は SD 尺度、下限 0）。初期値は対角 0.3、非対角 0。L-BFGS-B（中心差分の勾配）で $-\mathrm{IL}(\theta)$ を最小化する。分散パラメータが 1 つなら有界の 1 次元探索（Brent）。
4. 内側: $\theta$ ごとに $(u, \beta)$ をニュートン法（ステップ半減）で最大化。直前の解から始める。収束は PPL の変化と勾配の大きさで判定。
5. 収束後、$\hat b = \Lambda\hat u$、共分散、df、各対数尤度を計算する。

## 5. 数値上の注意

- 分散が 0 の境界では $\Lambda_t = 0$ で、$u$ は 0 に縮む。$\Sigma^{-1}$ を使わないので計算は止まらない。
- 計算量は 1 回の評価で $O(\mathrm{nnz}(W) + E P^2)$。グループ数が大きいと $P \times P$ の密行列が重くなる（後述の `coxme` の疎近似は使わない）。
- 内側の解を前回の解から始めると、中心差分の勾配にわずかな経路依存が入る。内側の収束判定を厳しくする（勾配のノルム $10^{-9}$ 以下）。
- $\log|\Lambda^\top I_{bb}\Lambda + I|$ は Cholesky 分解で計算する。

## 6. R と出力を合わせるための公開仕様（出力比較で決めた点）

- `coxme` の `loglik` は `NULL`、`Integrated`、`Penalized` の 3 つ。`print` の「Fitted」は罰則なしの $\ell(\hat\beta, \hat b)$。
- `vcov(fit)` は固定効果のブロック。`fit$variance` は $(b, \beta)$ の順の全体。
- `VarCorr(fit)` は相関つきの項で、対角が分散、非対角が相関の行列を返す。
- `coxme` の公式 `(x | g)` は切片なしの傾きだけを表す。statract は lme4 と同じく `(x | g)` を `(1 + x | g)` とみなす。傾きだけなら `(0 + x | g)` と書く。
- `(1 | a/b)` の R の項名は `a/b` と `a`。statract は `a` と `a:b`。
- 水準数が多い項（`coxme.control(sparse = c(50, 0.02))` の既定で 50 水準超）では、R は情報行列の一部を省く近似を使う。statract は常に厳密に計算する。fixture は `sparse` を大きくして近似を切った R と比べる。

## 7. 検証

`tests/r_oracle/test_cox_mixed.py`（fixture `cox_mixed.json`、スクリプト `cox_mixed.R`、データ `coxme_a.csv`・`coxme_b.csv`）。11 例: ランダム切片（Efron、Breslow）、交差、入れ子、層、切片と傾きを別の項、相関つきの傾き、重み、計数過程、80 水準（厳密と疎近似）。

R の分散の探索は最適点の手前で止まることがあるので、比べ方を 2 つに分ける。

- 分散を R の推定値に固定した当てはめ（R は `vfixed`、statract は `variances=`）: 係数 rtol 1e-6、共分散 rtol 1e-5、3 つの対数尤度 atol 1e-6、有効自由度 rtol 1e-6、BLUP atol 1e-6。全例で通る（実測は 1e-8 程度）。
- 分散も推定する当てはめ: statract の積分対数尤度が R 以上で差 0.01 未満、分散 rtol 2e-2 / atol 3e-3、係数 atol 2e-3。差は多くの例で 1e-6 未満。入れ子の例（病棟の分散が 0 に近い）で 0.007、傾きの例で 0.0008 だけ statract が高い。

速度（同じ機械、R は疎近似なし）: n = 5,000・100 施設で 2.3 秒（R 4.4 秒）、n = 20,000・300 施設で 5 秒（R 107 秒）、交差 2 項で 13 秒（R 315 秒）。

## 8. R と違うところ、未確認の点

- 6 節の公式の書き方と疎近似。
- `variances=` で分散を固定したとき、積分対数尤度の自由度に分散の数を入れない（R の `vfixed` と同じ）。
- REML 型の推定（`coxme` にもない）、分散の標準誤差、ランダム効果の予測区間、`||`（相関なしの傾き）は未実装。
- グループ数が非常に多いと $P \times P$ の密な情報行列が重い。次の手は、$b$ のブロックが「疎な行列 − 低ランク」であることを使う Woodbury 型の解法。
