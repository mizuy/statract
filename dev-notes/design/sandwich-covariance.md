# サンドイッチ共分散（不均一分散頑健とクラスタ頑健）

## 1. 目的

回帰係数のサンドイッチ型共分散を計算する。

- 不均一分散頑健（heteroskedasticity-consistent、HC）共分散 HC0–HC5、HC4m、定数型。出力を比べる相手は R の `sandwich::vcovHC()`。
- 一元・多元のクラスタ頑健共分散。出力を比べる相手は R の `sandwich::vcovCL()`。Cox モデルにも使う。

実装: `src/statract/models/covariance.py` の `_hc_omega`、`_score_scalars`、`cluster_covariance`、`_label_codes`、`_leverage_adjusted_sums`。

## 2. 出典

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica* 48(4), 817–838.（HC0）
- MacKinnon, J. G. & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics* 29(3), 305–325.（HC1、HC2、HC3）
- Long, J. S. & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician* 54(3), 217–224.（HC3 の推奨）
- Cribari-Neto, F. (2004). Asymptotic inference under heteroskedasticity of unknown form. *Computational Statistics & Data Analysis* 45(2), 215–233.（HC4）
- Cribari-Neto, F. & da Silva, W. B. (2011). A new heteroskedasticity-consistent covariance matrix estimator for the linear regression model. *AStA Advances in Statistical Analysis* 95(2), 129–146.（HC4m）
- Cribari-Neto, F., Souza, T. C. & Vasconcellos, K. L. P. (2007). Inference under heteroskedasticity and leveraged data. *Communications in Statistics – Theory and Methods* 36(10), 1877–1888.（HC5）
- Zeileis, A. (2004). Econometric computing with HC and HAC covariance matrix estimators. *Journal of Statistical Software* 11(10). 2 節（一般形 式 6、表 1 の $\omega_i$）。
- Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika* 73(1), 13–22.（クラスタ頑健分散）
- Arellano, M. (1987). Computing robust standard errors for within-groups estimators. *Oxford Bulletin of Economics and Statistics* 49(4), 431–434.
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics* 29(2), 238–249. 2 節（包除の式 2.13）。
- Bell, R. M. & McCaffrey, D. F. (2002). Bias reduction in standard errors for linear regression with multi-stage samples. *Survey Methodology* 28(2), 169–181.（クラスタ版 HC2）
- Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources* 50(2), 317–372. VI.B 節。
- Zeileis, A., Köll, S. & Graham, N. (2020). Various versatile variances: an object-oriented implementation of clustered covariances in R. *Journal of Statistical Software* 95(1). 2–3 節。

## 3. モデルと式

$n$ 行、係数 $k$ 個。推定方程式の行ごとのスコアを $s_i \in \mathbb{R}^k$ とする。GLM と最小二乗では $s_i = r_i x_i$ と書ける（$r_i$ は作業重みを含む残差のスカラー）。$h_i$ は重み付きハット行列の対角。ブレッド $B = n (X^\top W X)^{-1}$（GLM では分散パラメータ倍）。

一般形（Zeileis 2004, 式 6）:

$$\widehat{V} = \frac{1}{n} B \left(\frac{1}{n} X^\top \Omega X\right) B, \quad \Omega = \mathrm{diag}(\omega_1, \dots, \omega_n)$$

$\omega_i$（Zeileis 2004, 表 1）。$\bar p = \mathrm{round}(\sum_i h_i)$:

| 種類 | $\omega_i$ |
|---|---|
| HC0 | $r_i^2$ |
| HC1 | $r_i^2 \, n/(n-k)$ |
| HC2 | $r_i^2/(1-h_i)$ |
| HC3 | $r_i^2/(1-h_i)^2$ |
| HC4 | $r_i^2/(1-h_i)^{\delta_i}$、$\delta_i = \min(4, n h_i/\bar p)$ |
| HC4m | $\delta_i = \min(1, n h_i/\bar p) + \min(1.5, n h_i/\bar p)$ |
| HC5 | $r_i^2/\sqrt{(1-h_i)^{\delta_i}}$、$\delta_i = \min(n h_i/\bar p, \max(4, 0.7 n h_{\max}/\bar p))$ |
| const | $\sum_j r_j^2/(n-k)$（全行で同じ） |

クラスタ頑健（Liang & Zeger 1986; Arellano 1987）。クラスタ $g$ のスコア和 $U_g = \sum_{i \in g} s_i$。多元クラスタでは、クラスタ次元の空でない部分集合 $S$ について包除する（Cameron, Gelbach & Miller 2011, 式 2.13）:

$$M = \frac{1}{n} \sum_{S} (-1)^{|S|+1} c_S \sum_{g \in G(S)} U_g U_g^\top$$

$G(S)$ は $S$ の次元の交差で実際に現れるセル。$c_S = G_S/(G_S - 1)$（調整あり、$G_S > 1$）、それ以外は 1。HC1 ではさらに $(n-1)/(n-k)$ を掛ける。結果は $\widehat V = B M B / n$。

クラスタ版 HC2 / HC3（Bell & McCaffrey 2002; Cameron & Miller 2015, VI.B 節）。$a_i = \sqrt{w_i} x_i$、$e_i = r_i/\sqrt{w_i}$、$H_{gg} = A_g (A^\top A)^{-1} A_g^\top$ として

$$U_g = A_g^\top (I - H_{gg})^{-\alpha} e_g, \quad \alpha = 1/2 \ (\text{HC2}),\ 1 \ (\text{HC3})$$

## 4. 推定・計算の手順

### HC（`_hc_omega`）

1. スコア行列 $s_i$ と計画行列の行 $x_i$ から $r_i = \langle s_i, x_i\rangle / \langle x_i, x_i\rangle$ を求める。
2. 表の式で $\omega_i$ を作る。種類 `"HC"` は HC0 の別名。
3. 一般形に代入する（既存の `_sandwich_from_omega`）。

### クラスタ（`cluster_covariance`）

1. 種類の既定は最小二乗が HC1、それ以外が HC0。
2. クラスタラベルを列ごとに整数コードにする。並べ替えできない混在型のラベルは初出順でコード化する。
3. 部分集合 $S$ ごとに、列コードの組を一意化して交差セルを作り、$U_g$ を足し上げる。
4. 符号 $(-1)^{|S|+1}$ と $c_S$ を掛けて足す。
5. Cox モデルでは、スコア残差に症例重みを掛けたものを $s_i$ とする。

## 5. 数値上の注意

- スコアが全部 0 の行は $r_i = 0$。非有限の $r_i$ も 0 にする。
- 多元クラスタの結果は正定値とは限らない。三元の例では対角が負になる（R も標準誤差が NaN）。補正はしない。
- クラスタ版 HC2 / HC3 で $I - H_{gg}$ の固有値が 0 に近いとき（$h = 1$ の行）は、その成分を落とす（一般化逆行列）。閾値は $100\, \varepsilon \times$ クラスタの行数。
- ラベルの欠損は `ValueError("cluster has missing values; drop those rows before fitting")`。

## 6. R と出力を合わせるための公開仕様

- `?vcovHC`: `type` は "HC3"（既定）、"const"、"HC"、"HC0"–"HC5"、"HC4m"。
- `?vcovCL`: `cadjust = TRUE` で $G/(G-1)$。`type = "HC1"` で $(n-1)/(n-k)$。既定の `type` は線形モデルで HC1、それ以外で HC0。`multi0` などの正定値補正は使わない。
- `?coxph`: `cluster` を指定したときの頑健分散は、症例重みを掛けたスコア残差のクラスタ和から作る。調整なしの一元クラスタでこれと一致する。

## 7. 検証

- HC: `tests/r_oracle/test_glm_nb.py`（HC0、rtol 1e-6）、`test_risk_ratio.py`（rtol 1e-6、atol 1e-14）、`test_standardize_glm.py`（HC0 / HC3）、`tests/models/test_weighting_outcome.py`。
- クラスタ: `tests/r_oracle/test_cox_two_way.py`、fixture `cox_two_way.json`（8 例、rtol 1e-6、atol 1e-12）。調整なし一元クラスタが `cox_ph(cluster=)` と rtol 1e-10。ラベル配列と列名で rtol 1e-12。`test_risk_ratio.py`（5）、`test_standardize_glm.py`（`*-fixed-site`）。
- クラスタ版 HC2 / HC3 は R の fixture がない。1 行 1 クラスタ・調整なしで `hc_covariance` の HC2 / HC3 と一致すること（相対 1e-15、重み付き最小二乗、ポアソン、負の二項）を手で確かめた。

## 8. R と違うところ、未確認の点

- クラスタ版 HC2 / HC3 は一元クラスタだけ。多元クラスタと Cox では `ValueError` にする。R との数値照合はしていない。
- 作業時点では Cox のスコア残差が未実装（別作業）だったため、`test_cox_two_way.py` は通していない。Breslow 法の簡易スコア残差（コミットしない）で Breslow の 2 例を照合し、相対 3e-14 で一致した。Efron 法の例、三元、HC1 は Cox 残差の実装待ち。
