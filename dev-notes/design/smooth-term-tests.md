# 平滑項の検定（edf、参照自由度、統計量、p 値）

## 1. 目的

`summary.gam` の `s.table` に当たる表を出す。平滑ごとに edf、参照自由度（Ref.df）、F またはカイ二乗の統計量、p 値。帰無仮説は「その平滑の関数が 0」。`gam`（GamFit / GeneralFit）と `gamm`（GammFit）の両方で使う。

mgcv は GPL なので、論文と公開ヘルプからこのノートを書き、コードはこのノートから書く。

## 2. 出典

- Wood SN (2013b). On p-values for smooth components of an extended generalized additive model. *Biometrika* 100:221–228. 主要な検定。
- Wood SN (2013a). A simple test for random effects in regression models. *Biometrika* 100:1005–1010（零空間のない項。ここでは扱わない）。
- Marra G, Wood SN (2012). Coverage properties of confidence intervals for generalized additive model components. *Scand J Stat* 39:53–74。
- Nychka D (1988). Bayesian confidence intervals for smoothing splines. *JASA* 83:1134–1143。
- Wood SN (2017). *Generalized Additive Models*, 2nd ed. sec. 6.12.1。
- `?summary.gam` の Details 節。

## 3. モデルと式

平滑 $j$ について、$\hat\beta_j$ は係数、$V_j$ はベイズ事後共分散 $V_\beta$ の対応するブロック、$X_j$ はデータでの基底行列。$\hat f_j = X_j\hat\beta_j$。

**統計量。** 階数 $r$ の擬似逆行列を使う。

$$T_r = \hat f_j^\top (X_j V_j X_j^\top)^{r-}\hat f_j .$$

$X_j = QR$（薄い QR）とすると、$T_r = \hat\beta_j^\top R^\top (R V_j R^\top)^{r-} R\hat\beta_j$ で、$n$ 次元ではなく $p_j$ 次元で計算できる。`?summary.gam` によると、mgcv は統計量の成分を反復重みで重み付けする。つまり $X_j$ の行に $\sqrt{w_i}$ を掛けてから QR をとる。

**$r$ の決め方。** $r$ は平滑の edf から決める。`?summary.gam`: 「p 値の計算には $\mathrm{tr}(2A - AA)$ にもとづく別の edf が適している」。平滑 $j$ のブロックでは、$F = V_\beta\mathcal{I}$ として $\mathrm{tr}(2F_{jj} - (FF)_{jj})$。これを Ref.df とする。

**$r$ が整数でないとき（Wood 2013, sec. 2.2）。** $R V_j R^\top = U\Lambda U^\top$（固有値は降順）、$d = U^\top R\hat\beta_j$ とする。$k = \lfloor r\rfloor + 1$、$\nu = r - k + 1$、$\rho = \sqrt{\nu(1-\nu)/2}$。上位 $k-2$ 成分はふつうの逆数、最後の 2 成分は

$$\tilde B = \Lambda_{2}^{-1/2}\begin{pmatrix}1 & \rho\\ \rho & \nu\end{pmatrix}\Lambda_{2}^{-1/2}$$

でまとめる。$T_r = \sum_{i \le k-2} d_i^2/\lambda_i + d_{(2)}^\top\tilde B\,d_{(2)}$。$r$ が整数なら $\nu = 0$ で、ふつうの階数 $r$ の擬似逆行列になる。

論文の $T_r$ は、最後の 2 つの固有ベクトルの符号で値が変わる（固有値ソルバーの符号は任意）。`s.table` と合うのは、交差項が負にならない向き（$|d_{(2)}|$ を使う）だった（2026-10-22、mgcv 1.9-1 と rtol 1e-3 で一致）。

**帰無分布（sec. 2.3）。**

- 尺度が既知（二項、ポアソン、gamm）: $T_r \sim \chi^2_{k-2} + \nu_1\chi^2_1 + \nu_2\chi^2_1$、$\nu_1 = \{\nu + 1 + \sqrt{1-\nu^2}\}/2$、$\nu_2 = \nu + 1 - \nu_1$。p 値は正規変数 2 つを Gauss–Legendre 求積で積分して求める（Imhof 型の積分は精度が足りなかった）。
- 尺度を推定（ガウス、ガンマ）: p 値は $P(Q > T_r W/\kappa)$、$W \sim \chi^2_\kappa$、$\kappa = n - \mathrm{edf}$。$W$ の分位点で積分する。表には $F = T_r/r$ を出す。$r$ が整数なら $F(r, \kappa)$ 検定と一致する。

## 4. 計算の手順

1. 平滑ごとに $X_j$（作業重みを掛ける）、$\hat\beta_j$、$V_j$、$F$ を用意する。GeneralFit は `blocks[j].columns`、GammFit は `_slices[j]` が列範囲。
2. Ref.df $= \mathrm{tr}(2F_{jj} - (FF)_{jj})$。
3. QR、固有分解、階数 $r$ の擬似逆行列、$T_r$。
4. p 値。重み付きカイ二乗の和の分布関数は、自前で実装する（Davies 1980 のアルゴリズム、または Liu et al. 2009 のモーメント近似）。
5. `smooth_table()` に `ref_df`、`statistic`、`p_value` 列を足す。

## 5. 数値上の注意

- edf が 1 に近い平滑（ほぼ直線）では $r \approx 1$ になる。端数の扱いがこの範囲で安定していることを確かめる。
- テンソル積や `by` の平滑は列ブロックがそのまま 1 つの平滑なので、同じ手順でよい。
- `re` 基底（零空間なし）は Wood (2013a) の別の検定になる。最初は `NaN` を返し、docstring に書く。

## 6. R と出力を合わせるための公開仕様

- `?summary.gam`: `s.table` の列 `edf`、`Ref.df`、`F` または `Chi.sq`、`p-value`。p 値は平滑化パラメータを既知として扱う。
- `gamm4` の `summary(g$gam)` も同じ表を返す。

## 7. 検証

- `gam_summary.R` に `summary(m)$s.table` を足す。二項（カイ二乗）、重み付きガウス（F）、ポアソン、ガンマ、edf が 1 に近い平滑、2 本の平滑。
- `gamm4.json` に `summary(g$gam)$s.table` を足す。gamm4 0.2-6 と Matrix 1.6 の組では $V_p$ が約 1% ずれるので（`gamm-edf.md`）、比べる相手は補正した $V_p$ から作った値にする。

## 8. R と違うところ、未確認の点

- 2026-10-22: 論文 PDF（mizuy から受け取った）で §3 の式を確認し、実装した（`models/gam/smooth_test.py`）。
- `ref_df` と統計量は mgcv の `s.table` と rtol 1e-3 で一致する。ガンマの尺度は、はじめ REML 基準の尺度（0.2771）を使っていて、mgcv の Pearson 推定（0.2944）と約 6% ずれていた。2026-10-22 に、表示の尺度、共分散、検定を Pearson 推定 $\sum w_i (y_i-\hat\mu_i)^2/\hat\mu_i^2 / (n - \mathrm{edf})$ に替え、F は rtol 3e-3 で一致するようになった。
- p 値は mgcv より小さい（例: gam の age 項 0.29 に対し mgcv 0.37、b1 で 1.88e-5 に対し 2.03e-5）。statract の p 値は論文の帰無分布どおりで、モンテカルロで確かめた。mgcv 1.9-1 は論文と別の計算をしていると考えられるが、GPL のソースは読まないので理由は特定しない。
- gamm4 の `s.table` は `Ref.df = edf` を使う。`GammFit.smooth_table` もそれに合わせた。gamm4 の pivot の問題で $V_p$ が少しずれるため、比較は rtol 5%。
- `re` 基底の項は `NaN`。因子 `by` の平滑は全水準をまとめて 1 つの検定にする（mgcv は水準ごと）。
