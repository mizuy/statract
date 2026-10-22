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

**$r$ が整数でないとき。** $R V_j R^\top = U\Lambda U^\top$（固有値は降順）とする。$k = \lceil r\rceil$ として、上位 $k$ 個の固有成分を使う。$r$ の端数の扱いは Wood (2013b) の非整数階数の擬似逆行列による。最後の 2 成分を一つの $2\times2$ ブロックにまとめ、帰無分布が $\chi^2_{\lfloor r\rfloor}$ と $\chi^2_{\lceil r\rceil}$ のあいだを滑らかにつなぐようにする。ブロックの具体的な形は論文 sec. 3 で確認してから書く（§8）。

**帰無分布。**

- 尺度が既知（二項、ポアソン、gamm）: $T_r$ の帰無分布は、重み付きのカイ二乗の和で近似する（整数 $r$ なら $\chi^2_r$）。p 値は Davies または Liu–Tang–Zhang の方法で求める。
- 尺度を推定（ガウス、ガンマ）: $F = (T_r/r)/\hat\phi$ を $F(r, n - \mathrm{edf})$ と比べる。

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

- 非整数階数の擬似逆行列の $2\times2$ ブロックの形と、それに対応する帰無分布。論文の本文で確認する。クラウド環境から arXiv と出版社サイトに届かなかったため（2026-10-22）、このノートはまだ記憶にもとづく。確認が済むまで実装に入らない。
- 重み付きカイ二乗の和の p 値の方法（Davies の厳密法か、モーメント近似か）。R と合わせる精度に応じて決める。
