# gamm4 の事後共分散と有効自由度（テスト用 R スクリプト）

## 1. 目的

`tests/r_oracle/scripts/gamm4.R` の補助関数 `corrected_vp(fit)` が計算する量を定める。`gamm4::gamm4()` の当てはめ結果から、GAM 係数のベイズ事後共分散 $V_p$ と係数ごとの有効自由度（effective degrees of freedom、edf）を、ピボットなしの Cholesky 分解で求める。結果は fixture の `corrected.std_errors`、`corrected.edf`、`corrected.smooth_std_errors` になる。

R はここでは計算環境としてだけ使う。参照するのは公開されたアクセサ（`lme4::getME`、`weights(..., type = "working")`、`predict(..., type = "lpmatrix")`）だけ。

## 2. 出典

- Wood SN (2017). *Generalized Additive Models*, 2nd ed. sec. 6.6（平滑化と混合モデルの対応）、6.10–6.11（事後分布、有効自由度）。
- Wood SN, Scheipl F, `?gamm4` の Value 節（`gam` 部分の `Vp` の意味）。
- Wood SN (2004). Stable and efficient multiple smoothing parameter estimation for generalized additive models. *JASA* 99:673–686（ベイズ事後共分散）。

## 3. モデルと式

$X$: GAM の計画行列（lpmatrix、$n \times p$）。$S_\lambda$: 推定された平滑化パラメータによる罰則。$w$: 収束時の作業重み。審査者（examiner）のランダム切片による周辺作業共分散

$$V = \mathrm{diag}(1/w) + Z_e \Lambda_e \Lambda_e^\top Z_e^\top .$$

求める量（二項とポアソンなので分散パラメータは 1）は

$$V_p = (X^\top V^{-1} X + S_\lambda)^{-1},\qquad \mathrm{edf}_i = \left[V_p X^\top V^{-1} X\right]_{ii}.$$

### 混合モデル表現による計算

平滑化項は混合モデルでは「罰則なし部分（固定効果）$X_f$」と「罰則あり部分（ランダム効果）$Z_s$」に書き直される（Wood 2017, sec. 6.6）。相対共分散因子 $\Lambda_s$ で尺度をそろえると、係数 $u_s$ の事前分布は $N(0, I)$。計画行列

$$A = [X_f,\; Z_s \Lambda_s]$$

について

$$V_u = \left(A^\top V^{-1} A + \mathrm{blockdiag}(0, I)\right)^{-1}.$$

$A$ と $X$ は同じ列空間を張るので $A = XM$ となる $p \times p$ 行列 $M$ がある。元の係数は $\beta = M(\beta_f, u_s)$ なので

$$V_p = M V_u M^\top .$$

$M$ が正則なら、これは $(X^\top V^{-1}X + M^{-\top}\,\mathrm{blockdiag}(0, I)\,M^{-1})^{-1}$ で、上の定義と一致する（罰則 $S_\lambda = M^{-\top}\mathrm{blockdiag}(0, I)M^{-1}$）。

## 4. 計算の手順

1. `getME` で $X_f$、$Z$、$\Lambda$、項ごとの列範囲を得る。グループ名 `examiner` の列を $e$、それ以外（平滑化項）の列を $s$ とする。
2. 作業重み $w$ を `weights(mer, type = "working")` で得る。
3. $V$ を密行列で作り、ピボットなしの Cholesky 分解 $V = R^\top R$ を求める。
4. $R^{-\top}A$、$R^{-\top}X$ を後退代入で求め、$V_u$ を計算する。
5. $M$ を最小二乗 $XM = A$（QR）で求め、$XM = A$ が成り立つことを確かめる。
6. $V_p = MV_uM^\top$（対称化する）、$\mathrm{edf} = \mathrm{diag}(V_p X^\top V^{-1}X)$。

この方法では、平滑化項の再パラメータ化を当てはめ結果のフィールドから読む必要がない。

## 5. 数値上の注意

- ピボットありの分解でピボットを無視すると、$V_p$ が約 1 % ずれる。これがこの関数を作った理由。
- 分散成分が 0 の平滑化項では $\Lambda_s$ の列が 0 になる。$M$ が特異になるが、$V_p = MV_uM^\top$ はその方向の分散 0 として正しい極限を与える（逆行列 $\Lambda_s^{-1}$ は使わない）。
- $n = 800$ なので $n \times n$ の密行列で十分。

## 6. R と出力を合わせるための公開仕様

- `?gamm4`: `gam` 部分の `Vp` は GAM 係数のベイズ事後共分散。
- `?lme4::getME`: `"X"`、`"Z"`、`"Lambda"`、`"Gp"`、`"cnms"`。
- `?lme4::merMod`: `weights(type = "working")`。

## 7. 検証

- 利用先: `tests/r_oracle/test_gamm4.py::test_gamm_matches_gamm4`（SE rtol 1e-4）。
- R が作業環境にないため、このスクリプトはまだ実行していない。fixture `tests/r_oracle/fixtures/gamm4.json` は変えていない。確認するには `Rscript tests/r_oracle/scripts/gamm4.R` で再生成し、`corrected.*` を今の fixture と比べる。

## 8. R と違うところ、未確認の点

- 未実行。次の前提を置いている: グループ名が `examiner`、mer の作業重みが gam の作業重みと同じ、lpmatrix と mer の行の並びが同じ（データに欠測なし）。
- $A$ が $X$ の列空間にあることは `stopifnot` で確かめる。
