# PBC 反復検査における対数ビリルビン（LMM と GAM）

## 0. なぜこの切り口か

indo の混合は二項 `fit_mixed(..., family="binomial")` である。ガウス `fit_mixed`（`lmer` 相当）、患者クラスタ SE、および現行制限つき GAM を縦断検査で本編に載せる。

## 1. Clinical Question

無作為化 PBC の反復検査で、病日と D-ペニシラミンは対数ビリルビンと関連するか。患者内相関を変量切片で入れたとき、集団の病日カーブ（GAM）はどうか。

## 2. 現象・既知

PBC ではビリルビンが予後と強く結びつく。同一患者の反復測定は独立ではない。

## 3. 仮説

病日が進むと対数ビリルビンは上昇しうる。治療の固定効果は小さく、患者間分散は残差より無視できない。

## 4. 結論の形

変量切片 LMM の固定効果、同一平均構造の OLS + クラスタ SE、病日の GAM 平滑（主解析の一部）。

## 5. 撤退基準

無作為化患者が復元できない、または `bili>0` の訪問が極端に少ないとき止める。GAM が平滑1本しか受け付けなくても LMM があれば例は成立するが、GAM は本編から外さない。

## 6. Key messages（予定）

- ガウス LMM は `fit_mixed` の既定。二項は `family="binomial"`（indo）。
- GAM はこの版では `family="gaussian"`・`cr`・平滑1本・REML。治療と変量は GAM に入れられない。

## 7. 対象のイメージ

`pbcseq` の無作為化患者と、その反復行。

## 8. 限界の先取り

GAM に `dp` も `(1|id)` も無い。クラスタ SE は平均構造だけを OLS にした比較。GPL データ。

## 9. 統計解析手法

訪問 $ij$、患者 $i$。主解析の LMM:

$$
\log(\mathrm{bili}_{ij})=\beta_0+\beta_1\mathrm{day\_years}_{ij}+\beta_2\mathrm{dp}_i+u_i+\varepsilon_{ij},\quad u_i\sim N(0,\sigma_u^2).
$$

比較: 同じ平均構造の OLS に患者クラスタのサンドイッチ。GAM（主解析）:

$$
\log(\mathrm{bili})=\beta_0+s(\mathrm{day\_years})+\varepsilon,
$$

$s$ は立方回帰スプライン（`smooth`、`k` は一意な病日数以下）。実装: `fit_mixed`、`fit_ols`、`cluster_covariance`、`gam` / `smooth`。
