# 多重代入の統合（Rubin のルール）

対象: `src/statract/models/impute.py` の `_barnard_rubin` と `_fraction_missing_information`（呼び出し元 `pool`）。

## 1. 目的

$m$ 個の代入データでの推定を統合するときの自由度（Barnard–Rubin の小標本自由度）と、欠測情報の割合（fraction of missing information, FMI）を計算する。R の `mice::pool()` の出力（`df`、`fmi`）と比べる。

## 2. 出典

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley, sec. 3.1（式 3.1.6 の自由度、式 3.1.10 の欠測情報の割合）。
- Barnard, J. & Rubin, D. B. (1999). Small-sample degrees of freedom with multiple imputation. *Biometrika* 86:948–955。
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. CRC, sec. 2.3.6（各量の定義の整理）。

## 3. モデルと式

各項について、代入内分散の平均を $\bar U$、代入間分散を $B$、全分散を $T=\bar U+(1+1/m)B$ とする。完全データの自由度を $\nu_{com}$ とする。

- $\lambda = (1+1/m)B/T$（欠測に由来する分散の割合）。
- $r = (1+1/m)B/\bar U$（分散の相対増加、relative increase in variance, RIV）。
- 大標本の自由度（Rubin 1987, 式 3.1.6）: $\nu_{old} = (m-1)/\lambda^2$。
- 観測データの自由度（Barnard & Rubin 1999）: $\nu_{obs} = \dfrac{\nu_{com}+1}{\nu_{com}+3}\,\nu_{com}\,(1-\lambda)$。
- 統合の自由度: $\nu = \dfrac{\nu_{old}\,\nu_{obs}}{\nu_{old}+\nu_{obs}}$。$\nu_{com}=\infty$ なら $\nu=\nu_{old}$。
- 欠測情報の割合（Rubin 1987, 式 3.1.10; van Buuren 2018, sec. 2.3.6）: $\gamma = \dfrac{r+2/(\nu+3)}{r+1}$。

## 4. 計算の手順

1. $\lambda$ を求め、下限 $10^{-4}$ で切る（5 節）。
2. $\nu_{old}$ を求める。$\nu_{com}$ が無限なら $\nu_{old}$ を返す。
3. $\nu_{obs}$ と調和型の結合で $\nu$ を返す。
4. FMI は 3 節の式をそのまま使う。$\nu$ は 1–3 の値。

## 5. 数値上の注意

- $B=0$（どの代入でも推定が同じ）だと $\lambda=0$、$\nu_{old}=\infty$ になる。$\lambda$ に下限 $10^{-4}$ を置き、$\nu_{old}$ を有限（約 $10^8(m-1)$）にする。このとき $\nu\approx\nu_{obs}$ になり、欠測がない場合の正しい極限（$\nu_{com}$ に近い値）になる。下限の値は自分で決めた。fixture にこの場合はない。
- `pool` が表に出す `lambda` 列は下限を当てる前の値。

## 6. R と出力を合わせるための公開仕様

- `mice::pool` のヘルプページ: `dfcom` は完全データの自由度。既定ではモデルから取る（線形モデルと GLM は残差自由度）。
- 表の列 `ubar`、`b`、`t`、`dfcom`、`df`、`riv`、`lambda`、`fmi` は `mice::pool` の出力と同じ意味。

## 7. 検証

- `tests/r_oracle/test_mice_pool.py::test_pool_matches_mice`（ols、logit、cox。全列 rtol $10^{-6}$。fixture `tests/r_oracle/fixtures/mice_pool.json`、`scripts/mice_pool.R` で作成）。
- `::test_pool_options`（$\nu_{com}=\infty$ で $\nu = 4/\lambda^2$、$m=5$）。
- `::test_impute_chained_recovers_estimates_under_mar`（推定値 atol 0.05）。

## 8. R と違うところ、未確認の点

- $\lambda$ の下限は fixture では確かめられない。$B$ がほぼ 0 の項で R と値が違う可能性がある。
