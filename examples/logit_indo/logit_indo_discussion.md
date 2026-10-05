# logit_indo_discussion

数値は [logit_indo_results.md](logit_indo_results.md) の生成物を読む。

## CQ への答え

n=602 の全例で施設変量 `fit_mixed(..., family="binomial")` を主解析とした。インドメタシンの固定効果オッズ比は約 0.47（95% CI 約 0.28–0.78）。クラスタ分散は約 0.30、MOR は約 1.68（`median_odds_ratio`）。施設 BLUP（`plot_random_effects`）では `1_UM` が正、`2_IU` が負。同じ共変量の固定効果調整 GLM の OR は約 0.47 で向きは一致する。

## 当たり外れ

未調整 / 調整 GLM の OR が 1 を下回る方向は Elmunzer NEJM 2012 と一致する。lme-python `glmer` で表・BLUP・MOR まで書けた。

## 限界

- 主エンジンは lme-python。R `glmer` との数値照合は vs-r のガウス fixture ほどきつくない。
- 教学パッケージデータであり、原論文の解析コードの完全再現ではない。
- ホールドアウト較正 / DCA は予測モデル向けであり、本例では載せていない。
