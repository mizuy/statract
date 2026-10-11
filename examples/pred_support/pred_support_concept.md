# SUPPORT2 における 180 日死亡の予測確率（較正 / DCA）

## 0. なぜこの切り口か

`logit_indo` は RCT の効果推定（施設 GLMM）であり、ホールドアウト較正は外した。較正曲線と決定曲線（DCA）は **予測（点推定の確率）** の例が必要である。SUPPORT2 は公開の重症予後コホートで、二項 GLM の train / hold-out を教えやすい。

## 1. Clinical Question

SUPPORT2 の研究登録時点（主に day 3 生理）の臨床・生理変数から、180 日死亡確率 $\hat\pi$ を推定したとき、ホールドアウトでの較正・Brier・決定曲線は、切片のみや年齢+性別と比べてどうか。見かけ（学習標本）較正はホールドアウトより楽観的か。

## 2. 現象・既知

Knaus / Harrell らの SUPPORT は 2 か月・6 か月生存の予後モデルの定番データである。教学 CSV は hbiostat。本例はライブラリの `sps` / `surv6m` を使わず、生の共変量から GLM を組む（スコアの再利用はしない）。

## 3. 仮説

多変量 GLM は null / 年齢+性別より Brier が小さく、中程度の閾値で net benefit が treat-all を上回る。見かけ較正はホールドアウトより対角線に近い。

## 4. 結論の形

学習標本の OR forest、ホールドアウトの較正図 / DCA / Brier、閾値 0.40 の `binary_perf`、見かけ vs 検証の較正対比。

## 5. 撤退基準

SUPPORT2 が取得できない、または予後共変量の完全例が極端に少ないとき中止。

## 6. Key messages（予定）

- 較正と DCA は予測モデルの話であり、効果推定例（`logit_indo`）には載せない。
- 評価は学習に使っていない hold-out。見かけ較正は楽観の教材として併記する。

## 7. 対象のイメージ

5 施設の重症成人（教学表。PHI なしとされている公開ファイル）。

## 8. 限界の先取り

完全例、内部分割（外部検証ではない）、閾値 0.40 は教学、線形 logit、欠測は完全例で落とすだけ。臨床運用の予後ツールではない。

## 9. 統計解析手法

アウトカム $Y$ は 180 日死亡（$T\le 180$ かつ死亡）。学習標本で

$$
\operatorname{logit}(\pi)=\beta_0+\beta^\top x
$$

の二項 GLM（`fit_glm(..., family="binomial")`）。予測確率 $\hat\pi$ を検証標本で評価。Brier は $\mathrm{E}[(Y-\hat\pi)^2]$。較正はビン平均の予測 vs 観測。決定曲線の net benefit は閾値 $t$ で

$$
\mathrm{NB}=\frac{\mathrm{TP}}{n}-\frac{\mathrm{FP}}{n}\cdot\frac{t}{1-t}.
$$

実装: `write_probability_artifacts` / `plot_calibration` / `plot_dca` / `binary_perf` / `threshold_tradeoff`。
