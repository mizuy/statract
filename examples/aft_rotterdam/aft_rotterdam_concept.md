# ロッテルダム乳がんコホートの死亡時間（Cox と AFT）

## 0. なぜこの切り口か

`surv_colon` は PH を仮定した Cox だけである。ここでは PH 診断と、崩れたときの加速故障時間（Weibull）を同じ公開コホートで見せる。

## 1. Clinical Question

ロッテルダム乳がんコホートで、ホルモン療法は死亡までの時間と関連するか。Cox の比例ハザードが疑わしいとき、Weibull AFT では時間比はどうか。

## 2. 現象・既知

観察コホートであり割付は無作為ではない。ホルモン療法は予後因子と交絡する。PH が崩れると Cox の単一 HR の解釈が難しい。

## 3. 仮説

調整後もホルモン療法の HR は 1 からずれうる。`proportional_hazards_test` の全体 p が小さい項があれば AFT を主たる代替として読む。

## 4. 結論の形

KM、log-rank、Cox HR、zph 表、Weibull AFT の時間比。因果の確定主張はしない。

## 5. 撤退基準

`dtime` / `death` / `hormon` が欠けたら止める。再発時間は競合と重なるため使わない。

## 6. Key messages（予定）

- アウトカムは死亡であり再発ではない。
- PH 検定は `proportional_hazards_test`（`cox.zph` 相当）。AFT は `accelerated_failure(..., distribution="weibull")`。

## 7. 対象のイメージ

`survival::rotterdam`、約 2982 人。教学エクスポート。

## 8. 限界の先取り

AFT の分布は Weibull に固定（収束しないときだけ lognormal を併記）。LGPL データ。

## 9. 統計解析手法

Cox:

$$
h(t \mid x)=h_0(t)\exp(\beta_{\mathrm{hormon}}\mathrm{hormon}+\beta_{\mathrm{age}}\mathrm{age}+\beta_{\mathrm{nodes}}\mathrm{nodes}+\beta_{\mathrm{size}}\mathrm{size}+\beta_{\mathrm{grade}}\mathrm{grade}).
$$

PH のスコア検定は時間変換 Kaplan–Meier（R の `km`）。Weibull AFT は log-time 上の線形予測子（`survreg` パラメータ化）。実装: `cox_ph`、`proportional_hazards_test`、`accelerated_failure`。
