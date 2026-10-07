# 糖尿病網膜症の眼単位生存とクラスター頑健 Cox

## 0. なぜこの切り口か

`cox_ph` の Lin–Wei sandwich（`cluster(id)` / `cluster=`）を、両眼が同一患者に属する公開 RCT で示す。独立観測を仮定したモデルベース SE では過大・過小になりうる。

## 1. Clinical Question

増殖前糖尿病網膜症において、片眼レーザー光凝固は対側の未治療眼に比べ、視力喪失までの時間を延長するか。

## 2. 現象・既知

DRS / ETDRS 系の教学データ（R `survival::retinopathy`）は患者あたり両眼 1 行ずつ（n=394、id=197）。治療は患者内で片眼に無作為化される。同一 id の両眼は共有する frailty を持つ。

## 3. 仮説

治療眼は対照眼より視力喪失ハザードが低い。患者単位共変量（病型）の SE は sandwich で膨らみ、治療対比はペア構造のため sandwich SE がモデルベースより小さくなることがある。

## 4. 結論の形

KM（治療ラベル）、log-rank、クラスター頑健 Cox の HR、モデルベース SE との比。

## 5. 撤退基準

`id` / `futime` / `status` / `trt` が欠ける、または患者あたり 2 行でないときは止める。

## 6. Key messages（予定）

- 解析単位は眼、分散推定のクラスターは患者 `id`。
- 主解析は `cox_ph(..., cluster=)` の Lin–Wei。OLS/GLM の `hc_covariance` ではない。

## 7. 対象のイメージ

R `survival::retinopathy`。施設カルテではない。

## 8. 限界の先取り

LGPL 教学エクスポート。counting process の sandwich はこの API では right-censored のみ（entry=0）。

## 9. 統計解析手法

Kaplan–Meier $\hat S(t)$（群 = `trt_label`）。2 標本 log-rank。Cox:

$$
h(t \mid x) = h_0(t)\exp(\beta_{\mathrm{trt}} + \beta_{\mathrm{type}}\mathrm{type} + \beta_{\mathrm{risk}}\mathrm{risk}).
$$

分散は Lin–Wei meat（スコア残差を `id` で合算）。実装: `cox_ph(..., "Surv(futime, event) ~ trt_label + type + risk + cluster(id)")` または `cluster="id"`。
