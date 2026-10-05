# Colon 補助化学療法と再発時間

## 0. なぜこの切り口か

`statract` の生存 API（Kaplan–Meier、log-rank、Cox）を、公開 RCT の患者単位データで一連の文書（concept → protocol → results）に載せる。

## 1. Clinical Question

切除後 stage B/C 大腸癌において、補助化学療法 `rx`（観察 / levamisole / levamisole+5-FU）は再発までの時間を延長するか。

## 2. 現象・既知

Moertel らの補助療法 RCT は、levamisole+5-FU が観察群より再発・死亡を減らしたと報告した。R `survival::colon` は患者あたり再発と死亡の2行を持つ。

## 3. 仮説

Lev+5FU は観察群に比べ再発ハザードが低い。Lev 単独の効果は小さい。

## 4. 結論の形

治療群間の KM、log-rank p、Cox のハザード比（`rx` + 年齢・性別・陽性リンパ節）。

## 5. 撤退基準

患者単位に畳めない、または `time`/`status`/`rx` が欠けるときは解析を止める。競合リスク（再発 vs 死亡）は本カットの対象外。

## 6. Key messages（予定）

- 解析単位は再発レコード（`etype=1`）の1患者1行である。
- 主解析は Cox、群間の記述は KM と log-rank である。

## 7. 対象のイメージ

`survival::colon` の公開教学セット。施設カルテではない。

## 8. 限界の先取り

GPL パッケージデータの再エクスポートであり原資料の完全再現ではない。打ち切りはパッケージ定義に従う。AFT / Fine–Gray は出さない。

## 9. 統計解析手法

Kaplan–Meier 生存関数 $\hat S(t)$（群 = `rx`）。k 標本 log-rank（Mantel–Haenszel、$\rho=0$）。Cox:

$$
h(t \mid x) = h_0(t)\exp(\beta_{\mathrm{rx}} + \beta_{\mathrm{age}}\mathrm{age} + \beta_{\mathrm{sex}}\mathrm{sex} + \beta_{\mathrm{nodes}}\mathrm{nodes}).
$$

式の実装は `cox_ph(..., "Surv(time, event) ~ rx + age + sex + nodes")`。参照水準は `rx=Obs`（`pl.Enum` 順）。
