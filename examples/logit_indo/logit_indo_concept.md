# 直腸インドメタシンと PEP

## 0. なぜこの切り口か

公開 RCT で、施設変量切片の二項 GLMM を主解析として同じ文書フローに載せる。

## 1. Clinical Question

ERCP 時の直腸インドメタシン 100 mg は、プラセボと比べ post-ERCP pancreatitis（PEP）を減らすか。施設を変量切片にしたとき固定効果 OR と施設間異質性（MOR）はどうか。

## 2. 現象・既知

Elmunzer, Higgins ら NEJM 2012 は多施設 RCT を中間解析で中止し、PEP がプラセボ 16% 対インドメタシン 9% と報告した。

## 3. 仮説

主推定は施設変量 GLMM の固定効果（インドメタシン OR）。施設間異質性は MOR で要約する。固定効果のみの二項 GLM は比較対照であり、向きは大きくは違わない。

## 4. 結論の形

`fit_mixed(..., family="binomial")` の固定効果 OR、施設 BLUP、MOR。比較として未調整・調整 GLM。ホールドアウト較正 / DCA は使わない（予測モデル用）。

## 5. 撤退基準

`outcome` / `rx` が復元できないとき中止。GLMM が失敗したら例の主解析は欠ける。

## 6. Key messages（予定）

- 主解析は全例の施設 GLMM（`fit_mixed` + `family="binomial"`）と MOR / BLUP。
- 固定効果 GLM はライブラリ比較用。較正ホールドアウトは載せない。
- 除外が無いので flowchart は置かない。

## 7. 対象のイメージ

`medicaldata::indo_rct`、n=602、4 施設。

## 8. 限界の先取り

gpboost は使わない。lme-python `glmer` の lme4 照合はエンジン側の CBPP 級であり、本例の fixture ではない。クラスタロバスト標準誤差は本カットの主結果にしない。

## 9. 統計解析手法

PEP 確率 $\pi$。主解析（施設 $j$）:

$$
\operatorname{logit}(\pi_{ij})=\beta_0+\beta_1\mathrm{indomethacin}_{ij}+\beta^\top z_{ij}+u_j,\quad u_j\sim N(0,\sigma^2)
$$

$z$ は年齢、リスクスコア、（あれば）SOD・膵管ステント。実装は `fit_mixed("pep ~ ... + (1 | site)", family="binomial")`。

$$
\mathrm{MOR}=\exp\bigl(\sqrt{2\sigma^2}\,\Phi^{-1}(0.75)\bigr)
$$

（`median_odds_ratio`）。施設 BLUP は `glmm_random_effects` / `plot_random_effects`。

比較:

$$
\operatorname{logit}(\pi_i)=\beta_0+\beta_1\mathrm{indomethacin}_i+\beta^\top z_i
$$

（`fit_glm(..., family="binomial")`）。
