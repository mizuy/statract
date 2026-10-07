# PBC 試験における肝死の累積発生（競合リスク）

## 0. なぜこの切り口か

単一イベントの KM/Cox は `surv_colon` で示した。ここでは移植を競合とした **肝死** の累積発生と Fine–Gray を、同じ `statract` 公開 API で通す。

## 1. Clinical Question

無作為化 PBC 試験で、D-ペニシラミンはプラセボと比べ、移植を競合事象としたときの **肝死の累積発生** を下げるか。

## 2. 現象・既知

Mayo PBC 試験では肝移植が死亡と並んで観察される。移植を打ち切りにした KM は肝死の確率を過大評価しうる。

## 3. 仮説

D-ペニシラミンの subdistribution ハザードはプラセボと大きくは違わない（歴史的試験は生存利益が明確ではない）。重症度（ビリルビン等）は肝死 CIF と関連する。

## 4. 結論の形

治療群別の Aalen–Johansen CIF（肝死）、誤用 KM との対比、Fine–Gray の subdistribution HR（`dp` + 年齢・性別・ビリルビン・アルブミン・浮腫・病期）。

## 5. 撤退基準

無作為化フラグ `trt` が復元できない、または `status` が 0/1/2 でないときは止める。非無作為化 106 例は主解析に入れない。

## 6. Key messages（予定）

- 関心事象は肝死（`status=2`）。移植（`status=1`）は競合であり打ち切りではない。
- Fine–Gray は `fine_gray` で展開し、推定は `cox_ph(..., weights="fgwt")` である。

## 7. 対象のイメージ

`survival::pbc` の無作為化 312 例。施設カルテではない。

## 8. 限界の先取り

LGPL 教学エクスポート。CIF 公式プロットは薄く、ステップ図は自前。計数過程の一般化（`split_follow_up`）は出さない。

## 9. 統計解析手法

原因 $k=2$（肝死）。Aalen–Johansen 累積発生 $\hat F_2(t)$。誤用として移植を打ち切りにした Kaplan–Meier $1-\hat S_{\mathrm{naive}}(t)$ を対比する。

Fine–Gray の subdistribution ハザード:

$$
\lambda_2(t \mid x)=\lambda_{2,0}(t)\exp(\beta_{\mathrm{dp}}\mathrm{dp}+\beta^\top z),
$$

$z$ は年齢、性別、ビリルビン、アルブミン、浮腫、病期。実装は `fine_gray(..., cause=2)` のあと `cox_ph("Surv(fgstart, fgstop, fgstatus) ~ ...", weights="fgwt")`。
