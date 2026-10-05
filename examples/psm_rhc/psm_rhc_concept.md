# 初日 RHC と 30 日死亡（PS マッチ）

## 0. なぜこの切り口か

観察研究で交絡を減らす MatchIt 流の `match_sample` を、公開の重症患者教学データで示す。

## 1. Clinical Question

ICU 初日の肺動脈カテーテル（RHC）は、交絡を傾向スコア最近傍マッチで揃えたあと、30 日死亡と関連するか。

## 2. 現象・既知

Connors et al. JAMA 1996 は SUPPORT 由来の観察データで RHC と死亡・資源の関連を報告し、議論を呼んだ。本セットは Vanderbilt hbiostat の教学 CSV である。

## 3. 仮説

未マッチでは RHC 群の重症度が高く死亡が多い。マッチ後も残差関連がありうるが、SMD は小さくなる。

## 4. 結論の形

マッチ前後の Table 1、Love plot、マッチ標本での `dth30_bin ~ rhc` のオッズ比。因果の確定主張はしない。

## 5. 撤退基準

`swang1` または `dth30` が無い、またはマッチ共変量の完全例が極端に少ないとき中止。利用条件が取得を禁じるならこの例は落とす。

## 6. Key messages（予定）

- API は `match_sample`。
- 推定対象は ATT、順序はデータ順で決定的にする。

## 7. 対象のイメージ

成人 ICU 教学コホート（PHI なしとされている公開表）。

## 8. 限界の先取り

未測定交絡、キャリパーによる除外、OSI ライセンスが明示されない教学データの再配布不可。

## 9. 統計解析手法

傾向スコア $e(x)=P(\mathrm{rhc}=1\mid x)$ をロジットで推定。最近傍 1:1、ATT、`order=data`、標準化キャリパー 0.2。Love plot はマッチ前後の標準化平均差。アウトカムモデル:

$$
\operatorname{logit}P(\mathrm{dth30}=1)=\beta_0+\beta_1\mathrm{rhc}
$$

実装は `match_sample` と `fit_glm(..., family="binomial")`。重み列があれば `weights=`。
