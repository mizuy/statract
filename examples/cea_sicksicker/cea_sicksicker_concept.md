# Sick-Sicker 仮想疾患の費用対効果（cohort Markov）

## 0. なぜこの切り口か

`statract.cea` は疾患非依存の核だけを持つ。公開教学パラメータで **Markov → ICER → DSA/PSA** を通し、病態をライブラリの外に置く。

## 1. Clinical Question

25 歳で Healthy の仮想コホートに対し、標準治療（SoC）と比べて治療 A・B・AB はいくらの増分費用対効果か。決定の不確実性（one-way / PSA）はどうか。

## 2. 現象・既知

DARTH の Sick-Sicker は 4 状態（H / S1 / S2 / D）の教学モデルである。実患者データではなく、論文 Table 1 のスカラーが入力である。

## 3. 仮説

A は S1 の効用を上げ費用を足す。B は S1→S2 のオッズを下げ費用を足す。AB は併用。frontier 上の ICER と、WTP $100,000/QALY での NMB が主結果になる。

## 4. 結論の形

4 戦略の `calculate_icers`（ND / D / ED）、最適戦略 NMB の tornado、PSA の CE plane / CEAC / EVPI。

## 5. 撤退基準

遷移行列が行和 1 を満たさない、または効用の順序が崩れてモデルが破綻するとき止める。

## 6. Key messages（予定）

- 核は `simulate_cohort_markov` / `calculate_icers` / `one_way_dsa` / `run_psa`
- 病態（P 行列・戦略別報酬）はこの example 側
- 半サイクル補正・遷移報酬・年齢別死亡は V1 核に無いので入れない

## 7. 対象のイメージ

閉じた仮想コホート。全員 25 歳・状態 H から開始。除外なし。

## 8. 限界の先取り

教学の仮想疾患。生命表も校正も無い。論文の tunnels / transition rewards は再現しない。

## 9. 統計解析手法

サイクル $t$（年齢 $a=25+t$）の所属ベクトル $m_t$。割引 $\delta=0.03$:

$$
C=\sum_t (1+\delta)^{-t} m_t^\top c,\qquad
Q=\sum_t (1+\delta)^{-t} m_t^\top u,\qquad
m_{t+1}=m_t P.
$$

S1/S2 死亡確率は $p_{HD}$ を率に変換してハザード比を掛ける。治療 B の進行確率は logit 上で $\mathrm{OR}=0.6$。ICER は費用・効果ベクトルに対する `calculate_icers`。NMB $= Q\cdot\mathrm{WTP}-C$。実装: `statract.cea`。
