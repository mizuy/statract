# surv_colon_discussion

数値は [surv_colon_results.md](surv_colon_results.md) の生成物を読む。再集計しない。

## CQ への答え

再発レコードに畳んだ 929 人で、補助療法 `rx` は再発時間と関連する（log-rank 統計量 23.06、df=2、p≈9.8×10⁻⁶）。Cox（年齢・性別・陽性リンパ節調整）では Lev+5FU 対 Obs の HR が約 0.58（95% CI 約 0.46–0.74）。Lev 単独は Obs と大きく違わない（HR≈0.93）。これは教学データの reproductions であり、診療方針の更新を主張しない。

## 当たり外れ

Lev+5FU の HR が 1 を下回る方向は、Moertel らの歴史的報告と一致する。

## 限界

- 患者単位は再発行のみ。死亡を競合リスクにした Fine–Gray は出していない。
- `nodes` 欠損は Cox だけ落とす（`cox_n.md`）。
- データは GPL パッケージ由来の教学エクスポートである。
