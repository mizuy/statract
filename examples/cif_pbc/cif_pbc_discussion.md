# cif_pbc_discussion

数値は [cif_pbc_results.md](cif_pbc_results.md) の生成物を読む。再集計しない。

## CQ への答え

無作為化 312 例（非無作為化 106 を除外）。肝死の 5 年 CIF は D-ペニシラミン約 0.28、プラセボ約 0.28（`cif_death_at.csv`、`state=2`）。Fine–Gray の `dp` subdistribution HR は約 0.97（95% CI 約 0.68–1.40）。`crr`（`fine_gray_regression`）でも 0.97（0.67–1.41）、Gray 検定は肝死 p = 0.80。関心事象は肝死、移植は競合である。教学データの reproductions であり、診療方針の更新を主張しない。

## 当たり外れ

治療の点推定が 1 付近なのは、歴史的試験で生存利益が明確でなかったことと方向が一致する。ビリルビンの SHR は約 1.13（1 mg/dL あたり）で、高いほど肝死が多い。

## 限界

- 非無作為化例は除外した。
- 誤用 KM は対比用であり主結果ではない。
- CIF 図は公式 `plot_survival` ではなくステップの自前図である。
- データは LGPL パッケージ由来の教学エクスポートである。
