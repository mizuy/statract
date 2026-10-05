# cif_pbc_results

解釈は [cif_pbc_discussion.md](cif_pbc_discussion.md)。

## 対象集団の流れ

@import "cif_pbc_out/text_flowchart.md"

## Table 1（ベースライン特性）

| 内容 | CSV |
|---|---|
| Table 1 by `trt_label` | [cif_pbc_out/table1.csv](cif_pbc_out/table1.csv) |

@import "cif_pbc_out/table1_csv.md"

## 誤用 Kaplan–Meier（移植を打ち切り）

@import "cif_pbc_out/km_naive_at_csv.md"

## Aalen–Johansen CIF（肝死）

@import "cif_pbc_out/cif_death_at_csv.md"

![CIF of liver death](cif_pbc_out/figures/cif_death.png)

## Fine–Gray（subdistribution HR）

@import "cif_pbc_out/finegray_n.md"

@import "cif_pbc_out/finegray_tidy_csv.md"

![Fine–Gray forest (subdistribution HR)](cif_pbc_out/figures/finegray_forest.png)
