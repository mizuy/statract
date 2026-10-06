# cif_pbc_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/cif_pbc.md`](../../docs/stat/examples/cif_pbc.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [cif_pbc_concept.md](cif_pbc_concept.md)

## 1. 目的・デザイン

無作為化 PBC 試験における D-ペニシラミンと肝死累積発生。デザインは歴史的 RCT の二次解析。競合は肝移植。

## 2. 対象コホート

R パッケージ `survival` の `pbc`。取得は `build.py`（R が使えれば `data(pbc, package="survival")`、否则 Rdatasets CSV）。単位は患者 1 行。主解析は無作為化例のみ。

## 3. Inclusion criteria

1. `trt` 非欠損（無作為化 312 例。1 = D-ペニシラミン、2 = プラセボ）。
2. `time` と `status` が非欠損。

## 4. Exclusion criteria

1. Fine–Gray のみ: `age` / `sex` / `bili` / `albumin` / `edema` / `stage` 欠損行を完全例削除。CIF と Table 1 には残す。
2. 非無作為化例（`trt` 欠損）は入れない。

## 5. `pp.flowchart`

| 段 | flowchart キー | 対応基準 |
|----|----------------|----------|
| 無作為化 | `Randomized (trt not missing)` | Inclusion 1 |
| 時間 | `Time and status present` | Inclusion 2 |

成果物: `cif_pbc_out/text_flowchart.md`

## 6. Outcome

- 時間: 列 `time`（日）。
- 原因コード `status`: 0 打ち切り、1 移植（競合）、2 死亡（関心事象 = 肝死）。
- 誤用 KM 用: `death_naive` = (`status == 2`)。移植は打ち切り扱い。

## 7. Exposure

- `trt_label`: `D-penicillamine` / `placebo`。回帰では `dp`（D-ペニシラミン = 1）。参照はプラセボ。

## 8. 統計解析

concept §9。列と関数:

- Table 1: `tableone(..., hue="trt_label", add_pvalue=True)`。ビリルビン・アルブミンは `agg_median_iqr`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- 誤用 KM: `survival_curve(..., event="death_naive")`
- CIF: `survival_curve(..., event="status", kind="aalen_johansen")`。表は `state=="2"`（肝死）
- Fine–Gray: `fine_gray(..., "Surv(time, status) ~ dp + age + sex + bili + albumin + edema + stage", cause=2)` → `cox_ph(..., "Surv(fgstart, fgstop, fgstatus) ~ ...", weights="fgwt")` → `tidy(exponentiate=True)`
- CIF 図: matplotlib ステップ（`plot_survival` は使わない）

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `cif_pbc_out/text_flowchart.md` | 対象集団 |
| `cif_pbc_out/table1.csv` | Table 1 |
| `cif_pbc_out/km_naive_at.csv` | 誤用 KM |
| `cif_pbc_out/cif_death_at.csv`, `figures/cif_death.png` | AJ CIF |
| `cif_pbc_out/finegray_tidy.csv`, `finegray_n.md`, `figures/finegray_forest.png` | Fine–Gray |
