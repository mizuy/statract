# surv_colon_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/surv_colon.md`](../../docs/stat/examples/surv_colon.md)。本ファイルはローカル workflow（列名・inclusion / exclusion・出力対応）の正本です。

問い・式の正本: [surv_colon_concept.md](surv_colon_concept.md)

## 1. 目的・デザイン

補助化学療法 `rx` と再発時間の関連を、公開 RCT 教学データで記述・回帰する。デザインは歴史的 RCT の二次解析。

## 2. 対象コホート

R パッケージ `survival` の `colon`（Laurie's / Moertel adjuvant colon studies に近い公開版）。取得は `build.py`（R が使えれば `data(colon, package="survival")`、否则 Rdatasets CSV）。単位は再発レコードに畳んだ患者。

## 3. Inclusion criteria

1. `etype == 1`（再発エンドポイントの行。患者1行）。
2. `time` と `status` が非欠損。
3. `rx` が非欠損。

## 4. Exclusion criteria

1. Cox のみ: `age` / `sex` / `nodes` 欠損行を完全例削除。KM と Table 1 には残す。

## 5. `pp.flowchart`

| 段 | flowchart キー | 対応基準 |
|----|----------------|----------|
| ソース | （開始 n = 長表の行） | 再発+死亡の2行形式 |
| 患者単位 | `Recurrence record (etype=1, patient-level)` | Inclusion 1 |
| 時間 | `Time and status present` | Inclusion 2 |
| 治療 | `Treatment rx present` | Inclusion 3 |

成果物: `surv_colon_out/text_flowchart.md`, `surv_colon_out/mermaid_flowchart.md`（`mermaid_flowchart`）

## 6. Outcome

- 時間: 列 `time`（日）。原点はパッケージ定義（登録/手術後）。
- イベント: `status==1` を再発、0 を打ち切り。解析列 `event`（bool）。
- 死亡行（`etype=2`）は使わない。

## 7. Exposure

- `rx`: `Obs`, `Lev`, `Lev+5FU`。参照は `Obs`。

## 8. 統計解析

concept §9。列と関数:

- Table 1: `tableone(..., hue="rx", add_pvalue=True)`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- KM: `survival_curve(..., by="rx")`
- log-rank: `log_rank(..., by="rx")`
- 図: `plot_survival(..., hue="rx")`（number-at-risk 表は既定 ON）
- Cox: `cox_ph(..., "Surv(time, event) ~ rx + age + sex + nodes")` → `tidy(exponentiate=True)` → `plot_forest(..., layout="table")`
- PH: `proportional_hazards_test(fit)` + `write_cox_diagnostic_suite(...)`

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `surv_colon_out/text_flowchart.md`, `mermaid_flowchart.md` | 対象集団 |
| `surv_colon_out/table1.csv` | Table 1 |
| `surv_colon_out/km_curve.csv`, `figures/km_rx.png` | KM（NAR 付き） |
| `surv_colon_out/logrank.md` | log-rank |
| `surv_colon_out/cox_tidy.csv`, `cox_n.md`, `figures/cox_forest.png` | Cox |
| `surv_colon_out/ph_test.csv`, `figures/cox_*.png` | PH / 診断 |
