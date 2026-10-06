# aft_rotterdam_protocol

読者向けの一続き版（推奨）: [`docs/examples/aft_rotterdam.md`](../../docs/examples/aft_rotterdam.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [aft_rotterdam_concept.md](aft_rotterdam_concept.md)

## 1. 目的・デザイン

ホルモン療法と死亡時間。観察コホートの二次解析。

## 2. 対象コホート

`survival::rotterdam`。取得は `build.py`。単位は患者 1 行。時間は `dtime`（日）、イベントは `death`。

## 3. Inclusion criteria

1. `dtime` と死亡ステータスが非欠損。
2. `hormon` が非欠損。

## 4. Exclusion criteria

1. Cox / AFT のみ: `age` / `nodes` / `size` / `grade` 欠損を完全例削除。
2. 再発（`rtime` / `recur`）は使わない。

## 5. `pp.flowchart`

使用しない（除外ゼロ）。コホート n は `aft_rotterdam_out/n.md`。

## 6. Outcome

- 腫瘍径 `size` はパッケージにより区間ラベル（`<=20` / `20-50` / `>50`）なのでカテゴリとして回帰する。
- イベント: `death==1` → `event`。

## 7. Exposure

- `hormon`（0/1）。表示は `hormon_label`。参照はホルモン療法なし。

## 8. 統計解析

concept §9。

- Table 1: `tableone(..., hue="hormon_label", add_pvalue=True)`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- KM: `survival_curve` / `plot_survival`（NAR 既定 ON）/ `log_rank`
- Nelson–Aalen: `survival_curve(..., kind="nelson_aalen")` を指定時点の 1 表
- Cox: `cox_ph(...)` → `plot_forest(..., layout="table")`
- PH: `proportional_hazards_test(fit)` + `write_cox_diagnostic_suite(...)`
- AFT: `accelerated_failure(..., distribution="weibull")` を第一。documented な `lognormal` を併記（Weibull が収束しないときの読み用）
## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `aft_rotterdam_out/n.md` | 対象集団 |
| `aft_rotterdam_out/table1.csv` | Table 1 |
| `aft_rotterdam_out/km_at.csv`, `figures/km_hormon.png` | KM（NAR 付き） |
| `aft_rotterdam_out/logrank.md` | log-rank |
| `aft_rotterdam_out/nelson_aalen_at.csv` | Nelson–Aalen |
| `aft_rotterdam_out/cox_tidy.csv`, `ph_test.csv`, `ph_test.md`, `figures/cox_forest.png`, `figures/cox_*.png` | Cox / zph / 診断 |
| `aft_rotterdam_out/aft_weibull_tidy.csv`, `aft_lognormal_tidy.csv`, `aft_n.md` | AFT |
