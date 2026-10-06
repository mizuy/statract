# iptw_nhefs_protocol

読者向けの一続き版（推奨）: [`docs/examples/iptw_nhefs.md`](../../docs/examples/iptw_nhefs.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [iptw_nhefs_concept.md](iptw_nhefs_concept.md)

## 1. 目的・デザイン

禁煙と体重変化の観察研究。推定対象は **ATE**（安定化 IPTW）。ATT は推定しない。

## 2. 対象コホート

NHEFS 教学表。取得は `build.py`（R `causaldata`、否则 Rdatasets、否则 Hernán CSV）。単位は参加者 1 行。

## 3. Inclusion criteria

1. `qsmk` と `wt82_71` が非欠損。
2. PS 共変量がすべて非欠損: `age`, `sex`, `race`, `education`, `smokeintensity`, `smokeyrs`, `exercise`, `active`, `wt71`。

## 4. Exclusion criteria

追加の臨床除外はしない（教学解析セットの完全例）。

## 5. `pp.flowchart`

| 段 | flowchart キー | 対応基準 |
|----|----------------|----------|
| 曝露とアウトカム | `Quit indicator and weight change present` | Inclusion 1 |
| 共変量 | `Complete propensity covariates` | Inclusion 2 |

成果物: `iptw_nhefs_out/text_flowchart.md`, `iptw_nhefs_out/mermaid_flowchart.md`

## 6. Outcome

- `wt82_71`: 1982−1971 の体重変化（kg）。

## 7. Exposure

- `qsmk`: 追跡中の禁煙 1、継続喫煙 0。

## 8. 統計解析

concept §9。

- Table 1: `tableone(..., hue="qsmk", add_pvalue=True)`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- PS: `fit_glm(..., "qsmk ~ age + sex + race + education + smokeintensity + smokeyrs + exercise + active + wt71", family="binomial")`
- 安定化 ATE 重みをスクリプトで計算。上限 10 で切り詰め（`sw_trunc`）
- 主: `fit_ols(..., "wt82_71 ~ qsmk", weights="sw_trunc")` + `hc_covariance(..., kind="HC3")`
- 未調整 OLS を併記
- 感度: `match_sample(..., method="cem")` の `balance` / `love_plot` のみ。アウトカム回帰はしない（第二推定対象にしない）
- **禁止**: `iptw()`（無い）、ATT 用の最近傍マッチ。マッチングを足すときは `match_sample`

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `iptw_nhefs_out/text_flowchart.md`, `iptw_nhefs_out/mermaid_flowchart.md` | 対象 |
| `iptw_nhefs_out/table1.csv` | Table 1 |
| `iptw_nhefs_out/ps_glm.csv`, `iptw_weight_summary.csv`, `iptw_n.md` | PS / 重み |
| `iptw_nhefs_out/ols_unadjusted.csv`, `ols_iptw_hc3.csv`, `figures/ols_iptw_forest.png` | ATE |
| `iptw_nhefs_out/cem_sensitivity.md`, `figures/cem_love.png` | CEM 感度 |
