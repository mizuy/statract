# psm_rhc_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/psm_rhc.md`](../../docs/stat/examples/psm_rhc.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [psm_rhc_concept.md](psm_rhc_concept.md)

## 1. 目的・デザイン

初日 RHC と 30 日死亡の関連を、傾向スコア最近傍マッチ後の二項 GLM で記述する。観察研究の教学二次解析。

## 2. 対象コホート

Vanderbilt hbiostat の RHC CSV（`https://hbiostat.org/data/repo/rhc.csv`）。Connors JAMA 1996 系の教学セット。単位は入院/患者 1 行。**リポジトリに CSV を置かない。再配布しない。**

## 3. Inclusion criteria

1. `swang1`（RHC / No RHC）と `dth30` が非欠損。解析列は `rhc`（0/1）と `dth30_bin`（0/1）。
2. protocol に列挙したマッチ共変量がすべて非欠損（完全例）。

## 4. Exclusion criteria

1. マッチ共変量欠損。
2. キャリパー 0.2（標準化）で相手が付かないユニットはウェイト 0（アウトカムモデルから除外）。

## 5. `pp.flowchart`

使用しない（除外ゼロ）。コホート n は `psm_rhc_out/n.md`。

## 6. Outcome

- **主**: `dth30`（Yes/No）→ `dth30_bin`。Connors の主アウトカムに近い 30 日死亡。
- `death` はより長い追跡の死亡フラグ。主モデルには使わない。

## 7. Exposure

- `swang1 == "RHC"` → `rhc=1`（初日 RHC）。対照は No RHC。

## 8. 統計解析

concept §9。共変量（存在する列のみ）: `age`, `sex`, `race`, `edu`, `cat1`, `ca`, `aps1`, `scoma1`, `meanbp1`, `hrt1`, `resp1`, `temp1`, `pafi1`, `alb1`, `hema1`, `bili1`, `crea1`, `sod1`, `cardiohx`, `chfhx`, `chrpulhx`, `dnr1`。

- 未マッチ Table 1: `tableone(..., hue="rhc", add_pvalue=True)`
- `match_sample(..., method="nearest", distance="logit", order="data", estimand="ATT", caliper=0.2)`
- `balance()` / `love_plot()`
- マッチ後 Table 1（`weights>0`）も `tableone`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- `fit_glm("dth30_bin ~ rhc", family="binomial", weights="weights")`
- マッチング API は `match_sample`。固定効果の回帰は `fit_glm`

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `psm_rhc_out/n.md` | 対象集団 |
| `psm_rhc_out/table1_unmatched.csv` | 未マッチ Table 1 |
| `psm_rhc_out/balance.csv`, `figures/love_plot.png` | バランス |
| `psm_rhc_out/table1_matched.csv` | マッチ後 Table 1 |
| `psm_rhc_out/glm_matched.csv`, `glm_unmatched.csv`, `figures/glm_matched_forest.png` | ロジスティック |
