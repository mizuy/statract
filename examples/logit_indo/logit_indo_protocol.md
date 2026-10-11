# logit_indo_protocol

読者向けの一続き版（推奨）: [`docs/examples/logit_indo.md`](../../docs/examples/logit_indo.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [logit_indo_concept.md](logit_indo_concept.md)

## 1. 目的・デザイン

直腸インドメタシンと PEP の関連。原研究は多施設 RCT。本ディレクトリは教学用の二次解析。主解析は施設変量 GLMM。

## 2. 対象コホート

`medicaldata::indo_rct`（Elmunzer et al. NEJM 2012 の教学データ、602 行）。取得は `build.py`（R `medicaldata`、否则 Rdatasets CSV）。単位は参加者 1 行。

## 3. Inclusion criteria

1. teaching セットの全行（`outcome` / `rx` / `age` / `risk` が揃っている前提）。

## 4. Exclusion criteria

追加の臨床除外はしない。**行は落とさない**（除外ゼロのため flowchart も置かない）。

## 5. `pp.flowchart`

使用しない（除外ゼロ）。コホート n は `logit_indo_out/n.md`。

## 6. Outcome

- 列 `outcome`（`0_no` / `1_yes`）を `pep`（0/1）に変換。PEP = post-ERCP pancreatitis。

## 7. Exposure

- `rx`: `0_placebo` / `1_indomethacin`。解析では `indomethacin`（0/1）。参照はプラセボ。
- 施設: `site`（変量切片グループ）。

## 8. 統計解析

concept §9。

- Table 1: `tableone(..., hue="rx", add_pvalue=True)`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- **主**: `fit_mixed("pep ~ ... + (1 | site)", family="binomial")` — lme-python `glmer`。固定効果 OR を `plot_forest(..., layout="table")`。施設 BLUP を `plot_random_effects`。MOR を `median_odds_ratio`。
- **比較**: `fit_glm(..., family="binomial")`（未調整 `pep ~ indomethacin`、調整は GLMM と同じ共変量）
- **禁止**: `glmm_gpboost` を主解析にしない。固定効果の比較は `fit_glm`。ホールドアウト較正 / DCA は載せない（本例は効果推定）

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `logit_indo_out/n.md` | 対象集団 |
| `logit_indo_out/table1.csv` | Table 1 |
| `logit_indo_out/glmm_tidy.csv`, `glmm.md`, `figures/glmm_fixed_forest.png` | 主解析（固定効果） |
| `logit_indo_out/glmm_random_effects.csv`, `figures/glmm_site_blups.png` | 施設 BLUP |
| `logit_indo_out/glmm_mor.csv` | MOR |
| `logit_indo_out/glm_unadjusted.csv`, `glm_adjusted.csv`, `figures/glm_adjusted_forest.png` | 固定効果比較 |
