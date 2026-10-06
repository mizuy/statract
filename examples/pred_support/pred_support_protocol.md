# pred_support_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/pred_support.md`](../../docs/stat/examples/pred_support.md)。本ファイルは既存の付随文書（列名・inclusion / exclusion・出力対応）です。

問い・式の正本: [pred_support_concept.md](pred_support_concept.md)

## 1. 目的・デザイン

重症成人コホートでの **180 日死亡確率モデル**。主目的は効果の OR ではなく、train でフィットした二項 GLM の **ホールドアウト較正 / DCA**。

## 2. 対象コホート

SUPPORT2（hbiostat teaching CSV、n=9105）。取得は `build.py`（`support2csv.zip`、失败时 UCI CSV）。単位は患者 1 行。**fetch only**（生 CSV は git に入れない）。

## 3. Inclusion criteria

1. `death` と `d_time` が非欠損（教学表では全例）。
2. 予後共変量がすべて非欠損: `age`, `sex`, `race`, `dzclass`, `num_co`, `diabetes`, `dementia`, `ca`, `scoma`, `meanbp`, `hrt`, `resp`, `temp`, `crea`, `sod`。

## 4. Exclusion criteria

共変量欠損のみ。SUPPORT の生理スコア（`sps`, `aps`, `surv2m`, `surv6m`）は予測因子に使わない。

## 5. `pp.flowchart`

| 段 | flowchart キー | 対応基準 |
|----|----------------|----------|
| 共変量 | `Complete prognostic covariates` | Inclusion 2（アウトカム含む） |

成果物: `pred_support_out/text_flowchart.md`, `pred_support_out/mermaid_flowchart.md`

## 6. Outcome

- `death_180`: `(death == 1) & (d_time <= 180)`。180 日時点の打ち切りによる不完全追跡は本表では 0 件。

## 7. Predictors

上記共変量。`sex` / `race` / `dzclass` / `ca` は因子。`diabetes` / `dementia` は 0/1。

## 8. 統計解析

concept §9。

- Table 1: 公開 API は `tableone(..., hue="death_180_label")`。`write_tableone_artifacts` は `*_out/` への書き出し
- 層化 70/30（seed=2026）で train / validation
- train に `fit_glm` binomial: null `death_180 ~ 1`、年齢+性別、多変量 `FULL_FORMULA`
- 多変量の OR forest: `plot_forest(..., layout="table")`（train）
- 検証確率: `write_probability_artifacts`（内部で `plot_calibration` / `plot_dca` / Brier）
- 見かけ vs 検証: `calibration_table` + `plot_calibration`（多変量のみ）
- `binary_perf` は閾値 0.40。`threshold_tradeoff` を 1 表
- **禁止**: SUPPORT 組込みスコアを共変量にする、検証データで再フィットして較正する

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `pred_support_out/text_flowchart.md`, `mermaid_flowchart.md` | 対象 |
| `pred_support_out/table1.csv` | Table 1 |
| `pred_support_out/glm_full_train.csv`, `figures/glm_full_forest.png` | 学習標本 GLM |
| `pred_support_out/table_brier.csv`, `table_calibration.csv`, `figures/fig_calibration.png` | ホールドアウト較正 |
| `pred_support_out/table_calibration_apparent.csv`, `figures/fig_calibration_apparent.png` | 見かけ vs 検証 |
| `pred_support_out/table_dca.csv`, `figures/fig_dca.png` | DCA |
| `pred_support_out/binary_perf_val.csv`, `threshold_tradeoff_val.csv` | 閾値性能 |
