# cox_retinopathy_protocol

読者向けの一続き版（推奨）: [`docs/examples/cox_retinopathy.md`](../../docs/examples/cox_retinopathy.md)。本ファイルは既存の付随文書（列名・出力対応）です。

問い・式の正本: [cox_retinopathy_concept.md](cox_retinopathy_concept.md)

## 1. 目的・デザイン

片眼レーザーと視力喪失時間の関連を、患者内クラスターを考慮した Cox で記述・回帰する。歴史的 RCT 教学データの二次解析。

## 2. 対象コホート

R パッケージ `survival` の `retinopathy`。取得は `build.py`（R が使えれば `data(retinopathy, package="survival")`、否则 Rdatasets CSV）。単位は眼 1 行（患者あたり 2 行）。

## 3. Inclusion criteria

公開表の全行。`futime` / `status` / `trt` / `id` はソースで非欠損。

## 4. Exclusion criteria

なし（行を落とさない）。flowchart は置かない。

## 5. `pp.flowchart`

使わない。コホート n は `cox_retinopathy_out/n.md`。

## 6. Outcome

- 時間: `futime`（月）。視力喪失または最終観察。
- イベント: `status==1` → `event`（bool）。

## 7. Exposure

- `trt_label`: `control` / `treated`（`trt` 0/1）。参照は `control`（`pl.Enum` 順）。
- 調整: `type`（adult / juvenile）、`risk`（眼単位リスクスコア）。

## 8. 統計解析

concept §9。列と関数:

- Table 1: `tableone(..., hue="trt_label", add_pvalue=False)`（CSV は `tableone_raw`。患者内割付けのため群間 p は付けない）
- KM: `survival_curve(..., by="trt_label")`
- log-rank: `log_rank(..., by="trt_label")`
- 図: `plot_survival(..., hue="trt_label")`（number-at-risk 既定 ON）
- Cox 主解析: `cox_ph(..., "Surv(futime, event) ~ trt_label + type + risk + cluster(id)")`
- 比較: 同じ平均構造で `cluster` なし（モデルベース SE）と `cluster="id"`
- forest: `plot_forest(..., layout="table")`（sandwich tidy）
- PH: `proportional_hazards_test(fit)` + `write_cox_diagnostic_suite(...)`

`hc_covariance` / `fit_mixed` は使わない。

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `cox_retinopathy_out/n.md` | 対象 |
| `cox_retinopathy_out/table1.csv` | Table 1 |
| `cox_retinopathy_out/km_curve.csv`, `figures/km_trt.png` | KM（NAR 付き） |
| `cox_retinopathy_out/logrank.md` | log-rank |
| `cox_retinopathy_out/cox_tidy.csv`, `cox_se_compare.csv`, `figures/cox_forest.png` | Cox sandwich |
| `cox_retinopathy_out/ph_test.csv`, `figures/cox_*.png` | PH / 診断 |
