# lmm_pbcseq_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/lmm_pbcseq.md`](../../docs/stat/examples/lmm_pbcseq.md)。本ファイルはローカル workflow（列名・inclusion / exclusion・出力対応）の正本です。

問い・式の正本: [lmm_pbcseq_concept.md](lmm_pbcseq_concept.md)

## 1. 目的・デザイン

反復検査の対数ビリルビン。歴史的 RCT の縦断二次解析。

## 2. 対象コホート

`survival::pbcseq`。取得は `build.py`。患者単位のベースラインは `id` ごとに最小 `day` の行。モデルは訪問行。本セットの `trt` は 1 = D-ペニシラミン、0 = プラセボ（`pbc` 横断表の 1/2 とはコードが違う）。

## 3. Inclusion criteria

1. `trt` 非欠損の無作為化患者。
2. ベースライン `bili > 0`。
3. モデル行: 当該患者の `log_bili`・`day_years`・`dp` が非欠損。

## 4. Exclusion criteria

1. 非無作為化患者。
2. `bili <= 0` または欠損の訪問（対数変換）。

## 5. `pp.flowchart`

使用しない（患者単位で除外ゼロ）。患者 n は `lmm_pbcseq_out/n.md`。訪問数は `visit_n.md`。

## 6. Outcome

- `log_bili` = $\log(\mathrm{bili})$。`bili` は mg/dL。
- 時間: `day`（日）と `day_years` = `day/365.25`（モデル用）。

## 7. Exposure

- `dp`（D-ペニシラミン = 1）。表示 `trt_label`。患者内で一定とみなす。

## 8. 統計解析

concept §9。**いずれも主解析**（GAM を副注にしない）。

- Table 1: 患者単位、`tableone(..., hue="trt_label", add_pvalue=True)`、検査は `agg_median_iqr`。CSV/HTML/md の書き出しは `write_tableone_artifacts`（`*_out/` 同期用）
- LMM: `fit_mixed(..., "log_bili ~ day_years + dp + (1 | id)")`
- 比較: `fit_ols(..., "log_bili ~ day_years + dp")` + `cluster_covariance(..., cluster="id", data=visits)`。HC は IPTW 例に回す
- GAM: `gam(visits, "log_bili", [smooth("day_years", k=...)])`。protocol 上、混合も治療項も GAM API では扱えない

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `lmm_pbcseq_out/n.md`, `visit_n.md` | 対象 |
| `lmm_pbcseq_out/table1.csv` | Table 1 |
| `lmm_pbcseq_out/lmm_fixed.csv`, `lmm_variance.csv` | LMM |
| `lmm_pbcseq_out/ols_naive.csv`, `ols_cluster.csv` | クラスタ OLS |
| `lmm_pbcseq_out/gam_smooth.csv`, `gam_n.md`, `figures/gam_day.png` | GAM |
