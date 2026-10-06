# 競合リスク解析

`cif_pbc` — 競合リスク（CIF / Fine–Gray）

単一イベントの colon に対し、競合リスク（肝死 vs 移植）の正本。Fine–Gray forest まで。

[← ギャラリー](index.md) · [実行用ディレクトリと手順（GitHub）](https://github.com/mizuy/statract/tree/main/examples/cif_pbc)

## 目的概説

単一イベントの KM/Cox は [`surv_colon`](surv_colon.md) で示した。ここでは移植を競合とした **肝死** の累積発生（Aalen–Johansen）と Fine–Gray を、同じ `statract` 公開 API で通すための例です。

## データと列の説明

| 項目 | 内容 |
|------|------|
| ソース | R `survival::pbc`（無作為化 312 例）。GPL-2/3 |
| 取得 | `build.py`（R `survival`、否则 Rdatasets CSV） |
| 単位 | 患者 1 行。非無作為化 106 例は主解析に入れない |

主な解析列:

| 列 | 意味 |
|----|------|
| `time` | 追跡日数 |
| `status` | 0 打ち切り、1 移植（競合）、2 死亡（関心事象 = 肝死） |
| `death_naive` | 誤用 KM 用（移植を打ち切り扱い） |
| `trt` / `trt_label` / `dp` | 治療（D-ペニシラミン = 1）。参照はプラセボ |
| `age`, `sex`, `bili`, `albumin`, `edema`, `stage` | Fine–Gray 共変量 |

## CQ と大まかな解析方針

**CQ** — 無作為化 PBC 試験で、D-ペニシラミンはプラセボと比べ、移植を競合事象としたときの **肝死の累積発生** を下げるか。

方針:

1. flowchart で無作為化例を確定
2. Table 1（`hue=trt_label`）
3. 誤用 KM（対比）と Aalen–Johansen CIF（肝死）
4. Fine–Gray（`fine_gray` → 重み付き `cox_ph`）と `plot_forest(..., layout="table")`

## flowchart / tableone

### Flowchart

| 段 | flowchart キー | 対応基準 |
|----|----------------|----------|
| 無作為化 | `Randomized (trt not missing)` | `trt` / `trt_label` 非欠損 |
| 時間 | `Time and status present` | `time` / `status` 非欠損 |

=== "図"

    ```mermaid
    flowchart TB
      classDef keep fill:#e8f5e9,stroke:#2e7d32,color:#111
      classDef drop fill:#f5f5f5,stroke:#9e9e9e,color:#333
      A["418 rows"]:::keep
      B["312 rows"]:::keep
      X1["not Randomized (trt not missing)<br/>excluded: 106 rows"]:::drop
      C["Analysis cohort<br/>312 rows"]:::keep
      X2["not Time and status present<br/>excluded: 0 rows"]:::drop
      A --> B
      A -.-> X1
      B --> C
      B -.-> X2
    ```

    同一内容は `assets/cif_pbc/mermaid_flowchart.md`（`cif_pbc_out/` から sync）。

=== "コード"

    ```python
    import io
    import polars as pl
    from statract.reporting import markdown_flowchart, mermaid_flowchart

    buf = io.StringIO()
    cohort = target.pp.flowchart(
        {
            "Randomized (trt not missing)": (
                pl.col("trt").is_not_null() & pl.col("trt_label").is_not_null()
            ),
            "Time and status present": (
                pl.col("time").is_not_null() & pl.col("status").is_not_null()
            ),
        },
        out=buf,
    )
    flow_text = buf.getvalue()
    markdown_flowchart(flow_text)
    mermaid_flowchart(flow_text, final_label="Analysis cohort")
    ```

### Table 1

=== "表"

    --8<-- "examples/assets/cif_pbc/table1_gt.md"

    [CSV](assets/cif_pbc/table1.csv)

=== "コード"

    ```python
    from statract import agg_category, agg_mean_sd, agg_median_iqr, tableone

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Bilirubin (mg/dL)": ("bili", agg_median_iqr),
        "Albumin (g/dL)": ("albumin", agg_median_iqr),
        "Edema score": ("edema", agg_category),
        "Histologic stage": ("stage", agg_category),
    }
    gt = tableone(cohort, params, hue="trt_label", add_all=True, add_pvalue=True)
    # CSV/HTML/md を *_out/ に書くときだけ write_tableone_artifacts（ギャラリー同期用）
    ```

## メインの解析方法とそのコア

原因 $k=2$（肝死）。Aalen–Johansen 累積発生 $\hat F_2(t)$。誤用として移植を打ち切りにした Kaplan–Meier を対比。Fine–Gray の subdistribution ハザード:

$$
\lambda_2(t \mid x)=\lambda_{2,0}(t)\exp(\beta_{\mathrm{dp}}\mathrm{dp}+\beta^\top z),
$$

$z$ は年齢、性別、ビリルビン、アルブミン、浮腫、病期。実装: `fine_gray(..., cause=2)` のあと `cox_ph("Surv(fgstart, fgstop, fgstatus) ~ ...", weights="fgwt")` → `plot_forest(..., layout="table")`（論文用白黒は `style="bw"`）。CIF 図は matplotlib ステップ（`plot_survival` は使わない）。偽の PH 診断スイートは載せない。


## 結果

無作為化 312 例。サイト掲載は `assets/cif_pbc/`。

### Aalen–Johansen CIF（肝死）

=== "図"

    ![CIF for death](assets/cif_pbc/cif_death.png)

=== "コード"

    ```python
    from statract import survival_curve

    aj = survival_curve(
        cohort, "time", "status", by="trt_label", kind="aalen_johansen"
    )
    cif = aj.frame()
    # state == "2" が肝死。図はステッププロット（例スクリプトの _plot_cif）
    ```

### CIF 時点値

=== "表"

    | time | estimate | std_error | conf_low | conf_high | n_risk | group | state |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | 365 | 0.05696 | 0.01892 | 0.01988 | 0.09404 | 150 | D-penicillamine | 2 |
    | 730 | 0.08861 | 0.02359 | 0.04238 | 0.1348 | 144 | D-penicillamine | 2 |
    | 1825 | 0.2844 | 0.04359 | 0.199 | 0.3698 | 83 | D-penicillamine | 2 |
    | 3650 | 0.5424 | 0.07883 | 0.3879 | 0.6969 | 17 | D-penicillamine | 2 |
    | 365 | 0.08442 | 0.02331 | 0.03873 | 0.1301 | 142 | placebo | 2 |
    | 730 | 0.1234 | 0.02818 | 0.06814 | 0.1786 | 136 | placebo | 2 |
    | 1825 | 0.2823 | 0.04357 | 0.1969 | 0.3677 | 78 | placebo | 2 |
    | 3650 | 0.514 | 0.08248 | 0.3524 | 0.6757 | 17 | placebo | 2 |

    [CSV](assets/cif_pbc/cif_death_at.csv)

=== "コード"

    ```python
    death_cif = cif.filter(pl.col("state").cast(pl.Utf8) == "2")
    # aj_death.at([365, 730, 1825, 3650])
    ```

### Fine–Gray（tidy、指数化）

=== "表"

    | term | estimate | std_error | statistic | p_value | conf_low | conf_high | exp_estimate | exp_conf_low | exp_conf_high |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | dp | -0.02607 | 0.1846 | -0.1412 | 0.8877 | -0.3879 | 0.3358 | 0.9743 | 0.6785 | 1.399 |
    | age | 0.03255 | 0.009486 | 3.431 | 6.005e-04 | 0.01396 | 0.05114 | 1.033 | 1.014 | 1.052 |
    | sexm | 0.4924 | 0.2527 | 1.949 | 0.05132 | -0.00283 | 0.9875 | 1.636 | 0.9972 | 2.685 |
    | bili | 0.124 | 0.01524 | 8.137 | 0 | 0.09414 | 0.1539 | 1.132 | 1.099 | 1.166 |
    | albumin | -0.8716 | 0.2482 | -3.511 | 4.456e-04 | -1.358 | -0.3851 | 0.4183 | 0.2572 | 0.6804 |
    | edema | 1.042 | 0.3119 | 3.34 | 8.363e-04 | 0.4306 | 1.653 | 2.835 | 1.538 | 5.224 |
    | stage | 0.45 | 0.1308 | 3.44 | 5.819e-04 | 0.1936 | 0.7064 | 1.568 | 1.214 | 2.027 |

    [CSV](assets/cif_pbc/finegray_tidy.csv)

=== "コード"

    ```python
    from statract import cox_ph, fine_gray

    expanded = fine_gray(fg_df, FG_FORMULA, cause=2)
    fg_fit = cox_ph(expanded, FG_COX, weights="fgwt")
    fg_tidy = fg_fit.tidy(exponentiate=True)
    ```

### Fine–Gray forest（subdistribution HR）

=== "図"

    ![Fine–Gray forest (subdistribution HR)](assets/cif_pbc/finegray_forest.png)

=== "コード"

    ```python
    from statract import plot_forest

    plot_forest(
        fg_tidy,
        out / "figures" / "finegray_forest.png",
        title="Fine–Gray (subdistribution HR)",
        xlabel="Hazard ratio",
        layout="table",
    )
    ```

## 解釈と解説

肝死の 5 年 CIF は D-ペニシラミン約 0.28、プラセボ約 0.28。Fine–Gray の `dp` subdistribution HR は約 **0.97**（95% CI 約 0.68–1.40）。治療の点推定が 1 付近なのは、歴史的試験で生存利益が明確でなかったことと方向が一致します。ビリルビンの SHR は約 1.13（1 mg/dL あたり）。

限界: 非無作為化例は除外。誤用 KM は対比用。CIF 図は公式 `plot_survival` ではない。GPL 教学データ。

## 実行と成果物

```bash
cd examples/cif_pbc
task all
```

既存の付随文書: [concept](https://github.com/mizuy/statract/blob/main/examples/cif_pbc/cif_pbc_concept.md) · [protocol](https://github.com/mizuy/statract/blob/main/examples/cif_pbc/cif_pbc_protocol.md) · [results](https://github.com/mizuy/statract/blob/main/examples/cif_pbc/cif_pbc_results.md) · [discussion](https://github.com/mizuy/statract/blob/main/examples/cif_pbc/cif_pbc_discussion.md)
