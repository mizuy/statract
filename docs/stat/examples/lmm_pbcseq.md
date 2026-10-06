# lmm_pbcseq — ガウス LMM / クラスタ SE / GAM

indo の二項混合に対し、ガウス `fit_mixed`（lmer 相当）の正本。GAM は主解析（ガウス・平滑1本）。

[← ギャラリー](../examples.md) · [実行用ディレクトリと手順（GitHub）](https://github.com/mizuy/statract/tree/main/examples/lmm_pbcseq)

## 目的概説

[`logit_indo`](logit_indo.md) の混合は二項 `fit_mixed(..., family="binomial")` です。ここではガウス `fit_mixed`（`lmer` 相当）、患者クラスタ SE、および現行制限つき GAM を縦断検査で本編に載せるための例です。

## データと列の説明

| 項目 | 内容 |
|------|------|
| ソース | R `survival::pbcseq`。GPL-2/3 |
| 取得 | `build.py` |
| 単位 | 患者単位 Table 1 + 訪問行のモデル。`trt` は 1 = D-ペニシラミン、0 = プラセボ（横断 `pbc` の 1/2 とは違う） |

主な解析列:

| 列 | 意味 |
|----|------|
| `id` | 患者 ID（変量 / クラスタ） |
| `bili` / `log_bili` | ビリルビン（mg/dL）と対数 |
| `day` / `day_years` | 病日と年換算（`day/365.25`） |
| `dp` / `trt_label` | 治療（患者内で一定とみなす） |
| `age`, `sex`, `albumin` | ベースライン記述 |

## CQ と大まかな解析方針

**CQ** — 無作為化 PBC の反復検査で、病日と D-ペニシラミンは対数ビリルビンと関連するか。患者内相関を変量切片で入れたとき、集団の病日カーブ（GAM）はどうか。

方針:

1. 患者単位 Table 1（除外なしのため flowchart 省略）
2. 変量切片 LMM（主）
3. 同平均構造の OLS + `cluster_covariance`（比較）
4. GAM `s(day_years)`（主解析の一部。治療・変量は API 制限で入れられない）

## flowchart / tableone

解析セットに除外はありません（全例解析）。ギャラリー方針どおり **flowchart / Mermaid は省略**します。

### Table 1（患者単位）

=== "コード"

    ```python
    from statract import agg_category, agg_mean_sd, agg_median_iqr, tableone

    params = {
        "Age (years)": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Baseline bilirubin": ("bili", agg_median_iqr),
        "Baseline albumin": ("albumin", agg_median_iqr),
        "Baseline day": ("day", agg_median_iqr),
    }
    gt = tableone(cohort_p, params, hue="trt_label", add_all=True, add_pvalue=True)
    # CSV/HTML/md を *_out/ に書くときだけ write_tableone_artifacts（ギャラリー同期用）
    ```

## メインの解析方法とそのコア

訪問 $ij$、患者 $i$。主解析の LMM:

$$
\log(\mathrm{bili}_{ij})=\beta_0+\beta_1\mathrm{day\_years}_{ij}+\beta_2\mathrm{dp}_i+u_i+\varepsilon_{ij},\quad u_i\sim N(0,\sigma_u^2).
$$

比較: 同じ平均構造の OLS に患者クラスタのサンドイッチ。GAM（主解析）:

$$
\log(\mathrm{bili})=\beta_0+s(\mathrm{day\_years})+\varepsilon,
$$

$s$ は立方回帰スプライン（`smooth`、`k` は一意な病日数以下）。この版の GAM は `family="gaussian"`・`cr`・平滑1本・REML。


## 結果

無作為化 312 人、訪問 1945 行。サイト掲載は `assets/lmm_pbcseq/`。

### GAM 平滑（病日）

=== "図"

    ![GAM day smooth](assets/lmm_pbcseq/gam_day.png)

=== "コード"

    ```python
    from statract import gam, smooth
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    k = min(8, int(visits["day_years"].n_unique()))
    fitted_gam = gam(visits, "log_bili", [smooth("day_years", k=k)])
    grid = pl.DataFrame({
        "day_years": np.linspace(
            float(visits["day_years"].min()),
            float(visits["day_years"].max()),
            80,
        )
    })
    pred = fitted_gam.predict(grid)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(visits["day_years"], visits["log_bili"], s=8, alpha=0.15, color="gray")
    ax.plot(grid["day_years"], pred, color="C0", label="GAM s(day_years)")
    ax.set_xlabel("Day (years)")
    ax.set_ylabel("log bilirubin")
    fig.savefig(out / "figures" / "gam_day.png", dpi=150, bbox_inches="tight")
    ```

### LMM 固定効果

=== "表"

    | term | estimate | std_error | statistic | p_value | conf_low | conf_high |
    | --- | --- | --- | --- | --- | --- | --- |
    | (Intercept) | 0.6265 | 0.09081 | 6.899 | 5.236e-12 | 0.4485 | 0.8045 |
    | day_years | 0.09509 | 0.00433 | 21.96 | 0 | 0.0866 | 0.1036 |
    | dp | -0.1107 | 0.1271 | -0.8712 | 0.3837 | -0.3598 | 0.1384 |

    [CSV](assets/lmm_pbcseq/lmm_fixed.csv)

=== "コード"

    ```python
    from statract import fit_mixed

    mixed = fit_mixed(visits, "log_bili ~ day_years + dp + (1 | id)")
    lmm_tidy = mixed.tidy()
    # mixed.variance_table() → 患者間分散 / 残差
    ```

### OLS + クラスタ SE

=== "表"

    | term | estimate | std_error | statistic | p_value | conf_low | conf_high |
    | --- | --- | --- | --- | --- | --- | --- |
    | (Intercept) | 0.5747 | 0.08942 | 6.427 | 1.635e-10 | 0.3993 | 0.7501 |
    | day_years | 0.01405 | 0.0161 | 0.8729 | 0.3828 | -0.01752 | 0.04562 |
    | dp | -0.03103 | 0.1235 | -0.2514 | 0.8016 | -0.2732 | 0.2111 |

    [CSV](assets/lmm_pbcseq/ols_cluster.csv)

=== "コード"

    ```python
    from statract import cluster_covariance, fit_ols

    ols = fit_ols(visits, "log_bili ~ day_years + dp")
    ols.covariance = cluster_covariance(ols, "id", data=visits)
    ols_clust = ols.tidy()
    ```

### GAM tidy

=== "表"

    | term | estimate | std_error | statistic | p_value |
    | --- | --- | --- | --- | --- |
    | (Intercept) | 0.6031 | 0.02516 | 23.97 | 0 |
    | day_years[0] | -0.01174 | 0.007787 | -1.508 | 0.1315 |
    | day_years[1] | -0.001792 | 0.007204 | -0.2487 | 0.8036 |
    | day_years[2] | 0.01733 | 0.01304 | 1.328 | 0.1841 |
    | day_years[3] | 0.03579 | 0.02175 | 1.646 | 0.09985 |
    | day_years[4] | 0.04911 | 0.02864 | 1.715 | 0.08642 |
    | day_years[5] | 0.07977 | 0.04686 | 1.702 | 0.08867 |
    | day_years[6] | 0.1419 | 0.1019 | 1.393 | 0.1636 |

    [CSV](assets/lmm_pbcseq/gam_tidy.csv)

=== "コード"

    ```python
    from statract import gam, smooth

    fitted_gam = gam(visits, "log_bili", [smooth("day_years", k=k)])
    gam_tidy = fitted_gam.tidy()
    ```

## 解釈と解説

変量切片 LMM では `day_years` が対数ビリルビンと正の関連（約 **+0.095 / 年**）。`dp` は約 −0.11（区間は 0 を含む）。患者間分散 1.20 に対し残差 0.24 で、独立 OLS をそのまま読む根拠は弱い。GAM（`s(day_years)`、k=8、edf≈2.1）は治療を含まない集団平滑です。クラスタ OLS の `day_years` は点推定が小さく区間が広い（変量切片の有無で平均構造の意味が違う）。

限界: GAM API はこの版では平滑1本・ガウスのみ。欠測は完全例。`cif_pbc` と同じ疾患だが解析単位が違う。GPL 教学データ。

## 実行と成果物

```bash
cd examples/lmm_pbcseq
task all
```

既存の付随文書: [concept](https://github.com/mizuy/statract/blob/main/examples/lmm_pbcseq/lmm_pbcseq_concept.md) · [protocol](https://github.com/mizuy/statract/blob/main/examples/lmm_pbcseq/lmm_pbcseq_protocol.md) · [results](https://github.com/mizuy/statract/blob/main/examples/lmm_pbcseq/lmm_pbcseq_results.md) · [discussion](https://github.com/mizuy/statract/blob/main/examples/lmm_pbcseq/lmm_pbcseq_discussion.md)
