# 傾向スコアマッチング

`psm_rhc` — 傾向スコア最近傍マッチ → ロジスティック

1:1 最近傍マッチで処置群での平均処置効果（average treatment effect on the treated、ATT）を推定する例の正本。API は `match_sample`。マッチ後オッズ比（odds ratio、OR）の forest まで。

[← ギャラリー](index.md) · [実行用ディレクトリと手順（GitHub）](https://github.com/mizuy/statract/tree/main/examples/psm_rhc)

## 目的概説

観察研究で交絡を減らす MatchIt 流の `match_sample` を、公開の重症患者教学データで示すための例です。推定対象は ATT、API は `match_sample` です。因果の確定主張はしません。

解説は Learn の [第 5 章 傾向スコアマッチング](../learn/05_propensity-score-matching.md) を参照してください。

## データと列の説明

| 項目 | 内容 |
|------|------|
| ソース | Vanderbilt の右心カテーテルデータ（Connors JAMA 1996 系）。**利用条件が未確認のため再配布しない** |
| 取得 | `build.py` → `https://hbiostat.org/data/repo/rhc.csv` |
| 単位 | 入院/患者 1 行 |

主な解析列:

| 列 | 意味 |
|----|------|
| `swang1` / `rhc` | 初日の右心カテーテル（Yes/No → 0/1） |
| `dth30` / `dth30_bin` | 30 日死亡 |
| `age`, `sex`, `race`, `edu`, `cat1`, `ca`, … | マッチ共変量（protocol 列挙の完全例） |
| `weights` | マッチ後ウェイト（キャリパー外は 0） |

## CQ と大まかな解析方針

**CQ** — 集中治療室（intensive care unit、ICU）入室初日の右心カテーテル（right heart catheterization、RHC。肺動脈カテーテル）は、交絡を傾向スコア最近傍マッチで揃えたあと、30 日死亡と関連するか。

方針:

1. 完全例コホートを確定（除外なしのため flowchart 省略）
2. 未マッチ Table 1 → `match_sample`（nearest / logit / ATT / キャリパー 0.2、ロジット尺度）
3. `balance` / `love_plot` とマッチ後 Table 1
4. マッチ標本で `dth30_bin ~ rhc` の二項一般化線形モデル（generalized linear model、GLM）で OR を推定し、`plot_forest(..., layout="table")` で描く

## flowchart / tableone

解析セットに除外はありません（全例解析）。ギャラリー方針どおり **flowchart / Mermaid は省略**します。

### Table 1（未マッチ / マッチ後）

=== "未マッチ"

    --8<-- "examples/assets/psm_rhc/table1_unmatched_gt.md"

    [CSV](assets/psm_rhc/table1_unmatched.csv)

=== "マッチ後"

    --8<-- "examples/assets/psm_rhc/table1_matched_gt.md"

    [CSV](assets/psm_rhc/table1_matched.csv)

=== "コード"

    ```python
    from statract import agg_category, agg_mean_sd, tableone

    table1_params = {
        "Age": ("age", agg_mean_sd),
        "Sex": ("sex", agg_category),
        "Race": ("race", agg_category),
        "Education (years)": ("edu", agg_mean_sd),
        "Primary disease": ("cat1", agg_category),
        "Cancer": ("ca", agg_category),
        "APS1": ("aps1", agg_mean_sd),
        "Mean BP": ("meanbp1", agg_mean_sd),
        "DNR": ("dnr1", agg_category),
        "30-day death": ("dth30_bin", agg_mean_sd),
    }
    unmatched = tableone(
        cohort, table1_params, hue="rhc", add_all=True, add_pvalue=True,
    )
    matched_gt = tableone(
        kept,
        {k: v for k, v in table1_params.items() if k != "30-day death"},
        hue="rhc",
        add_all=True,
        add_pvalue=True,
    )
    # CSV/HTML/md を *_out/ に書くときだけ write_tableone_artifacts
    # （stem="table1_unmatched" / "table1_matched"。ギャラリー同期用）
    ```

## メインの解析方法とそのコア

傾向スコア $e(x)=P(\mathrm{rhc}=1\mid x)$ をロジスティック回帰で推定し、最近傍 1:1・非復元で ATT を推定します（組を作る順はデータの並び順、`order="data"`）。距離とキャリパーは傾向スコアのロジット尺度で測り（`link="linear.logit"`）、キャリパーはロジットの標準偏差の 0.2 倍です（[Learn 5.5](../learn/05_propensity-score-matching.md#matching) が勧める尺度）。Love plot はマッチ前後の標準化差（standardized mean difference、SMD）です（[Learn 5.6](../learn/05_propensity-score-matching.md#balance)）。アウトカムモデル:

$$
\operatorname{logit}P(\mathrm{dth30}=1)=\beta_0+\beta_1\mathrm{rhc}
$$

実装は `match_sample` と `fit_glm(..., family="binomial", weights="weights")` → `plot_forest(..., layout="table")`（論文用白黒は `style="bw"`）。


## 結果

完全例 5735 人（RHC あり 2184 人）、マッチ後（weights>0）3374 人（RHC あり 1687 人）。RHC あり 497 人（約 23%）はキャリパーで外れたので、推定値は「組ができた RHC あり患者」での効果です（[Learn 5.5 の注意](../learn/05_propensity-score-matching.md#matching)）。サイト掲載は `assets/psm_rhc/`。

### Love plot（マッチ前後の SMD）

=== "図"

    ![Love plot](assets/psm_rhc/love_plot.png)

=== "コード"

    ```python
    fig = matched.love_plot()
    fig.savefig(out / "figures" / "love_plot.png", dpi=150, bbox_inches="tight")
    ```

### バランス（要約）

=== "表"

    | term | smd_all | smd_matched | n_matched | pair_distance |
    | --- | --- | --- | --- | --- |
    | distance | 1.027 | 0.01169 | 5735 | 0.01415 |
    | age | -0.06135 | 0.0145 | 5735 | 1.119 |
    | aps1 | 0.5014 | -0.00876 | 5735 | 1.012 |
    | meanbp1 | -0.4551 | 0.002638 | 5735 | 0.9114 |
    | pafi1 | -0.4332 | -0.0174 | 5735 | 0.9746 |

    抜粋。完全表: [CSV](assets/psm_rhc/balance.csv)。`distance` 行はロジット尺度の傾向スコアです。SMD の分母は既定の、調整前の両群をあわせた標準偏差です（定義は [Learn 5.6](../learn/05_propensity-score-matching.md#balance)）。

=== "コード"

    ```python
    balance = matched.balance()
    balance.write_csv(out / "balance.csv")
    ```

### ロジスティック（マッチ後、OR）

=== "表"

    | term | estimate | std_error | statistic | p_value | conf_low | conf_high | exp_estimate | exp_conf_low | exp_conf_high |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | (Intercept) | -0.8363 | 0.05301 | -15.78 | 4.583e-56 | -0.9402 | -0.7324 | 0.4333 | 0.3905 | 0.4807 |
    | rhc | 0.2883 | 0.07324 | 3.937 | 8.260e-05 | 0.1448 | 0.4319 | 1.334 | 1.156 | 1.54 |

    [CSV](assets/psm_rhc/glm_matched.csv)

=== "コード"

    ```python
    from statract import fit_glm

    glm = fit_glm(kept, "dth30_bin ~ rhc", family="binomial", weights="weights")
    glm_tidy = glm.tidy(exponentiate=True)
    ```

### マッチ後 forest（OR）

=== "図"

    ![Matched logistic forest (OR)](assets/psm_rhc/glm_matched_forest.png)

=== "コード"

    ```python
    from statract import plot_forest

    plot_forest(
        glm_tidy,
        out / "figures" / "glm_matched_forest.png",
        title="Matched logistic (OR)",
        xlabel="Odds ratio",
        layout="table",
    )
    ```

### 参考: 未マッチロジスティック

| term | estimate | exp_estimate | exp_conf_low | exp_conf_high |
| --- | --- | --- | --- | --- |
| rhc | 0.3276 | 1.388 | 1.241 | 1.552 |

[CSV](assets/psm_rhc/glm_unmatched.csv)

## 解釈と解説

未マッチの 30 日死亡 OR は約 **1.39**、マッチ後は約 **1.33**（いずれも `rhc` の OR>1）。Love plot で SMD の縮小を確認できます。観察された交絡を傾向スコアマッチで揃えても正の関連は残り、Connors JAMA 1996 の問題意識と方向は矛盾しません。教学用 ATT 近似であり、RHC 適応の根拠にはしません。

限界:

- 未測定交絡（大きさの目安は [Learn 7.3 E-value](../learn/07_assumptions-sensitivity.md#e-value)）
- キャリパーでの RHC あり 497 人の除外
- 組を無視した標準誤差（組を考えた区間は [Learn 5.7](../learn/05_propensity-score-matching.md#matched-inference)）
- 利用条件が未確認で再配布しない CSV

## 実行と成果物

```bash
cd examples
task psm_rhc:all
```

既存の付随文書: [concept](https://github.com/mizuy/statract/blob/main/examples/psm_rhc/psm_rhc_concept.md) · [protocol](https://github.com/mizuy/statract/blob/main/examples/psm_rhc/psm_rhc_protocol.md) · [results](https://github.com/mizuy/statract/blob/main/examples/psm_rhc/psm_rhc_results.md) · [discussion](https://github.com/mizuy/statract/blob/main/examples/psm_rhc/psm_rhc_discussion.md)
