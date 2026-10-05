# 解析例ギャラリー

`statract` と `statract.cea` を公開教学データで **入力 → 解析 → results → 図** まで通すスタンドアロン例です。配置と文書は [ANALYSIS_WORKFLOW.md](https://github.com/mizuy/endolab/blob/main/ANALYSIS_WORKFLOW.md)。生データ CSV はリポジトリに入れていません（各 `build.py` が snapshot または論文表のスカラーを cache）。リポジトリ側の目次は [`examples/README.md`](https://github.com/mizuy/endolab/blob/main/examples/README.md) です。

方針の正本は [example-gallery-policy.md](example-gallery-policy.md)。各カードの**主リンクは docs 内の単一ページ**です。読み順は **目的概説 → データと列 → CQ と方針 → flowchart / tableone → メイン解析 → 結果（図/表＋コードタブ）→ 解釈**（7 節。独立のライブラリコード節は置かない）。Results の図・表は **`examples/<stem>/<stem>_out/` が正本**で、選別コピーを `docs/stat/examples/assets/<stem>/` に載せます（別経路で再描画しない）。同期は `uv run python scripts/sync_example_assets.py`。完全な `*_out/` は gitignore のままローカル再生成用です。実行はリポジトリルートで `uv sync` のあと、各 `examples/<stem>/` で `task all`（手順は各 README）。

## ギャラリー {#gallery}

### 生存時間 {#survival}

<div class="grid cards" markdown>

-   __[surv_colon](examples/surv_colon.md)__ — KM / log-rank / Cox

    ---

    補助化学療法 `rx` と再発時間

    **データ** — R `survival::colon`（患者単位）  
    **API** — `plot_survival`（NAR 既定）、`cox_ph`, `proportional_hazards_test`, `write_cox_diagnostic_suite`, `plot_forest`（表一体）

    [ドキュメント →](examples/surv_colon.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/surv_colon)

-   __[cif_pbc](examples/cif_pbc.md)__ — 競合リスク（CIF / Fine–Gray）

    ---

    D-ペニシラミンと肝死の累積発生（移植は競合）

    **データ** — R `survival::pbc`（無作為化例）  
    **API** — `survival_curve(..., kind="aalen_johansen")`, `fine_gray`, 重み付き `cox_ph`, `plot_forest`

    [ドキュメント →](examples/cif_pbc.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/cif_pbc)

-   __[aft_rotterdam](examples/aft_rotterdam.md)__ — Cox / PH 検定 / AFT

    ---

    ホルモン療法と死亡時間（PH が怪しいときの AFT）

    **データ** — R `survival::rotterdam`  
    **API** — `cox_ph`, `proportional_hazards_test`, `write_cox_diagnostic_suite`, `accelerated_failure`（Weibull）, `plot_forest`（表一体）

    [ドキュメント →](examples/aft_rotterdam.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/aft_rotterdam)

-   __[cox_retinopathy](examples/cox_retinopathy.md)__ — クラスター頑健 Cox（Lin–Wei）

    ---

    片眼レーザーと視力喪失時間（患者内クラスター）

    **データ** — R `survival::retinopathy`（眼単位、id クラスター）  
    **API** — `cox_ph(..., cluster=)` / `cluster(id)`、`plot_survival`（NAR）、`proportional_hazards_test`, `write_cox_diagnostic_suite`, `plot_forest`（表一体）、`tableone`

    [ドキュメント →](examples/cox_retinopathy.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/cox_retinopathy)

</div>

### 回帰と混合モデル {#regression}

<div class="grid cards" markdown>

-   __[logit_indo](examples/logit_indo.md)__ — 施設 GLMM / MOR / 固定効果比較

    ---

    直腸インドメタシンと PEP（RCT）

    **データ** — `medicaldata::indo_rct`  
    **API** — `fit_mixed` binomial, `median_odds_ratio`, `glmm_random_effects`, `plot_random_effects`, `plot_forest`, `fit_glm` binomial

    [ドキュメント →](examples/logit_indo.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/logit_indo)

-   __[lmm_pbcseq](examples/lmm_pbcseq.md)__ — ガウス LMM / クラスタ SE / GAM

    ---

    病日・治療と対数ビリルビン（反復検査）

    **データ** — R `survival::pbcseq`  
    **API** — `fit_mixed`, `cluster_covariance`, `gam`, `smooth`

    [ドキュメント →](examples/lmm_pbcseq.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/lmm_pbcseq)

</div>

### 予測モデル {#prediction}

<div class="grid cards" markdown>

-   __[pred_support](examples/pred_support.md)__ — 二項 GLM / 較正 / DCA

    ---

    SUPPORT2 の 180 日死亡確率（train / hold-out）

    **データ** — hbiostat SUPPORT2（fetch only）  
    **API** — `fit_glm` binomial, `write_probability_artifacts`, `plot_calibration`, `plot_dca`, `binary_perf`, `threshold_tradeoff`, `plot_forest`

    [ドキュメント →](examples/pred_support.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/pred_support)

</div>

### 因果推定 {#causal}

<div class="grid cards" markdown>

-   __[psm_rhc](examples/psm_rhc.md)__ — PS 最近傍マッチ → ロジスティック

    ---

    初日 RHC と 30 日死亡（ATT）

    **データ** — Vanderbilt RHC（Connors JAMA 1996）。教学・非再配布  
    **API** — `match_sample`, `balance` / `love_plot`, マッチ後 `fit_glm`, `plot_forest`

    [ドキュメント →](examples/psm_rhc.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/psm_rhc)

-   __[iptw_nhefs](examples/iptw_nhefs.md)__ — 安定化 IPTW（ATE）

    ---

    禁煙と追跡時体重変化

    **データ** — NHEFS（fetch only）  
    **API** — PS `fit_glm`, 手計算の安定化重み, `fit_ols` + `hc_covariance`, `plot_forest`（名前付き `iptw()` は無い）

    [ドキュメント →](examples/iptw_nhefs.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/iptw_nhefs)

</div>

### 費用対効果 {#cea}

<div class="grid cards" markdown>

-   __[cea_sicksicker](examples/cea_sicksicker.md)__ — cohort Markov / ICER / DSA / PSA

    ---

    仮想 Sick-Sicker の SoC・A・B・AB

    **データ** — DARTH 教学パラメータ（Alarid-Escudero et al. MDM 2023 Table 1）。仮想疾患、PHI なし  
    **API** — `simulate_cohort_markov`, `calculate_icers`, `one_way_dsa`, `run_psa`, `ce_plane`, `ceac`, `evpi`

    [ドキュメント →](examples/cea_sicksicker.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/cea_sicksicker)

</div>

### レガシー {#legacy}

<div class="grid cards" markdown>

-   __[legacy_heart](examples/legacy_heart.md)__ — Stanford Heart（ノートブック）

    ---

    時間依存共変量の旧例（n=172）。ワークフロー正本ではない

    **データ** — lifelines / survival Stanford Heart  
    **API** — 既存ノートブック（counting process）

    [ドキュメント →](examples/legacy_heart.md) · [コード](https://github.com/mizuy/endolab/tree/main/examples/legacy_heart)

</div>

## 例の詳細 {#details}

各例の本文は上のカードから開く **docs 単一ページ**（上記 7 節）にあります。サイトを離れずに読めます。実行用の薄い README / concept·protocol 分割は `examples/<stem>/` に残しています。

| 例 | docs ページ | 実行コード |
|----|-------------|------------|
| surv_colon | [surv_colon](examples/surv_colon.md) | [examples/surv_colon/](https://github.com/mizuy/endolab/tree/main/examples/surv_colon) |
| cif_pbc | [cif_pbc](examples/cif_pbc.md) | [examples/cif_pbc/](https://github.com/mizuy/endolab/tree/main/examples/cif_pbc) |
| aft_rotterdam | [aft_rotterdam](examples/aft_rotterdam.md) | [examples/aft_rotterdam/](https://github.com/mizuy/endolab/tree/main/examples/aft_rotterdam) |
| cox_retinopathy | [cox_retinopathy](examples/cox_retinopathy.md) | [examples/cox_retinopathy/](https://github.com/mizuy/endolab/tree/main/examples/cox_retinopathy) |
| logit_indo | [logit_indo](examples/logit_indo.md) | [examples/logit_indo/](https://github.com/mizuy/endolab/tree/main/examples/logit_indo) |
| lmm_pbcseq | [lmm_pbcseq](examples/lmm_pbcseq.md) | [examples/lmm_pbcseq/](https://github.com/mizuy/endolab/tree/main/examples/lmm_pbcseq) |
| psm_rhc | [psm_rhc](examples/psm_rhc.md) | [examples/psm_rhc/](https://github.com/mizuy/endolab/tree/main/examples/psm_rhc) |
| iptw_nhefs | [iptw_nhefs](examples/iptw_nhefs.md) | [examples/iptw_nhefs/](https://github.com/mizuy/endolab/tree/main/examples/iptw_nhefs) |
| pred_support | [pred_support](examples/pred_support.md) | [examples/pred_support/](https://github.com/mizuy/endolab/tree/main/examples/pred_support) |
| cea_sicksicker | [cea_sicksicker](examples/cea_sicksicker.md) | [examples/cea_sicksicker/](https://github.com/mizuy/endolab/tree/main/examples/cea_sicksicker) |
| legacy_heart | [legacy_heart](examples/legacy_heart.md) | [examples/legacy_heart/](https://github.com/mizuy/endolab/tree/main/examples/legacy_heart) |

## API から探す {#api-index}

| API | 例 |
|-----|-----|
| Table 1 / flowchart / `write_csv_companion` | 各スタンドアロン例 |
| KM / log-rank / Cox / `plot_survival` | [`surv_colon`](examples/surv_colon.md)（正本）、[`aft_rotterdam`](examples/aft_rotterdam.md)、[`cox_retinopathy`](examples/cox_retinopathy.md) |
| Cox `cluster=` / `cluster(id)`（Lin–Wei sandwich） | [`cox_retinopathy`](examples/cox_retinopathy.md) |
| Aalen–Johansen / Fine–Gray | [`cif_pbc`](examples/cif_pbc.md) |
| `proportional_hazards_test` / `accelerated_failure` | [`aft_rotterdam`](examples/aft_rotterdam.md) |
| 二項 GLMM / MOR / RE plot / `fit_mixed` | [`logit_indo`](examples/logit_indo.md) |
| `plot_forest`（HR / OR / OLS coef） | Cox・Fine–Gray・二項 GLM・IPTW OLS の各例 |
| 較正 / DCA / Brier / `binary_perf` / `threshold_tradeoff` | [`pred_support`](examples/pred_support.md) |
| `match_sample` nearest | [`psm_rhc`](examples/psm_rhc.md) |
| `match_sample` CEM（バランス感度のみ） | [`iptw_nhefs`](examples/iptw_nhefs.md) |
| `fit_mixed` / `gam` / `cluster_covariance` | [`lmm_pbcseq`](examples/lmm_pbcseq.md) |
| `fit_ols` + `hc_covariance` / IPTW 手計算 | [`iptw_nhefs`](examples/iptw_nhefs.md) |
| `simulate_cohort_markov` / `calculate_icers` / DSA / PSA | [`cea_sicksicker`](examples/cea_sicksicker.md) |

## 使わないもの {#out-of-scope}

`psmatch` と `GLMHelper` は使わない。R の `forest.R` は Python gallery の正本にしない（matplotlib の `plot_forest` / `glmm_forestplot` を使う）。較正 / DCA は [`pred_support`](examples/pred_support.md)（`logit_indo` の効果推定には載せない）。

Quickstart は合成データのままです。公開データでの Python 対 R の壁時計と係数差は [Benchmarks](benchmarks.md)。
