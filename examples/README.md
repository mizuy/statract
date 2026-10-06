# statract 解析例

実行スクリプトは各ディレクトリの `build.py` と `{stem}.py`。例専用のパス・parquet cache・フローチャート補助は `examples/support.py`（公開 API ではない）。掲載図と表は `docs/examples/assets/`。

# 解析例

公開臨床データで **Table 1 → 解析 → results → 図** までを示す。データ CSV は git に入れない。各 `build.py` が取得し、`snapshot/` と `cache/build/` は gitignore する。

**読める入口（カードギャラリー + 各例の単一ページ）は公開ドキュメント側:**  
[解析例ギャラリー](https://mizuy.github.io/statract/examples/)（ソース: [`docs/examples/index.md`](../docs/examples/index.md)）。  
各例は docs の **1 HTML ページ**にまとめてあります（例: [surv_colon](https://mizuy.github.io/statract/examples/surv_colon/)）。本ディレクトリには実行コードと既存の付随文書を置きます。完全な `*_out/` は gitignore です。サイト掲載用の選別図・表は **`*_out/` と同一ファイル**を [`docs/examples/assets/`](../docs/examples/assets/) へコピーします（`uv run python scripts/sync_example_assets.py`）。

リポジトリルートで `uv sync` したあと、各ディレクトリで `task all`（または `uv run python build.py` → `uv run python {stem}.py`）。

| ディレクトリ | docs ページ | 問い | 見る API | データ |
|--------------|-------------|------|----------|--------|
| [surv_colon/](surv_colon/) | [docs](../docs/examples/surv_colon.md) | 補助化学療法 `rx` と再発時間 | `plot_survival`（NAR）、`cox_ph`, `proportional_hazards_test`, `write_cox_diagnostic_suite`, `plot_forest`（表一体） | R `survival::colon`（患者単位）。ライセンス GPL-2/3。git 非収載 |
| [logit_indo/](logit_indo/) | [docs](../docs/examples/logit_indo.md) | 直腸インドメタシンと PEP | `fit_mixed` binomial（主）、`median_odds_ratio`、`plot_random_effects`、比較用 `fit_glm` | `medicaldata::indo_rct`（MIT + Elmunzer NEJM 2012 引用） |
| [psm_rhc/](psm_rhc/) | [docs](../docs/examples/psm_rhc.md) | 初日 RHC と 30 日死亡 | `match_sample`, `balance` / `love_plot`, マッチ後 `fit_glm` | Vanderbilt RHC（Connors JAMA 1996）。**再配布しない。教学のみ** |
| [cif_pbc/](cif_pbc/) | [docs](../docs/examples/cif_pbc.md) | D-ペニシラミンと肝死 CIF（移植は競合） | `survival_curve(..., kind="aalen_johansen")`, `fine_gray`, 重み付き `cox_ph` | R `survival::pbc`（無作為化）。GPL-2/3 |
| [aft_rotterdam/](aft_rotterdam/) | [docs](../docs/examples/aft_rotterdam.md) | ホルモン療法と死亡時間 | `cox_ph`, `proportional_hazards_test`, `accelerated_failure`（Weibull） | R `survival::rotterdam`。GPL-2/3 |
| [cox_retinopathy/](cox_retinopathy/) | [docs](../docs/examples/cox_retinopathy.md) | 片眼レーザーと視力喪失（患者クラスター） | `cox_ph` + `cluster(id)` / `cluster=`（Lin–Wei）、`plot_survival`（NAR）、`plot_forest`（表一体）、`tableone` | R `survival::retinopathy`。GPL-2/3。git 非収載 |
| [lmm_pbcseq/](lmm_pbcseq/) | [docs](../docs/examples/lmm_pbcseq.md) | 病日・治療と対数ビリルビン | `fit_mixed`, `cluster_covariance`, `gam` / `smooth`（主解析） | R `survival::pbcseq`。GPL-2/3 |
| [iptw_nhefs/](iptw_nhefs/) | [docs](../docs/examples/iptw_nhefs.md) | 禁煙と体重変化（ATE） | 手計算の安定化 IPTW、`fit_ols` + `hc_covariance`。`iptw()` は無い | NHEFS（`causaldata` / Rdatasets / Hernán CSV）。**fetch only** |
| [pred_support/](pred_support/) | [docs](../docs/examples/pred_support.md) | 180 日死亡確率の較正 / DCA | `fit_glm` binomial、`write_probability_artifacts`、`plot_calibration`、`plot_dca`、`binary_perf` | SUPPORT2（hbiostat）。**fetch only** |
| [cea_sicksicker/](cea_sicksicker/) | [docs](../docs/examples/cea_sicksicker.md) | 仮想 Sick-Sicker の 4 戦略 CEA | `simulate_cohort_markov`, `calculate_icers`, `one_way_dsa`, `run_psa` / `ce_plane` / `ceac` / `evpi` | DARTH 教学パラメータ（Alarid-Escudero et al. MDM 2023 Table 1）。git に CSV なし |

`fit_mixed(..., family="binomial")` が indo の施設 GLMM。ガウス LMM は同じ `fit_mixed` の既定（`family="gaussian"`）。マッチングは `match_sample`、固定効果の回帰は `fit_glm` です。係数 forest は matplotlib の `plot_forest`（Cox HR / GLM OR / OLS）。R の `forest.R` は正本にしない。

速度と R との差: [`docs/models/benchmarks.md`](../docs/models/benchmarks.md)。
