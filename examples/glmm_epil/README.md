# glmm_epil — てんかん RCT の計数 GLMM

`MASS::epil`（Thall & Vail 1990、progabide RCT、59 人 × 4 回）の発作回数に、患者変量切片のポアソン GLMM・負の二項 GLMM・`ar1(period + 0 | subject)`・ゼロ過剰負の二項を `fit_mixed` で当てはめ、AIC と対数尤度で比べます。progabide の率比は `plot_forest(..., layout="table")`、患者 BLUP は `plot_random_effects`。

**読む:** [docs 単一ページ](../../docs/examples/glmm_epil.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/glmm_epil/)）。

## データ

`MASS::epil`（MASS は GPL-2 | GPL-3）。`build.py` が R（`MASS`）か Rdatasets の CSV（`https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv`）から取得します。生データはリポジトリに入れません（`snapshot/` と `cache/` は git 管理外）。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/glmm_epil
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python glmm_epil.py
```

完全な成果物は `glmm_epil_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/examples/assets/glmm_epil/`](../../docs/examples/assets/glmm_epil/) にあります（`uv run python scripts/sync_example_assets.py --stem glmm_epil`）。
