# htest_licorice — 甘草うがい RCT の基本検定

群ごとの発生割合（`binom_test` の Clopper–Pearson 区間）、群間のリスク差（`prop_test`）、痛みスコアの Wilcoxon 順位和（`wilcox_test`）と Welch の平均差（`t_test`）、群内の時点比較（`mcnemar_test`、対応のある `wilcox_test`）、4 時点の Holm 補正（`p_adjust`）。

**読む:** [docs 単一ページ](../../docs/examples/htest_licorice.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/htest_licorice/)）。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/htest_licorice
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python htest_licorice.py
```

データは R の `medicaldata` があればそこから、なければ Rdatasets の CSV、それも届かなければパッケージの `.rda` を GitHub から取って R で読みます。完全な成果物は `htest_licorice_out/`（git 管理外）。サイト掲載用は [`docs/examples/assets/htest_licorice/`](../../docs/examples/assets/htest_licorice/) です。
