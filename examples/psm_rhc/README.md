# psm_rhc — RHC の傾向スコアマッチ例

`match_sample` 最近傍 → Love plot → マッチ後ロジスティック。

**読む:** [docs 単一ページ](../../docs/examples/psm_rhc.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/psm_rhc/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples
task psm_rhc:all
# または
cd psm_rhc
uv run python build.py
uv run python psm_rhc.py
```

完全な成果物は `psm_rhc_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/examples/assets/psm_rhc/`](../../docs/examples/assets/psm_rhc/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[psm_rhc_concept.md](psm_rhc_concept.md) · [psm_rhc_protocol.md](psm_rhc_protocol.md) · [psm_rhc_results.md](psm_rhc_results.md) · [psm_rhc_discussion.md](psm_rhc_discussion.md)
