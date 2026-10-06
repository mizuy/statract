# aft_rotterdam — Rotterdam の Cox / PH / AFT 例

Cox、PH 検定、Weibull AFT を同じ公開コホートで通す。

**読む:** [docs 単一ページ](../../docs/stat/examples/aft_rotterdam.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/aft_rotterdam/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/aft_rotterdam
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python aft_rotterdam.py
```

完全な成果物は `aft_rotterdam_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/aft_rotterdam/`](../../docs/stat/examples/assets/aft_rotterdam/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[aft_rotterdam_concept.md](aft_rotterdam_concept.md) · [aft_rotterdam_protocol.md](aft_rotterdam_protocol.md) · [aft_rotterdam_results.md](aft_rotterdam_results.md) · [aft_rotterdam_discussion.md](aft_rotterdam_discussion.md)
