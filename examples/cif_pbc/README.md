# cif_pbc — PBC の競合リスク例

Aalen–Johansen CIF と Fine–Gray（移植を競合イベント）。

**読む:** [docs 単一ページ](../../docs/stat/examples/cif_pbc.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/cif_pbc/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/cif_pbc
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python cif_pbc.py
```

完全な成果物は `cif_pbc_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/cif_pbc/`](../../docs/stat/examples/assets/cif_pbc/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[cif_pbc_concept.md](cif_pbc_concept.md) · [cif_pbc_protocol.md](cif_pbc_protocol.md) · [cif_pbc_results.md](cif_pbc_results.md) · [cif_pbc_discussion.md](cif_pbc_discussion.md)
