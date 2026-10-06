# surv_colon — colon 補助療法の生存例

KM / log-rank / Cox。`etype=1` で患者単位に畳み、`hue=rx` の Table 1 から KM 図まで。

**読む:** [docs 単一ページ](../../docs/stat/examples/surv_colon.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/surv_colon/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/surv_colon
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python surv_colon.py
```

完全な成果物は `surv_colon_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/surv_colon/`](../../docs/stat/examples/assets/surv_colon/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[surv_colon_concept.md](surv_colon_concept.md) · [surv_colon_protocol.md](surv_colon_protocol.md) · [surv_colon_results.md](surv_colon_results.md) · [surv_colon_discussion.md](surv_colon_discussion.md)
