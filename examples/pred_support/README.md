# pred_support — SUPPORT2 の予測確率・較正 / DCA 例

180 日死亡の二項 GLM を train にフィットし、ホールドアウトで `write_probability_artifacts` / `plot_calibration` / `plot_dca` / `binary_perf` を通す。

**読む:** [docs 単一ページ](../../docs/stat/examples/pred_support.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/pred_support/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/pred_support
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python pred_support.py
```

完全な成果物は `pred_support_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/pred_support/`](../../docs/stat/examples/assets/pred_support/) にあります。生の SUPPORT2 CSV はリポジトリに入れません。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[pred_support_concept.md](pred_support_concept.md) · [pred_support_protocol.md](pred_support_protocol.md) · [pred_support_results.md](pred_support_results.md) · [pred_support_discussion.md](pred_support_discussion.md)
