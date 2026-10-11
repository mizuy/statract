# pred_support — SUPPORT2 の予測確率・較正 / DCA 例

180 日死亡の二項 GLM を train にフィットし、ホールドアウトで `write_probability_artifacts` / `plot_calibration` / `plot_dca` / `binary_perf` を通す。ROC は `roc_curve` / `roc_test`（DeLong）、train だけの内部検証は `validate_logistic` / `calibrate_logistic`（ブートストラップ B=200）。

**読む:** [docs 単一ページ](../../docs/examples/pred_support.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/pred_support/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples
task pred_support:all
# または
cd pred_support
uv run python build.py
uv run python pred_support.py
```

完全な成果物は `pred_support_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/examples/assets/pred_support/`](../../docs/examples/assets/pred_support/) にあります。生の SUPPORT2 CSV はリポジトリに入れません。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[pred_support_concept.md](pred_support_concept.md) · [pred_support_protocol.md](pred_support_protocol.md) · [pred_support_results.md](pred_support_results.md) · [pred_support_discussion.md](pred_support_discussion.md)
