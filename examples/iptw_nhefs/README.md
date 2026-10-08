# iptw_nhefs — NHEFS の安定化 IPTW 例

手計算の安定化 IPTW + `fit_ols` / HC3。CEM はバランス感度のみ。欠損アウトカムの多重代入感度（`impute_chained` + `pool`）つき。

**読む:** [docs 単一ページ](../../docs/examples/iptw_nhefs.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/iptw_nhefs/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/iptw_nhefs
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python iptw_nhefs.py
```

完全な成果物は `iptw_nhefs_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/examples/assets/iptw_nhefs/`](../../docs/examples/assets/iptw_nhefs/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[iptw_nhefs_concept.md](iptw_nhefs_concept.md) · [iptw_nhefs_protocol.md](iptw_nhefs_protocol.md) · [iptw_nhefs_results.md](iptw_nhefs_results.md) · [iptw_nhefs_discussion.md](iptw_nhefs_discussion.md)
