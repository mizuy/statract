# lmm_pbcseq — pbcseq のガウス LMM / GAM 例

`fit_mixed`、クラスタ SE、GAM 平滑（病日）。

**読む:** [docs 単一ページ](../../docs/stat/examples/lmm_pbcseq.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/lmm_pbcseq/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/lmm_pbcseq
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python lmm_pbcseq.py
```

完全な成果物は `lmm_pbcseq_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/lmm_pbcseq/`](../../docs/stat/examples/assets/lmm_pbcseq/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[lmm_pbcseq_concept.md](lmm_pbcseq_concept.md) · [lmm_pbcseq_protocol.md](lmm_pbcseq_protocol.md) · [lmm_pbcseq_results.md](lmm_pbcseq_results.md) · [lmm_pbcseq_discussion.md](lmm_pbcseq_discussion.md)
