# logit_indo — インドメタシン RCT のロジスティック例

施設変量 `fit_mixed(..., family="binomial")`（lme-python / lme-rs）+ MOR / 施設 BLUP。固定効果二項 GLM は比較。ホールドアウト較正は使わない。

**読む:** [docs 単一ページ](../../docs/examples/logit_indo.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/examples/logit_indo/)）。データ出典・プロトコル詳細・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/logit_indo
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python logit_indo.py
```

完全な成果物は `logit_indo_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/examples/assets/logit_indo/`](../../docs/examples/assets/logit_indo/) にあります。

## 付随文書

実行・再生成用の分割版（読者向け要約ではない）:

[logit_indo_concept.md](logit_indo_concept.md) · [logit_indo_protocol.md](logit_indo_protocol.md) · [logit_indo_results.md](logit_indo_results.md) · [logit_indo_discussion.md](logit_indo_discussion.md)
