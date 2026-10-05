# cox_retinopathy — 網膜症のクラスター頑健 Cox

KM / log-rank / Cox（Lin–Wei sandwich、`cluster(id)`）。両眼が同一患者に属する公開 RCT。

**読む:** [docs 単一ページ](../../docs/stat/examples/cox_retinopathy.md)（ギャラリー: [解析例](https://mizuy.github.io/endolab/stat/examples/cox_retinopathy/)）。データ出典・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/cox_retinopathy
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python cox_retinopathy.py
```

完全な成果物は `cox_retinopathy_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/cox_retinopathy/`](../../docs/stat/examples/assets/cox_retinopathy/) にあります。

## ローカル workflow 文書

[cox_retinopathy_concept.md](cox_retinopathy_concept.md) · [cox_retinopathy_protocol.md](cox_retinopathy_protocol.md) · [cox_retinopathy_results.md](cox_retinopathy_results.md) · [cox_retinopathy_discussion.md](cox_retinopathy_discussion.md)
