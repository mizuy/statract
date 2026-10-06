# cea_sicksicker — DARTH Sick-Sicker の cohort Markov CEA

`statract.cea` の公開教学例。仮想疾患の 4 状態 STM で SoC / A / B / AB を ICER・DSA・PSA まで通す。病態はライブラリ外（このディレクトリ）に置く。

**読む:** [docs 単一ページ](../../docs/stat/examples/cea_sicksicker.md)（ギャラリー: [解析例](https://mizuy.github.io/statract/stat/examples/cea_sicksicker/)）。パラメータ出典・プロトコル・図つき Results はそちら。

## 実行

リポジトリルートで `uv sync` のあと:

```bash
cd examples/cea_sicksicker
task all
# または
PYTHONPATH=. uv run python build.py
PYTHONPATH=. uv run python cea_sicksicker.py
```

完全な成果物は `cea_sicksicker_out/`（git 管理外）。サイト掲載用に選んだ図・表だけが [`docs/stat/examples/assets/cea_sicksicker/`](../../docs/stat/examples/assets/cea_sicksicker/) にあります。

## 付随文書

[cea_sicksicker_concept.md](cea_sicksicker_concept.md) · [cea_sicksicker_protocol.md](cea_sicksicker_protocol.md) · [cea_sicksicker_results.md](cea_sicksicker_results.md) · [cea_sicksicker_discussion.md](cea_sicksicker_discussion.md)
