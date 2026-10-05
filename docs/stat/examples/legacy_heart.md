# legacy_heart — Stanford Heart（ノートブック）

時間依存共変量の旧例（n=172）。**ワークフロー正本ではない。**

**データ** — lifelines / survival Stanford Heart  
**API** — 既存ノートブック（counting process）

[← ギャラリー](../examples.md) · [実行用ディレクトリ（GitHub）](https://github.com/mizuy/endolab/tree/main/examples/legacy_heart)

## 概要

Stanford Heart Transplant（n=172）の counting process 例を jupyter ノートブックとして残しています。新規の正本はギャラリー上のスタンドアロン例（特に [`surv_colon`](surv_colon.md)）を見てください。

## Concept（短縮）

時間依存共変量（移植状態）を counting process 形式で扱う旧デモです。Table 1 と生存解析のノートブックがあり、`ANALYSIS_WORKFLOW.md` の concept / protocol / results / discussion 4 文書フローは採用していません。

## Protocol / 実行

```bash
cd examples/legacy_heart
uv sync
python build.py
# main.ipynb / r.ipynb を開いて解析
```

- `build.py` — snapshot → `cache/build/`
- `main.ipynb` / `r.ipynb` — Table 1 と生存解析

## Results / Discussion

新規の教学正本としては使わない想定です。生存 API の正本は [`surv_colon`](surv_colon.md)、競合リスクは [`cif_pbc`](cif_pbc.md)、PH / AFT は [`aft_rotterdam`](aft_rotterdam.md) を参照してください。

詳細はリポジトリの [README](https://github.com/mizuy/endolab/blob/main/examples/legacy_heart/README.md) です。
