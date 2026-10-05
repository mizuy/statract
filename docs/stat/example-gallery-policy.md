# 解析例ギャラリー方針

`docs/stat/examples/` に載せる公開教学例の読み方・書き方・成果物同期の正本です。ギャラリー索引は [Examples / Gallery](examples.md)。ローカル実行の文書分割は [ANALYSIS_WORKFLOW.md](https://github.com/mizuy/statract/blob/main/examples/README.md) を参照してください。

## 読み取り UX とローカル実行の分担

| 層 | 役割 |
|----|------|
| **mkdocs 単一ページ**（`docs/stat/examples/<stem>.md`） | 読者がサイト上で完結して読む本文。1 例 = 1 ページ |
| **`examples/<stem>/` の concept / protocol / results / discussion** | ローカルで `task all` するときの解析ワークフロー分割。読み取り UX の正本ではない |
| **薄い README** | 目的・実行手順・docs へのリンクのみ |

ギャラリーカードの**主リンクは docs 内ページ**。GitHub のコードツリーは補助リンク。

## ページ構成（7 節）

各メイン例は次の順。**独立した「ライブラリ呼び出し時のコード」節は置かない**（呼び出し例は結果の Material タブ内に置く）。

1. **目的概説**
2. **データと列の説明**
3. **CQ と大まかな解析方針**
4. **flowchart / tableone**（除外が無い例は tableone のみ。下記）
5. **メインの解析方法とそのコア**
6. **結果**
7. **解釈と解説**

末尾に短い「実行と成果物」（`task all`・ワークフロー分割へのリンク）を置いてよい。

### Flowchart は除外があるときだけ

解析セットから行が落ちない（全段 `excluded: 0`、inclusion 100%）ときは **`flowchart` / Mermaid / `text_flowchart` を載せない**。コホート n を一文で述べ、Table 1 に進む。実際に除外・畳み込みがある例（例: `surv_colon` の患者単位畳み、`cif_pbc` の非無作為化除外、`iptw_nhefs` の欠損除外）だけ flowchart を残し、適用できるとき Mermaid 化する。

## 結果: Material タブ

図・表ごとに Material tabs を使う。

- **図** / **表** / **コード**（当該成果物に応じて必要なタブだけ）
- コードタブはコピー可能な実 API（`statract` 等）。疑似コードや「別節にまとめたライブラリ一覧」で代替しない

## 成果物の同一性（`*_out/` ↔ docs assets）

| 項目 | 方針 |
|------|------|
| 正本 | `examples/<stem>/<stem>_out/`（`task analysis` / `task all` が生成） |
| サイト掲載 | 選別コピーを `docs/stat/examples/assets/<stem>/` に置く |
| 同期 | `uv run python scripts/sync_example_assets.py`（または同等）。手作業で別経路の図を描き直して差を作らない。2026-10-05 に runners を再実行し、掲載ファイルは再生成後も同一だった |
| git | 完全な `*_out/` は gitignore。生データ CSV もリポジトリに入れない。掲載用アセットのみ commit |

## Forest（GLM / Cox 系）

HR / OR / 係数の forest は再実装した `plot_forest(..., layout="table")`（表 + forest 一体。既定も `"table"`。論文向けのフォント・行揃え・マーカー＋CI・参照線・HR/OR の対数軸）。docs 掲載は `style="color"`（既定）。印刷用白黒は KM と同じ `style="bw"`。docs のコードタブでも `layout="table"` を明示する。R の `forest.R` を gallery 正本にしない。

対象の目安: Cox、Fine–Gray（重み付き Cox）、二項 GLM、IPTW OLS など。LMM / GAM だけの例に無理に forest を足さない。

## 生存時間まわり

単一イベント生存（および Cox を主にする例）では次を揃える。

| 要素 | 方針 |
|------|------|
| KM | `plot_survival` 既定の **number-at-risk**（in-house `survival_curve`）。NAR は曲線と **同じ x 軸**（`sharex` の表パネル）。`style="bw"` で白黒（線種+打点） |
| Forest | `plot_forest` 既定の **表一体**（`layout="table"`）。`style="bw"` で白黒（黒マーカー＋ヒゲ） |
| PH | `proportional_hazards_test` |
| 診断 | survminer 風スイート（`write_cox_diagnostic_suite`: Schoenfeld / log-log / dfbeta / martingale / deviance 等） |
| Flowchart | **除外がある例だけ** `examples/support.py` の `flowchart`。適用できるとき `mermaid_flowchart` を docs の図タブと `*_out/` に残す。除外ゼロなら省略 |

競合リスク（CIF / Fine–Gray）では偽の PH スイートを載せない。KM の代わりに AJ CIF を示し、Fine–Gray forest は `layout="table"`。非生存例（二項 GLM・PSM・IPTW・LMM）にも偽の PH / NAR スイートを付けない。除外がある例の flowchart は Mermaid 化を推奨する。

## データ

公開教学データのみ（Rdatasets、medicaldata、hbiostat 教学 CSV、NHEFS fetch など）。再配布制約があるものは docs に明記する。

## 対象ステム

| stem | 備考 |
|------|------|
| `surv_colon` | 単一イベント生存の正本（KM+NAR、PH、診断、table forest） |
| `aft_rotterdam` | PH 検定 + AFT の正本 |
| `cox_retinopathy` | Cox Lin–Wei sandwich（`cluster(id)`）。両眼クラスター |
| `cif_pbc` | 競合リスク（CIF / Fine–Gray） |
| `logit_indo` | 施設 GLMM（主）+ MOR / RE plot。固定効果 GLM は比較。較正ホールドアウトなし |
| `pred_support` | 予測（点モデル）の正本。train / hold-out 較正・DCA・`binary_perf` |
| `psm_rhc` | PS 最近傍マッチ |
| `iptw_nhefs` | 安定化 IPTW（手計算重み） |
| `lmm_pbcseq` | ガウス LMM / クラスタ SE / GAM |
| `cea_sicksicker` | `statract.cea`（DARTH Sick-Sicker 教学パラメータ。患者 tableone / flowchart なし） |

## チェックリスト（新規・改訂時）

- [ ] docs が 7 節構成で、独立のライブラリコード節が無い
- [ ] 結果の図・表に Material タブと実 API コードがある
- [ ] GLM/Cox 系 forest が `layout="table"`（スクリプトと docs）。論文用白黒は `style="bw"`
- [ ] 生存例は NAR（軸揃え）・PH・診断スイートが揃っている（型に応じて）
- [ ] 除外がある例だけ flowchart + Mermaid（適用可なら）を docs + `*_out/` に持つ。除外ゼロなら flowchart を置かない
- [ ] `scripts/sync_example_assets.py` 経由で assets が `*_out/` と同一
- [ ] 生 CSV / 完全 `*_out/` を commit していない
- [ ] ギャラリーカードの主リンクが docs ページになっている
