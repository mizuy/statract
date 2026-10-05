# Legacy: Stanford Heart Transplant（ノートブック）

**レガシー。** ワークフロー正本（Taskfile、concept / protocol / results、`{stem}.py` → `{stem}_out/`）は [`examples/surv_colon/`](../surv_colon/) ほか3本を見る。本ディレクトリは既存の Stanford Heart（n=172、counting process）と jupyter を残したもの。削除していない。

**docs 単一ページ:** [`docs/stat/examples/legacy_heart.md`](../../docs/stat/examples/legacy_heart.md)。

このプロジェクトは、`endolab`ライブラリを使用した生存解析の例です。

## 概要

- **データソース**: lifelinesライブラリのStanford Heart Transplantデータセット
- **解析内容**:
  1. データの前処理と型変換（snapshot_cacheで自動的に固定化）
  2. Table Oneによるbaseline characteristicsの表示（Python）
  3. 生存解析

## ワークフロー

新しいワークフローでは、データの前処理と分析を分離しています:

1. **build.py**: データを前処理し、`cache/build/`にparquetファイルとして保存
2. **r.ipynb**: `load_parquet_dir`を使用してデータをロードし、分析を実行

```python
# build.py でデータを保存
python build.py

# r.ipynb でデータをロード
from endolab import load_parquet_dir
data = load_parquet_dir("cache/build")
target = data["target"]
```

## プロジェクト構造

```
examples/legacy_heart/
├── Makefile            # ビルドと実行のワークフロー
├── pyproject.toml      # 依存関係の定義
├── README.md           # このファイル
├── project.py          # ProjectPathインスタンスの定義
├── config.py           # パス定数とヘルパー関数の定義
├── build.py            # データの取得と前処理、cache/buildに保存
├── main.ipynb          # Python分析ノートブック（Table One）
├── r.ipynb             # 分析ノートブック（load_parquet_dirでデータロード）
├── snapshot/           # 固定化されたデータ（gitignore推奨）
├── cache/              # キャッシュファイル（gitignore推奨）
│   └── build/          # 前処理済みデータ（parquetファイル）
└── log/                # 実行ログ（gitignore推奨）
```

## セットアップ

1. リポジトリルートで `uv sync`（workspace member）。

2. このディレクトリの依存関係をインストール:
   ```bash
   cd examples/legacy_heart
   uv sync
   ```

## 実行方法

### 全ワークフローの実行

```bash
make all
```

または

```bash
make build analysis
```

これは以下を順に実行します:
1. `make build`: データの取得と前処理（snapshot_cacheで自動的に固定化）、キャッシュ保存
2. `make analysis`: Python分析ノートブックとR分析ノートブックの実行

### 個別実行

```bash
# buildのみ実行（データの取得と前処理、キャッシュ保存）
make build

# PythonとRの分析を実行
make analysis

# 解析結果をHTMLに変換して保存
make save

# キャッシュをクリア
make clear
```

## 出力

分析結果は`log/`ディレクトリに保存されます:

### 分析結果（`log/analysis_YYYYMMDD_HHMMSS/`）
- `table1_baseline.csv`: Table One（baseline characteristics）
- `coxph_model1.csv`: Cox回帰モデル1の結果
- `coxph_model2.csv`: Cox回帰モデル2の結果

`make save`を実行すると、ノートブックのHTML版も生成されます:
- `main.html`: Python分析ノートブックのHTML版
- `main_r.html`: R分析ノートブックのHTML版

## データセットについて

Stanford Heart Transplantデータセットは、心臓移植患者の生存データです。

- **サイズ**: 172行、8列
- **特徴**: time-varying covariates（時間依存共変量）を含む
- **カラム**:
  - `start`: 開始時間
  - `stop`: 終了時間
  - `event`: イベント（1=死亡、0=打ち切り）
  - `age`: 年齢（正規化済み）
  - `year`: 年（正規化済み）
  - `surgery`: 手術（0/1）
  - `transplant`: 移植（0/1）
  - `id`: 患者ID

## 参考

- [ANALYSIS_WORKFLOW.md](../ANALYSIS_WORKFLOW.md): 解析ワークフローと文書標準（正本）
- [lifelines documentation](https://lifelines.readthedocs.io/): 生存解析ライブラリ

