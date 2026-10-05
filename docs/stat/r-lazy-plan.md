# R 依存を遅延にする

`rpy2` が要るコードと要らないコードを分け、`endolab` を入れただけでは `rpy2` も埋め込み R も起動しないようにする。この文書は境界と受け入れ基準を固定する。以下は実装済みの境界である。

## 0. いま

実行時の import は、すでに呼び出しまで遅延している。

- `statract.r` は `run`、`assign`、`get`、`init`、`save_session`、`has_r_package`、`assign_dfs`、`assign_via_parquet` の中で `_require_rpy2()` を呼ぶ。モジュールを import しただけでは `rpy2` を読まない。
- `statract.__init__` は `r` を import しない。`import endolab` も `import statract` も `rpy2` を `sys.modules` に載せない。
- 推定器（`fit_ols`、`cox_ph`、`fit_mixed` など）、`statract.cea`、図、ローダは R を呼ばない。数値の照合は `tests/r_oracle` の JSON を読む。再生成は `Rscript` で、`rpy2` ではない。
- `glmm_forestplot` / `plot_forest` は matplotlib 実装で、R を import しない。

インストールも分かれている。

- `rpy2` と `rpy2-arrow` は `[project.optional-dependencies]` の extra `r` にある。`uv sync` だけでは入らない。ブリッジは `uv sync --extra r` または `pip install statract[r]` で入れる。extra が無いときに `run` や `assign` を呼ぶと `RNotAvailableError` になり、メッセージに `statract[r]` と動く R が要ることが入る。
- `tests/conftest.py` は `_rpy2_env` を import しない。`RPY2_CFFI_MODE` は `r.py` の `_require_rpy2()` が置く。
- `stat/r.py` は解析ワークフローが明示的に呼ぶブリッジである。オラクル生成は `Rscript` と JSON である。

システム R（`scripts/cloud-install-r.sh`、`Rscript tests/r_oracle/scripts/generate.R`、`bench/stat/run_r.R`）は Python パッケージの依存ではない。fixture とベンチマークを作り直すときだけ使う。

## 1. 境界

### rpy2 が要る

| 場所 | 役割 |
|------|------|
| `src/endolab/stat/r.py` | 唯一の呼び出し口。`_require_rpy2()` だけが `rpy2` と `rpy2_arrow.polars` を import する |
| `src/endolab/stat/_rpy2_env.py` | その import の直前に `RPY2_CFFI_MODE=ABI` を置く。他からは import しない |
| `tests/test_r_bridge.py` | ブリッジのテスト。`rpy2` が無ければ skip する |
| 解析ワークフロー | `from statract.r import assign, run, get, init, save_session` と書いたときだけ |

公開関数は今の名前のままにする。`statract` の `__all__` には入れない。

### rpy2 が要らない

- `import endolab` と `import statract`、推定器、生存、マッチング、GAM、混合、式、CEA、可視化、ローダ
- `tests/r_oracle`（JSON を読む pytest）。再生成用の `Rscript` は開発機のシステム R
- `bench/stat` の Python 側。R 側は `Rscript bench/stat/run_r.R`
- `glmm_forestplot` / `plot_forest`。matplotlib 実装。`r.py` を import しない

`rpy2` という文字列をソースに書いてよいのは `stat/r.py` と `stat/_rpy2_env.py` だけにする。テストは `tests/test_r_bridge.py` と、境界を固定する 1 本だけ。

## 2. インストール

`rpy2` と `rpy2-arrow` を `dependencies` から外し、optional extra に移す。

```toml
[project.optional-dependencies]
r = [
    "rpy2>=3.6.4",
    "rpy2-arrow>=0.1.3",
]
```

- 通常のインストールは `uv sync`。`rpy2` は入らない。
- ブリッジを使うときは `uv sync --extra r`、または `pip install statract[r]`。システムに R があることも必要である。
- `uv.lock` は gitignore されている。変えるのは `pyproject.toml` で、ロックの更新は環境側で行う。
- extra が無い状態で `run` や `assign` を呼ぶと、今の `RNotAvailableError` を出す。メッセージに `statract[r]` と、動く R が要ることを書く。

dev グループには入れない。ブリッジのテストは extra が無い環境では skip し、ある環境では今どおり走る。

## 3. import の固定

1. `tests/conftest.py` から `_rpy2_env` の import を外す。設定は `r.py` の `_require_rpy2()` がすでに行っている。
2. `statract.__init__` からは `r` を import しない。今どおり。
3. テストを 2 つ足す。
   - `import endolab` のあと `sys.modules` に `rpy2` が無い。続けて `import statract.r` してもまだ無い。`run` の直前まで遅延する、という既存の構造をこのテストで固定する。`run` 自体は呼ばない（R が無い環境で落ちるため）。
   - `src/endolab` を走査し、`rpy2` への import が `stat/r.py` と `stat/_rpy2_env.py` 以外に無いことを確認する。
4. `tests/test_r_bridge.py` の `pytest.importorskip("rpy2")` は残す。

## 4. 文書

- `docs/api/stat/r.md` の先頭に、`statract[r]` とシステム R が要ることを書く。
- `docs/stat/r-parity-plan.md` の `stat/r.py` の行を、オラクル生成ではなく解析ブリッジに直す。オラクルは `Rscript` のまま。
- `docs/cea/vs-r.md` の「本体が rpy2 を持つ」は、optional extra になったあとの文に変える。
- `ANALYSIS_WORKFLOW.md` と `.claude/skills/endolab-analysis-workflow/reference.md`、`.agents/skills/endolab-analysis-workflow/reference.md` の `from statract.r import ...` の前に、`statract[r]` が要ることを 1 行足す。

## 5. 受け入れ

- `rpy2` を入れていない環境で `import endolab` と `uv run python -m pytest tests/test_stat_estimators.py tests/r_oracle/test_parity.py` が通る。
- 同じ環境で `statract.r.run("1+1")` は `RNotAvailableError` になり、メッセージに `statract[r]` がある。
- `rpy2` と R がある環境では `tests/test_r_bridge.py` が今どおり通る。
- 推定器の数値と公開関数の引数は変えない。

## 6. 今回やらない

- `statract.r` の削除や、解析ワークフローから R を外すこと。
- `Rscript` による fixture 再生成と `bench/stat/run_r.R` を Python に置き換えること。システム R のインストール手順（`scripts/cloud-install-r.sh`）も残す。
- `scikit-learn` 経由で `import endolab` が pandas を読むこと。これは別件である。lifelines は実行時依存から削除済み。gpboost は optional extra で、`import statract` は読まない。
- （済）`plot_forest` / `glmm_forestplot` の matplotlib 実装。

## 7. 順序

1. 境界テスト（import とソース走査）を先に置き、今の遅延 import が壊れないことを見る。
2. `conftest.py` から `_rpy2_env` を外す。
3. `pyproject.toml` の extra へ移し、`RNotAvailableError` の文を変える。
4. 文書を §4 のとおり直す。
5. extra 無しで推定器テスト、extra 有りで `tests/test_r_bridge.py` を見る。
