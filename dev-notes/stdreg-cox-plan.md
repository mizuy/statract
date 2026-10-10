# Cox 回帰標準化の実装計画

`stdReg2::standardize_coxph` の標準化生存関数と制限付き平均生存時間（RMST）を `statract` に足す。数値は R と同じにする。実行時に R は起動しない。R は fixture の再生成だけに使う。

合わせる版は stdReg2 1.0.8 の `R/coxph_methods.R` と `R/utils.R` である。生存関数の分散は Sjölander (2016) のサンドイッチ、RMST は Chen and Tsiatis (2001) である。

jweb の手術比較が使うのは生存関数と、参照 0 との差だけである。

```r
standardize_coxph(
  formula, data,
  values = list(ope = c(0, 1)),
  times = seq(min_event_time_ceil, 8, step),
  contrasts = "difference", reference = 0
)
```

時点 0 の生存 1 は jweb 側で足している。

## 0. 方針

1. 段階を分ける。段階 1 で jweb が要るものを出し、そこで jweb を切り替えられるようにする。RMST は段階 3 で、別の PR にする。
2. R の挙動に合わせる。R の癖のうち、合わせると誤りになるものは合わせず、`docs/models/vs-r.md` に書く。どれがそうかは下の各節に挙げる。
3. R で検証できないものは入れない。生存関数のケースウェイトはこれに当たる。`standardize_coxph` には weights 引数が無く、中の `coxph` にも渡していないので、R では常にウェイト 1 である。
4. 既存の `cox_ph` を使う。点推定、スコア残差、情報行列は既存のものを使い、新しく書くのは基底ハザードのサンドイッチと RMST の分散だけにする。
5. 図は作らない。`standardize_glm`、GEE、`standardize_parfrailty` も入れない。

## 1. 呼び出し

推定と表示を分ける。R の `standardize_coxph` と `tidy()` の関係に合わせ、`cox_ph` の `.tidy()` とも形をそろえる。

```python
from statract.surv import standardize_cox

std = standardize_cox(
    data,
    "Surv(time, status) ~ ope * age + sex",
    values={"ope": [0, 1]},
    times=[1.0, 3.0, 5.0],
)
std.tidy()                                          # 曲線
std.tidy(contrast="difference", reference=0)        # 差
std.covariance(3.0)                                 # 水準間の共分散行列

rmst = standardize_cox(data, "Surv(time, status) ~ ope + age", values={"ope": [0, 1]}, times=5.0, measure="rmean")
```

`standardize_cox(data, formula, *, values, times, measure="survival", cluster=None)`

| 引数 | 内容 |
|------|------|
| `data` | Polars の `DataFrame`。式が使う列（と `cluster`）に欠損がある行は落とす |
| `formula` | `Surv(time, status) ~ ...`。既存の Wilkinson 式 |
| `values` | 曝露名 1 つから水準の列への辞書 |
| `times` | 生存関数は評価時点の列。RMST は制限時刻 1 つ |
| `measure` | `"survival"` または `"rmean"` |
| `cluster` | 生存関数だけ。クラスタ列の名前 |

`.tidy(*, contrast=None, reference=None, transform=None, ci="plain", level=0.95)` は Polars の表を返す。列は `time`、曝露、`estimate`、`std_error`、`conf_low`、`conf_high`。`contrast` を渡したら `reference` が要る。順番は R と同じで、変換をかけてから対比を取る。

両 measure で次を拒否する。

- 式の `strata()`、`cluster()`、`tt()`、`frailty`
- `Surv(start, stop, status)`
- 曝露が 2 つ以上
- `ci="log"` と `contrast="difference"` の組み合わせ。参照行の推定値が 0 になり、R では区間が NaN になるため

曝露の水準を置き換えるときは、`data` の曝露列を `values` の値に置き換え、`build_design(new, fit.design)` で設計行列を組み直す。交互作用、因子、スプラインはこれで扱える。`pl.Enum` の曝露は dtype を保つ。

## 2. 生存関数

### 点推定

`cox_ph(data, formula, ties="breslow", cluster=None)` を当てる。曝露を \(x\) に置き換えた行列で `predict(kind="survival", times=[t])` を出し、行で平均する。

\[
\hat\theta(t,x)=\frac1n\sum_i \hat S(t\mid X=x,Z_i).
\]

R は中心化した基底ハザードと中心化した risk の積を使う。中心化しない側の積と同じ値なので、点推定には既存の `predict` が使える。

最初の評価時点以前にモデルの事象時刻が 1 つも無いときは、R と同じく `"No events before first value in times"` で止める。時点 0 を入れても、最初の事象時刻が 0 より後なら止まる。

### 分散

`utils.R` の `sandwich()`（Cox 分岐）と、`standardize_coxph` の時点ループを写す。ここだけは中心化が要る。\(m\) を共変量の標本平均（`fit$means`）とする。

- 係数の肉は `score_contributions()`、パンは \(-\mathrm{vcov}^{-1}/n\)。
- 基底ハザードの肉 `UH` と微分 `IH` は R の式どおりに作る。R は `expand(dH / nevent, data[, t2])` で、時刻の値で行に割り当てる。このため、事象時刻と同じ時刻で打ち切られた行にも \(dH/\text{nevent}\) が入る。これは R に合わせる。時刻が整数に丸めてあるデータでは同順位が多く、ここがずれると数値が合わないからである。
- `tempmat` は \(m\) で中心化した設計行列を使う。中心化するのは観測した \(X\) の平均で、置き換えた \(x\) の平均ではない。
- 残差 \(\hat S_i-\hat\theta\) と係数・基底ハザードの肉を横に結び、`var`（分母 \(n-1\)）で \(J\) を作る。クラスタがあるときは先にクラスタで合計し、最後に \(n_\text{cluster}/n^2\) を掛ける。クラスタが無いときは \(1/n\)。
- 時点 0 は推定値 1、分散 0。

### 表示

`summary_std_coxph` を写す。変換 `log`、`logit`、`odds` のヤコビアンを掛け、次に対比を取る。`difference` は参照を引き、`ratio` は参照で割る。参照行の分散は 0 にする。`ci="plain"` は \(\hat\theta\pm z\,\mathrm{se}\)、`ci="log"` は \(\hat\theta\exp(\pm z\,\mathrm{se}/\hat\theta)\)。

## 3. RMST

R の手順は次のとおりである。曝露は数値の 0/1。式から曝露の項を外し、群 0 と群 1 に別々の Cox（Efron）を当てる。プールした全員について、事象時刻 \(t_k\le t^\star\) ごとの期待事象数から \(\bar S_a(t_k)\) を出し、矩形積分（`rsum`）で RMST にする。分散は R の 3 項（係数の影響、基底ハザード、個人別 RMST の標本分散）と群間共分散である。人ごとのループはベクトル化する。

R と変えるところは次の 3 つで、どれも vs-r.md に書く。

1. **外す項。** R は `grep(曝露名, 項名)` で外すので、曝露が `ope` なら `operation` も消える。statract は、曝露の列を含む項だけを正確に外す。
2. **同順位。** R は `Ai <- data[[exp]][match(etimes, time)]` で、各事象時刻の群を「同じ時刻の最初の行」から取る。その行が打ち切りや別群でも拾う。statract は事象を起こした行そのものの群を使う。同じ時刻の事象が同じ群の中だけなら R と一致する。`etimes` の重複（事象 1 件ごとに 1 項）は R のまま残す。
3. **共変量なし。** 曝露を外すと共変量が 0 個になる式は拒否する。R はこの場合 `solve()` に 0×0 行列を渡して止まる（fixture 作成時に確認する）。

`times` が 2 個以上なら R は最大値を使って警告する。statract は `ValueError` にする。RMST は `cluster`、`logit`、`odds` を拒否する。

## 4. テスト

```
tests/r_oracle/
  scripts/stdreg_cox.R
  data/stdreg_*.csv
  fixtures/stdreg_cox.json
  test_stdreg_cox.py
```

データは `scripts/make_data.py` で作り、seed を固定する。pytest の実行時に R は要らない。

| 標本 | measure | 確かめること |
|------|---------|----------|
| 二値曝露、整数時刻 | survival | `ope * age` と因子。打ち切りと事象が同じ時刻に重なる。曲線、差、比。jweb の使い方に当たる |
| 連続曝露 | survival | 3 水準、交互作用、対比なし |
| クラスタ | survival | 1 列のクラスタ |
| 変換 | survival | `log`、`logit`、`odds` と `ci="log"` |
| 二値、連続時刻 | rmean | 共変量 1 つ、同順位なし、差と比 |
| 二値、交互作用あり | rmean | 式に `ope * age`。外したあと R と一致 |

比べるのは `estimate`、`std_error`、区間、水準間の共分散で、rtol 1e-6 である。R と変えたところ（項の外し方、同順位の群、共変量なし）は R と比べず、期待する挙動を単体テストで確かめる。生成スクリプトは stdReg2 と survival の版を fixture に書く。

## 5. 順序

| 段階 | 中身 | 受け入れ |
|------|------|----------|
| 1 | 生存関数の点推定とサンドイッチ、`.tidy()` の差と比（`ci="plain"`） | 二値・連続・クラスタの 3 標本が fixture と一致。jweb の呼び出しを置き換えられる |
| 2 | 変換と `ci="log"` | 変換の標本が一致 |
| 3 | RMST の点推定と分散 | rmean の 2 標本が一致。R と変えた 3 点に単体テスト |
| 4 | 文書 | `docs/models/vs-r.md`、`docs/api/models/surv.md`、`docs/models/overview.md` |

段階 1 と 2 は 1 つの PR、段階 3 は別の PR にする。置き場所は `src/statract/surv/standardize.py` で、`statract.surv` とパッケージの入口から出す。

## 6. 対象外

- ケースウェイト（R で検証できない）
- RMST のクラスタ、`logit`、`odds`、共変量なし
- 層別、時間依存共変量、左切り捨て、複数の曝露
- `plot.std_surv`、`standardize_glm`、GEE、`standardize_parfrailty`、`standardize_custom`

## 参考

- Sjölander A. Regression standardization with the R-package stdReg. *Eur J Epidemiol.* 2016;31(6):563-574.
- Chen PY, Tsiatis AA. Causal inference on the difference of the restricted mean lifetime between two groups. *Biometrics.* 2001;57(4):1030-1038.
- Sachs MC, Ohlendorff JS, Brand A, et al. The next generation of regression standardization with the R package stdReg2. *Ann Epidemiol.* 2025;105:66-74.
- `dev-notes/r-parity-plan.md`

## 実装メモ

段階 1〜3 を `src/statract/surv/standardize.py` に実装した。fixture は 7 例で、推定値、共分散、表がすべて rtol 1e-6 で一致する。実装中に見つかった R の癖（文字列のクラスタで止まる、`logit` と `odds` の変換で止まる、RMST の `values` の順）は `docs/models/vs-r.md` の「stdReg2 と違うところ」にまとめた。
