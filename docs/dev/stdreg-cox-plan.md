# Cox 回帰標準化の実装計画

`stdReg2::standardize_coxph` の生存関数と制限付き平均生存時間（RMST）を `statract` に足す。数値の目標は R と同じ出力で、呼び出し方は Python 側の流儀にする。実行時に R は起動しない。オラクルは `Rscript` と JSON である。

合わせる版は [stdReg2 の `R/coxph_methods.R` と `R/utils.R`](https://github.com/sachsmc/stdReg2) である。生存関数の分散は Sjölander (2016) のサンドイッチで、共変量の経験分布のばらつきを含む。Gail–Byar の条件付き分散には合わせない。RMST は Chen and Tsiatis (2001) で、群ごとに別の Cox を当てる。

jweb の手術比較は生存関数だけを使っている。

```r
standardize_coxph(
  formula, data,
  values = list(ope = c(0, 1)),
  times = seq(min_event_time_ceil, 8, step),
  contrasts = "difference", reference = 0
)
```

`tidy()` の `contrast == "none"` が曲線、`difference` が参照 0 との差である。時点 0 の生存 1 は呼び出し側が足している。

## 0. 方針

1. 新しい計算は R を起動しない。fixture の再生成だけ `Rscript` を使う。
2. テストは複数標本で、推定値・標準誤差・区間を比べる。呼び出しの形は比べない。
3. 同順位は measure ごとに R の実装へ合わせる。生存関数は関数内で Breslow に固定する。RMST は `coxph` の既定である Efron である。
4. 公開名は `standardize_cox`。第 1 引数は Polars の `DataFrame`、式は `Surv(time, status) ~ x`、オプションはキーワード専用である。
5. 戻りは Polars の表である。`contrast` を渡したときは、stdReg2 の `format_result_standardize` と同じく `none` と指定した対比の両方を返す。
6. 図は作らない。`plot.std_surv` は再現しない。
7. `standardize_glm`、GEE、`standardize_parfrailty` は入れない。

## 1. 呼び出し

```python
from statract import standardize_cox

curve = standardize_cox(
    data,
    "Surv(time, status) ~ ope + age + sex",
    values={"ope": [0, 1]},
    times=[1.0, 3.0, 5.0],
    measure="survival",
    contrast="difference",
    reference=0,
)

rmst = standardize_cox(
    data,
    "Surv(time, status) ~ ope * age + sex",
    values={"ope": [0, 1]},
    times=[5.0],
    measure="rmean",
    contrast="difference",
    reference=0,
)
```

| 引数 | 内容 |
|------|------|
| `data` | 第 1 引数。欠損のある行は、その式が使う列についてまとめて落とす |
| 式 | `Surv(time, status) ~ ...`。演算子は既存の Wilkinson 式 |
| `values` | 曝露名から水準の列への辞書。生存関数は 1 列の曝露に複数水準。RMST は 0 と 1 の 2 水準 |
| `times` | 生存関数は評価時点の列。RMST は制限時刻。複数なら最大値だけを使い、表は 1 行 |
| `measure` | `"survival"` または `"rmean"` |
| `contrast` | `None`、`"difference"`、`"ratio"`。指定したときは `reference` が要る |
| `reference` | `values` にある水準 |
| `transform` | `None`、`"log"`、`"logit"`、`"odds"`。RMST は `"logit"` と `"odds"` を拒否する |
| `weights` | 生存関数のケースウェイト列。RMST は受けない |
| `cluster` | 生存関数のクラスタ列。残差を先に合計する。RMST は受けない |
| `level` | 区間の被覆確率。既定 0.95 |
| `ci` | `"plain"` または `"log"`。既定は `"plain"` |

表の列は `time`、曝露、`estimate`、`std_error`、`conf_low`、`conf_high`、`contrast`、`transform`、`measure` である。各時点の曝露水準間の共分散は結果オブジェクトが持つ。差と比の標準誤差はそこからデルタ法で作る。

カテゴリの参照水準は既存の Cox と同じで、`pl.Enum` の先頭、文字列はソート順である。曝露を置き換えたあとは、交互作用を含めて設計行列を組み直す。

両 measure で次を拒否する。

- 式の `strata()`、`cluster()`、`tt()`、`frailty`
- `Surv(start, stop, status)` の左切り捨て
- 曝露が 2 列以上

生存関数で、最初の評価時点以前に事象が 1 つも無いときは `"No events before first value in times"` と同じ条件で止める。時点 0 を格子に含めても、最小時点が最初の事象より前なら止まる。jweb が時点 0 を外で足しているのはこのためである。

## 2. 生存関数

1 本の Cox を `ties="breslow"` で当てる。時点 \(t\)、曝露 \(x\) の推定値は

\[
\hat\theta(t,x)=\sum_i \hat S(t\mid X=x,Z_i)\,w_i\Big/\sum_i w_i.
\]

\(\hat S=\exp\{-\hat\Lambda(t)\exp(h)\}\) である。\(\hat\Lambda\) は `coxph.detail$hazard` の累積で、共変量を標本平均に中心化した基底ハザードである。リスクは `predict(type="risk")` で、同じ中心化の線形予測の指数である。`statract` の `baseline_hazard` は `basehaz(centered=FALSE)` なので、この積には使わない。`H(t)` は事象時刻で右連続な階段関数（`stepfun`）である。

分散は `utils.R` の `sandwich` と、`standardize_coxph` の時点ループを写す。

- 係数の肉はウェイト付きスコア残差。パンは \(-\mathrm{vcov}(\hat\beta)^{-1}/n\)。
- 各時点の基底ハザードの肉 `UH` と、係数に対する微分 `IH` は `sandwich()` の Cox 分岐である。同順位は Breslow で、タイの中のウェイトは等しいとして `nevent` で割る。
- 標準化生存の残差は \(w_i(\hat S_i-\hat\theta)\)。これを係数と基底ハザードのスコアと横に結ぶ。
- \(J\) は残差行列の `var`（分母 \(n-1\)）。情報行列 \(I\) は、\(\theta\) のヤコビアン `SI` の下に Cox の情報 `oI` を積む。
- クラスタが無いとき \(V=(I^{-1} J I^{-\top}/n)\) の \(\theta\) ブロック。クラスタがあるときは残差をクラスタで合計してから `var` し、\(n_{\mathrm{cluster}}/n^2\) を掛ける。
- 時点 0 は推定値 1、分散 0。ループの前の「最初の時点以前に事象がある」判定は残す。

対比と変換は `summary_std_coxph` のデルタ法である。`difference` は参照を引く。`ratio` は参照で割る。参照行の分散は 0 にする。`log`、`logit`、`odds` は変換後のヤコビアンを掛ける。`ci="plain"` は \(\hat\theta\pm z\cdot\mathrm{se}\)。`ci="log"` は \(\hat\theta\exp(\pm z\cdot\mathrm{se}/\hat\theta)\) である。

## 3. RMST

曝露は数値の 0/1 で、水準も 0 と 1 だけである。式から、曝露名を含む項（主効果と交互作用）を外す。外した式で群 0 と群 1 に別々の `cox_ph` を当てる。同順位は Efron である。両モデルの完全ケースの和集合を、あとの平均に使う。

各群のモデルで、プールした全員について事象時刻 \(t_k\le t^\star\) の期待事象数を出す。これは `predict(type="expected", reference="zero")` で、中心化しない線形予測と、そのモデルの累積ハザードの積である。`statract` の `predict(kind="expected")` がこの積と一致する範囲を使う。群 \(a\) の生存は、その期待事象数の指数の標本平均 \(\bar S_a(t_k)\) である。RMST は時刻 0 で生存 1 から始め、事象時刻の階段を \(t^\star\) まで矩形積分する（`rsum`）。

\[
\hat\mu_a(t^\star)=\sum_k \bar S_a(t_{k-1})\,(t_k-t_{k-1}),\quad t_0=0,\; t_{K+1}=t^\star.
\]

分散は次の 3 項を群ごとに足し、\(n\) で割ったものである。

1. 共変量係数の影響。リスク集合内の共変量分散 \(\Sigma_a\) と、個人別の積分影響 \(g_{ai}\) から \(g_a=\Sigma_a^{-1}\sum_i g_{ai}/n_a\) を作り、\((n_a/n)\,g_a^\top\Sigma_a g_a\)。
2. ハザード増分。群 \(a\) の事象について \(h_a(t)^2 / (S^{(0)}(t)\,\sum_i Y_i(t)r_i)\) を足す。
3. 個人別 RMST の標本分散。\((n-1)/n\cdot\mathrm{var}_i(\int_0^{t^\star}\exp(-\hat\Lambda_{a,i}))\)。

群間共分散は、2 本の個人別 RMST の `cov` に \((n-1)/n^2\) を掛けたものである。差の分散は \(\mathrm{var}_1+\mathrm{var}_0-2\mathrm{cov}\) で、`summary_std_coxph` が共分散行列から作る。R の実装は人ごとに行列を組む。同じ式をベクトル化し、fixture の \(n\) は数百行に留める。

`times` が 2 個以上のときは最大値を \(t^\star\) にし、警告を出す。表の `time` はその 1 点である。

## 4. テスト

```
tests/r_oracle/
  scripts/stdreg_cox.R
  fixtures/stdreg_cox.json
  test_stdreg_cox.py
```

合成データ、seed 固定。pytest の実行時に R は要らない。

| 標本 | measure | 中身 |
|------|---------|------|
| 連続曝露 | survival | 数時点、対比なし、交互作用 |
| 二値曝露 | survival | `ope * age` と因子、`difference` と `ratio`、参照 0 |
| ウェイトとクラスタ | survival | 1 列クラスタ、ケースウェイト |
| 二値、共変量なし | rmean | 水準 0/1、1 時点、差 |
| 二値、交互作用を落とす | rmean | 元の式に曝露の交互作用、差と比 |

比べる量は `estimate`、`std_error`、区間である。許容差は rtol 1e-6、p 値を出す段は無い。生存関数の標本では、中の Breslow Cox の係数が `cox_ph(..., ties="breslow")` と一致することも見る。RMST の 2 本は Efron の `cox_ph` と一致することを見る。

再生成スクリプトは stdReg2 の版を fixture に書く。

## 5. 順序と受け入れ基準

| 順 | 成果物 | 受け入れ |
|----|--------|----------|
| 1 | 結果表、plain / log の区間、差と比のデルタ法 | 既知の \(\theta\) と共分散から `summary_std_coxph` の表と一致 |
| 2 | 生存関数の点推定とサンドイッチ | 3 標本の推定値と標準誤差が fixture と一致。中心化しない基底ハザードを混ぜない |
| 3 | RMST の点推定 | 2 本の Efron Cox、矩形積分が fixture の `estimate` と一致 |
| 4 | RMST の 3 項の分散と群間共分散 | 同じ 2 標本の標準誤差と差の標準誤差が一致 |
| 5 | 文書 | 下の 3 ファイル |

配置は `src/statract/surv/standardize.py`。`standardize_cox` を `statract.surv` とパッケージの公開入口から出す。

実装が終わったら次を更新する。

- `docs/models/vs-r.md` に 2 行。生存は Breslow とサンドイッチ、RMST は Efron と Chen–Tsiatis
- `docs/api/models/surv.md`
- `docs/models/overview.md` の生存の行

## 6. 対象外

- `measure` 以外の stdReg2（`standardize_glm`、GEE、`standardize_parfrailty`、`standardize_custom`）
- RMST のウェイト、クラスタ、`logit`、`odds`
- 層別、時間依存共変量、左切り捨て、複数の曝露列
- `plot.std_surv`

## 参考

- Sjölander A. Regression standardization with the R-package stdReg. *Eur J Epidemiol.* 2016;31(6):563-574.
- Chen PY, Tsiatis AA. Causal inference on the difference of the restricted mean lifetime between two groups. *Biometrics.* 2001;57(4):1030-1038.
- Sachs MC, Ohlendorff JS, Brand A, et al. The next generation of regression standardization with the R package stdReg2. *Ann Epidemiol.* 2025;105:66-74.
- `docs/dev/r-parity-plan.md`
