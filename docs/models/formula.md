# Wilkinson 式

`fit_ols`、`fit_glm`、`fit_mixed`、`model_matrix` が同じ文字列を受けます。展開は R の `terms` と `model.matrix`（treatment contrast）に合わせ、列名と行列は `tests/r_oracle/fixtures/wilkinson.json` で固定しています。主効果だけのモデルは、列名のリストでも呼べます。

```python
from statract import fit_ols, model_matrix

fit_ols(data, "y ~ x * stage")
built = model_matrix("y ~ x * stage", data)
built.design.names  # 列名
built.y              # 応答
built.offset         # offset() の和。無ければ None
```

式を渡したときは `predictors`、`groups`、`slopes` を省きます。呼び出し例は [Models](overview.md)、R との対応は [R パッケージとの対応](vs-r.md) です。

## 対応一覧

基準は formulaic の [Formula Grammar](https://matthew.wardrop.casa/formulaic/latest/guides/grammar/) にある演算子と関数です。展開の意味は R の `terms` と `model.matrix` に合わせています。

### 演算子

| 記法 | 対応 | 内容 |
|------|------|------|
| `` `列名` `` | 対応 | 空白や記号を含む列名 |
| `(...)` | 対応 | 優先順位を変える |
| `.` | 対応 | 左辺に書いた列を除く、データフレームの残りの列 |
| `^` | 対応 | その次数までの交互作用。指数は 2 以上の整数。`x^2` は項 `x` |
| `:` | 対応 | 交互作用。数値で列を拡縮することはしない |
| `*` | 対応 | 左右の主効果と交互作用 |
| `/` | 対応 | ネスト。`a / b` は `a + b %in% a` |
| `%in%` | 対応 | 積だけ。`b %in% a` に `a` の主効果は入らない |
| `+` `-` | 対応 | 項を加える、項を除く |
| 単項 `+` `-` | 対応 | R の規則。`-1` は切片を外し、`-0` は切片を入れる |
| `~` | 対応 | 1 つ。左辺が必要 |
| `0` `1` | 対応 | 切片の切り替え。それ以外の数値リテラルはエラー |
| `"..."` | 非対応 | 文字列リテラル |
| `{...}` | 非対応 | Python 式の埋め込み |
| 任意の関数呼び出し | 非対応 | 下の表にある関数だけ |
| `**` | 非対応 | 同じ役割は `^` |
| `2.5:x` のような数値 | 非対応 | 計算は `I()` の中 |
| `\|` で設計行列を分ける | 非対応 | Formula パッケージの複数行列。`(1 \| g)` は下の変量効果 |
| `[. ~ .]` | 非対応 | 多段式 |
| `~` の省略 | 非対応 | 右辺だけの式 |

### 関数

| 関数 | 対応 | 内容 |
|------|------|------|
| `I(...)` | 対応 | 中の `+` `-` `*` `/` `^` は数値演算。引数は 1 つ。列名は `I(x^2)` のように R の deparse |
| `log` `log10` `exp` `sqrt` `abs` | 対応 | 引数は 1 つ。左辺でも右辺でも使える |
| `offset(...)` | 対応 | 設計行列の列にはせず、`FormulaModel.offset` に足す。formulaic はこの関数を持たない。`fit_mixed` は受けない |
| `Surv(time, status)` `Surv(start, stop, status)` | 対応 | `cox_ph`、`accelerated_failure`、`fine_gray` の左辺。`Surv(start, stop, status)` は `cox_ph` と `fine_gray` |
| `strata(...)` | 対応 | `cox_ph` と `conditional_logit` の層。設計行列の列にはしない |
| `cluster(...)` | 対応 | `cox_ph` のクラスター頑健分散。`conditional_logit` は `method="efron"` と `method="breslow"` で使う。exact では使えない |
| `(1 \| g)` `(1 + x \| g)` `(0 + x \| g)` | 対応 | lme4 の変量効果。formulaic の表には無い。推定は `fit_mixed`。グループは 1 つ。傾きは数値 1 列 |
| `Q(...)` | 非対応 | 列名はバッククォート |
| `C(...)` | 非対応 | 因子にするのは列の型。対比の種類は指定しない |
| `center` `scale` `standardize` | 非対応 | |
| `lag` | 非対応 | |
| `poly` | 非対応 | 二乗は `I(x^2)` |
| `bs` `cs` `cr` `cc` `te` | 非対応 | 平滑は式に書かず `smooth` |
| `hashed` | 非対応 | |
| `log2` `exp2` `exp10` | 非対応 | |
| `np` と、呼び出し側の関数 | 非対応 | 名前空間を式へ渡す口は無い |
| 引数のカンマ | 一部 | `Surv(time, status)` と `strata(a, b)` の区切り。`poly(x, 2)` や `log(x, 10)` は非対応 |
| `\|\|` | 非対応 | `(x \|\| g)` は読める。`fit_mixed` は無相関の傾きとしてエラーにする |
| `s` | 非対応 | 平滑は `smooth` |

### formulaic と違う動き

次の 3 点は R に合わせています。formulaic は項をアルファベット順に並べ、`(x - 1)` に切片を残し、patsy の方法で行列の階数を落とします。

- 項は次数のあと、式に現れた順です。
- `(x - 1)` も `x - 1` も切片がありません。
- 対比は treatment contrast の局所規則です。

左辺の `+` は数値の加算です。`x + z ~ y` の応答は `x + z` の 1 列で、複数の応答列にはなりません。

## 左辺と右辺

形は `応答 ~ 右辺` です。`~` は 1 つです。左辺は列名、上の表の関数、またはそれらの数値演算です。`log(y) ~ x` の応答は `log(y)` です。

右辺の演算子は、優先度の高いものから次のとおりです。

| 優先 | 演算子 | 展開 |
|------|--------|------|
| 1 | `^` | 同じ項集合の積を繰り返す。指数は 2 以上の整数 |
| 2 | `:` | 左右の項を組にして、変数の和集合にする |
| 3 | `%in%` | 右の項をすべて合わせ、その結果を左の各項に掛ける |
| 4 | `*` `/` | `*` は左、右、左右の積。`/` は左と、左全体を右の各項に掛けた項 |
| 5 | `+` `-` | 項を加える。`-` は右の項を左から除く |

`a * b:c` は `a * (b:c)` です。空白は無視します。識別子に使えない列名は `` `列名` `` で囲みます。

## 演算子が作る列

水準は `stage` が I, II, III、`arm` が A, B です。文字列なのでソート順で、参照は I と A です。

| 式 | 列 |
|----|----|
| `y ~ x + stage` | `(Intercept)`, `x`, `stageII`, `stageIII` |
| `y ~ x * stage` | 上に `x:stageII`, `x:stageIII` |
| `y ~ x:stage` | `(Intercept)`, `x:stageI`, `x:stageII`, `x:stageIII` |
| `y ~ stage:x` | `(Intercept)`, `stageI:x`, `stageII:x`, `stageIII:x` |
| `y ~ stage/arm` | `(Intercept)`, `stageII`, `stageIII`, `stageI:armB`, `stageII:armB`, `stageIII:armB` |
| `y ~ stage + arm %in% stage` | `y ~ stage/arm` と同じ列 |
| `y ~ arm %in% stage` | `(Intercept)` と `armA:stageI` から `armB:stageIII` まで。主効果は入らない |
| `y ~ (x + stage + arm)^2` | 主効果と二次の交互作用。三次は入らない |
| `y ~ x^2` | `(Intercept)`, `x` |
| `y ~ I(x^2)` | `(Intercept)`, `I(x^2)` |

`*` は主効果を残します。`:` と `%in%` は積だけです。`a/b` は `a + b %in% a` です。`stage + arm:stage` は、変数を式の左から登録するので、積の名前は `stageI:armB` になります。`arm:stage` は `arm` が先なので、名前は `armA:stageI` の順です。

`x^2` は項 `x` です。二乗の列は `I(x^2)` です。`(x + stage)^2` は `x * stage` と同じ項です。指数が 1 以下、または整数でないときはエラーです。

同じ項が二度出ると、先の項を残します。`y ~ x * x` の列は `(Intercept)`, `x` です。

## 切片

既定では先頭が `(Intercept)` です。`0` と `1` と `-` が切片を切り替えます。

| 式 | 列 |
|----|----|
| `y ~ 1` | `(Intercept)` |
| `y ~ x - 1` | `x` |
| `y ~ 0 + x` | `x` |
| `y ~ -1 + x + stage` | `x`, `stageI`, `stageII`, `stageIII` |
| `y ~ (x - 1)` | `x` |
| `y ~ x - 0` | `(Intercept)`, `x` |
| `y ~ x * 0` | `x` |
| `y ~ x + z - (z - 1)` | `(Intercept)`, `x` |

`y ~ 0` と `y ~ -1` は列が空なので、変量効果が無いときはエラーです。先頭の `-stage` は、まだ入っていない `stage` を除く操作です。その後の `.` が `stage` を足すので、`y ~ -stage + .` には `stageII` が残ります。

## 項の順序と対比

展開のあと、項を次数（含まれる名前の数）で安定ソートします。次数が同じ項は、式に現れた順です。項の中の名前は、式の左から最初に現れた順です。交互作用の列名は `:` でつなぎます。直積では、その項の最初の因子が最も速く動きます。

因子は treatment contrast です。主効果は最初の水準を落とします。ダミー名は列名と水準を連結します。切片の名前は `(Intercept)` です。

ある変数を除いた残りが、前にある項に含まれないとき、その変数は全水準を残します。

- `y ~ x:stage` は `stage` の主効果が前に無いので、`x:stageI` から全水準です。
- `y ~ x:stage + stage` は次数順で `stage` が先です。列は `(Intercept)`, `stageII`, `stageIII`, `x:stageI`, `x:stageII`, `x:stageIII` です。

切片が無いときは、応答の次に登録された変数から見て、最初の因子を全水準にします。変える因子は 1 つです。

| 式 | 列 |
|----|----|
| `y ~ x + stage - 1` | `x`, `stageI`, `stageII`, `stageIII` |
| `y ~ stage - 1 + x` | `stageI`, `stageII`, `stageIII`, `x` |
| `y ~ stage + arm - 1` | `stageI`, `stageII`, `stageIII`, `armB` |
| `y ~ (stage - 1) * x` | `stageI`, `stageII`, `stageIII`, `x`, `stageII:x`, `stageIII:x` |
| `y ~ b - 1` | `bFALSE`, `bTRUE` |

## 列の型

| 型 | 式の中での扱い |
|----|----------------|
| 浮動小数点、整数 | 数値 1 列 |
| 文字列 | 欠損を落としたあとの観測値をソートした水準。先頭が参照 |
| `pl.Enum`、`pl.Categorical` | 定義順。先頭が参照 |
| 論理値 | 水準 `FALSE`, `TRUE`。`y ~ b + x` のダミーは `bTRUE` |

水準が 2 未満の因子はエラーです。列名リストの `design_matrix` では、論理値と整数は 0/1 の数値です。式では論理値を因子にします。

## 関数と offset

右辺と左辺で使える関数は `I`、`log`、`exp`、`sqrt`、`log10`、`abs` です。それぞれ数値 1 列で、引数は 1 つです。列名は R の deparse に合わせます。

| 式の中の書き方 | 列名 |
|----------------|------|
| `I(x^2)` | `I(x^2)` |
| `I(x * z)` | `I(x * z)` |
| `I(x/z)` | `I(x/z)` |
| `I(x + z)` | `I(x + z)` |
| `I((x + z)^2)` | `I((x + z)^2)` |
| `log(w + x)` | `log(w + x)` |
| `I(-x)` | `I(-x)` |

`offset()` の値は、引数を足したベクトルとして `FormulaModel.offset` に入ります。設計行列にはその列を作りません。`y ~ offset(z) + x` と `y ~ x * offset(z)` の列はどちらも `(Intercept)`, `x` で、offset は `z` です。`offset(z)` はドットから `z` を除きません。

`fit_ols` と `fit_glm` は、式の `offset()` と引数 `offset=` を足します。`fit_mixed` は `offset()` を受けません。当てはめた行への `predict()` は、線形予測子に offset を足します。新しいデータへの `predict(data)` は、設計行列と係数の積です。

## ドット

`.` は、左辺に名前として出た列を除く、データフレームの残りの列です。並びはデータフレームの列順です。すでに右辺へ入った名前は飛ばします。`log(y) ~ .` は `y` を除きます。`y ~ x + .` は `x` を一度だけ残します。

右辺に応答と同じ主効果があると、その項を落として `UserWarning` を出します。

## 欠損と新しいデータ

式が使う列に欠損がある行は、まとめて落とします。水準は残った行から決めます。`weights=` と `offset=` の欠損も、そのあとで同じ行を落とします。

`predict` に新しいデータを渡すと、学習時のレシピで列を作り直します。学習時に無かった水準はエラーです。

## 変量効果

`(1 | g)` は lme4 の書き方で、`model_matrix` が `random_effects` に入れます。`fit_ols`、`fit_glm`、`formula_model_matrix` は変量効果があるとエラーにします。推定は `fit_mixed` です。

変量効果は `+` で他の項とつなぎます。`*`、`:`、`/`、`^`、`%in%`、引き算と組み合わせるとエラーです。傾きは列 1 本で、数値です。グループの列も 1 つです。`fit_mixed` が受け取るグループは 1 つです。

- `y ~ x + (1 | g)` は、固定効果が `x` と切片、変量効果が切片です。
- `y ~ x + (1 + x | g)` は、変量切片と `x` の変量傾きです。相関があります。`method="ml"` で最尤にします。
- `y ~ (x | g)` は、固定効果が切片、変量効果が切片と `x` の傾きです。
- `y ~ 0 + (1 | g)` は、固定効果が空、変量効果が切片です。
- `y ~ 0 + x + (0 + x | g)` は、固定効果が `x`、変量効果が `x` の傾きです。

`(x || g)` は解析できます。`fit_mixed` は無相関の傾きとしてエラーにします。因子や論理値の傾き、`offset()` もエラーです。

`cox_ph`、`accelerated_failure`、`fine_gray` は `Surv(time, status) ~ age + sex` を受けます。Cox は切片の列を外し、因子は treatment contrast のままです。`strata(site)` と `cluster(id)` は Cox の式に書けます。`conditional_logit` は `bleed ~ eolm + strata(set)` を受け、切片を外します。既定の `method="exact"` はマッチセット内の組合せ尤度です。`Surv(start, stop, status)` は counting process で、`cox_ph` と `fine_gray` が受けます。`fine_gray` の右辺は残す共変量で、`cause` は別引数です。加速故障時間は切片を残します。平滑は `smooth` です。列名での呼び出しも残します。

## 戻り値

`model_matrix(formula, data)` は `FormulaModel` を返します。

| 属性 | 内容 |
|------|------|
| `design` | 固定効果の `Design`。`names` が列名、`x` が行列、`row_index` が残った行 |
| `y` | 左辺を評価したベクトル |
| `offset` | `offset()` の和。無ければ `None` |
| `random_effects` | `(傾き \| グループ)` の列。各要素は `intercept`、`slopes`、`group`、`correlated` |
| `response_name` | 左辺が列名のときの名前。関数のときは `None` |

変量効果は `fit_mixed` に渡します（ガウスも二項 `family="binomial"` も同じ式）。固定効果だけなら `fit_glm` を使ってください。実験的な `glmm_gpboost` は optional extra で、同じ `model_matrix` 展開を使います。
