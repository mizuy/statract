# Wilkinson 式の項展開

## 1. 目的

右辺の式を「項」の順序つき列に展開する。項は変数ラベルの集合。各項の各因子を対比（treatment contrast）で符号化するか、全水準の指示変数で符号化するかも決める。出力を比べる相手は R の公開関数 `terms()`、`model.matrix()`。
実装は `src/statract/models/formula.py` の `_encode`、`_delete`、`_trim`、`_by_degree`、`_install`、`_contrast_codes`。

## 2. 出典

- Wilkinson, G. N. & Rogers, C. E. (1973). Symbolic description of factorial models for analysis of variance. *Applied Statistics* 22(3):392–399.
- Chambers, J. M. & Hastie, T. J. (eds.) (1992). *Statistical Models in S*. Wadsworth. 第 2 章（2.3 節 式の演算子、2.4.1 節 因子の符号化と周辺性）。
- R の公開ヘルプ `?formula`、`?terms.object`（`factors` 属性の 0/1/2 の意味）。

## 3. モデルと式

変数ラベルの集合を $V$ とする。項は $T \subseteq V$。式は項の集合 $\mathcal{M}$ と、定数項（切片）の有無 $c \in \{0,1\}$ の組。空集合の項 $\emptyset$ が定数項にあたる。

演算子（Wilkinson & Rogers 1973; Chambers & Hastie 1992, 2.3 節）:

- 和: $A + B = A \cup B$
- 積（交互作用）: $A : B = \{\, S \cup T : S \in A,\ T \in B \,\}$
- 交差: $A * B = A + B + A{:}B$
- 入れ子: $A / B = A + \{\, \mathrm{vars}(A) \cup T : T \in B \,\}$。$\mathrm{vars}(A) = \bigcup_{S\in A} S$
- `%in%`: $B \mathbin{\%in\%} A = \{\, T \cup \mathrm{vars}(A) : T \in B \,\}$
- 冪: $A^k = A * A * \cdots * A$（$k$ 個）。$k$ 個以下の項の積すべて。次数の上限ではない
- 差: $A - B = A \setminus B$
- 数値 `1` は $c=1$、`0` は $c=0$。差の右辺の中では意味が反転する

符号化（Chambers & Hastie 1992, 2.4.1 節）: 項 $T$ の変数 $v$ について、周辺 $T \setminus \{v\}$ が

$$ T\setminus\{v\} = \emptyset \quad\text{または}\quad \exists\, T' \text{（$T$ より前の項）}: T\setminus\{v\} \subseteq T' $$

を満たすとき、$v$ は対比（最初の水準を落とす、コード 1）。満たさないとき全水準（コード 2）。周辺がすでに張られていれば対比で足り、張られていなければ全水準が要る、という周辺性の考え方。

## 4. 推定・計算の手順

1. 構文木を再帰的にたどる。列名は変数として登録し（初出順）、単一の項を返す。`.` は応答以外のデータ列すべて。
2. `+`、`:`、`*`、`/`、`%in%`、`^` は 3 節の集合演算。結果から空の項と重複を除き、最初の出現を残す。
3. `:` の積は左側の項が速く動く順に並べる。`(a+b):(c+d)` は `a:c, b:c, a:d, b:d`。
4. `^k` は「これまでの結果」と「元の項」の積を $k-1$ 回くり返す。`(a+b+c)^2` は `a, b, c, a:b, a:c, b:c`。
5. 差 `A - B` は左を通常に、右を「引かれる側」で評価する。引かれる側ではフラグを反転し、`1` は切片を外し `0` は戻す。入れ子の差では二重に反転する。`y ~ x + z - (z - 1)` は切片あり。
6. 関数呼び出し: `log` `exp` `sqrt` `log10` `abs` `I` は 1 引数の計算列。`offset()` は項を返すが、後で設計行列から除き、和を offset に回す。`ns` `bs` `rcs` はスプライン基底列。`strata()`、`cluster()` は項を作らず状態に記録する。変量効果 `(… | g)` も項を作らない。
7. 全体を次数（変数の数）で安定ソートする。同じ次数は式に現れた順。項の中の変数の並びは登録順。
8. 各項の各変数に 3 節の規則でコードを付ける。切片が無いときは、変数の登録順で最初の因子を持つ最初の項について、その因子をコード 2 にする（定数項の代わりに全水準の列が定数を張る）。変える因子は 1 つ。

## 5. 数値上の注意

- 数値リテラルは 0 と 1 だけ。それ以外はエラー（計算は `I()` の中）。
- `^` の指数は 2 以上の整数。`x^2` は項 `x`（同じ項の積は同じ項）。
- `x * x` は項 `x` だけ。
- 項が空で切片も無く、変量効果も Surv 応答も無いときはエラー。
- 先頭の `-stage` はまだ無い項を除くだけで、何もしない。

## 6. R と出力を合わせるための公開仕様

- `?formula`: `-1` と `+0` で切片を外す。`a/b` は `a + b %in% a`。`(a+b+c)^2` は二次までの交互作用。
- `?terms.object`: `factors` 属性の 1 は対比、2 は全水準のダミー。項は次数順。
- 列名は変数名と水準名の連結、交互作用は `:` でつなぐ。直積では項の最初の因子が最も速く動く。

## 7. 検証

- `tests/models/test_estimators.py`: `test_formula_model_matrix_uses_treatment_contrasts`、`test_formula_model_matrix_drops_null_rows`（`x * x`）、`test_wilkinson_formula_matches_the_column_interface`、`test_fit_mixed_accepts_a_wilkinson_formula`、`test_survival_accepts_a_surv_formula`。
- `tests/r_oracle/test_splines.py::test_model_matrix`（R の列名と並び、rtol 1e-10）。
- 式を使うテスト約 400 件が間接に通る。
- `docs/models/formula.md` の表（演算子、切片、項の順序と対比）にある式は手で確かめて一致。

## 8. R と違うところ、未確認の点

- 「前の項に含まれる」を部分集合で判定している。R が完全一致で判定するなら、`a:b + a:c` のような式で符号化が変わりうる。未確認。
- 切片なしで最初に因子を含む項が交互作用（例 `0 + x + x:f`）のとき、その因子を全水準にする。R と同じかは未確認。
- `(a+b)*(c+d)` の交互作用の順（左が速い）は記憶による。fixture では未検証。
- `docs/models/formula.md` が挙げる `tests/r_oracle/fixtures/wilkinson.json` はリポジトリに無い。
