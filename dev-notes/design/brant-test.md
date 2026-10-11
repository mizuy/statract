# Brant の比例オッズ検定

対象: `src/statract/models/ordinal.py` の `brant_test` と補助関数 `_weighted_logistic`。

## 1. 目的

比例オッズモデル（proportional odds model、logit リンクの累積リンクモデル）の「傾きがどの切断点でも同じ」という仮定を、Wald 検定で調べる。全体（omnibus）と共変量ごとの統計量、自由度、p 値を返す。R の公開関数としては `brant::brant()` があるが、テストの fixture は Brant (1990) の式を R で手書きした結果。

## 2. 出典

- Brant, R. (1990). Assessing proportionality in the proportional odds model for ordinal logistic regression. *Biometrics* 46:1171–1178, sec. 3（二値ロジスティック回帰の同時分散と Wald 検定）。
- McCullagh, P. (1980). Regression models for ordinal data. *JRSS-B* 42:109–142（比例オッズモデル）。

## 3. モデルと式

応答の水準を $1,\dots,K$、共変量を $x_i\in\mathbb R^p$、切断点を $k=1,\dots,K-1$ とする。各 $k$ で二値応答 $z_{ik}=1\{Y_i>k\}$ にロジスティック回帰

$$\mathrm{logit}\,P(Y_i>k) = \alpha_k + x_i'\beta_k + o_i$$

を当てはめる（$o_i$ はオフセット）。$\tilde X=[1, X]$、$\pi_{ik}=P(Y_i>k)$ の推定値、$w_i$ を度数重みとする。

- $k\le l$ のとき $1\{Y>k\}$ と $1\{Y>l\}$ の共分散は $\pi_l-\pi_k\pi_l$。
- $W_{kl}=\mathrm{diag}\{w_i(\pi_{il}-\pi_{ik}\pi_{il})\}$。$W_{kk}$ は通常の $w\pi(1-\pi)$。
- 推定量 $(\hat\alpha_k,\hat\beta_k)$ と $(\hat\alpha_l,\hat\beta_l)$ の漸近共分散（Brant 1990, sec. 3）:
  $$V_{kl} = (\tilde X'W_{kk}\tilde X)^{-1}\,\tilde X'W_{kl}\tilde X\,(\tilde X'W_{ll}\tilde X)^{-1}.$$
- 対比 $D$: $k=2,\dots,K-1$ について $\hat\beta_k-\hat\beta_1$（切片は除く）。$d=D\hat\theta$、$\Sigma = D V D'$。
- 全体の統計量 $X^2 = d'\Sigma^{-1}d$、自由度 $p(K-2)$。
- 共変量 $c$ ごとの統計量は、$d$ と $\Sigma$ を $c$ 番目の傾きの差 $K-2$ 個に絞ったもの。自由度 $K-2$。
- p 値は $\chi^2$ の上側確率。

## 4. 推定・計算の手順

1. link が logit でなければ ValueError（メッセージに "logit"）。水準が 3 未満、共変量が 0 個でも ValueError。
2. 各切断点 $k$ で、Newton 法（ロジットでは Fisher scoring と同じ）で重みつきロジスティック回帰を解く。切片の初期値は重みつき割合の logit。ステップの最大成分が $10^{-12}(1+\max|\beta|)$ 以下で止める。
3. 最適点での $\pi_{ik}$ と $(\tilde X'W_{kk}\tilde X)^{-1}$ を保存する。
4. 3 節の式で $(K-1)(p+1)$ 次の同時共分散行列を組む（$k>l$ は転置）。
5. 対比 $D$ を作り、全体と共変量ごとの Wald 統計量を `solve` で計算する。
6. 表（`term`, `statistic`, `df`, `p_value`）、二値回帰の係数（$K-1$ 行 $\times$ $(1+p)$ 列、切片が先）、切断点ラベル `"<水準 k>|<水準 k+1>"`、名前 `["(Intercept)", ...]` を返す。

## 5. 数値上の注意

- 重みは度数重みとして扱う（情報行列と $W_{kl}$ の両方に $w_i$ を掛ける）。
- 分離（ある切断点で完全に分かれる）があると Newton 法が発散する。今は対策していない。
- 同時共分散を作ってから対比を取るので、対比後の $\Sigma$ は対称になる。

## 6. R と出力を合わせるための公開仕様

- fixture は Brant (1990) の式を R で直接書いたもの。二値回帰は R の `glm(..., family = binomial)` を厳しい収束条件で当てはめた値。
- 共変量ごとの行の名前は `fit.names`（モデル行列の列名）。

## 7. 検証

- `tests/r_oracle/test_ordinal.py::test_brant_matches_hand_coded_r`（a_logit、b_weighted、c_interaction）。二値回帰の係数 rtol $10^{-6}$・atol $10^{-9}$、統計量 rtol $10^{-6}$、自由度は完全一致、p 値 rtol $10^{-5}$・atol $10^{-12}$。fixture `tests/r_oracle/fixtures/ordinal.json`。
- `::test_brant_needs_logit`。

## 8. R と違うところ、未確認の点

- 分離がある場合の挙動は確かめていない。
