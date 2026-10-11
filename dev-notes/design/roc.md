# ROC 曲線: 閾値、最適カットポイント、座標、DeLong 分散と検定

対象: `src/statract/models/roc.py` の `_thresholds`、`_optim_crit`、`_paired_rocs`、`_placements`、`RocCurve.var_auc`、`RocCurve.ci_auc`、`RocCurve._calc_coords`、`roc_test`。

## 1. 目的

- `_thresholds`: 経験 ROC 曲線（receiver operating characteristic curve）の候補閾値を作る。出力は R の `pROC::roc()` の `thresholds` と比べる。
- `_optim_crit`: 各曲線点の最適性の基準値を返す。大きいほど良い。`pROC::coords(x = "best")` の `best.method`、`best.weights` と出力を比べる。
- `_paired_rocs`: 2 本の曲線が同じ被験者で測られた（対応がある）かを判定する。対応があれば、共通の行で作り直した 2 本を返す。`pROC::are.paired(return.paired.rocs = TRUE)` と同じ意味。結果は `roc_test` の DeLong 検定に使う。
- `_placements`: DeLong の配置値（placement value）を作る。分散と検定の共通部品。
- `var_auc`: AUC の DeLong 分散。`pROC::var()` と出力を比べる。
- `ci_auc`: AUC の DeLong 信頼区間。`pROC::ci.auc()` と比べる。
- `_calc_coords`: 閾値ごとの混同行列と派生指標。`pROC::coords()` と比べる。
- `roc_test`: 2 本の AUC の DeLong 検定（対応あり Z 検定、対応なし Welch 型 t 検定）。`pROC::roc.test(method = "delong")` と比べる。

## 2. 出典

- Fawcett T (2006). An introduction to ROC analysis. *Pattern Recognition Letters* 27:861–874. 第 5 節（経験 ROC 曲線の効率的な作り方）。
- Youden WJ (1950). Index for rating diagnostic tests. *Cancer* 3:32–35.
- Perkins NJ, Schisterman EF (2006). The inconsistency of "optimal" cutpoints obtained using two criteria based on the receiver operating characteristic curve. *American Journal of Epidemiology* 163:670–675. Youden 指数と左上最近点の定義、費用と有病率による重みづけ。
- DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 44:837–845. 対応のある AUC（area under the curve）の比較。
- Sun X, Xu W (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters* 21:1389–1393. 中央順位（midrank）による配置値の $O(N \log N)$ 計算（式 6–7 付近）。
- Hanley JA, McNeil BJ (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143:29–36. AUC と Mann–Whitney 統計量の関係、正規近似の区間。
- Hanley JA, McNeil BJ (1983). A method of comparing the areas under receiver operating characteristic curves derived from the same cases. *Radiology* 148:839–843. 背景のみ。
- Fawcett T (2006) の第 2 節、図 1。混同行列と派生指標（感度、特異度、適合率など）の定義。
- Welch BL (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika* 34:28–35。Satterthwaite FE (1946). *Biometrics Bulletin* 2:110–114。対応なし検定の自由度。
- Robin X ほか (2011). pROC: an open-source package for R and S+ to analyze and compare ROC curves. *BMC Bioinformatics* 12:77. 公開仕様の確認だけに使う。

## 3. モデルと式

記号: 対照群のスコア $X_1,\dots,X_n$、症例群のスコア $Y_1,\dots,Y_m$。閾値 $t$。

向き（direction）:
- `"<"`: 対照の方が低い。$s \ge t$ を陽性とする。
- `">"`: 対照の方が高い。$s \le t$ を陽性とする。

`"<"` のとき感度と特異度は $\mathrm{Se}(t) = \#\{Y_j \ge t\}/m$、$\mathrm{Sp}(t) = \#\{X_i < t\}/n$。

候補閾値（Fawcett 2006, 第 5 節）: プールしたスコアの相異なる値を昇順に $v_1 < \dots < v_k$ とする。

$$T = \{-\infty,\ (v_1+v_2)/2,\ \dots,\ (v_{k-1}+v_k)/2,\ +\infty\}$$

長さは $k+1$。曲線は相異なる値の間でしか変わらないので、これで全ての点が出る。

重み: 費用 $c$、有病率 $p$ として $r = (1-p)/(c\,p)$（Perkins & Schisterman 2006）。

- 重みつき Youden 基準: $J_r = \mathrm{Se} + r\,\mathrm{Sp}$。$r = 1$ のとき Youden (1950) の $J = \mathrm{Se} + \mathrm{Sp} - 1$ と定数差だけ違い、最大点は同じ。
- 左上最近点（closest to (0, 1)）の基準: $-\{(1-\mathrm{Se})^2 + r\,(1-\mathrm{Sp})^2\}$。符号を反転して「大きいほど良い」にそろえる。

配置値（DeLong ほか 1988, 第 2 節）: $\psi(y, x) = 1$（$y > x$）、$1/2$（$y = x$）、$0$（$y < x$）とする。向き `"<"` で

$$V_{10}(Y_j) = \frac{1}{n}\sum_{i=1}^{n} \psi(Y_j, X_i), \qquad V_{01}(X_i) = \frac{1}{m}\sum_{j=1}^{m} \psi(Y_j, X_i).$$

$V_{10}$ は症例ごとの「その症例より低い対照の割合」、$V_{01}$ は対照ごとの「その対照より高い症例の割合」。どちらの平均も AUC $\hat\theta$ に等しい（Mann–Whitney 統計量、Hanley & McNeil 1982）。向き `">"` ではスコアの符号を反転してから同じ式を使う。

DeLong 分散（DeLong ほか 1988, 第 2 節）: $S_{10}$、$S_{01}$ を $V_{10}$、$V_{01}$ の標本分散（分母 $m-1$、$n-1$）として

$$\widehat{\mathrm{Var}}(\hat\theta) = \frac{S_{10}}{m} + \frac{S_{01}}{n}.$$

信頼区間（正規近似）: $\hat\theta \pm z_{(1+\alpha')/2}\sqrt{\widehat{\mathrm{Var}}}$。$\alpha'$ は信頼水準。3 つの値は $[0, 1]$ に切り詰める。

対応ありの比較（DeLong ほか 1988, 第 3 節）: 2 本の曲線 $r = 1, 2$ の配置値 $V_{10}^{(r)}$、$V_{01}^{(r)}$ から $2\times 2$ の標本共分散行列 $S_{10}$、$S_{01}$ を作る。$S = S_{10}/m + S_{01}/n$、$L = (1, -1)$ として

$$Z = \frac{\hat\theta_1 - \hat\theta_2}{\sqrt{L S L^\top}}.$$

$Z$ は帰無仮説の下で近似的に標準正規分布に従う。差の信頼区間は $(\hat\theta_1-\hat\theta_2) \pm z_{1-(1-c)/2}\sqrt{L S L^\top}$（$c$ は信頼水準）。

対応なしの比較: 各曲線の DeLong 分散 $V_1$、$V_2$、観測数（症例 + 対照）$N_1$、$N_2$ として

$$t = \frac{\hat\theta_1 - \hat\theta_2}{\sqrt{V_1 + V_2}}, \qquad \nu = \frac{(V_1 + V_2)^2}{V_1^2/(N_1 - 1) + V_2^2/(N_2 - 1)}.$$

$\nu$ は Welch–Satterthwaite 型の自由度。各曲線の分母を「観測数 − 1」にする選び方は、fixture の自由度に合うものとして決めた（第 8 節）。$t$ を自由度 $\nu$ の $t$ 分布で評価する。

p 値: 分布関数を $F$（正規または $t$）として、両側 $2F(-|s|)$、`greater` は $F(-s)$、`less` は $F(s)$。

座標（Fawcett 2006, 第 2 節）: 症例数 $m$、対照数 $n$、閾値での感度 Se、特異度 Sp から

- $TP = \mathrm{Se}\,m$、$FN = m - TP$、$TN = \mathrm{Sp}\,n$、$FP = n - TN$。
- $\mathrm{NPV} = TN/(TN+FN)$、$\mathrm{PPV} = TP/(TP+FP)$、$\mathrm{FNR} = FN/(TP+FN)$、$\mathrm{FDR} = FP/(TP+FP)$。
- 正解率 $= (TP+TN)/(m+n)$。
- TPR = recall = Se、TNR = Sp、FPR = $1-\mathrm{Sp}$、precision = PPV。`1-x` の列は補数。
- `youden` は $J_r$、`closest.topleft` は重みつき距離（正の値）。

## 4. 計算の手順

`_thresholds`:
1. スコアの相異なる値を昇順に並べる。
2. 隣り合う値の中点を取る。
3. 両端に $-\infty$ と $+\infty$ を付ける。

`_optim_crit`:
1. 有病率 $p$ が $(0,1)$ の内側か確かめる。
2. $r$ を計算する。
3. 方法に応じて第 3 節の式をベクトルで評価する。`"topleft"` は `"closest.topleft"` の別名。

`_paired_rocs`（DeLong ほか 1988 の「同じ被験者」の条件を、同じ応答ベクトルとして判定する）:
1. 水準（levels）が違えば対応なし（None）。
2. 欠測除去後の応答の値が同じなら、そのまま 2 本を返す。系列名は無視する。
3. 元の応答の長さと値が同じなら、共通の行で作り直す。共通の行は、応答が欠測でなく水準に含まれ、どちらの元の予測値も欠測でない行。各曲線は自分の水準と向きを保つ。
4. それ以外は対応なし（None）。

`_placements`（Sun & Xu 2014）:
1. 向き `">"` ならスコアの符号を反転する。
2. 症例と対照をまとめた中央順位、症例内の中央順位、対照内の中央順位を求める。同順位は平均順位。
3. 症例の配置値 = （全体での順位 − 症例内での順位）/ $n$。順位の差は、その症例より低い対照の数（同点は 1/2）になる。
4. 対照の配置値 = 1 − （全体での順位 − 対照内での順位）/ $m$。順位の差は、その対照より低い症例の数（同点は 1/2）。1 から引くと「高い症例の割合」になる。
5. 計算量は並べ替えの $O(N \log N)$。

`var_auc`:
1. 症例か対照が 2 未満なら NaN。
2. 完全な曲線（ある点で Se + Sp = 2）なら警告を出す。
3. 配置値の標本分散から第 3 節の式で計算する。

`ci_auc`:
1. `method` が `"delong"` でないとき、または `level` が $[0,1]$ の外のとき `ValueError`。
2. 症例か対照が 2 未満なら NaN を 3 つ。
3. 完全な曲線なら「区間が 1 の 1 点につぶれる」旨の警告。
4. 正規分位点で区間を作り、3 つの値を $[0,1]$ に切り詰める。

`_calc_coords`: 第 3 節の式をベクトルで評価する。0/0 は NaN のまま返し、警告は出さない。

`roc_test`:
1. `method` と `alternative` を確かめる。不正なら `ValueError`。
2. 向きが違えば警告（DeLong 検定は同じ向きを前提にする）。
3. `_paired_rocs` で対応を判定する。`paired=None` なら判定結果に従う。`paired=True` なのに対応がなければ `ValueError`。`paired=False` なのに対応があれば、対応ありの検定を勧める警告。
4. 両方が完全な曲線なら警告。
5. 対応あり: 作り直した 2 本の配置値から共分散行列を作り、$Z$ と差の信頼区間を求める。
6. 対応なし: 各曲線の DeLong 分散で $t$ と $\nu$ を求める。信頼区間は返さない。
7. 第 3 節の規則で p 値を求める。

## 5. 数値上の注意

- オーバーフロー: どちらかの値の絶対値が $10^{100}$ を超えるときは $v_i/2 + v_{i+1}/2$ で中点を取る。$(v_i+v_{i+1})/2$ だと $10^{308}$ 付近であふれる。
- 近接した同順位（near ties）: 隣り合う 2 値が浮動小数点で 1 ulp ほどしか違わないと、計算した中点が片方の値と一致する。そのときは判定規則の下で 2 値を分ける側に置き換える。
  - `"<"`: 上側の値 $v_{i+1}$。$v_i < t$ かつ $v_{i+1} \ge t$ になる。
  - `">"`: 下側の値 $v_i$。$v_i \le t$ かつ $v_{i+1} > t$ になる。
  - 置き換え後も分けられなければ `RuntimeError("thresholds do not separate near-tied predictor values")`。相異なる有限値ではこの分岐にはまず入らない。
- 最適点の同順位: `coords(x = "best")` は基準が最大値と等しい点を全て返す。

- 完全な曲線の判定: $|\mathrm{Se} + \mathrm{Sp} - 2| < \sqrt{\varepsilon}$（$\varepsilon$ は機械イプシロン）。このとき配置値は全て 1 なので分散は 0。
- 検定で差も標準誤差も 0 のときは統計量を 0 とする（0/0 を避ける）。差が 0 でなく標準誤差が 0 なら $\pm\infty$。
- 対応ありの共分散行列 $L S L^\top$ が丸め誤差で負になったら 0 に切り上げる。
- 分散が両方 0 のとき自由度は NaN になる。

## 6. R と出力を合わせるための公開仕様

- `?pROC::roc`: 閾値は「並べた一意な値の中点と、−Inf、Inf」。
- `?pROC::coords`: `best.weights = c(cost, prevalence)`。既定は `c(1, 0.5)` で、このとき $r = 1$。`best.method` は `"youden"` か `"closest.topleft"`。
- `coords` の `youden` 列は $\mathrm{Se} + r\,\mathrm{Sp}$ として出る（$-1$ はしない）。`closest.topleft` 列は距離そのもの（正の値）で、`_optim_crit` の値の符号を反転したもの。
- `?pROC::are.paired`: 応答が同じなら対応あり。`return.paired.rocs = TRUE` で共通の行の曲線を返す。
- `?pROC::ci.auc`: DeLong 区間は $[0, 1]$ に収める。既定の信頼水準は 0.95。
- `?pROC::roc.test`: `paired` を省くと自動判定。`method = "delong"` で対応なしなら t 検定（自由度つき）、対応ありなら Z 検定。`alternative` は `"two.sided"`、`"less"`、`"greater"`。表示される方法名は `"DeLong's test for two correlated ROC curves"`（対応あり）と `"DeLong's test for two ROC curves"`（対応なし）。
- `?pROC::coords`: 各列の定義（tp = se × 症例数 など）。
- 警告の文言は自分の言葉で書いた（完全な曲線、向きの違い、対応の勧め）。
- エラーの文言: 有病率が範囲外なら `best_weights[1] (prevalence) must be in (0, 1)`。方法が不正なら `best_method must be 'youden' or 'closest.topleft'`。

## 7. 検証

- fixture: `tests/r_oracle/fixtures/proc.json`（`scripts/proc.R`、pROC 1.18.5 の出力）。
- `tests/r_oracle/test_proc.py`:
  - `test_curve_and_auc`: 閾値、感度、特異度、AUC を rtol 1e-10 で比べる。
  - `test_coords`: rtol 1e-10。
  - `test_roc_test`: 統計量 rtol 1e-8、p 値 atol 1e-8、AUC rtol 1e-10、自由度 rtol 1e-8。
  - `test_paired_flag_errors`、`test_delong_var_ci`、`test_direction_and_levels`、`test_plot_roc`。
- 許容誤差: 分散 rtol 1e-8・atol 1e-15、区間 rtol 1e-8（水準 0.95、0.90）、座標 rtol 1e-10。
- 試料: `balanced`、`small`、`ties`。曲線 `roc1`、`roc2`（同じ応答）、`roc_other`（別の応答）。
- 結果: `test_proc.py` の 40 件が全て通過。許容誤差の変更なし。

## 8. R と違うところ、未確認の点

- 近接した同順位の処理は、自分で決めた規則（第 5 節）。fixture には近接した同順位がないので、R と比べていない。
- 対応なし検定の自由度: Welch–Satterthwaite の各項の分母を「その曲線の観測数（症例 + 対照）− 1」とした。3 試料とも fixture の自由度と rtol 1e-8 以内で一致した。「症例数と対照数の小さい方 − 1」や「観測数」そのものでは合わない。理論上の最良の選び方かは未確認。
- 欠測除去後の応答が一致するだけで対応ありとする（判定手順 2）。2 本で除去された行が違うと、配置値の並びが被験者ごとにそろわない恐れがある。fixture には欠測がないので R と比べていない。
- 応答の一致判定は値だけを見る。型（整数 0/1 と実数 0.0/1.0）は区別しない。同じ位置の欠測は一致として扱う。
