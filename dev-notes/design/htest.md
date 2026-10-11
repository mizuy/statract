# 古典的検定: t 検定、Wilcoxon 検定、McNemar 検定、二項検定、比率の検定

対象: `src/statract/models/htest.py` の `t_test`、`wilcox_test`（補助 `_signed_rank_parts`、`_rank_sum_parts`、`_wilcox_exact_interval`、`_signrank_z`、`_ranksum_z`、`_normal_interval`、`_signrank_normal_interval`、`_ranksum_normal_interval`）、`mcnemar_test`、`_binom_two_sided_p`、`_wilson`、`prop_test`。

## 1. 目的

- `t_test`: 1 標本、対応あり、Student のプール 2 標本、Welch 2 標本の t 検定と平均（差）の信頼区間。R の `t.test()` と出力を比べる。
- `wilcox_test`: Wilcoxon 符号付き順位検定（1 標本、対応あり）と順位和検定（Mann–Whitney）。正確法と正規近似。`conf_int=True` で Hodges–Lehmann（HL）推定量と区間。R の `wilcox.test()` と比べる。
- `mcnemar_test`: 2×2 の McNemar 検定と r×r の Bowker 対称性検定。R の `mcnemar.test()` と比べる。
- `_binom_two_sided_p`: 正確二項検定の両側 p 値。`binom_test` が $0<p<1$ の両側のときだけ呼ぶ。R の `binom.test()` と比べる。
- `_wilson`: 1 標本比率の Wilson スコア区間（連続修正つき可）。`prop_test` の 1 標本と `tableone` の `proportion_ci(method="wilson")` が使う。
- `prop_test`: k 群の比率の Pearson カイ二乗検定。R の `prop.test()` と比べる。

## 2. 出典

- Student (1908). The probable error of a mean. *Biometrika* 6:1–25.
- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika* 34:28–35.
- Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance components. *Biometrics Bulletin* 2:110–114.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin* 1:80–83.
- Mann, H. B. & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Ann Math Stat* 18:50–60.
- Hollander, M., Wolfe, D. A. & Chicken, E. (2014). *Nonparametric Statistical Methods*, 3rd ed. 3.1–3.3 節（符号付き順位、正規近似、同順位補正、HL 推定と区間）、4.1–4.3 節（順位和）。
- Hodges, J. L. & Lehmann, E. L. (1963). Estimates of location based on rank tests. *Ann Math Stat* 34:598–611.
- Bauer, D. F. (1972). Constructing confidence sets using rank statistics. *JASA* 67:687–690.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12:153–157.
- Edwards, A. L. (1948). Note on the "correction for continuity" in testing the significance of the difference between correlated proportions. *Psychometrika* 13:185–187.
- Bowker, A. H. (1948). A test for symmetry in contingency tables. *JASA* 43:572–574.
- Sterne, T. E. (1954). Some remarks on confidence or fiducial limits. *Biometrika* 41:275–278.（最小尤度法による両側 p 値）
- Clopper, C. J. & Pearson, E. S. (1934). *Biometrika* 26:404–413.（区間は既存コードで、本ノートの対象外）
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22:209–212.
- Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion: comparison of seven methods. *Stat Med* 17:857–872, method 4（連続修正つきスコア区間）。
- Yates, F. (1934). Contingency tables involving small numbers and the χ² test. *JRSS Suppl* 1:217–235.
- Fleiss, J. L., Levin, B. & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions*, 3rd ed. 2–4 章、9 章（k 群の比較）、4.3 節（差の区間）。

## 3. モデルと式

### t 検定

標本 $x_1,\dots,x_{n_x}$、$y_1,\dots,y_{n_y}$。平均 $\bar x,\bar y$、不偏分散 $s_x^2,s_y^2$（分母 $n-1$）。帰無値 $\mu_0$。

- 1 標本: $SE=\sqrt{s_x^2/n_x}$、$t=(\bar x-\mu_0)/SE$、自由度 $\nu=n_x-1$。
- 対応あり: 差 $d_i=x_i-y_i$ に 1 標本の式を使う。
- プール（Student）: $s_p^2=\{(n_x-1)s_x^2+(n_y-1)s_y^2\}/(n_x+n_y-2)$、$SE=\sqrt{s_p^2(1/n_x+1/n_y)}$、$\nu=n_x+n_y-2$。
- Welch: $a=s_x^2/n_x$、$b=s_y^2/n_y$、$SE=\sqrt{a+b}$、Satterthwaite の自由度 $\nu=(a+b)^2/\{a^2/(n_x-1)+b^2/(n_y-1)\}$。
- 2 標本の統計量は $t=(\bar x-\bar y-\mu_0)/SE$。

信頼区間は統計量の尺度で $t+[q_L,q_U]$ を作り、$\mu_0+(\cdot)\,SE$ で平均の尺度に戻す。両側は $q=\pm t_{\nu,(1+\gamma)/2}$、"less" は $(-\infty,\,t+t_{\nu,\gamma}]$、"greater" は $[t-t_{\nu,\gamma},\infty)$（$\gamma$ は信頼係数）。

### Wilcoxon 検定

符号付き順位（Hollander ら 3.1 節）: 差 $d_i$（$x-\mu_0$、または $x-y-\mu_0$）のうち 0 を除いた $n$ 個。$|d_i|$ に中間順位 $R_i$ をつける。

$$V=\sum_{d_i>0}R_i,\quad E_0V=\frac{n(n+1)}{4},\quad \mathrm{Var}_0V=\frac{n(n+1)(2n+1)}{24}-\frac{\sum_g(t_g^3-t_g)}{48}$$

$t_g$ は同順位の組の大きさ。

順位和（4.1 節）: $m=n_x$、$n=n_y$、$N=m+n$。$x-\mu_0$ と $y$ を合わせた中間順位の $x$ 側の和を $S$ とし、

$$W=S-\frac{m(m+1)}{2},\quad E_0W=\frac{mn}{2},\quad \mathrm{Var}_0W=\frac{mn}{12}\left\{(N+1)-\frac{\sum_g(t_g^3-t_g)}{N(N-1)}\right\}$$

$W$ は Mann–Whitney の数え上げ $\#\{x_i-\mu_0>y_j\}$（同順位は 1/2）。

標準化統計量 $Z=(T-E_0T-\text{cc})/\sqrt{\mathrm{Var}_0T}$。連続修正 cc は "greater" で $+1/2$、"less" で $-1/2$、両側で $\mathrm{sign}(T-E_0T)/2$（`correct=False` なら 0）。

HL 推定と区間（Hodges & Lehmann 1963、Bauer 1972、Hollander ら 3.3、4.3 節）:
- 符号付き順位: Walsh 平均 $A_{ij}=(x_i+x_j)/2$（$i\le j$、計 $M=n(n+1)/2$ 個）。
- 順位和: 差 $A_{ij}=x_i-y_j$（計 $M=mn$ 個）。
- 推定量は $A$ の中央値。昇順の $A_{(1)}\le\dots\le A_{(M)}$ で、区間 $[A_{(c)},A_{(M+1-c)}]$ の被覆確率は $1-2P_0(T\le c-1)$。

### McNemar と Bowker

r×r 表 $n_{ij}$。Bowker (1948):

$$X^2=\sum_{i<j}\frac{(n_{ij}-n_{ji})^2}{n_{ij}+n_{ji}},\quad \text{自由度 } r(r-1)/2$$

r=2 は McNemar (1947)。連続修正（Edwards 1948）は 2×2 だけ: $(|n_{12}-n_{21}|-1)^2/(n_{12}+n_{21})$。

### 正確二項検定（両側）

$X\sim\mathrm{Bin}(n,p)$、観測値 $k$、$f(j)=P(X=j)$。Sterne (1954) の考え方で、観測値より起こりにくい結果の確率の合計を p 値とする。

$$p=\sum_{j:\,f(j)\le f(k)} f(j)$$

### Wilson 区間

$\hat p=k/n$、$z$ は正規分位点。Wilson (1927) の限界は $(\hat p-\pi)^2=z^2\pi(1-\pi)/n$ の 2 根。

$$L,U=\frac{\hat p+\frac{z^2}{2n}\mp z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+z^2/n}$$

連続修正 $c$ つき（Newcombe 1998, method 4）は、上限を $\hat p+c/n$、下限を $\hat p-c/n$ で上式に代入したものと一致する。

### 比率の検定

群 $g=1,\dots,k$、成功 $x_g$、試行 $n_g$、$\hat p_g=x_g/n_g$。帰無比率は与えられた $p_g$、または等しい比率の検定ではプール $\bar p=\sum x_g/\sum n_g$。$2k$ 個のセル（成功と失敗）で期待度数 $E$ を作り

$$X^2=\sum_{\text{cells}}\frac{(|O-E|-c)^2}{E}$$

自由度は等比率で $k-1$、与えられた $p$ で $k$。$c$ は Yates (1934) の修正（6 節）。

2 群の差の区間（Fleiss ら 4.3 節）: $\hat p_1-\hat p_2\pm\{z\sqrt{\hat p_1(1-\hat p_1)/n_1+\hat p_2(1-\hat p_2)/n_2}+c(1/n_1+1/n_2)\}$。

## 4. 計算の手順

### t 検定

1. 欠測の除去: 1 標本と 2 標本は標本ごとに NaN を除く。対応ありは両方そろった組だけ残し、差をとる。
2. 標本サイズを確かめる（足りなければ `ValueError`）。
3. 3 節の式で $SE$、$\nu$、$t$ を計算する。プールでサイズ 1 の群は平方和に寄与しない。
4. 「ほぼ定数」の判定: $SE\le 10\,\varepsilon_{\text{mach}}\cdot\max(|\bar x|,|\bar y|)$ なら `ValueError`。
5. p 値: less は $P(T\le t)$、greater は $P(T\ge t)$、両側は $2P(T\le-|t|)$。
6. 区間は 3 節のとおり。

### Wilcoxon 検定

1. 符号付き順位: 差を作り、0 の差を除く。0 の差がある、または $|d|$ に同順位があると正確法は使えない。順位和: $x-\mu_0$ と $y$ を合わせて同順位を見る。
2. `exact=None` のとき、符号付き順位は $n<50$、順位和は両群とも 50 未満で正確法。同順位（または 0）があれば警告して正規近似にする。
3. 正確法の p 値: 既存の `_psignrank` / `_pwilcox` で $P(T\le t)$ と $P(T\ge t)=P(T>t-1)$。両側は小さい方の 2 倍を 1 で切る。
4. 正規近似の p 値: 3 節の $Z$ から。両側は $2\min\{\Phi(Z),1-\Phi(Z)\}$。
5. 区間の標本: 符号付き順位では 0 の差（$d_i=0$）の観測を区間の計算からも除く（検定と同じ標本を使う）。

### Wilcoxon 区間: 正確法

1. Walsh 平均（または全差）を作り、昇順に並べる。
2. 片側確率 $a=\alpha/2$（両側）または $\alpha$（片側）。$c$ = 帰無分布の $a$ 分位点（`_qsignrank` / `_qwilcox`、$P(T\le c)\ge a$ となる最小の $c$）。
3. $c=0$ なら要求水準に届かない。$c=1$ にして警告する。
4. 下端 $A_{(c)}$、上端 $A_{(M+1-c)}$。片側は反対側を $\pm\infty$。
5. 届かなかったときに報告する水準は $1-2P_0(T\le c-1)$（片側は $1-P_0(T\le c-1)$）。
6. 推定量は中央値。

### Wilcoxon 区間: 正規近似

1. 探索範囲: 符号付き順位は $[\min x,\max x]$、順位和は $[\min x-\max y,\ \max x-\min y]$。
2. $Z(\delta)$ は「$x-\delta$ に対する標準化統計量」。$\delta$ ごとに順位をつけ直し、同順位補正も計算し直す。符号付き順位では $x_i-\delta=0$ の観測をその $\delta$ では除き、$n$ を数え直す。$Z$ は $\delta$ について非増加の階段関数。
3. 連続修正は検定と同じ向き（3 節の cc、対立仮説で決まる）。
4. 下端は $Z(\delta)=z_{1-a}$、上端は $Z(\delta)=-z_{1-a}$ の根。`scipy.optimize.brentq`（`xtol=tol_root`）で求める。
5. 範囲の端で $Z-$目標の符号が変わらないときは、その端を限界にし、端での $Z$ から届く水準（両側 $2\Phi(\min(Z_{\text{左端}},-Z_{\text{右端}}))-1$）を報告して警告する。
6. 推定量は連続修正なしの $Z(\delta)=0$ の根。見つからなければ範囲の中点。範囲の幅が 0 ならその値。

### McNemar

1. 入力が表なら正方、2 行以上、有限で非負かを確かめる。
2. 2 列の入力なら、どちらかが欠測の組を除く。水準は polars Enum なら Enum の順（出現する水準だけ）、それ以外は値の昇順。両列の水準数が等しく 2 以上であることを確かめ、位置で対応させてクロス表を作る。
3. 上三角と下三角の対を取り、3 節の式で和をとる。$n_{ij}+n_{ji}=0$ の対は情報がないので和から除く（自由度は変えない）。全部 0 なら統計量と p 値は NaN。
4. 2×2 で `correct=True` なら $|n_{12}-n_{21}|$ から 1 を引く（0 未満にはしない）。
5. p 値はカイ二乗の上側。

### 正確二項検定

1. $k=np$ なら 1。
2. 二項確率は単峰なので、$f(j)\le f(k)$ の結果は「$k$ 側の裾」と「最頻値の反対側の裾」になる。反対側で $f(j)\le f(k)(1+10^{-7})$ を満たす最初の点を探し、両側の裾の和を分布関数（cdf/sf）で求める。
3. 1 で上限を切る。

### Wilson 区間

1. $z$ は両側なら $(1+\gamma)/2$、片側なら $\gamma$ の分位点。
2. 上限: $\hat p+c/n\ge1$ なら 1、それ以外は $\hat p+c/n$ で 3 節の上限。下限: $\hat p-c/n\le0$ なら 0、それ以外は $\hat p-c/n$ で下限。
3. $[0,1]$ に切る。"greater" は $(L,1)$、"less" は $(0,U)$。

### 比率の検定

1. 入力を（成功、試行）の組にそろえる。`n=None` なら $x$ は（成功、失敗）の対か k×2 表。欠測のある群は除く。
2. $p$ の検査（各群 1 つ、$(0,1)$ の中）。1 群で $p$ がなければ 0.5。
3. 対立仮説は 1 群、または $p$ なしの 2 群のときだけ使う。それ以外は "two.sided" と報告する。
4. 修正量 $c$: `correct=True` かつ $k\le2$ のとき $\min(1/2,\ \text{乖離})$。乖離は 1 群で $|x-np_0|$、2 群で $|\hat p_1-\hat p_2|/(1/n_1+1/n_2)$（与えられた $p$ の 2 群も同じ）。$k>2$ では 0。
5. 統計量は 3 節の式。期待度数に 5 未満があれば警告。
6. p 値: 両側はカイ二乗の上側。片側は $\mathrm{sign}(\hat p-p_0)$（2 群は $\mathrm{sign}(\hat p_1-\hat p_2)$）をつけた $\sqrt{X^2}$ を標準正規で評価する。
7. 区間: 1 群は Wilson（修正 $c$）、$p$ なしの 2 群は 3 節の差の区間を $[-1,1]$ に切る。それ以外はなし。
8. method 文字列の末尾は $c>0$ なら " with continuity correction"、それ以外は " without continuity correction"。

## 5. 数値上の注意

- t: 定数データでは $SE$ がほぼ 0 になる。4 の判定で止める。
- Wilcoxon 正規近似: $Z$ は階段関数なので根は跳びの位置にある。brentq は括弧の幅が `tol_root` を下回ると止まる。R の `uniroot` も同じ種類の停止規則だが、たどる点が違うので、答えは最大 $2\cdot$`tol_root` ずれうる。
- Wilcoxon 正確法: Walsh 平均は $n<50$ で高々 1275 個、全差は高々 2401 個。全列挙で十分。
- 二項: 確率が同じ結果（$p=1/2$ の対称な点など）は浮動小数で少しずれる。相対 $10^{-7}$ で同じとみなす。裾の和は cdf/sf を使うので、小さい p 値でも相対精度が保てる。
- Wilson: $\hat p\pm c/n$ が $[0,1]$ を出ると根号内が負になりうる。先に 0、1 に置き換える。
- 比率の検定: 修正量が乖離と等しいとき（例 15/50 対 16/52）、統計量は理論上 0 だが丸め誤差で $10^{-31}$ 程度になる。式をそのまま評価して、R の出力（2.448e-31）と一致した。

## 6. R と出力を合わせるための公開仕様

- `?t.test`: 既定 `alternative="two.sided"`、`mu=0`、`paired=FALSE`、`var.equal=FALSE`、`conf.level=0.95`。method 文字列は "One Sample t-test"、"Paired t-test"、"Two Sample t-test"、"Welch Two Sample t-test"（R は先頭に空白を入れるが、こちらは入れない）。推定値の名前は "mean of x"、"mean difference"、"mean of x"/"mean of y"。
- `?wilcox.test`: 正確法は 50 未満で、同順位や 0 があると使わない。`tol.root` は根の探索の許容誤差。推定値の名前は "(pseudo)median"、"difference in location"。method 文字列は "Wilcoxon signed rank exact test"、"Wilcoxon signed rank test"、"Wilcoxon rank sum exact test"、"Wilcoxon rank sum test"、正規近似で修正ありなら " with continuity correction" がつく。
- 正確法で水準に届かないとき、R の出力では符号付き順位の水準が有効数字 2 桁で出る（fixture: $n=4$ で $1-2/16=0.875$ が 0.88）。statract もそれに合わせて 2 桁に丸める。順位和は丸めない（fixture なし）。
- `?mcnemar.test`: 連続修正は 2×2 だけ。method "McNemar's Chi-squared test"、統計量名 "McNemar's chi-squared"。
- `?binom.test`: 両側 p 値は「観測値の確率以下の確率をもつ結果の確率の和」。
- `?prop.test`: 1 標本の区間は Wilson のスコア区間で、`correct=TRUE` なら連続修正つき。修正は標本と帰無の比率の差を超えない範囲でだけ使う。method 文字列は "1-sample proportions test"、"{k}-sample test for equality of proportions"、"{k}-sample test for given proportions"。
- エラーと警告の文面は statract 独自のもの。

## 7. 検証

- `tests/r_oracle/test_htest.py`、fixture `tests/r_oracle/fixtures/htest.json`（R 4.3）。`test_case` 0–9（t）、10–31（Wilcoxon）、32–35（McNemar）、36–43（二項）、44–56（比率）。
- 許容誤差: 統計量 rtol $10^{-10}$、p 値 rtol $10^{-9}$ / atol $10^{-14}$、自由度・区間・水準 rtol $10^{-10}$（atol $10^{-12}$）。正規近似の Wilcoxon 区間と推定値は atol $2\cdot$`tol_root`。
- `test_proportion_ci[wilson-*]`（rtol $10^{-10}$、atol $10^{-14}$）、`test_frame_shape`。
- 結果: 92 件すべて通過。許容誤差の変更なし。

## 8. R と違うところ、未確認の点

- Wilcoxon 符号付き順位の区間で、0 の差の観測を区間の計算から除くこと。除かないと fixture の case 10、12、15、17、20、25 が合わない（例: 整数データの case 25 で上端が 0.5 になり、R は 1.0）。除くとすべて合う。
- 正規近似の区間で水準に届かないときの扱い（端を返し、届く水準を報告）は R と比べていない。
- McNemar: 対の両セルが 0 のときは和から除く。R の出力とは比べていない。2×2 で $n_{12}=n_{21}$ のとき、修正後の差を 0 で止めるので統計量 0、p 値 1。
- 二項: 観測値と同じ側で $k$ より中心寄りなのに確率が $f(k)$ とほぼ等しい結果は数えない。fixture では起きない。
- 比率の検定で与えられた $p$ の 2 群にも差による修正の縮小を使う（case 56 で確認）。$k>2$ の与えられた $p$ は修正なし（fixture は 4 群の等比率だけ）。
