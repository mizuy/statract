# 臨床の問いから始める統計

臨床研究に使う統計の考え方を、学び直すテキストです。臨床の問いから出発し、記述、予測、因果の三つに分け、それぞれに合う解析を、統計モデリングという共通の土台の上で理解することを目指します。

関数の選び方だけを知りたいときは [手法の選び方](../guide/methods.md) を見てください。実データでの解析の例は [解析例ギャラリー](../examples/index.md) にあり、各例から関係する章へリンクしています。

## 中心のメッセージ

!!! note ""
    統計手法から研究を考えるのではなく、まず何を知りたいのかを決める。その問いが**記述**（どのくらい起きているか）、**予測**（この患者はどうなるか）、**因果**（変えたら変わるか）のどれなのかを区別する。そのうえで、合ったモデルと解析方法を選ぶ。

この三つの分け方は、Hernán ら [@Hernan2019-dl] と Shmueli [@Shmueli2010-bg] に基づきます。Hernán らは因果推論を「反事実の予測」と呼び、記述、予測と並べています。

臨床の疑問を研究の問い（リサーチクエスチョン）に仕上げる手順は、日本語では福原 [@Fukuhara2017-book] が詳しく説明しています。

| 問いの種類 | 例 | 本書 |
|------------|----|------|
| 記述 | 出血は何%に起きるか。検査の感度はいくらか | 第 1 章、第 17 章 |
| 予測 | この患者は出血するか | 第 III 部 |
| 因果 | クリップを使えば出血は減るか | 第 II 部 |

臨床家がまず区別できるようにしたいのは、取り違えやすい因果と予測です。本書の中心もこの二つです。統計モデリングは問いの種類ではなく、どの問いにも使う共通の言葉です。本書ではロジスティック回帰と確率モデルを通して、三つをつなげます（第 IV 部）。

## 通しの例

全章で同じ例を使います。**大腸内視鏡治療後の遅発性出血**と**予防的クリップ閉鎖**です。同じデータに、次の違う問いを繰り返し投げます。

| 問い | 種類 | 扱う部 |
|------|------|--------|
| 出血はどのくらい起きているか | 記述 | 第 I 部 |
| 遅発性出血の「独立した危険因子」は何か | あいまい（分けて考える） | 第 I 部 |
| 予防的クリップは出血を減らすか | 因果 | 第 II 部 |
| この患者は出血するか | 予測 | 第 III 部 |
| いつ出血するか | アウトカムが時間（記述、予測、因果のどれにも） | 第 V 部 |
| 検査が陽性ならどのくらい確かか | 記述（検査の性能）と予測 | 第 V 部 |

データは**シミュレーション**です。どの変数がどれに影響するかを表す有向非巡回グラフ（directed acyclic graph、DAG）をコードに書いて作っているので、各章で「推定した値」と「本当の値」を比べられます。データの中身は [第 1 章](descriptive.md)、作り方は [第 2 章](independent-risk-factor.md#simulation) と [`examples/theory_bleeding`](https://github.com/mizuy/statract/tree/main/examples/theory_bleeding) にあります。

## 読み方

- 本文であつかう数学のうち高校数学を超える部分は [数学補](math/probability.md) にまとめ、一段ずつ説明します。
- 本文では式の導出より、「その式が何を意味するか」「何をしているか」を重視します。
- 図は statract で描き、コードを添えます。
- 第 2 章以降の各章と数学補の多くのページに、スライダーやクリックで動かせる図があります。
- 「発展」と書いた章（第 6、7、11、12 章）は、初めて読むときは飛ばしてかまいません。

## 目次

### 第 I 部 導入

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 1. [記述：まずデータを見る](descriptive.md) | 記述という問い、Table 1、割合と信頼区間、平均と中央値、$P$ 値と信頼区間の要点、Table 1 の $P$ 値 | 年齢と病変径の分布 | `write_tableone_artifacts`、`proportion_ci` |
| 2. [「独立した危険因子」とは何か](independent-risk-factor.md) | 慣習的な解析（単変量 → 多変量）、ロジスティック回帰の基本、因果か予測か | ロジスティック曲線、forest、層別の出血割合 | `fit_glm`、`write_tableone_artifacts`、`plot_forest` |

### 第 II 部 因果推論

理論は短く紹介し、詳しくは文献に譲ります。中心は、背景を調整する方法の使い方です。

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 3. [因果推論の考え方](causal-basics.md) | Rubin の反事実モデル、交換可能性、DAG とバックドア基準、文献 | 群ごとの潜在アウトカム、DAG | `prop_test` |
| 4. [背景を調整する方法](adjustment-methods.md) | 限定、層別化、回帰と標準化、マッチング、重み付けの比較 | 方法ごとの推定値と本当の値 | `fit_glm`、`standardize_glm`、`match_sample`、`propensity_weights` |
| 5. [傾向スコアマッチング](propensity-score-matching.md) | バランシングスコア、傾向スコア、重なり、キャリパー、バランスの確認 | 傾向スコアの分布、Love plot、マッチ前後の Table 1 | `match_sample`、`plot_love`、`write_tableone_artifacts` |
| 6. [IPTW](iptw.md)（発展） | 重みで仮の集団を作る、g-formula と同じ量になる理由、安定化、極端な重み、推定する対象 | 重みの分布、重み付きの Love plot、重みごとの推定値 | `propensity_weights`、`plot_love` |
| 7. [仮定と感度分析](assumptions-sensitivity.md)（発展） | 正値性、未測定交絡、E-value、一致性、Bradford Hill の視点 | 未測定交絡の影響、E-value の図 | `standardize_glm` |

### 第 III 部 予測

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 8. [予測モデルとしてのロジスティック回帰](prediction-model.md) | 同じ式を予測に使う、係数より予測確率、因果でない変数も予測に役立つ | 予測確率の分布 | `fit_glm` |
| 9. [過学習と汎化](overfitting.md) | 見かけの成績、学習曲線、バイアスと分散、例数の目安、分割、交差検証、LOOCV、ブートストラップ | 学習曲線、交差検証の模式図と比較 | `fit_glm`、`roc_curve`、`validate_logistic` |
| 10. [識別と較正](discrimination-calibration.md) | ROC と C 統計量、較正の図と切片・傾き、Brier、決定曲線 | ROC、較正の図、DCA | `roc_curve`、`plot_roc`、`decision_curve_table`、`plot_dca` |
| 11. [正則化](regularization.md)（発展） | ridge、LASSO、λ を交差検証で選ぶ、一回では当てにならないこと、ベイズとのつながり | 係数の経路、繰り返しでの較正の傾き | statract に未実装（numpy で図示） |
| 12. [木、アンサンブル、ニューラルネットワーク](trees-ensembles.md)（発展） | 条件付き推測木（ctree）、ランダムフォレストと勾配ブースティングの紹介、ロジスティック回帰を重ねたニューラルネットワーク | 木の図、ネットワークの模式図 | `conditional_tree`、`plot_tree`（森、ブースティング、ネットワークは numpy） |

### 第 IV 部 統計モデリング

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 13. [確率モデルとしてのロジスティック回帰](probability-model.md) | データが生まれる仕組み、データの作り方の種明かし、モデルから作ったデータでの確認、スプライン、GLM | 生成過程の模式図、硬貨を投げ直したときの出血の人数、模擬データと実データ、直線とスプライン | `fit_glm`、`spline_test` |
| 14. [尤度と二つの推定法：最尤法とベイズ](likelihood.md) | 尤度、対数尤度と曲がり具合、標準誤差、ニュートン法、プロファイル尤度、三つの検定、ベイズ推論（事前分布、事後分布、信用区間）、MCMC、最尤法とベイズの比較、分離と Firth の方法 | 尤度曲線、プロファイル尤度、事前と事後、MCMC の連鎖 | `fit_glm`、`likelihood_ratio_test`（MCMC は numpy） |
| 15. [階層モデル](hierarchical.md) | 施設差、完全・部分・なしのプーリング、縮小、ICC と MOR、施設の順位づけ、ベイズ推論で読む部分プーリング（経験ベイズ） | 施設ごとの推定の縮み、施設の効果の区間、事前と事後 | `fit_mixed`、`glmm_cluster_variance`、`plot_random_effects` |

### 第 V 部 臨床でよく使う解析

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 16. [生存時間解析](survival.md) | 打ち切り、Kaplan–Meier 法、ハザードと人年法、Poisson 回帰、Cox 比例ハザードモデル、セミパラメトリックの意味と部分尤度、加速故障時間（AFT）モデル、モデルの比較と使い分け | 累積出血割合、区間ごとのハザード、リスク集合の模式図 | `plot_survival`、`cox_ph`、`proportional_hazards_test`、`fit_glm`（poisson）、`accelerated_failure` |
| 17. [診断の指標](diagnostic-metrics.md) | 2×2 表、感度と特異度、陽性・陰性的中率と有病割合、尤度比と検査後確率、ROC と Youden 指数、信頼区間、同じ患者での二つの検査の比較、スペクトラムバイアスと検証バイアス、STARD | 有病割合と的中率、ROC 曲線、研究デザインごとの ROC | `threshold_tradeoff`、`roc_curve`、`roc_test`、`mcnemar_test`、`proportion_ci` |

### 数学補

| 項 | 内容 |
|----|------|
| A1. [確率と確率変数](math/probability.md) | 確率空間、確率変数、確率分布（全体） |
| A2. [オッズと対数](math/odds-log.md) | オッズ、オッズ比、指数と対数、ロジット。第 2 章で使う道具 |
| A3. [条件付き確率とベイズの定理](math/conditional-probability.md) | 条件付き独立、陽性的中率、ベイズ推論（第 II 部、第 10 章、第 14 章、第 17 章） |
| A4. [期待値と分散](math/expectation-variance.md) | 繰り返し期待値と g-formula、和の分散、標準誤差、1.96（全体） |
| A5. [微分と最大化](math/derivatives-maximization.md) | 頂上では傾きが 0、曲がり具合と標準誤差、ニュートン法（第 14 章） |
| A6. [ベクトルと行列](math/vectors-matrices.md) | 計画行列、情報行列と分散共分散行列、係数の組み合わせ、LOOCV の近道（第 9、14 章） |
| A7. [確率分布](math/distributions.md) | ベルヌーイと二項、Poisson（二項とのつながり）、負の二項（過分散）、正規、指数と Weibull、ベータ、ガンマ、分布の選び方（第 13、14、16、17 章） |
| A8. [情報量規準（AIC、WAIC）](math/information-criteria.md) | LOOCV の得点、AIC、WAIC と実効的なパラメータの数、正則なモデルと特異なモデル（第 9、11、14 章） |

### 付録

| 項 | 内容 |
|----|------|
| B. [報告ガイドライン](reporting-guidelines.md) | CONSORT、SPIRIT、STROBE、RECORD、TARGET、TRIPOD+AI、STARD、PRISMA と日本語訳 |
| C. [P 値の読み方と書き方](p-value.md) | 統計モデルと検定統計量、モデル全体の合わなさとしての $P$ 値、よくある読み違い、ASA 声明、NEJM の報告指針、NEJM、JAMA、Lancet、BMJ の書き方（大文字か小文字か、斜体、0 の有無、桁数） |

## 文献 {#references}

::: {#refs}
:::

## 略語 {#abbreviations}

各章でも初出で説明します。

| 略語 | 英語 | 日本語 |
|------|------|--------|
| AFT | accelerated failure time | 加速故障時間 |
| AIC | Akaike information criterion | 赤池情報量規準 |
| ASA | American Statistical Association | 米国統計学会 |
| ATE | average treatment effect | 平均処置効果 |
| ATO | average treatment effect in the overlap population | 重なり集団での平均処置効果 |
| ATT | average treatment effect on the treated | 処置群での平均処置効果 |
| AUC | area under the curve | 曲線下面積（ROC 曲線の下の面積） |
| CART | classification and regression trees | 分類回帰木 |
| CI | confidence interval | 信頼区間 |
| ctree | conditional inference tree | 条件付き推測木 |
| CONSORT | Consolidated Standards of Reporting Trials | ランダム化比較試験の報告指針 |
| CV | cross-validation | 交差検証 |
| DAG | directed acyclic graph | 有向非巡回グラフ |
| DCA | decision curve analysis | 決定曲線分析 |
| EPV | events per variable | 1 変数あたりのイベント数 |
| EQUATOR | Enhancing the QUAlity and Transparency Of health Research | 報告指針をまとめる国際的な取り組み |
| ESS | effective sample size | 有効サンプルサイズ |
| GEE | generalized estimating equations | 一般化推定方程式 |
| GLM | generalized linear model | 一般化線形モデル |
| GLMM | generalized linear mixed model | 一般化線形混合モデル |
| HR | hazard ratio | ハザード比 |
| ICC | intraclass correlation coefficient | 級内相関係数 |
| IPTW | inverse probability of treatment weighting | 治療の逆確率による重み付け |
| IPW | inverse probability weighting | 逆確率重み付け |
| IRLS | iteratively reweighted least squares | 反復重み付き最小 2 乗法 |
| LASSO | least absolute shrinkage and selection operator | 係数の絶対値の和に罰をつける正則化 |
| LOOCV | leave-one-out cross-validation | 一つ抜き交差検証 |
| LR | likelihood ratio | 尤度比 |
| MCMC | Markov chain Monte Carlo | マルコフ連鎖モンテカルロ法 |
| MOR | median odds ratio | オッズ比の中央値 |
| NEJM | New England Journal of Medicine | （医学雑誌の名前） |
| NPV | negative predictive value | 陰性的中率 |
| OR | odds ratio | オッズ比 |
| PPV | positive predictive value | 陽性的中率 |
| PRISMA | Preferred Reporting Items for Systematic reviews and Meta-Analyses | 系統的レビューの報告指針 |
| PSIS-LOO | Pareto-smoothed importance sampling leave-one-out cross-validation | パレート平滑化重点サンプリングによる一つ抜き交差検証 |
| RCT | randomized controlled trial | ランダム化比較試験 |
| RECORD | REporting of studies Conducted using Observational Routinely-collected health Data | 診療データを使った観察研究の報告指針 |
| RHC | right heart catheterization | 右心カテーテル |
| RMST | restricted mean survival time | 制限付き平均生存時間 |
| ROC | receiver operating characteristic | 受信者動作特性 |
| RR | risk ratio | リスク比 |
| SD | standard deviation | 標準偏差 |
| SE | standard error | 標準誤差 |
| SMD | standardized mean difference | 標準化差 |
| SPIRIT | Standard Protocol Items: Recommendations for Interventional Trials | 試験計画書の報告指針 |
| STARD | Standards for Reporting of Diagnostic Accuracy Studies | 診断精度研究の報告指針 |
| STROBE | Strengthening the Reporting of Observational Studies in Epidemiology | 観察研究の報告指針 |
| TARGET | TrAnsparent ReportinG of observational studies Emulating a Target trial | 標的試験の模倣の報告指針 |
| TRIPOD+AI | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence | 予測モデル研究の報告指針（回帰と機械学習の両方を扱う 2024 年版） |
| WAIC | widely applicable information criterion | 広く使える情報量規準（渡辺–赤池情報量規準とも） |

**次へ：**[第 1 章 記述：まずデータを見る](descriptive.md)
