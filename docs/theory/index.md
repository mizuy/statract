# 臨床の問いから始める統計

臨床研究に使う統計の考え方を、高校数学から学び直すテキストです。手法の一覧ではありません。「この研究で何を知りたいのか」から出発し、因果推論、予測、統計モデリングを一つの枠組みで理解することを目指します。

関数の選び方だけを知りたいときは [手法の選び方](../guide/methods.md) を見てください。

## 中心のメッセージ

!!! note ""
    統計手法から研究を考えるのではなく、まず何を知りたいのかを決める。その問いが**因果**なのか**予測**なのかを区別する。そのうえで、合ったモデルと解析方法を選ぶ。

臨床家にとっての重要度は、おおむね次の順と考えます。

**因果推論 > 機械学習・予測 > 統計モデリング**

ただし三つは別々の分野ではありません。どれもロジスティック回帰と確率モデルを共通の言葉としてつながっています。

## 通しの例

全章で同じ例を使います。**大腸内視鏡治療後の遅発性出血**と**予防的クリップ閉鎖**です。同じデータに、次の三つの違う問いを繰り返し投げます。

| 問い | 種類 | 扱う部 |
|------|------|--------|
| 遅発性出血の「独立した危険因子」は何か | あいまい（分けて考える） | 第 I 部 |
| 予防的クリップは出血を減らすか | 因果 | 第 II 部 |
| この患者は出血するか | 予測 | 第 III 部 |

データは**シミュレーション**です。どの変数がどれに影響するか（DAG: directed acyclic graph、有向非巡回グラフ）をコードに書いて作っているので、各章で「推定した値」と「本当の値」を比べられます。作り方は [第 1 章](independent-risk-factor.md#data) と [`examples/theory_bleeding`](https://github.com/mizuy/statract/tree/main/examples/theory_bleeding) にあります。

## 読み方

- 本文は**高校数学まで**を前提にします。それを超える数学は [数学補](math/odds-log.md) にまとめ、一段ずつ説明します。
- 本文では式の導出より、「その式が何を意味するか」「何をしているか」を重視します。
- 図は statract で描き、コードを添えます。

## 目次

章のリンクがないものは、これから書く章です。図と関数は予定です。

### 第 I 部 導入

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 1. [「独立した危険因子」とは何か](independent-risk-factor.md) | 慣習的な解析（単変量 → 多変量）、ロジスティック回帰の基本、因果か予測か | ロジスティック曲線、forest、層別の出血率 | `fit_glm`、`tableone`、`plot_forest` |

### 第 II 部 因果推論

理論は短く紹介し、詳しくは文献に譲ります。中心は、背景を調整する方法の使い方です。

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 2. [因果推論の考え方](causal-basics.md) | Rubin の反事実モデル、交換可能性、DAG とバックドア基準、文献 | 群ごとの潜在アウトカム、DAG | `prop_test` |
| 3. [背景を調整する方法](adjustment-methods.md) | 限定、層別化、回帰と標準化、マッチング、重み付けの比較 | 方法ごとの推定値と本当の値 | `standardize_glm`、`match_sample`、`propensity_weights` |
| 4. [傾向スコアマッチング](propensity-score-matching.md) | バランシングスコア、傾向スコア、重なり、キャリパー、バランスの確認 | 傾向スコアの分布、Love plot、マッチ前後の Table 1 | `match_sample`、`plot_love` |
| 5. [IPTW（逆確率重み付け）](iptw.md) | 重みで仮の集団を作る、g-formula と同じ量になる理由、安定化、極端な重み、推定する対象 | 重みの分布、重み付きの Love plot、重みごとの推定値 | `propensity_weights`、`plot_love` |
| 6. [仮定と感度分析](assumptions-sensitivity.md) | 正値性、未測定交絡、E-value、一致性、Bradford Hill の視点 | 未測定交絡の影響、E-value の図 | `standardize_glm` |

### 第 III 部 予測と機械学習

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 7. [予測モデルとしてのロジスティック回帰](prediction-model.md) | 同じ式を予測に使う、係数より予測確率、因果でない変数も予測に役立つ | 予測確率の分布 | `fit_glm` |
| 8. [過学習と汎化](overfitting.md) | 見かけの成績、学習曲線、例数の目安、分割、交差検証、LOOCV、AIC と WAIC、ブートストラップ | 学習曲線、楽観度 | `validate_logistic` |
| 9. [識別と較正](discrimination-calibration.md) | ROC と C 統計量、較正の図と切片・傾き、Brier、決定曲線 | ROC、較正の図、DCA | `roc_curve`、`plot_roc`、`threshold_tradeoff`、`decision_curve_table` |
| 10. [正則化](regularization.md) | ridge、LASSO、λ を交差検証で選ぶ、一回では当てにならないこと、ベイズとのつながり | 係数の経路、繰り返しでの較正の傾き | statract に未実装（numpy で図示） |
| 11. [木、アンサンブル、ニューラルネットワーク](trees-ensembles.md) | 条件付き推測木（ctree）、ランダムフォレストと勾配ブースティングの紹介、ロジスティック回帰を重ねたニューラルネットワーク | 木の図、ネットワークの模式図 | `conditional_tree`、`plot_tree`（森、ブースティング、ネットワークは numpy） |

### 第 IV 部 統計モデリング

| 章 | 内容 | 主な図 | statract |
|----|------|--------|----------|
| 12. 確率モデルとしてのロジスティック回帰 | $Y_i \sim \mathrm{Bernoulli}(p_i)$、データの生成過程 | 生成過程の模式図 | `fit_glm` |
| 13. 尤度と最尤法 | 尤度、対数尤度、最尤推定、標準誤差 | 尤度曲線 | `fit_glm`、`likelihood_ratio_test` |
| 14. ベイズ推論 | 事前分布、事後分布、事前の強さ | 事前と事後 | — |
| 15. 事後予測分布 | 予測の不確かさ、モデルの確認 | 事後予測チェック | — |
| 16. 階層モデル | 施設差、部分プーリング | 施設ごとの推定の縮み | `fit_mixed`、`plot_random_effects` |

### 数学補

| 項 | 内容 |
|----|------|
| A1. [確率、オッズ、対数](math/odds-log.md) | 第 1 章で使う道具 |
| A2. 条件付き確率とベイズの定理 | 第 II 部、第 IV 部 |
| A3. 期待値と分散 | 全体 |
| A4. 微分と最大化 | 最尤法 |
| A5. ベクトルと行列 | 多変量のモデル |

## 略語 {#abbreviations}

各章でも初出で説明します。

| 略語 | 英語 | 日本語 |
|------|------|--------|
| AIC | Akaike information criterion | 赤池情報量規準 |
| ATE | average treatment effect | 平均処置効果 |
| ATO | average treatment effect in the overlap population | 重なり集団での平均処置効果 |
| ATT | average treatment effect on the treated | 処置群での平均処置効果 |
| AUC | area under the curve | 曲線下面積（ROC 曲線の下の面積） |
| CI | confidence interval | 信頼区間 |
| CV | cross-validation | 交差検証 |
| DAG | directed acyclic graph | 有向非巡回グラフ |
| DCA | decision curve analysis | 決定曲線分析 |
| EPV | events per variable | 1 変数あたりのイベント数 |
| ESS | effective sample size | 有効サンプルサイズ |
| IPTW | inverse probability of treatment weighting | 治療の逆確率による重み付け |
| LASSO | least absolute shrinkage and selection operator | （係数の絶対値に罰をつける正則化） |
| LOOCV | leave-one-out cross-validation | 一つ抜き交差検証 |
| OR | odds ratio | オッズ比 |
| RCT | randomized controlled trial | ランダム化比較試験 |
| ROC | receiver operating characteristic | 受信者動作特性 |
| RR | risk ratio | リスク比 |
| SMD | standardized mean difference | 標準化差 |
| TRIPOD | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis | 予測モデル研究の報告指針 |
| WAIC | widely applicable information criterion | 広く使える情報量規準（渡辺–赤池情報量規準とも） |

## これから作るもの

- 残りの章（上の順に書きます）
- 必要な所の対話型の図
- 本文と同じ図と記法を使ったスライド
