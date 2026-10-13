# 付録 B. 報告ガイドライン

研究を論文にするときは、研究デザインごとに決まった**報告ガイドライン**（reporting guideline）に沿って書きます。読者が研究の確からしさを判断するのに必要な情報を、漏れなく書くためのチェックリストです。多くの医学雑誌は、投稿のときにチェックリストの提出を求めます。

読む側にも役に立ちます。論文を読むとき、ガイドラインの項目が書かれているかを見れば、どこに偏りが入りうるかがわかります。報告ガイドラインは EQUATOR Network（Enhancing the QUAlity and Transparency Of health Research）のサイトにまとまっています。

[← 目次](index.md) · [← 数学補 A8](math/information-criteria.md)

## B.1 問いとデザインから選ぶ {#choose}

| 研究 | ガイドライン | 本書の関連する章 |
|------|--------------|------------------|
| ランダム化比較試験 | CONSORT 2025 | [第 1 章](descriptive.md#pvalue) |
| ランダム化比較試験の計画書 | SPIRIT 2025 | — |
| 観察研究（コホート、症例対照、横断） | STROBE | 第 1–7 章 |
| 診療データベースを使った観察研究 | RECORD | 第 2–7 章 |
| 標的試験を模倣した観察研究 | TARGET | [第 3 章](causal-basics.md)、[第 7 章](assumptions-sensitivity.md#consistency) |
| 予測モデルの開発と検証 | TRIPOD+AI | 第 8–12 章 |
| 診断精度の研究 | STARD 2015 | [第 17 章](diagnostic-metrics.md) |
| 系統的レビューとメタ解析 | PRISMA 2020 | — |

一つの研究に二つ以上が当てはまることもあります。たとえば、電子カルテのデータで治療効果を推定する研究なら、STROBE と RECORD、標的試験の形で設計したなら TARGET も見ます。

## B.2 ランダム化比較試験：CONSORT と SPIRIT {#consort}

**CONSORT**（Consolidated Standards of Reporting Trials）は、ランダム化比較試験の報告指針です。2025 年に改訂されました [@Hopewell2025-el]。割付けの方法、盲検化、参加者の流れ図（何人が組み入れられ、何人が脱落したか）、事前に決めた主要アウトカムなどを書きます。第 1 章で見たように、Table 1 で群の差を検定しないことも求めています。

**SPIRIT**（Standard Protocol Items: Recommendations for Interventional Trials）は、試験を始める前の計画書（プロトコル）の指針です。CONSORT と同じ年に改訂されました [@Chan2025-kx]。

## B.3 観察研究：STROBE と RECORD {#strobe}

**STROBE**（Strengthening the Reporting of Observational Studies in Epidemiology）は、コホート研究、症例対照研究、横断研究の報告指針です [@von-Elm2007-wg]。対象集団の選び方、曝露とアウトカムの定義、交絡因子とその選び方、欠測の扱いなどを書きます。本書の第 I–II 部の解析は、多くがこの型の研究です。

**RECORD**（REporting of studies Conducted using Observational Routinely-collected health Data）は、STROBE を、レセプトや電子カルテなど診療のために集めたデータの研究向けに拡張したものです [@Benchimol2015-qo]。データベースの中身、患者や病名を拾い出すコードとアルゴリズム、データの結合の方法などを加えて書きます。

## B.4 標的試験の模倣：TARGET {#target}

**TARGET**（TrAnsparent ReportinG of observational studies Emulating a Target trial）は、観察データで「もし試験をしたら」を模倣する研究（標的試験の模倣、target trial emulation）の報告指針です。2025 年に発表されました [@Cashin2025-mp]。模倣したい試験の計画（対象、治療の定義、追跡の開始、アウトカム、推定したい量）を先に書き、観察データでそれをどう再現したかを書きます。第 7 章の一致性で見たように、「何を治療とするか」を先に決めることは、因果を推定する研究の土台です。

## B.5 予測モデル：TRIPOD+AI {#tripod}

**TRIPOD+AI**（Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence）は、予測モデルの開発と検証の報告指針です [@Collins2024-qy]。2024 年に、回帰モデルと機械学習モデルの両方を扱う形に改訂されました。予測因子の選び方、例数、欠測の扱い、過学習への対処（第 9 章）、識別と較正（第 10 章）、内部検証と外部検証を書きます。

## B.6 診断精度：STARD {#stard}

**STARD**（Standards for Reporting of Diagnostic Accuracy Studies）は、診断精度研究の報告指針です [@Bossuyt2015-ea]。患者の集め方、参照基準、2×2 表、感度や特異度とその信頼区間を書きます。主な項目は[第 17 章](diagnostic-metrics.md)にまとめています。

## B.7 系統的レビュー：PRISMA {#prisma}

**PRISMA**（Preferred Reporting Items for Systematic reviews and Meta-Analyses）は、系統的レビューとメタ解析の報告指針です [@Page2021-bp]。文献の探し方、選び方の流れ図、研究ごとの偏りの評価、結果の統合の方法を書きます。

## B.8 日本語訳 {#japanese}

| ガイドライン | 日本語訳 |
|--------------|----------|
| CONSORT | 2025 年版の訳は確認できませんでした。2010 年版はライフサイエンス出版のサイトにあります |
| SPIRIT | 2025 年版の公式の訳は確認できませんでした |
| STROBE | 上岡洋晴、津谷喜一郎による訳が EQUATOR Network にあります |
| RECORD | 奥山ら（医療の質・安全学会誌 2017;12(4):413–417） |
| TARGET | 確認できませんでした |
| TRIPOD+AI | 有志による訳があります（佐藤ら、チェックリストの拡張版） |
| STARD | 2015 年版の訳は確認できませんでした |
| PRISMA | 上岡ら（薬理と治療 2021;49(6):831–842）。PRISMA のサイトに PDF があります |

日本語訳は、EQUATOR Network の翻訳の一覧と、日本医学会の「医学雑誌編集ガイドライン 2022」にもまとまっています。

## 文献 {#references}

::: {#refs}
:::

## まとめ

- 報告ガイドラインは、研究デザインごとに書くべき項目を決めたチェックリストです。
- 試験は CONSORT、観察研究は STROBE（診療データなら RECORD、標的試験の模倣なら TARGET）、予測モデルは TRIPOD+AI、診断精度は STARD、系統的レビューは PRISMA です。
- 論文を読むときも、項目が書かれているかで研究の確からしさを判断できます。

**次へ：**[付録 C P 値の書き方](p-value-format.md)
