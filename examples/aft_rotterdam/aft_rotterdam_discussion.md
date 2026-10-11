# aft_rotterdam_discussion

数値は [aft_rotterdam_results.md](aft_rotterdam_results.md) の生成物を読む。再集計しない。

## CQ への答え

死亡アウトカム、n=2982。log-rank はホルモン療法群で差がある（統計量 23.7、p≈1.1×10⁻⁶）。調整 Cox の `hormon` HR は約 0.95（95% CI 約 0.80–1.13）で 1 と区別しにくい。節数・腫瘍径・グレード・年齢は HR>1。`proportional_hazards_test` の全体 p≈3.8×10⁻⁴（主に `age`）。Weibull AFT は収束しなかった。収束した lognormal AFT を併記する。観察コホートの教学 reproductions であり、治療適応の更新を主張しない。

## 当たり外れ

未調整 KM と調整 Cox の向きが食い違うのは交絡（ホルモン療法が予後不良側に偏る）として読む。zph 全体が有意なので単一 HR だけに頼らない、という予定どおりの読み方をする。

## 限界

- 再発は競合になりやすいので使っていない。
- AFT の分布は Weibull を第一選択。収束しないときは lognormal を併記。
- 未測定交絡。LGPL 教学データ。
