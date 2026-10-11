# weighting

二値の処置の逆確率重み付け（IPTW）です。公開名は `propensity_weights`、`PropensityWeights`、`balance_table`、`weighted_glm`、`WeightedGLM`、`weighted_effect` です。

- 傾向スコアは `fit_glm` のロジスティック回帰です。重みの式、`stabilize`、`trim` は `WeightIt::weightit(method = "glm")` と `WeightIt::trim` に合わせています。
- `balance()` と `balance_table` は `cobalt::bal.tab` と同じ行と列の表です。因子は水準ごとに分けます。差は 2 値も連続も SMD で、連続には分散比も出します。SMD の定義は下の「実装メモ: SMD」です。
- `effective_sample_size()` は群ごとの Kish の有効標本サイズ `(Σw)² / Σw²` です。
- 図は [`plot_love`](../viz/balance.md) です。

- アウトカムのモデルは `weighted_glm`（`WeightIt::glm_weightit`）と `PropensityWeights.effect` です。既定の分散は傾向スコアの推定を含めます（下の「アウトカムのモデルと分散」）。
- Cox モデルは `frame()` の `weights` 列を `cox_ph(..., weights="weights")` に渡します。使い方は [Models](../../models/overview.md) の「逆確率重み付け」です。

## アウトカムのモデルと分散

```python
from statract import propensity_weights, weighted_glm

ipw = propensity_weights(frame, "clip ~ age + size_mm + anticoag", estimand="ATE")
ipw.effect("bleed", measure=["difference", "ratio", "odds_ratio"])  # 周辺効果
ipw.effect("bleed", vcov="bootstrap", n_boot=2000, seed=1)  # PS を毎回推定し直す
fit = weighted_glm(ipw, "bleed ~ clip + age", family="binomial", cluster="hospital")
fit.tidy()
```

- `weighted_glm(weights, formula, family=, vcov=, cluster=)` は、重み × サンプリング重みを事前重みにした GLM です。データは `weights.data` の重みのある行です。アウトカムの式で行が落ちるとエラーです。
- `PropensityWeights.effect(outcome, measure=)`（関数は `weighted_effect`）は、群ごとの重み付き平均 `mean_treated`、`mean_control` と、その差（`"difference"`）、比（`"ratio"`）、オッズ比（`"odds_ratio"`）です。推定対象は重みの推定対象です。飽和モデル `y ~ treat` と同じで、2 値のアウトカムの差は重み付き線形モデルのリスク差です。比とオッズ比の検定と区間は対数尺度で、`std_error` も対数の SE です。

`vcov` は 3 通りです。

| `vcov` | 中身 | R |
| --- | --- | --- |
| `"asympt"`（既定） | M 推定。アウトカムの推定方程式を傾向スコアのロジスティック回帰のスコアと重ね、全体のサンドイッチ $A^{-1} B A^{-\top}$ の左上を取る | `glm_weightit(vcov = "asympt")` |
| `"HC0"` | アウトカムの部分だけ。重みを既知とみなす | `glm_weightit(vcov = "HC0")`、`sandwich::vcovHC(type = "HC0")` |
| `"bootstrap"` | 行（`cluster` があればクラスタ）を復元抽出し、毎回傾向スコア、重み、アウトカムを推定し直す。SE は複製の標準偏差、区間はパーセンタイル | `glm_weightit(vcov = "BS")` に近い（区間の作り方が違う） |

- $A$ の交差項は、重みを傾向スコアの係数で微分したものです。$\partial w / \partial \gamma = (\partial w / \partial e)\, e(1-e)\, z$ で、$\partial w / \partial e$ は ATE が処置群 $-1/e^2$、対照群 $1/(1-e)^2$、ATT が対照群 $1/(1-e)^2$（処置群 0）、ATC が処置群 $-1/e^2$（対照群 0）、ATO が処置群 $-1$、対照群 $1$ です（Lunceford と Davidian 2004）。
- `stabilize=True` の処置群の割合 $p_1$ も推定値として重ねます（推定方程式 $s(a - p_1)$）。
- `cluster` はクラスタの中で推定方程式を足し、$G/(G-1)$ を掛けます（`sandwich::vcovCL` の `cadjust`）。
- ATE では、傾向スコアの推定を含めた SE は HC0 より小さくなります。ATT などでは大きくなることもあります。
- 切り詰めた重み（`trim=`）は傾向スコアの係数で滑らかに動かないので、`vcov="asympt"` はエラーです。`"HC0"` と `"bootstrap"`（毎回切り詰め直す）は使えます。
- ブートストラップは `numpy.random.default_rng(seed)` です。片方の群がない複製は引き直します。

## 実装メモ: SMD

statract のバランス表（`PropensityWeights.balance()`、`balance_table`、`MatchedSample.balance()`）は、既定で同じ SMD を使います。

$$
\mathrm{SMD} = \frac{\bar x_1 - \bar x_0}{\sqrt{(s_1^2 + s_0^2)/2}}
$$

- 分散 $s_1^2, s_0^2$ は調整前（重み付けやマッチングの前）の標本で一度だけ計算し、調整前と調整後の行で同じ分母を使います。サンプリング重みは残します。
- 2 値の変数も標準化し、分散は $p(1-p)$ です（Austin 2009）。tableone の 2 水準カテゴリの式と同じです。
- 推定対象（ATE、ATT、ATC、ATO）で既定は変わりません。

分母を固定するのは、調整前後の SMD の変化を平均の差の変化として読むためです。調整後の標本で分散を計算し直すと、分散が変わっただけで SMD が動きます。

R のパッケージとの違いと、その理由です。

- **MatchIt** は ATT の分母に処置群の標準偏差を使います（ATC は対照群）。ATT のマッチングでは処置群がほぼそのまま残るので、目標集団の単位で測った固定の尺度になります。
- **cobalt** は 2 値を割合の差のまま出します。2 値の標準偏差は割合で決まるので、生の差のほうが読みやすいという考えです（Greifer）。ATE の分母は調整前の pooled SD です。ATT は処置群、ATC は対照群、ATO は重み付き全標本の標準偏差です。
- **tableone** は渡された表の中で pooled SD を計算し直します。マッチ後の標本で作った Table 1 の SMD は、分母が違うのでバランス表の `smd_matched` / `diff_adjusted` と一致しないことがあります。

R の流儀はオプションで再現できます。

| 再現したいもの | statract の指定 |
| --- | --- |
| cobalt `bal.tab`（ATE） | `balance(binary="raw", sd_denominator="pooled")` |
| cobalt `bal.tab`（ATT / ATC） | `balance(binary="raw", sd_denominator="treated")` / `"control"` |
| cobalt `bal.tab`（ATO） | `balance(binary="raw", sd_denominator="weighted")` |
| MatchIt `summary`（ATT / ATC） | `MatchedSample.balance(sd_denominator="treated")` / `"control"` |
| tableone `ExtractSmd`（調整前） | 既定のまま（調整前の行が一致します） |

`sd_denominator` には `"all"`（群を分けない全標本）もあります。`continuous="raw"` は連続変数を標準化しません。

::: statract.models.weighting

::: statract.models.weighting_outcome
