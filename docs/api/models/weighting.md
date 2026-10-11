# weighting

二値の処置の逆確率重み付け（IPTW）です。公開名は `propensity_weights`、`PropensityWeights`、`balance_table` です。

- 傾向スコアは `fit_glm` のロジスティック回帰です。重みの式、`stabilize`、`trim` は `WeightIt::weightit(method = "glm")` と `WeightIt::trim` に合わせています。
- `balance()` と `balance_table` は `cobalt::bal.tab` と同じ行と列の表です。因子は水準ごとに分けます。差は 2 値も連続も SMD で、連続には分散比も出します。SMD の定義は下の「実装メモ: SMD」です。
- `effective_sample_size()` は群ごとの Kish の有効標本サイズ `(Σw)² / Σw²` です。
- 図は [`plot_love`](../viz/balance.md) です。

アウトカムのモデルはここにありません。`frame()` の `weights` 列を `fit_glm(..., weights="weights")` と `hc_covariance(fit, "HC0")`、または `cox_ph(..., weights="weights")` に渡します。使い方は [Models](../../models/overview.md) の「逆確率重み付け」です。

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
