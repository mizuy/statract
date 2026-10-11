# matching

傾向スコアと距離に基づくマッチングです。公開名は `match_sample` です。`total_distance` はマッチした辺の距離の和で、full matching の重みは MatchIt の `normalize=TRUE` に合わせています。

## 実装メモ: SMD

`balance()` の `smd_all` と `smd_matched` は、重み付けのバランス表と同じ SMD です。既定の分母は、マッチング前の全標本の pooled SD $\sqrt{(s_1^2 + s_0^2)/2}$ で、マッチ前とマッチ後で同じ値を使います。分母を固定するので、SMD の変化は平均の差の変化を表します。2 値の変数は分散 $p(1-p)$ で標準化します（Austin 2009）。`pair_distance` も同じ分母で割ります。`distance` の行は常に標準化します。

MatchIt は ATT の分母に処置群の標準偏差を使います。ATT のマッチングでは処置群がほぼそのまま残るので、目標集団の単位の固定した尺度になるためです。これは `balance(sd_denominator="treated")`（ATC は `"control"`）で再現できます。cobalt のように 2 値を割合の差のまま出すには `binary="raw"` です。tableone はマッチ後の表の中で pooled SD を計算し直すので、マッチ後の Table 1 の SMD は `smd_matched` と一致しないことがあります。理由と対応表は [weighting の実装メモ](weighting.md#smd) にあります。

::: statract.models.matching
