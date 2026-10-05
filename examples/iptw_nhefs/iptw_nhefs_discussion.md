# iptw_nhefs_discussion

数値は [iptw_nhefs_results.md](iptw_nhefs_results.md) の生成物を読む。再集計しない。

## CQ への答え

完全例 n=1566、P(qsmk=1)≈0.26。安定化 IPTW（ATE、上限 10、実際に cap された重みは 0、HC3）では禁煙の体重変化差は約 +3.28 kg（95% CI 約 2.25–4.31）。未調整は約 +2.54 kg。ライブラリに `iptw()` は無く、重みは例スクリプトの手計算である。教学 reproductions であり、禁煙指導の体重効果の確定推定ではない。

## 当たり外れ

「禁煙後に体重が増える」方向は教科書の NHEFS 例と一致する。IPTW 後も差は残る。

## 限界

- ATT は出していない（ユーザー決定）。
- CEM はバランス感度であり、実装が ATT 風の重みなので因果効果として読まない。
- Positivity と切り詰め。未測定交絡。fetch-only データ。
