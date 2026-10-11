# psm_rhc_discussion

数値は [psm_rhc_results.md](psm_rhc_results.md) の生成物を読む。

## CQ への答え

完全例 5735 人、マッチ後（weights>0）3350 人（治療 1675）。未マッチの 30 日死亡 OR は約 1.39、マッチ後は約 1.34（いずれも `rhc` の OR>1）。Love plot と `balance.csv` で SMD の縮小を見る。教学用 ATT 近似であり、RHC 適応の根拠にはしない。

## 当たり外れ

観察された交絡を nearest/logit で揃えても、30 日死亡との正の関連は残った。Connors JAMA 1996 の問題意識（観察下での有害関連）と方向は矛盾しない。

## 限界

- 未測定交絡と適応による交絡が残る。
- 標準化キャリパー 0.2 で落ちる症例は ATT から外れる。
- データは OSI ライセンスが薄い公開教学 CSV であり、再配布しない。
