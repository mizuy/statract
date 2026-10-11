# cea_sicksicker_discussion

数値は [cea_sicksicker_results.md](cea_sicksicker_results.md) の生成物を読む。再集計しない。

## CQ への答え

仮想 25 歳コホート、76 サイクル。基本ケース費用 / QALY は SoC $1.506\times 10^5$ / 21.83、B $2.524\times 10^5$ / 23.27、AB $3.662\times 10^5$ / 24.14、A $2.783\times 10^5$ / 22.56。A は強支配（D）。B vs SoC の ICER は $7.081\times 10^4$/QALY、AB vs B は $1.307\times 10^5$。WTP $100{,}000$ では B の NMB が最大（$2.075\times 10^6$）。PSA（n=1000）の CEAC は同 WTP で B 0.762。EVPI は約 $3.047\times 10^3$ 人あたり。

## 当たり外れ

遷移は行和 1。A は効用を足すが進行を止めないので B に支配される、という読みと ICER 表は一致する。

## 限界

- 仮想疾患。生命表・校正・実費用は無い
- 論文の年齢依存死亡、tunnels、遷移報酬、半サイクル補正は核が持たないため入れていない
- tornado は WTP $100,000/QALY における**最適戦略の NMB**（4 戦略の max）であり、単一 ICER の OWSA ではない
- PSA の効用は順序制約（H ≥ trtA ≥ S1 ≥ S2）を入れた
