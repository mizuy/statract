# cox_retinopathy_discussion

数値は [cox_retinopathy_results.md](cox_retinopathy_results.md) の生成物を読む。再集計しない。

## CQ への答え

197 人・394 眼で、レーザー治療眼は対照眼より視力喪失ハザードが低い（log-rank 22.25、df=1、p≈2.4×10⁻⁶）。クラスター Cox の treated 対 control の HR は約 0.46（sandwich 95% CI 約 0.34–0.62）。教学データの reproduction であり、現行の光凝固適応を更新しない。

## 当たり外れ

患者内無作為化のため、治療 HR の sandwich SE がモデルベースより小さくなることがある（ペア情報）。病型など患者単位共変量では比が 1 を超えやすい。`cox_se_compare.csv` がその対比である。

## 限界

- 眼を独立とみなした KM / log-rank は記述。推論の主はクラスター Cox。
- counting-process（entry>0）ではこの実装は sandwich を掛けない。
- LGPL パッケージ由来の教学エクスポートである。
