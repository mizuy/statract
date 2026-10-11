# 予測モデルのブートストラップ検証: 指標、ロジスティック当てはめ、生存の較正

対象: `src/statract/models/validation.py` の `_surv_concordance`、`_lrm_fit`、`_logistic_index`、`_emax`、`_cox_index`、`_groupkm`、`_calibration_gap`（補助の `_binary_deviance`、`_km_at`、`_cut_groups` を含む）。

## 1. 目的

- 予測モデルの性能指標を、見かけ（apparent）、訓練（training）、検証（test）の 3 通りで計算する。楽観度（optimism）は訓練の平均から検証の平均を引いたもの。補正済み指標は見かけの値から楽観度を引く（Efron 1983 の楽観度ブートストラップ）。
- 公開関数 `validate_logistic`、`calibrate_logistic`、`validate_cox`、`calibrate_cox`、`somers_dxy` の内部。出力は R の `rms::validate()`（`lrm`、`cph`）、`rms::calibrate()`（`cmethod = "KM"` を含む）、`Hmisc::somers2()` と比べる。

## 2. 出典

- Harrell FE Jr (2015). *Regression Modeling Strategies*, 2nd ed. Springer. 4.10 節（予測精度の指標、Nagelkerke $R^2$、$D$、$U$、$Q$、Brier、$g$）、5.3 節（ブートストラップによるモデル検証）、10.9 節（ロジスティックモデルの検証、較正の切片と傾き、Emax）、20.9 節と 20.6 節（Cox モデルの検証と較正）。
- Harrell FE, Lee KL, Mark DB (1996). Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Statistics in Medicine* 15:361–387. 第 5 節（判別、較正、Dxy、$D$、$U$、$Q$）。5.5 節（打ち切りのある一致指数）。
- Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA (1982). Evaluating the yield of medical tests. *JAMA* 247:2543–2546. 打ち切りデータの一致指数 C。
- Efron B (1983). Estimating the error rate of a prediction rule: improvement on cross-validation. *Journal of the American Statistical Association* 78:316–331. 楽観度ブートストラップ。
- Steyerberg EW (2019). *Clinical Prediction Models*, 2nd ed. Springer. 第 15 章（較正の切片と傾き、Brier）。
- Kaplan EL, Meier P (1958). Nonparametric estimation from incomplete observations. *JASA* 53:457–481. 積極限推定量と Greenwood の分散。
- McCullagh P, Nelder JA (1989). *Generalized Linear Models*, 2nd ed. 4.4 節（ロジスティック回帰の Newton–Raphson）。Hosmer DW, Lemeshow S (2000). *Applied Logistic Regression*, 2nd ed. 1.2 節。
- Nagelkerke NJD (1991). A note on a general definition of the coefficient of determination. *Biometrika* 78:691–692.
- David HA (1968). Gini's mean difference rediscovered. *Biometrika* 55:573–575.

## 3. モデルと式

記号: 標本の大きさ $n$。線形予測子 $\eta_i$（xb）。二値の結果 $y_i$。$\operatorname{expit}(t) = 1/(1+e^{-t})$。

### 3.1 一致指数

二値の結果: Somers の $D_{xy} = 2(C - 1/2)$。$C$ は症例と対照の組のうち、症例の予測が高い割合（同順位は 1/2）。

打ち切りのある結果（Harrell ほか 1982、Harrell, Lee & Mark 1996, 5.5 節）: 組 $(i, j)$ は、短い方の時間がイベントのとき使える。イベント側のリスクが高ければ一致 $C$、低ければ不一致 $D$、等しければリスク同順位 $T$。

$$D_{xy} = \frac{C - D}{C + D + T}$$

これは同順位を 1/2 と数えた $2(c - 1/2)$ と同じ。

### 3.2 ロジスティックの最尤推定

対数尤度 $\ell(\beta) = \sum_i \{y_i \eta_i - \log(1 + e^{\eta_i})\}$、$\eta = X\beta$（切片列つき）。スコア $U = X^\top(y - \mu)$、情報行列 $I = X^\top W X$、$W = \operatorname{diag}\{\mu_i(1-\mu_i)\}$（McCullagh & Nelder 1989, 4.4 節）。Newton–Raphson の更新は $\beta \leftarrow \beta + I^{-1} U$。

### 3.3 ロジスティックの指標（Harrell 2015, 4.10 節・10.9 節）

$L_0$ を切片だけのモデルの $-2\log L$、$\mathrm{LR}$ をモデルの尤度比統計量とする。

- Nagelkerke (1991) の $R^2 = \{1 - \exp(-\mathrm{LR}/n)\}/\{1 - \exp(-L_0/n)\}$。
- 較正の切片 $a$ と傾き $b$: $y$ を $\eta$ に回帰したロジスティックモデルの係数（Steyerberg 2019, 第 15 章）。
- 判別指数 $D = (\mathrm{LR} - 1)/n$。
- 信頼性の欠如 $U = (\mathrm{LR}_{\mathrm{cal}} - 2)/n$。$\mathrm{LR}_{\mathrm{cal}}$ は「$a = 0, b = 1$」対「$a, b$ を推定」の尤度比統計量。
- 品質指数 $Q = D - U$。
- Brier スコア $B = n^{-1}\sum(\operatorname{expit}\eta_i - y_i)^2$。
- $g$: 線形予測子の Gini 平均差。$g_p$: 予測確率の Gini 平均差（David 1968）。
- Emax: 補正済みの較正直線について $\max_{0 \le p \le 1} |p - \operatorname{expit}(a + b\operatorname{logit} p)|$。

### 3.4 Cox モデルの指標（Harrell 2015, 20.9 節）

$\ell(\beta)$ を部分対数尤度、$L_0 = -2\ell(0)$、$\mathrm{LR} = 2\{\ell(\hat\beta) - \ell(0)\}$。

- $R^2 = \{1 - \exp(-\mathrm{LR}/n)\}/\{1 - \exp(-L_0/n)\}$。$n$ は被験者数。
- 傾き: 見かけと訓練では 1。検証では $\eta$ だけを共変量にした Cox モデルの係数 $\hat\gamma$。
- $D = (\mathrm{LR} - 1)/L_0$。打ち切りデータでは $n$ の代わりに $L_0$ で割る。
- $U$: 見かけと訓練では $-2/L_0$。検証では $\mathrm{LR}_{\mathrm{cal}}/L_0$、$\mathrm{LR}_{\mathrm{cal}} = 2\{\ell(\hat\gamma) - \ell(1)\}$（8 節を参照）。
- $Q = D - U$。$g$ は $\hat\gamma\,\eta$ の Gini 平均差。

### 3.5 生存確率の較正（Harrell 2015, 20.6 節）

時点 $u$ での予測生存確率 $\hat S_i(u)$ を区間に分ける。各群で Kaplan–Meier 推定値（Kaplan & Meier 1958）

$$\hat S(u) = \prod_{t \le u} (1 - d_t/r_t)$$

と、Greenwood の式による $\log \hat S(u)$ の標準誤差

$$\mathrm{SE} = \Bigl\{\sum_{t \le u} \frac{d_t}{r_t (r_t - d_t)}\Bigr\}^{1/2}$$

を出す。$d_t$ は時点 $t$ のイベント数、$r_t$ はリスク集合の大きさ。差は「KM − 平均予測」（`what = "observed-predicted"`）または KM そのもの（`"observed"`）。

## 4. 推定・計算の手順

### 4.1 `_surv_concordance`

1. 時間の降順に並べ、同じ時間の塊ごとに処理する。
2. 各塊では、まず打ち切り例をリスク順位の Fenwick 木（二分索引木）に入れる。打ち切り時間とイベント時間が同じなら打ち切りを長い方とみなすため。
3. 次に塊のイベント例ごとに、木の中（自分より長い時間の人）で順位が自分より低い数を $C$、等しい数を $T$、高い数を $D$ に足す。
4. 最後に塊のイベント例を木に入れる。同時刻の 2 つのイベントは組にならない。
5. 計算量は $O(n \log n)$。

### 4.2 `_lrm_fit`

1. 開始値は切片だけのモデル: $\beta_0 = \operatorname{logit}\bar y$、傾き 0。
2. Newton–Raphson の 1 歩を計算する。$-2\log L$ が `eps/10` より大きく悪化したら歩幅を半分にする（最大 30 回）。この半減の規則は fixture の出力に合うように選んだ。
3. $-2\log L$ の変化が `eps`（既定 0.025）未満になったら収束と判定する。
4. 収束を判定した位置でもう一度 Newton の 1 歩を計算し、それを適用した値を推定値とする。$-2\log L$ もその点で計算し直す。この追加の 1 歩は fixture の出力に合うように選んだ。
5. 特異性: 情報行列を対角で正規化し、最小固有値が `tol` 未満なら失敗。

### 4.3 `_logistic_index`

- 当てはめ（`fit`）があるとき（見かけ、訓練）: $\mathrm{LR}$ と $L_0$ はその当てはめのもの。$a = 0$、$b = 1$、$\mathrm{LR}_{\mathrm{cal}} = 0$。
- ないとき（検証）: $y$ を固定した $\eta$ に回帰し直す（`_lrm_fit` を既定の設定で使う）。$\mathrm{LR}$、$L_0$、$a$、$b$ はこの再較正の当てはめから取る。$\mathrm{LR}_{\mathrm{cal}}$ は「$\eta$ をそのまま使った $-2\log L$」から再較正の $-2\log L$ を引いたもの。$g$ と $g_p$ は再較正した $a + b\eta$ で計算する。Brier は再較正しない $\operatorname{expit}\eta$ で計算する。
- 予測変数が 1 つのとき、検証の $R^2$、$D$、$g$ は見かけの値に一致する（$\eta$ が $x$ の 1 次変換のため）。fixture でもそうなっている。

### 4.4 `_emax`

$p = 0, 0.0005, \dots, 1$ の 2001 点で $|p - \operatorname{expit}(a + b\operatorname{logit}p)|$ の最大を取る。

### 4.5 `_cox_index`

検証では $\eta$ だけの Cox モデルを `_cox_fit` で当てはめ直す（Efron または Breslow の同順位処理は元のモデルと同じ）。$\ell(1)$ は係数 1 で反復 0 回の評価で得る。当てはめが失敗したら `RuntimeError`。

### 4.6 `_groupkm` と `_calibration_gap`

1. 区間は $[c_k, c_{k+1})$、最後だけ閉区間。外側の切断点はデータの範囲まで広げる（`Hmisc::cut2` のヘルプに書かれた規則）。
2. 群ごとに平均予測、人数、イベント数、KM、SE を出す。区間の数だけ行を作る。
3. `_calibration_gap` は、標本のイベントが 5 未満なら `None` を返す。

## 5. 数値上の注意

- 同順位: 打ち切りの一致指数では、イベントとイベントの時間の同順位は組にしない。打ち切りとイベントの同時刻は、打ち切りを長い方とする。
- 対数尤度は $\log(1 + e^{\eta})$ を `logaddexp` で計算してオーバーフローを避ける。
- `_lrm_fit` の境界: 結果が全部同じなら `fail=True`、係数は NaN、逸脱度は $(0, 0)$。予測変数がなければ切片だけで `fail=False`。尤度が有限でない、情報行列が特異、`maxit` 回で収束しない、のどれかで `fail=True`。
- Emax の格子: $p = 0, 1$ では $\operatorname{logit}$ が $\pm\infty$ になる。傾きが正なら較正値は $p$ と一致し差は 0。
- $\eta$ が定数（Cox の検証）: 情報がないので再当てはめしない。$R^2 = 0$、$D = -1/L_0$、$U = 0$、$Q = D$、$g = 0$、傾きは NaN、$D_{xy} = 0$（使える組はすべてリスク同順位）。
- KM: $u$ が群の最後の追跡時間より後なら $S(u)$ は推定できないので NaN。$u$ までにイベントがなければ $S = 1$、SE = 0。$S(u) = 0$ のとき SE は無限大。
- 空の区間: 人数 0、他は NaN。ブートストラップの平均では NaN の回は除かれる。

## 6. R と出力を合わせるための公開仕様

- `lrm.fit` のヘルプ: `eps` は収束を宣言する $-2\log L$ の変化（既定 0.025）、`maxit` は反復の上限（既定 12）、`tol` は特異性の基準（既定 1e-7）。`calibrate` からは `tol = 1e-13` で呼ぶ。
- `cph` のヘルプ（`coxph.control` の既定）: `eps = 1e-4`（部分尤度の相対変化）、`iter.max = 10`。`_cox_fit` は既存のものを使う。
- `validate.lrm` のヘルプ: 指標の並び Dxy、R2、Intercept、Slope、Emax、D、U、Q、B、g、gp。Emax は補正済みの較正直線から求め、見かけと訓練の列は 0。
- `validate.cph` のヘルプ: 並び Dxy、R2、Slope、D、U、Q、g。
- `calibrate.cph` のヘルプ: `cmethod = "KM"` で予測生存確率の群ごとに KM と比べる。群は約 `m` 人ずつ、`cut2` で切る。
- `groupkm` のヘルプ: 列 x、n、events、KM、std.err（$\log S$ の標準誤差）。

## 7. 検証

- テスト: `tests/r_oracle/test_rms.py`（`test_fixture_versions` と `test_lowess_matches_r` 以外）。
- fixture: `tests/r_oracle/fixtures/rms.json`（rms 6.7-1、Hmisc 5.1.1、survival 3.5.8、R 4.3.3。スクリプト `tests/r_oracle/scripts/rms.R`）。データ、R が引いたブートストラップの行番号、rms の結果を含む。
- 許容誤差（変更なし）:
  - `validate_logistic`、`validate_cox` の全列: rtol 1e-10、atol 1e-12。
  - `_lrm_fit` の係数: rtol 1e-12。
  - `somers_dxy`: rel 1e-9。
  - `calibrate_cox` の全列: atol 1e-12。
  - `calibrate_logistic`: atol 1e-9。
- 乱数だけのテスト（形と範囲）と、不正な `indices` の `ValueError` も通る。

## 8. R と違うところ、未確認の点

- `_lrm_fit` の停止位置: 収束判定の後にもう 1 歩進める規則と歩幅半減の規則は、fixture の 3 例（反復 3、4、5 回）の出力に合うように選んだ。完全に収束した最尤推定値とは小標本で最大約 4e-7 相対だけ違う。
- Emax: 2001 点の格子は fixture の 3 例から決めた。真の最大とは最大約 5e-7 相対違う（格子の方が小さい）。
- Cox の検証での $U$: Harrell (2015) の定義どおり「$-2$」を引くと fixture と $2/L_0$ だけずれる。fixture は見かけと訓練では $-2/L_0$、検証では $\mathrm{LR}_{\mathrm{cal}}/L_0$（$-2$ なし）。ロジスティックの検証では $-2$ を引いた値が fixture と合う。この非対称は fixture に合わせたもので、理論上の根拠は確認していない。
- 較正で標本を捨てるイベント数の下限 5 は自分で決めた。fixture ではどの反復も捨てられていない（`n` 列が B と同じ）ので、下限の値は検証されていない。
- 空の区間、群の最後の時間より後の $u$、切断点のごく近くの予測値の扱いは fixture では起きない。R の出力と一致するかは未確認。
- R の乱数は再現しない。`indices` を渡したときだけ rms と同じ結果になる。
