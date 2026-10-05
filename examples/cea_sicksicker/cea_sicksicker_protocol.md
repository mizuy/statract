# cea_sicksicker_protocol

読者向けの一続き版（推奨）: [`docs/stat/examples/cea_sicksicker.md`](../../docs/stat/examples/cea_sicksicker.md)。本ファイルはローカル workflow（入力・出力対応）の正本です。

問い・式の正本: [cea_sicksicker_concept.md](cea_sicksicker_concept.md)

## 1. 目的・デザイン

仮想疾患 Sick-Sicker の時間一定 cohort STM。4 戦略の CEA + 感度分析。患者マイクロデータは使わない。

## 2. 対象コホート

DARTH 教学の閉じたコホート。開始年齢 25、終了 100（両端含む、76 サイクル）。初期分布は H=1。`build.py` が論文 Table 1 のスカラーを cache する（CSV を git に置かない）。

出典: Alarid-Escudero et al. *Med Decis Making* 2023;43(1):3-20. doi:10.1177/0272989X221103163（OA）。コード: https://github.com/DARTH-git/Cohort-modeling-tutorial

## 3. Inclusion criteria

1. 仮想コホート全員（開始時 H）。

## 4. Exclusion criteria

なし。行が落ちないので `pp.flowchart` は使わない。

## 5. `pp.flowchart`

使わない。コホート記述は `cea_sicksicker_out/cohort_n.md`。

## 6. Outcome

- 割引済み費用（USD、教学単位）
- 割引済み QALY
- 割引済み LY（参考）
- ICER と支配ステータス
- WTP = $100,000/QALY の NMB
- PSA: CE plane、CEAC、EVPI

## 7. Exposure

戦略:

| 名前 | 効果 |
|------|------|
| Standard of care | ベースの P と状態報酬 |
| Strategy A | S1/S2 に `c_trtA`、S1 効用 `u_trtA` |
| Strategy B | S1→S2 に OR 0.6、S1/S2 に `c_trtB` |
| Strategy AB | A と B を併用 |

## 8. 統計解析

concept §9。

- 病態と P 行列はこの example（`cea_sicksicker.py`）
- `simulate_cohort_markov`（状態報酬のみ。遷移報酬・半サイクルなし）
- `calculate_icers` / `net_monetary_benefit`
- `one_way_dsa`（`nmb` = 最適戦略の NMB、WTP $100k）→ `tornado_table`
- `run_psa`（n=1000, seed=2026）→ `ce_plane` / `ceac` / `evpi`
- Table 1 相当は入力パラメータ表。患者 tableone は無い

## 9. 出力 ↔ results

| ファイル | results 章 |
|----------|------------|
| `cohort_n.md`, `params.csv` | 対象・入力 |
| `trace_soc.csv`, `figures/trace_soc.png` | SoC 所属 |
| `icer.csv`, `figures/ce_frontier.png` | 基本ケース ICER |
| `dsa.csv`, `figures/tornado.png` | one-way DSA |
| `ce_plane.csv`, `ceac.csv`, `evpi.csv` と対応図 | PSA |
