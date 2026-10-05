# statract.cea — Quickstart

依存は Python のみです（`numpy`, `polars`）。R は不要です。

```python
from statract.cea import (
    calculate_icers,
    one_way_dsa,
    run_psa,
    simulate_cohort_markov,
    ceac,
    evpi,
)
```

## 1. 3 状態 cohort Markov

Healthy → Sick → Dead の教科書型です。

```python
import numpy as np
from statract.cea import simulate_cohort_markov

# 行=from, 列=to
P = np.array(
    [
        [0.94, 0.05, 0.01],  # Healthy
        [0.00, 0.90, 0.10],  # Sick
        [0.00, 0.00, 1.00],  # Dead
    ]
)

res = simulate_cohort_markov(
    states=("Healthy", "Sick", "Dead"),
    initial={"Healthy": 1.0},
    ages=range(50, 81),
    transition=P,
    utility={"Healthy": 1.0, "Sick": 0.6, "Dead": 0.0},
    cost={"Healthy": 0.0, "Sick": 1000.0, "Dead": 0.0},
    discount_rate=0.03,
    record_trace=True,
)
print(res.cost, res.qaly, res.ly)
print(res.trace.head())
```

### 介入を `on_cycle` に載せる

検査・切除など、状態報酬だけでは表せないイベントはフックに書きます。返す費用・QALY は**未割引**です。

```python
def on_cycle(age: int, mass: np.ndarray):
    if age != 55:
        return mass, 0.0, 0.0
    m = mass.copy()
    # 例: Healthy の 10% を「検診」で費用だけ計上
    return m, 5000.0 * float(m[0]), 0.0

res = simulate_cohort_markov(
    states=("Healthy", "Sick", "Dead"),
    initial=[1.0, 0.0, 0.0],
    ages=range(50, 70),
    transition=P,
    utility=[1.0, 0.5, 0.0],
    discount_rate=0.03,
    on_cycle=on_cycle,
)
```

年齢依存遷移は callable か `dict[age, matrix]` で渡せます。

```python
def transition(age: int, mass: np.ndarray) -> np.ndarray:
    p_hs = min(0.02 + 0.001 * (age - 50), 0.2)
    P_age = P.copy()
    P_age[0, 0] = 1.0 - p_hs - 0.01
    P_age[0, 1] = p_hs
    return mass @ P_age
```

## 2. ICER 表（モデル非依存）

```python
from statract.cea import calculate_icers, net_monetary_benefit

icers = calculate_icers(
    cost=[10_000, 25_000, 40_000],
    effect=[5.0, 6.0, 6.2],
    strategies=["A", "B", "C"],
)
print(icers)

nmb = net_monetary_benefit(
    icers["cost"],
    icers["effect"],
    wtp=5_000_000,
)
```

`status` が `ND` / `D` / `ED` です。延長支配の例は単体テスト `tests/test_cea_summarize.py` を参照してください。

## 3. One-way DSA と PSA

```python
import numpy as np
from statract.cea import one_way_dsa, run_psa, ce_plane, ceac, evpi


def evaluate(params: dict[str, float]):
    # SOC vs INT（例）
    c0, e0 = 1000.0, 10.0
    c1 = params["c_int"]
    e1 = 10.0 + params["delta_e"]
    return [c0, c1], [e0, e1]


dsa = one_way_dsa(
    strategies=["SOC", "INT"],
    base_params={"c_int": 5000.0, "delta_e": 1.0},
    ranges={"c_int": (3000.0, 8000.0), "delta_e": (0.5, 1.5)},
    evaluate=evaluate,
    outcome="delta_nmb",
    comparator="SOC",
    wtp=50_000.0,
)
print(dsa)  # spread 降順 → tornado 用

rng = np.random.default_rng(0)
draws = {
    "c_int": rng.normal(5000.0, 500.0, size=500),
    "delta_e": rng.normal(1.0, 0.1, size=500),
}
psa = run_psa(strategies=["SOC", "INT"], param_draws=draws, evaluate=evaluate)
plane = ce_plane(psa, comparator="SOC")
ac = ceac(psa, wtp=[0, 1e4, 5e4, 1e5])
voi = evpi(psa, wtp=[0, 1e4, 5e4, 1e5])
```

## 4. 疾患モデルはプロジェクト側

SSL スクリーニング CEA の実行例:

```bash
cd profile
task cea:treeage   # 自然史入力
task cea:run       # Detection 比較（statract.cea を内部利用）
```

費用の Gamma 引き（SAP: SE = 平均の 20%）は `profile/cea/core/psa_draws.py` が担当し、要約だけ `statract.cea` に渡します。

## 関連

- [概要](overview.md)
- [R との対応](vs-r.md)
- 公開教学例: [cea_sicksicker](../stat/examples/cea_sicksicker.md)（DARTH Sick-Sicker）
- API: [markov](../api/cea/markov.md) · [summarize](../api/cea/summarize.md) · [sensitivity](../api/cea/sensitivity.md)
