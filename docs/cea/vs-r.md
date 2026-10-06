# statract.cea と R パッケージの対応

公開品質の目標は「R の定番と同じ**役割分担**を Python で持つ」ことです。R パッケージをラップしたり、依存したりはしません。

## 役割の対応表

| 役割 | R | `statract.cea` | 備考 |
|------|---|---------------|------|
| cohort Markov の定義・実行 | **heemod** | `markov.simulate_cohort_markov` | 時間依存 `P(a)`・状態報酬・`on_cycle`。半サイクル補正は未実装 |
| 個体 / DES / PartSA | **hesim** | （なし） | V1 対象外 |
| ICER・強支配・延長支配 | **dampack** `calculate_icers` | `summarize.calculate_icers` | `ND` / `D` / `ED` |
| NMB | dampack / BCEA | `summarize.net_monetary_benefit` | |
| One-way / tornado | dampack OWSA | `sensitivity.one_way_dsa` | `nmb` / `delta_nmb` / `icer` |
| PSA 実行の器 | （モデル側） | `sensitivity.run_psa` | 分布のサンプリングは呼び出し側 |
| CE plane / CEAC / EVPI | dampack / **BCEA** | `ce_plane` / `ceac` / `evpi` | モデル非依存の後段。`ceac` は非有限の draw を確率から外す |
| EVPPI | BCEA | （なし） | V1 対象外 |

## 意図的に寄せない点

- **heormodel**（PyPI）が heemod 相当のフル HTA を既に狙っているため、同じ土俵で全部を再実装しない
- 病態・費用・生命表はライブラリに持たず、利用者のモデル側で与える

## status ラベル（dampack 互換の意味）

`calculate_icers` の `status` 列:

| 値 | dampack 的意味 |
|----|----------------|
| `ND` | non-dominated（frontier） |
| `D` | strongly dominated |
| `ED` | extended / weakly dominated |

## 依存関係

```text
statract.cea
  └── numpy, polars
      （scipy / rpy2 / R パッケージは不要）
```

R ブリッジは optional extra `statract[r]` です。`import statract.cea` や CEA の計算経路は R を起動しません。
