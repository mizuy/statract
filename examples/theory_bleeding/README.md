# theory_bleeding

[臨床の問いから始める統計](../../docs/theory/index.md) の通しの例です。大腸内視鏡治療後の遅発性出血と予防的クリップを、DAG を決めたシミュレーションで作ります。実データではありません。

- `build.py`: データを作り `cache/build/bleeding.parquet` に保存する。本当の係数は `TRUE_BLEED`。
- `theory_bleeding.py`: 各章の図と表を `theory_bleeding_out/` に書く。

```bash
cd examples
task theory_bleeding:all
cd ..
uv run python tools/sync_example_assets.py --stem theory_bleeding
```
