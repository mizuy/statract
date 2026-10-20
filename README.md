<p align="center">
  <img src="docs/assets/statract-logo.png" alt="Statract" width="420">
</p>

# statract

A Python library for clinical research: analyze your data and make the tables and figures for your paper. Data go in as Polars data frames.

[日本語](README.ja.md)

- The core does not depend on pandas, statsmodels, or lifelines. Kaplan–Meier uses the bundled `survival_curve`.
- The R bridge `statract.r` comes with the extra `r`. `import statract` alone loads neither R nor rpy2.

## Installation

Requires Python 3.13 or later. The package is not on PyPI, so install it from GitHub.

```bash
pip install "git+https://github.com/mizuy/statract"
# To also use the R bridge
pip install "statract[r] @ git+https://github.com/mizuy/statract"
```

## Minimal example

```python
import polars as pl
from statract import cox_ph, fit_ols, survival_curve

frame = pl.DataFrame(
    {
        "y": [1.2, 0.4, 2.1],
        "x": [0.0, 0.5, 1.0],
        "time": [0.4, 1.2, 0.8],
        "event": [1, 0, 1],
    }
)
fit_ols(frame, "y", ["x"])
survival_curve(frame, "time", "event")
cox_ph(frame, "Surv(time, event) ~ x")
```

## Documentation

[https://mizuy.github.io/statract/](https://mizuy.github.io/statract/)

| Section | Contents |
|------------|------|
| [Guide](https://mizuy.github.io/statract/) | Choosing a method, Table One, Models, Figures, Cost-effectiveness |
| [Learn](https://mizuy.github.io/statract/learn/) | Statistics starting from clinical questions (causal inference, prediction, statistical modeling, survival analysis), in Japanese |
| [Examples](https://mizuy.github.io/statract/examples/) | Full worked analyses on public data |
| [Benchmarks](https://mizuy.github.io/statract/models/benchmarks/) | Speed, and numeric agreement with R |
| [Reference](https://mizuy.github.io/statract/api/models/fit/) | API |

## License

MIT License ([LICENSE](LICENSE)). For third-party files used to build the docs, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
