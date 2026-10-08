"""Write the seeded synthetic CSVs that the R oracle scripts read.

Run from the repository root: ``uv run python tests/r_oracle/scripts/make_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"


def _glmm_count(seed: int, n_groups: int, size: tuple[int, int], *, slope_sd: float, theta: float | None, base: float):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(size[0], size[1] + 1, n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    x = rng.normal(size=n)
    arm = rng.choice(["control", "low", "high"], size=n)
    years = rng.uniform(0.5, 3.0, n)
    b0 = rng.normal(0, 0.6, n_groups)
    b1 = rng.normal(0, slope_sd, n_groups)
    effect = np.select([arm == "low", arm == "high"], [0.3, -0.4], 0.0)
    mu = years * np.exp(base + 0.4 * x + effect + b0[group] + b1[group] * x)
    if theta is None:
        y = rng.poisson(mu)
    else:
        y = rng.negative_binomial(theta, theta / (theta + mu))
    return pl.DataFrame(
        {
            "y": y.astype(np.int64),
            "x": np.round(x, 6),
            "arm": arm,
            "years": np.round(years, 6),
            "site": [f"s{g:03d}" for g in group],
        }
    )


def _glmm_binary(seed: int, n_groups: int, size: tuple[int, int], *, slope_sd: float, base: float) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(size[0], size[1] + 1, n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    x = rng.normal(size=n)
    arm = rng.choice(["control", "low", "high"], size=n)
    b0 = rng.normal(0, 0.8, n_groups)
    b1 = rng.normal(0, slope_sd, n_groups)
    effect = np.select([arm == "low", arm == "high"], [0.4, -0.6], 0.0)
    p = 1.0 / (1.0 + np.exp(-(base + 0.7 * x + effect + b0[group] + b1[group] * x)))
    return pl.DataFrame(
        {
            "y": (rng.uniform(size=n) < p).astype(np.int64),
            "x": np.round(x, 6),
            "arm": arm,
            "site": [f"s{g:03d}" for g in group],
        }
    )


def _glmm_zero(seed: int, n_groups: int, *, structural: float, theta: float | None) -> pl.DataFrame:
    """Counts with extra zeros whose probability depends on z."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(15, 30, n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    years = rng.uniform(0.5, 3.0, n)
    b0 = rng.normal(0, 0.5, n_groups)
    mu = years * np.exp(0.8 + 0.4 * x + b0[group])
    count = rng.poisson(mu) if theta is None else rng.negative_binomial(theta, theta / (theta + mu))
    pi = 1.0 / (1.0 + np.exp(-(np.log(structural / (1 - structural)) + 0.6 * z)))
    y = np.where(rng.uniform(size=n) < pi, 0, count)
    return pl.DataFrame(
        {
            "y": y.astype(np.int64),
            "x": np.round(x, 6),
            "z": np.round(z, 6),
            "years": np.round(years, 6),
            "site": [f"s{g:03d}" for g in group],
        }
    )


def _cox_sample(seed: int, n: int, *, n_strata: int, weighted: bool) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.4, 0.35, 0.25])
    site = rng.choice([f"h{k}" for k in range(n_strata)], size=n)
    effect = np.select([stage == "II", stage == "III"], [0.4, 0.9], 0.0)
    rate = 0.15 * np.exp(0.5 * x + effect)
    event_time = rng.exponential(1.0 / rate)
    censor = rng.uniform(1.0, 8.0, n)
    # Whole months create tied event times.
    time = np.ceil(np.minimum(event_time, censor) * 12.0) / 12.0
    status = (event_time <= censor).astype(int)
    frame = pl.DataFrame(
        {"time": np.round(time, 6), "status": status, "x": np.round(x, 6), "stage": stage, "site": site}
    )
    if weighted:
        frame = frame.with_columns(pl.Series("w", np.round(rng.uniform(0.5, 2.0, n), 6)))
    return frame


def _counting_sample(seed: int, n_subjects: int) -> pl.DataFrame:
    """Each subject has one to three (start, stop] rows with a changing covariate."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        age = rng.normal()
        start = 0.0
        n_rows = int(rng.integers(1, 4))
        for k in range(n_rows):
            dose = float(rng.integers(0, 3))
            length = float(np.ceil(rng.exponential(2.0) * 4.0) / 4.0)
            stop = start + length
            hazard = 0.2 * np.exp(0.4 * age + 0.3 * dose)
            event = int(rng.uniform() < 1 - np.exp(-hazard * length)) if k == n_rows - 1 else 0
            rows.append(
                {"id": f"p{subject:03d}", "start": start, "stop": stop, "status": event, "age": round(age, 6), "dose": dose}
            )
            start = stop
    return pl.DataFrame(rows)


def _competing_sample(seed: int, n: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.4, 0.35, 0.25])
    effect = np.select([stage == "II", stage == "III"], [0.3, 0.7], 0.0)
    t1 = rng.exponential(1.0 / (0.12 * np.exp(0.5 * x + effect)))
    t2 = rng.exponential(1.0 / (0.08 * np.exp(-0.3 * x)))
    censor = rng.uniform(2.0, 10.0, n)
    time = np.ceil(np.minimum.reduce([t1, t2, censor]) * 12.0) / 12.0
    status = np.select([(t1 <= t2) & (t1 <= censor), (t2 < t1) & (t2 <= censor)], [1, 2], 0)
    return pl.DataFrame({"time": np.round(time, 6), "status": status, "x": np.round(x, 6), "stage": stage})


def _mi_sample(seed: int, n: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], size=n, p=[0.4, 0.35, 0.25])
    effect = np.select([g == "b", g == "c"], [0.5, -0.3], 0.0)
    y = 1.0 + 1.0 * x1 - 0.7 * x2 + effect + rng.normal(size=n)
    b = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.3)))).astype(int)
    hazard = 0.2 * np.exp(0.4 * x1 + 0.3 * b)
    t = rng.exponential(1 / hazard)
    censor = rng.uniform(1.0, 8.0, n)
    time = np.round(np.minimum(t, censor), 4)
    status = (t <= censor).astype(int)
    # Missing at random: x2 more often missing when y is high, g and b at random.
    p_x2 = 0.4 / (1 + np.exp(-(y - 1.0)))
    frame = pl.DataFrame(
        {
            "y": np.round(y, 6),
            "x1": np.round(x1, 6),
            "x2": np.round(x2, 6),
            "g": g,
            "b": b,
            "time": time,
            "status": status,
        }
    )
    return frame.with_columns(
        pl.when(pl.Series(rng.uniform(size=n) < p_x2)).then(None).otherwise(pl.col("x2")).alias("x2"),
        pl.when(pl.Series(rng.uniform(size=n) < 0.15)).then(None).otherwise(pl.col("g")).alias("g"),
        pl.when(pl.Series(rng.uniform(size=n) < 0.15)).then(None).otherwise(pl.col("b")).alias("b"),
    )


def _two_way_sample(seed: int, n_patients: int, n_examiners: int, *, weighted: bool) -> pl.DataFrame:
    """Right-censored exams, crossed by patient and examiner, with frailties for both."""
    rng = np.random.default_rng(seed)
    per_patient = rng.integers(1, 5, n_patients)
    patient = np.repeat(np.arange(n_patients), per_patient)
    n = len(patient)
    examiner = rng.integers(0, n_examiners, n)
    u_patient = rng.normal(0, 0.5, n_patients)
    u_examiner = rng.normal(0, 0.4, n_examiners)
    x = rng.normal(size=n)
    arm = rng.choice(["control", "low", "high"], size=n)
    effect = np.select([arm == "low", arm == "high"], [0.3, -0.4], 0.0)
    hazard = 0.15 * np.exp(0.5 * x + effect + u_patient[patient] + u_examiner[examiner])
    t = rng.exponential(1.0 / hazard)
    censor = rng.uniform(1.0, 6.0, n)
    # Round to a grid so that event times tie.
    time = np.ceil(np.minimum(t, censor) * 4.0) / 4.0
    status = (t <= censor).astype(int)
    frame = pl.DataFrame(
        {
            "id_patient": [f"p{i:03d}" for i in patient],
            "e_examiner": [f"e{i:02d}" for i in examiner],
            "time": time,
            "status": status,
            "x": np.round(x, 6),
            "arm": arm,
            "site": rng.choice(["A", "B"], size=n),
        }
    )
    if weighted:
        frame = frame.with_columns(pl.Series("w", np.round(rng.uniform(0.5, 2.0, n), 3)))
    # A few missing covariates check that clusters follow the rows the fit used.
    gone = rng.uniform(size=n) < 0.03
    return frame.with_columns(pl.when(pl.Series(gone)).then(None).otherwise(pl.col("x")).alias("x"))


def _crossed_sample(seed: int, n_patients: int, n_examiners: int, *, theta: float, rho: float) -> pl.DataFrame:
    """Counts and a binary outcome with patients crossed with examiners, and
    examiner effects that drift over years as an AR(1) process."""
    rng = np.random.default_rng(seed)
    years = np.arange(2015, 2021)
    per_patient = rng.integers(2, 6, n_patients)
    patient = np.repeat(np.arange(n_patients), per_patient)
    n = len(patient)
    examiner = rng.integers(0, n_examiners, n)
    year = rng.choice(years, size=n)
    u_patient = rng.normal(0, 0.5, n_patients)
    # AR(1) over the years for each examiner, stationary SD 0.4.
    v = np.zeros((n_examiners, len(years)))
    v[:, 0] = rng.normal(0, 0.4, n_examiners)
    for k in range(1, len(years)):
        v[:, k] = rho * v[:, k - 1] + rng.normal(0, 0.4 * np.sqrt(1 - rho**2), n_examiners)
    x = rng.normal(size=n)
    arm = rng.choice(["control", "low", "high"], size=n)
    effect = np.select([arm == "low", arm == "high"], [0.3, -0.4], 0.0)
    exposure = rng.uniform(0.5, 2.0, n)
    eta = -0.2 + 0.3 * x + effect + u_patient[patient] + v[examiner, year - years[0]]
    mu = exposure * np.exp(eta)
    y = rng.negative_binomial(theta, theta / (theta + mu))
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(eta - 0.2)))).astype(int)
    return pl.DataFrame(
        {
            "id_patient": [f"p{i:03d}" for i in patient],
            "e_examiner": [f"e{i:02d}" for i in examiner],
            "year": year,
            "y": y,
            "yb": yb,
            "x": np.round(x, 6),
            "arm": arm,
            "exposure": np.round(exposure, 6),
        }
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    # (1) Poisson counts, random intercept, 30 sites.
    _glmm_count(20261007, 30, (15, 25), slope_sd=0.0, theta=None, base=0.2).write_csv(DATA / "glmm_count_a.csv")
    # (2) Overdispersed counts with a random slope.
    _glmm_count(20261008, 25, (20, 40), slope_sd=0.3, theta=1.5, base=0.5).write_csv(DATA / "glmm_count_b.csv")
    # (3) Many small sites and low counts (most outcomes are zero or one).
    _glmm_count(20261009, 120, (2, 4), slope_sd=0.0, theta=3.0, base=-1.2).write_csv(DATA / "glmm_count_c.csv")
    # Binary outcomes: random intercept, random slope, and many pairs.
    _glmm_binary(20261014, 40, (10, 25), slope_sd=0.0, base=-0.3).write_csv(DATA / "glmm_binary_a.csv")
    _glmm_binary(20261015, 30, (20, 40), slope_sd=0.5, base=0.2).write_csv(DATA / "glmm_binary_b.csv")
    _glmm_binary(20261016, 150, (2, 3), slope_sd=0.0, base=-1.0).write_csv(DATA / "glmm_binary_c.csv")
    # Zero-inflated counts: Poisson and negative binomial.
    _glmm_zero(20261017, 30, structural=0.25, theta=None).write_csv(DATA / "glmm_zero_a.csv")
    _glmm_zero(20261018, 25, structural=0.35, theta=2.0).write_csv(DATA / "glmm_zero_b.csv")
    # Cox inference: tied right-censored data, then strata and weights, then
    # counting-process rows with repeated subjects.
    _cox_sample(20261010, 300, n_strata=1, weighted=False).write_csv(DATA / "cox_a.csv")
    _cox_sample(20261011, 400, n_strata=3, weighted=True).write_csv(DATA / "cox_b.csv")
    _counting_sample(20261012, 200).write_csv(DATA / "cox_c.csv")
    # Competing risks for Fine-Gray: cause 1 of interest, cause 2 competing.
    _competing_sample(20261013, 300).write_csv(DATA / "fg_a.csv")
    # Multiple imputation: missing covariates, a binary column, and a factor.
    _mi_sample(20261019, 250).write_csv(DATA / "mi_a.csv")
    # Two-way clustered Cox: patients crossed with examiners.
    _two_way_sample(20261020, 150, 12, weighted=False).write_csv(DATA / "cox2way_a.csv")
    _two_way_sample(20261021, 200, 8, weighted=True).write_csv(DATA / "cox2way_b.csv")
    # Crossed GLMMs: patients crossed with examiners, examiner-by-year AR(1).
    _crossed_sample(20261022, 300, 30, theta=2.0, rho=0.6).write_csv(DATA / "glmm_crossed_a.csv")


if __name__ == "__main__":
    main()
