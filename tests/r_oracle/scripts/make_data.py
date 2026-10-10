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


def _gamm_binary(seed: int, n: int, n_examiners: int) -> pl.DataFrame:
    """A binary outcome with a smooth in lesion size and an examiner intercept."""
    rng = np.random.default_rng(seed)
    size = np.round(rng.uniform(3.0, 40.0, n), 1)
    examiner = rng.integers(0, n_examiners, n)
    u = rng.normal(0, 0.6, n_examiners)
    f = 1.2 * np.sin((size - 3.0) / 37.0 * np.pi) - 0.02 * (size - 20.0)
    age = np.round(rng.normal(60, 10, n), 1)
    eta = -0.8 + f + 0.02 * (age - 60) + u[examiner]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    count = rng.poisson(np.exp(0.2 + 0.5 * f + u[examiner]))
    return pl.DataFrame(
        {"y": y, "n": count, "pre_size_mm": size, "age": age, "examiner": [f"e{i:02d}" for i in examiner]}
    )


def _stdreg_sample(seed: int, n: int, *, exposure: str, whole_tenths: bool, max_follow_up: float) -> pl.DataFrame:
    """Cox data for standardization: a binary ``ope`` or a continuous ``dose``."""
    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 10.0, n)
    sex = rng.choice(["F", "M"], size=n)
    # Integer sites: stdReg2 passes the cluster column to data.table's ``by``,
    # which reads a character vector as column names.
    site = rng.integers(1, 31, n)
    if exposure == "ope":
        value = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(age - 65.0) / 10.0))).astype(int)
        effect = -0.5 * value + 0.01 * value * (age - 65.0)
    else:
        value = np.round(rng.uniform(0.0, 2.0, n), 6)
        effect = -0.3 * value + 0.01 * value * (age - 65.0)
    rate = 0.2 * np.exp(0.03 * (age - 65.0) + 0.3 * (sex == "M") + effect)
    event_time = rng.exponential(1.0 / rate)
    censor = rng.uniform(1.0, max_follow_up, n)
    time = np.minimum(event_time, censor)
    # Tenths of a year tie events with each other and with censored rows.
    time = np.ceil(time * 10.0) / 10.0 if whole_tenths else time
    return pl.DataFrame(
        {
            "time": np.round(time, 6),
            "status": (event_time <= censor).astype(int),
            exposure: value,
            "age": np.round(age, 6),
            "sex": sex,
            "site": site,
        }
    )


def _std_glm_sample(seed: int, n: int) -> pl.DataFrame:
    """Binary and count outcomes for GLM standardization.

    ``trt`` is a 0/1 exposure that depends on age, ``arm`` a three-level
    factor, and ``w`` a positive weight. ``site`` is an integer cluster.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(60.0, 10.0, n)
    sex = rng.choice(["F", "M"], size=n)
    site = rng.integers(1, 26, n)
    trt = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(age - 60.0) / 12.0))).astype(int)
    arm = rng.choice(["a", "b", "c"], size=n, p=[0.4, 0.35, 0.25])
    years = rng.uniform(0.5, 3.0, n)
    arm_effect = np.select([arm == "b", arm == "c"], [0.4, -0.5], 0.0)
    eta = -0.4 + 0.6 * trt + 0.03 * (age - 60.0) + 0.02 * trt * (age - 60.0) + 0.3 * (sex == "M") + arm_effect
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    count = rng.poisson(years * np.exp(-0.5 + 0.4 * trt + 0.02 * (age - 60.0) + 0.5 * arm_effect))
    return pl.DataFrame(
        {
            "y": y,
            "count": count.astype(np.int64),
            "trt": trt,
            "arm": arm,
            "age": np.round(age, 6),
            "sex": sex,
            "years": np.round(years, 6),
            "site": site,
            "w": np.round(rng.uniform(0.5, 2.0, n), 6),
        }
    )


def _ordinal_sample(seed: int, n: int, *, cuts: tuple[float, ...], labels: tuple[str, ...] | None, weighted: bool, holes: bool, non_po: float) -> pl.DataFrame:
    """Ordinal outcome from a latent logistic variable. ``non_po`` lets the x slope grow with the cutpoint."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    age = rng.uniform(30, 80, n)
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.4, 0.35, 0.25])
    latent = 0.8 * x + 0.03 * (age - 55) + np.select([stage == "II", stage == "III"], [0.5, 1.1], 0.0)
    latent = latent + non_po * x * (latent > 0.5) + rng.logistic(size=n)
    code = np.digitize(latent, cuts)
    y: list[object] = [labels[c] for c in code] if labels is not None else (code + 1).tolist()
    frame = pl.DataFrame({"y": y, "x": np.round(x, 6), "age": np.round(age, 2), "stage": stage})
    if weighted:
        frame = frame.with_columns(pl.Series("w", np.round(rng.uniform(0.5, 2.0, n), 4)))
    if holes:
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) % 37 == 5).then(None).otherwise(pl.col("x")).alias("x"),
            pl.when(pl.int_range(pl.len()) % 41 == 7).then(None).otherwise(pl.col("y")).alias("y"),
        )
    return frame


def _multinom_sample(seed: int, n: int, *, labels: tuple[str, ...], weighted: bool) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    age = rng.uniform(30, 80, n)
    sex = rng.choice(["F", "M"], size=n)
    k = len(labels)
    coef_x = np.linspace(0.0, 1.2, k)
    coef_age = np.linspace(0.0, -0.03, k)
    coef_sex = np.concatenate([[0.0], rng.normal(0, 0.6, k - 1)])
    base = np.concatenate([[0.0], rng.normal(0, 0.5, k - 1)])
    eta = base + np.outer(x, coef_x) + np.outer(age - 55, coef_age) + np.outer(sex == "M", coef_sex)
    prob = np.exp(eta - eta.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    code = np.array([rng.choice(k, p=row) for row in prob])
    frame = pl.DataFrame({"y": [labels[c] for c in code], "x": np.round(x, 6), "age": np.round(age, 2), "sex": sex})
    if weighted:
        frame = frame.with_columns(pl.Series("w", np.round(rng.uniform(0.5, 2.0, n), 4)))
    return frame


def _risk_sample(seed: int, n: int, *, base: float, n_sites: int, weighted: bool) -> pl.DataFrame:
    """Binary outcome with a log-linear risk, so log-binomial fits stay inside (0, 1)."""
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "treat"], size=n)
    age = rng.uniform(40, 80, n)
    sex = rng.integers(0, 2, n)
    site = rng.integers(1, n_sites + 1, n)
    risk = base * np.exp(-0.4 * (arm == "treat") + 0.01 * (age - 60) + 0.2 * sex)
    y = (rng.uniform(size=n) < np.clip(risk, 0, 0.95)).astype(np.int64)
    frame = pl.DataFrame({"y": y, "arm": arm, "age": np.round(age, 2), "sex": sex, "site": site})
    if weighted:
        frame = frame.with_columns(pl.Series("w", np.round(rng.uniform(0.5, 2.0, n), 4)))
    return frame


def _rmst_sample(seed: int, n: int, *, arms: tuple[object, object], rate: tuple[float, float], follow_up: float, digits: int, last_event: bool) -> pl.DataFrame:
    """Two arms with exponential times, uniform censoring, and rounding for ties."""
    rng = np.random.default_rng(seed)
    arm_code = rng.integers(0, 2, n)
    t = rng.exponential(1.0 / np.asarray(rate)[arm_code])
    c = rng.uniform(0.5, follow_up, n)
    time = np.round(np.maximum(np.minimum(t, c), 10.0 ** -digits), digits)
    status = (t <= c).astype(np.int64)
    if last_event:
        # The shorter arm ends with an event, so tau may go past its last time.
        last0 = time[arm_code == 0].max()
        status[(arm_code == 0) & (time == last0)] = 1
        time[arm_code == 1] = np.minimum(time[arm_code == 1], last0 + 1.0)
        status[(arm_code == 1) & (time == time[arm_code == 1].max())] = 0
    return pl.DataFrame({"time": time, "status": status, "arm": [arms[i] for i in arm_code]})


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
    # Smooth plus examiner intercept, binary (gamm4 and mgcv bs = "re").
    _gamm_binary(20261023, 800, 25).write_csv(DATA / "gamm_binary_a.csv")
    # Cox standardization: tied times with sites, a continuous exposure, and
    # untied times for the restricted mean.
    _stdreg_sample(20261024, 300, exposure="ope", whole_tenths=True, max_follow_up=8.0).write_csv(DATA / "stdreg_a.csv")
    _stdreg_sample(20261025, 250, exposure="dose", whole_tenths=False, max_follow_up=8.0).write_csv(DATA / "stdreg_b.csv")
    _stdreg_sample(20261026, 300, exposure="ope", whole_tenths=False, max_follow_up=6.0).write_csv(DATA / "stdreg_c.csv")
    # GLM standardization: binary and count outcomes, a factor, weights, sites.
    _std_glm_sample(20261027, 400).write_csv(DATA / "stdglm_a.csv")
    _std_glm_sample(20261028, 250).write_csv(DATA / "stdglm_b.csv")
    # Ordinal outcomes: labelled levels, integer levels with weights, missing values with non-proportional odds.
    _ordinal_sample(20261027, 300, cuts=(-0.5, 0.8, 2.0), labels=("none", "mild", "moderate", "severe"), weighted=False, holes=False, non_po=0.0).write_csv(DATA / "ordinal_a.csv")
    _ordinal_sample(20261028, 250, cuts=(-1.0, 0.0, 1.0, 2.0), labels=None, weighted=True, holes=False, non_po=0.0).write_csv(DATA / "ordinal_b.csv")
    _ordinal_sample(20261029, 400, cuts=(-0.3, 1.0, 2.2), labels=("low", "mid", "high", "top"), weighted=False, holes=True, non_po=0.9).write_csv(DATA / "ordinal_c.csv")
    # Multinomial outcomes: three labels, four labels with weights, and two labels.
    _multinom_sample(20261030, 300, labels=("A", "B", "C"), weighted=False).write_csv(DATA / "multinom_a.csv")
    _multinom_sample(20261031, 400, labels=("ctrl", "low", "mid", "high"), weighted=True).write_csv(DATA / "multinom_b.csv")
    _multinom_sample(20261032, 200, labels=("no", "yes"), weighted=False).write_csv(DATA / "multinom_c.csv")
    # Binary outcomes for risk ratios: common and rare outcomes, clusters, weights.
    _risk_sample(20261033, 500, base=0.35, n_sites=20, weighted=False).write_csv(DATA / "risk_a.csv")
    _risk_sample(20261034, 600, base=0.15, n_sites=12, weighted=True).write_csv(DATA / "risk_b.csv")
    _risk_sample(20261035, 300, base=0.45, n_sites=8, weighted=False).write_csv(DATA / "risk_c.csv")
    # RMST: tied times, string arms, and a shorter arm that ends with an event.
    _rmst_sample(20261036, 200, arms=(0, 1), rate=(0.25, 0.15), follow_up=8.0, digits=1, last_event=False).write_csv(DATA / "rmst_a.csv")
    _rmst_sample(20261037, 160, arms=("control", "treat"), rate=(0.4, 0.3), follow_up=6.0, digits=0, last_event=False).write_csv(DATA / "rmst_b.csv")
    _rmst_sample(20261038, 120, arms=(0, 1), rate=(0.5, 0.3), follow_up=5.0, digits=2, last_event=True).write_csv(DATA / "rmst_c.csv")


if __name__ == "__main__":
    main()
