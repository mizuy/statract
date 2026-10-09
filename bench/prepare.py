"""Download the public benchmark tables and write the analysis slices.

The canonical tables are Parquet. CSV copies sit beside them so R can read the
same rows without the arrow package. Nothing under the cache is committed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import polars as pl

CACHE = Path("/tmp/statract-bench")
RAW = CACHE / "raw"
OUT = CACHE / "prepared"
SEED = 20260927
# Bootstrap repetitions for validate/calibrate. R and Python draw different
# resamples, so only the apparent column is compared; both run the same B.
VALIDATE_B = 20
DATASETS = ("adult", "bike", "support", "star")

ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
BIKE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
SUPPORT_URL = "https://hbiostat.org/data/repo/support2csv.zip"
STAR_URL = "https://dataverse.harvard.edu/api/access/datafile/666716"

ADULT_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = Request(url, headers={"User-Agent": "statract-bench/0.1"})
    with urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        handle.write(response.read())


def _write_frame(name: str, frame: pl.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT / f"{name}.parquet")
    frame.write_csv(OUT / f"{name}.csv")
    factors = [col for col, dtype in zip(frame.columns, frame.dtypes, strict=True) if dtype == pl.String]
    return {
        "name": name,
        "parquet": str(OUT / f"{name}.parquet"),
        "csv": str(OUT / f"{name}.csv"),
        "n": frame.height,
        "factors": factors,
    }


def _stratified_indices(labels: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Take a stratified subset, preserving the source order of the chosen rows."""
    if target >= len(labels):
        return np.arange(len(labels))
    chosen: list[int] = []
    for level in sorted(set(labels.tolist())):
        pool = np.flatnonzero(labels == level)
        share = int(round(target * len(pool) / len(labels)))
        share = min(max(share, 1), len(pool))
        pick = rng.choice(pool, size=share, replace=False)
        chosen.extend(int(i) for i in pick)
    chosen_arr = np.array(chosen)
    if len(chosen_arr) > target:
        extra = rng.choice(len(chosen_arr), size=len(chosen_arr) - target, replace=False)
        keep = np.ones(len(chosen_arr), dtype=bool)
        keep[extra] = False
        chosen_arr = chosen_arr[keep]
    elif len(chosen_arr) < target:
        rest = np.setdiff1d(np.arange(len(labels)), chosen_arr, assume_unique=False)
        need = min(target - len(chosen_arr), len(rest))
        chosen_arr = np.concatenate([chosen_arr, rng.choice(rest, size=need, replace=False)])
    return np.sort(chosen_arr)


def _take(frame: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    """Keep rows at ``indices`` in the frame's current order."""
    chosen = set(int(i) for i in indices)
    mask = [i in chosen for i in range(frame.height)]
    return frame.filter(pl.Series(mask))


def _student_prefix(frame: pl.DataFrame, target: int, rng: np.random.Generator) -> pl.DataFrame:
    """Draw students, in one permutation, until the row count reaches ``target``."""
    counts: dict[str, int] = {}
    for student in frame["student"].to_list():
        counts[student] = counts.get(student, 0) + 1
    students = list(counts)
    if frame.height <= target:
        return frame
    order = rng.permutation(len(students))
    picked: list[str] = []
    total = 0
    for index in order:
        student = students[int(index)]
        picked.append(student)
        total += counts[student]
        if total >= target:
            break
    keep = set(picked)
    return frame.filter(pl.col("student").is_in(list(keep)))


def _load_adult() -> pl.DataFrame:
    frames = []
    for name in ("adult.data", "adult.test"):
        path = RAW / name
        text = path.read_text().splitlines()
        rows = []
        for line in text:
            if not line.strip() or line.startswith("|"):
                continue
            parts = [part.strip().rstrip(".") for part in line.split(",")]
            if len(parts) != 15:
                continue
            rows.append(parts)
        frames.append(pl.DataFrame({col: [row[i] for row in rows] for i, col in enumerate(ADULT_COLS)}))
    raw = pl.concat(frames)
    work = raw["workclass"].to_list()
    country = raw["native_country"].to_list()
    keep = [(w != "?" and c != "?") for w, c in zip(work, country, strict=True)]
    raw = raw.filter(pl.Series(keep))

    def marital3(value: str) -> str:
        if value == "Never-married":
            return "never_married"
        if value.startswith("Married"):
            return "married"
        return "other"

    def workclass4(value: str) -> str:
        if value == "Private":
            return "private"
        if value in {"Self-emp-not-inc", "Self-emp-inc"}:
            return "self_employed"
        if value in {"Federal-gov", "Local-gov", "State-gov"}:
            return "government"
        return "other"

    def relationship3(value: str) -> str:
        if value == "Husband":
            return "husband"
        if value == "Wife":
            return "wife"
        return "other"

    income = [1 if value == ">50K" else 0 for value in raw["income"].to_list()]
    frame = pl.DataFrame(
        {
            "hours_per_week": raw["hours_per_week"].cast(pl.Float64),
            "age": raw["age"].cast(pl.Float64),
            "education_num": raw["education_num"].cast(pl.Float64),
            "capital_gain": raw["capital_gain"].cast(pl.Float64),
            "capital_loss": raw["capital_loss"].cast(pl.Float64),
            "sex": raw["sex"],
            "race": raw["race"],
            "marital3": [marital3(value) for value in raw["marital_status"].to_list()],
            "workclass4": [workclass4(value) for value in raw["workclass"].to_list()],
            "relationship3": [relationship3(value) for value in raw["relationship"].to_list()],
            "us_native": [1.0 if value == "United-States" else 0.0 for value in raw["native_country"].to_list()],
            "fnlwgt": raw["fnlwgt"].cast(pl.Float64),
            "income_gt_50k": np.array(income, dtype=float),
        }
    )
    return frame.with_columns(pl.arange(0, frame.height).alias("row_id")).drop_nulls()


def _pad(value: str, width: int) -> str:
    return value.zfill(width)


def _load_bike() -> pl.DataFrame:
    with zipfile.ZipFile(RAW / "bike.zip") as archive:
        text = archive.read("hour.csv").decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    frame = pl.DataFrame(
        {
            "cnt": [float(row["cnt"]) for row in rows],
            "season": [_pad(row["season"], 1) for row in rows],
            "mnth": [_pad(row["mnth"], 2) for row in rows],
            "hr": [_pad(row["hr"], 2) for row in rows],
            "weekday": [row["weekday"] for row in rows],
            "workingday": [row["workingday"] for row in rows],
            "weathersit": [row["weathersit"] for row in rows],
            "temp": [float(row["temp"]) for row in rows],
            "hum": [float(row["hum"]) for row in rows],
            "windspeed": [float(row["windspeed"]) for row in rows],
            "yr": [row["yr"] for row in rows],
        }
    )
    return frame.with_columns(pl.arange(0, frame.height).alias("row_id"))


def _blank(value: str) -> bool:
    return value.strip() in {"", "NA", "NaN"}


def _support_records(parquet: Path | None = None) -> list[dict[str, str]]:
    """Rows of support2.csv as stripped strings, from the zip or from a local Parquet copy."""
    if parquet is not None:
        frame = pl.read_parquet(parquet)
        columns = {name: frame[name].cast(pl.String).to_list() for name in frame.columns}
        return [
            {name: ("" if columns[name][i] is None else columns[name][i].strip()) for name in frame.columns}
            for i in range(frame.height)
        ]
    with zipfile.ZipFile(RAW / "support2.zip") as archive:
        text = archive.read("support2.csv").decode()
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    records = []
    for record in reader:
        if len(record) == len(header) + 1:
            record = record[1:]
        if len(record) != len(header):
            continue
        records.append({name: value.strip() for name, value in zip(header, record, strict=True)})
    return records


def _cause(death: float, hospdead: float) -> float:
    """Competing-risk code: 1 death in hospital, 2 death after discharge, 0 alive (censored)."""
    if death == 0:
        return 0.0
    return 1.0 if hospdead == 1 else 2.0


def _load_support(records: list[dict[str, str]]) -> pl.DataFrame:
    source_cols = [
        "d.time",
        "death",
        "hospdead",
        "slos",
        "age",
        "sex",
        "num.co",
        "scoma",
        "meanbp",
        "hrt",
        "temp",
        "resp",
        "diabetes",
        "ca",
        "dzgroup",
    ]
    kept = [row for row in records if all(not _blank(row[col]) for col in source_cols)]
    frame = pl.DataFrame(
        {
            "d_time": [float(row["d.time"]) for row in kept],
            "death": [float(row["death"]) for row in kept],
            "age": [float(row["age"]) for row in kept],
            "sex": [row["sex"] for row in kept],
            "num_co": [float(row["num.co"]) for row in kept],
            "scoma": [float(row["scoma"]) for row in kept],
            "meanbp": [float(row["meanbp"]) for row in kept],
            "hrt": [float(row["hrt"]) for row in kept],
            "temp": [float(row["temp"]) for row in kept],
            "resp": [float(row["resp"]) for row in kept],
            "diabetes": [float(row["diabetes"]) for row in kept],
            "ca": [row["ca"] for row in kept],
            "dzgroup": [row["dzgroup"] for row in kept],
            "hospdead": [float(row["hospdead"]) for row in kept],
            "slos": [float(row["slos"]) for row in kept],
            "cause": [_cause(float(row["death"]), float(row["hospdead"])) for row in kept],
        }
    )
    positive = frame.filter(pl.col("d_time") > 0)
    return positive.with_columns(pl.arange(0, positive.height).alias("row_id"))


INCOME4 = {"under $11k": "inc1", "$11-$25k": "inc2", "$25-$50k": "inc3", ">$50k": "inc4"}


def _number_or_none(value: str) -> float | None:
    return None if _blank(value) else float(value)


def _load_support_mi(records: list[dict[str, str]]) -> pl.DataFrame:
    """SUPPORT2 with its missing laboratory values and income kept, for multiple imputation."""
    complete = ["hospdead", "age", "sex", "num.co", "meanbp", "hrt"]
    kept = [row for row in records if all(not _blank(row[col]) for col in complete)]
    frame = pl.DataFrame(
        {
            "hospdead": [float(row["hospdead"]) for row in kept],
            "age": [float(row["age"]) for row in kept],
            "sex": [row["sex"] for row in kept],
            "num_co": [float(row["num.co"]) for row in kept],
            "meanbp": [float(row["meanbp"]) for row in kept],
            "hrt": [float(row["hrt"]) for row in kept],
            "alb": [_number_or_none(row["alb"]) for row in kept],
            "bili": [_number_or_none(row["bili"]) for row in kept],
            "pafi": [_number_or_none(row["pafi"]) for row in kept],
            "wblc": [_number_or_none(row["wblc"]) for row in kept],
            "income4": [None if _blank(row["income"]) else INCOME4[row["income"]] for row in kept],
        },
        schema_overrides={name: pl.Float64 for name in ("alb", "bili", "pafi", "wblc")} | {"income4": pl.String},
    )
    return frame.with_columns(pl.arange(0, frame.height).alias("row_id"))


def _load_star() -> pl.DataFrame:
    with (RAW / "STAR_Students.tab").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    grades = (("gk", 0.0), ("g1", 1.0), ("g2", 2.0), ("g3", 3.0))
    built: dict[str, list] = {
        key: []
        for key in (
            "student",
            "read",
            "gender",
            "race",
            "freelunch",
            "birthyear",
            "grade",
            "classtype",
            "urban",
            "tyears",
            "tgen",
            "classsize",
            "school",
            "class_id",
        )
    }
    for row in rows:
        student = row["stdntid"].strip()
        gender = row["gender"].strip()
        race = row["race"].strip()
        birth = row["birthyear"].strip()
        if _blank(student) or _blank(gender) or _blank(race) or _blank(birth):
            continue
        for prefix, grade in grades:
            needed = {
                "read": row[f"{prefix}treadss"],
                "freelunch": row[f"{prefix}freelunch"],
                "classtype": row[f"{prefix}classtype"],
                "urban": row[f"{prefix}surban"],
                "tyears": row[f"{prefix}tyears"],
                "tgen": row[f"{prefix}tgen"],
                "classsize": row[f"{prefix}classsize"],
                "school": row[f"{prefix}schid"],
                "teacher": row[f"{prefix}tchid"],
            }
            if any(_blank(value) for value in needed.values()):
                continue
            school = needed["school"].strip()
            teacher = needed["teacher"].strip()
            built["student"].append(student)
            built["read"].append(float(needed["read"]))
            built["gender"].append(gender)
            built["race"].append(race)
            built["freelunch"].append(needed["freelunch"].strip())
            built["birthyear"].append(float(birth))
            built["grade"].append(grade)
            built["classtype"].append(needed["classtype"].strip())
            built["urban"].append(needed["urban"].strip())
            built["tyears"].append(float(needed["tyears"]))
            built["tgen"].append(needed["tgen"].strip())
            built["classsize"].append(float(needed["classsize"]))
            built["school"].append(school)
            built["class_id"].append(f"{school}-g{int(grade)}-{teacher}")
    frame = pl.DataFrame(built)
    return frame.with_columns(pl.arange(0, frame.height).alias("row_id"))


def _varying_predictors(frame: pl.DataFrame, predictors: list[str]) -> list[str]:
    kept = []
    for name in predictors:
        if frame[name].n_unique() >= 2:
            kept.append(name)
    return kept


def _tasks(slices: dict[str, dict]) -> list[dict]:
    tasks = []

    def add(slice_name: str, kind: str, task_id: str, **option: object) -> None:
        if slice_name not in slices:
            return
        tasks.append({"id": task_id, "slice": slice_name, "kind": kind, "option": option, "n": slices[slice_name]["n"]})

    for size in ("1000", "10000"):
        name = f"adult-{size}"
        add(name, "ols", "ols")
        add(name, "ols_w", "ols-w")
        add(name, "glm_bin", "glm-bin")
        add(name, "glm_gamma", "glm-gamma")
        add(name, "hc", "hc", types=["HC0", "HC1", "HC2", "HC3", "HC4", "HC5"])
        add(name, "wald", "wald")
        add(name, "lr", "lr")
        add(name, "bp", "bp", studentize=[True, False])
        add(name, "reset", "reset")
        add(name, "match", "m-logit", distance="logit")
        add(name, "match", "m-mah", distance="mahalanobis")
        add(name, "match", "m-exact", distance="exact")
        add(name, "match", "m-cem", distance="cem")
    for size in ("1000", "10000"):
        name = f"bike-{size}"
        add(name, "glm_pois", "glm-pois")
        add(name, "nw", "nw")
        add(name, "dw", "dw")
        add(name, "bg", "bg", orders=[1, 4])
        add(name, "gam", "gam-8", k=8)
        add(name, "gam", "gam-10", k=10)
    for name in ("support-1000", "support-large"):
        add(name, "km", "km")
        add(name, "km_sex", "km-sex")
        add(name, "na", "na")
        add(name, "lrk", "lrk")
        add(name, "cox", "cox-e", ties="efron", strata=False)
        add(name, "cox", "cox-b", ties="breslow", strata=False)
        add(name, "cox", "cox-s", ties="efron", strata=True)
        add(name, "aft", "aft-w", distribution="weibull")
        add(name, "aft", "aft-ln", distribution="lognormal")
        add(name, "aft", "aft-ex", distribution="exponential")
        # Functions added after the first round. The binary outcome is death in
        # hospital (hospdead); the competing-risk code is the column cause.
        add(name, "ttest", "ttest")
        add(name, "wilcox", "wilcox")
        add(name, "prop", "prop")
        add(name, "padj", "padj", methods=["holm", "hochberg", "hommel", "BH", "BY"])
        add(name, "roc", "roc")
        add(name, "val_lrm", "val-lrm", B=VALIDATE_B)
        add(name, "cal_lrm", "cal-lrm", B=VALIDATE_B)
        add(name, "val_cph", "val-cph", B=VALIDATE_B)
        add(name, "cal_cph", "cal-cph", B=VALIDATE_B, u=180.0, m=150)
        add(name, "std_cox", "std-cox", exposure="diabetes", values=[0.0, 1.0])
        add(name, "cuminc", "cuminc")
        add(name, "crr", "crr", cause=1)
        add(name, "gamm", "gamm", k=8)
        add(name, "glmm", "glmm-pois", family="poisson")
        add(name, "glmm", "glmm-nb", family="negative_binomial")
        add(name, "ctree", "ctree")
    for name in ("support-mi-1000", "support-mi-large"):
        # One timed call after the warm-up: impute_chained takes minutes on the
        # full slice (polyreg for income4 needs thousands of L-BFGS steps).
        add(name, "mice", "mice", m=5, maxit=5, repeats=1)
    for name in ("star-1000", "star-large"):
        add(name, "cl1", "cl1")
        add(name, "cl2", "cl2")
        add(name, "lmm", "lmm-ri", method="reml", slopes=False)
        add(name, "lmm", "lmm-ml", method="ml", slopes=False)
    for name in ("star-rs-1000", "star-rs-large"):
        add(name, "lmm", "lmm-rs", method="reml", slopes=True)
    return tasks


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="prepare only these data sets (and their tasks)",
    )
    parser.add_argument(
        "--support-parquet",
        type=Path,
        default=None,
        help="read SUPPORT2 from a local Parquet copy of support2.csv instead of downloading it",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    only = set(args.only)
    RAW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    slices = {}
    times: list[float] = []

    if "adult" in only:
        for name, url in ADULT_URLS.items():
            _download(url, RAW / name)
        adult = _load_adult()
        # Draw 10,000 with the published seed, then a stratified 1,000 from that slice.
        adult_10k = _take(adult, _stratified_indices(adult["income_gt_50k"].to_numpy(), 10000, rng))
        small_pos = _stratified_indices(adult_10k["income_gt_50k"].to_numpy(), 1000, np.random.default_rng(SEED))
        slices["adult-10000"] = _write_frame("adult-10000", adult_10k)
        slices["adult-1000"] = _write_frame("adult-1000", _take(adult_10k, small_pos))

    if "bike" in only:
        _download(BIKE_URL, RAW / "bike.zip")
        bike = _load_bike()
        bike_predictors = ["season", "mnth", "hr", "weekday", "workingday", "weathersit", "temp", "hum", "windspeed", "yr"]
        for size in (1000, 10000):
            part = bike.head(size)
            info = _write_frame(f"bike-{size}", part)
            info["predictors"] = _varying_predictors(part, bike_predictors)
            info["dropped_constant"] = [name for name in bike_predictors if name not in info["predictors"]]
            slices[f"bike-{size}"] = info

    if "support" in only:
        if args.support_parquet is None:
            _download(SUPPORT_URL, RAW / "support2.zip")
        records = _support_records(args.support_parquet)
        support_large = _load_support(records)
        support_pos = _stratified_indices(support_large["death"].to_numpy(), 1000, np.random.default_rng(SEED))
        slices["support-large"] = _write_frame("support-large", support_large)
        slices["support-1000"] = _write_frame("support-1000", _take(support_large, support_pos))
        times = np.quantile(support_large["d_time"].to_numpy(), [0.25, 0.5, 0.75]).tolist()
        mi_large = _load_support_mi(records)
        mi_pos = _stratified_indices(mi_large["hospdead"].to_numpy(), 1000, np.random.default_rng(SEED))
        slices["support-mi-large"] = _write_frame("support-mi-large", mi_large)
        slices["support-mi-1000"] = _write_frame("support-mi-1000", _take(mi_large, mi_pos))

    if "star" in only:
        _download(STAR_URL, RAW / "STAR_Students.tab")
        star = _load_star()
        star_rng = np.random.default_rng(SEED)
        star_large = _student_prefix(star, 10000, star_rng)
        star_small = _student_prefix(star_large, 1000, np.random.default_rng(SEED))
        slices["star-large"] = _write_frame("star-large", star_large)
        slices["star-1000"] = _write_frame("star-1000", star_small)
        multi = star.join(
            star.group_by("student").len().filter(pl.col("len") >= 2).select("student"), on="student", how="inner"
        )
        rs_large = _student_prefix(multi, 10000, np.random.default_rng(SEED))
        rs_small = _student_prefix(rs_large, 1000, np.random.default_rng(SEED))
        slices["star-rs-large"] = _write_frame("star-rs-large", rs_large)
        slices["star-rs-1000"] = _write_frame("star-rs-1000", rs_small)

    manifest = {
        "seed": SEED,
        "km_times": times,
        "slices": slices,
        "tasks": _tasks(slices),
        "notes": {
            "adult": "UCI Adult, CC BY 4.0. Question marks in workclass or native country are missing.",
            "bike": "First 1000 hours drop predictors that do not vary. season and yr are constant there.",
            "support": (
                "Vanderbilt SUPPORT2. d.time and num.co are renamed d_time and num_co. cause is 1 for death"
                " in hospital (hospdead), 2 for death after discharge, 0 for alive at last follow-up."
                " support-mi keeps missing alb, bili, pafi, wblc and income (income4 = inc1..inc4)."
            ),
            "star": "Harvard Dataverse 10.7910/DVN/SIWH9F. The tenth predictor is class size; special education is not recorded in grades 2 and 3.",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({name: {"n": info["n"], **{k: info[k] for k in info if k in {"dropped_constant", "predictors"}}} for name, info in slices.items()}, indent=2))


if __name__ == "__main__":
    main()
