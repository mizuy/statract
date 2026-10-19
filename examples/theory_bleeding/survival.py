"""Chapter 16 of the theory text: survival analysis.

Part V uses its own examples instead of the bleeding data of chapters 1-15.

Recurrence after curative surgery for stage II-III colon cancer: 2000
patients, followed for up to 5 years. Adjuvant chemotherapy is the treatment,
given mostly to stage III and younger patients. Death without recurrence is
censored at first and treated as a competing event in the last section.

The data are simulated, so the truth is known. Asset names keep the old chapter
number (``ch15_*``).

Run from the repo root::

    uv run python examples/theory_bleeding/survival.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl

# theory_bleeding runs its pipeline only under ``__main__``; importing it is safe.
from theory_bleeding import BLUE, GRAY, RED, _expit, _recolor, _write_csv

from statract import (
    accelerated_failure,
    cox_ph,
    cumulative_incidence,
    fine_gray_regression,
    fit_glm,
    plot_survival,
    proportional_hazards_test,
    survival_curve,
)

SHAPE = 0.8  # Weibull shape of the true baseline hazard: high at first, then falling
FOLLOW_UP = 5.0  # years
H5 = 0.20  # cumulative baseline hazard of recurrence at 5 years, lp = 0
DEATH_RATE = 0.02  # per year, death without recurrence at age 65 (women)
CUTS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
SURV_VARS = ["age", "male", "stage3", "t4", "cea_high", "right", "chemo"]
SURV_FORMULA = "Surv(time, event) ~ " + " + ".join(SURV_VARS)

# True log hazard ratios of recurrence (age per year from 65).
TRUE_LOG_HR = {
    "age": 0.03,
    "male": 0.10,
    "stage3": 0.90,
    "t4": 0.70,
    "cea_high": 0.40,
    "right": 0.0,
    "chemo_stage2": -0.10,
    "chemo_stage3": -0.45,
}


def _true_lp(d: pl.DataFrame) -> np.ndarray:
    """True log hazard ratio of recurrence for each patient."""
    b = TRUE_LOG_HR
    chemo_effect = np.where(d["stage3"] == 1, b["chemo_stage3"], b["chemo_stage2"])
    return (
        b["age"] * (d["age"].to_numpy() - 65)
        + b["male"] * d["male"].to_numpy()
        + b["stage3"] * d["stage3"].to_numpy()
        + b["t4"] * d["t4"].to_numpy()
        + b["cea_high"] * d["cea_high"].to_numpy()
        + b["right"] * d["right"].to_numpy()
        + d["chemo"].to_numpy() * chemo_effect
    )


def simulate_crc(n: int = 2000, seed: int = 16) -> pl.DataFrame:
    """Time to recurrence after colon cancer surgery, and death without recurrence.

    Recurrence follows a Weibull proportional hazards model. Death without
    recurrence has its own hazard, constant in time and rising with age, and
    chemotherapy does not change it. ``cause`` is 1 for recurrence, 2 for death
    without recurrence and 0 for censoring; ``event`` is recurrence only.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(67, 10, n), 35, 90).round()
    male = rng.binomial(1, 0.55, n)
    stage3 = rng.binomial(1, 0.5, n)
    t4 = rng.binomial(1, 0.12 + 0.08 * stage3)
    cea_high = rng.binomial(1, 0.25 + 0.1 * stage3)
    right = rng.binomial(1, 0.4, n)
    p_chemo = _expit(-1.4 + 2.6 * stage3 - 0.08 * (age - 65) + 0.5 * t4)
    chemo = rng.binomial(1, p_chemo)
    d = pl.DataFrame(
        {
            "age": age.astype(float),
            "male": male,
            "stage3": stage3,
            "t4": t4,
            "cea_high": cea_high,
            "right": right,
            "chemo": chemo,
        }
    )

    lam = H5 / FOLLOW_UP**SHAPE
    t_rec = (rng.exponential(1.0, n) / (lam * np.exp(_true_lp(d)))) ** (1 / SHAPE)
    death_rate = DEATH_RATE * np.exp(0.09 * (age - 65) + 0.3 * male)
    t_death = rng.exponential(1.0, n) / death_rate

    # Patients entered over 3 years; the study closed 6 years after the first,
    # and follow-up stops at 5 years.
    t_admin = np.minimum(FOLLOW_UP, 6.0 - rng.uniform(0, 3, n))
    t_loss = rng.exponential(1 / 0.02, n)
    t_cens = np.minimum(t_admin, t_loss)
    time = np.minimum.reduce([t_rec, t_death, t_cens])
    cause = np.select([time == t_rec, time == t_death], [1, 2], 0)
    return d.with_columns(
        pl.Series("time", np.maximum(time, 0.01)),
        pl.Series("event", (cause == 1).astype(int)),
        pl.Series("cause", cause),
    )


def _split(d: pl.DataFrame) -> pl.DataFrame:
    """One row per patient and interval: person-years and whether the event happened in it."""
    parts = []
    for a, b in itertools.pairwise(CUTS):
        part = d.filter(pl.col("time") > a).with_columns(
            (pl.min_horizontal(pl.col("time"), pl.lit(b)) - a).alias("years"),
            ((pl.col("event") == 1) & (pl.col("time") <= b))
            .cast(pl.Int64)
            .alias("event"),
            pl.lit(f"{a:g}-{b:g} y").alias("interval"),
        )
        parts.append(part)
    return pl.concat(parts).with_columns(pl.col("years").log().alias("log_years"))


def _risk_set_schematic(path: Path) -> None:
    """Partial likelihood: at each event, who was still at risk?"""
    times = [0.6, 1.1, 1.8, 2.2, 2.7, 3.3, 4.1, 4.8]
    status = [1, 0, 1, 0, 0, 1, 0, 0]
    fig, ax = plt.subplots(figsize=(7, 3))
    t_event = 1.8
    for i, (t, e) in enumerate(zip(times, status, strict=True)):
        at_risk = t >= t_event
        ax.plot(
            [0, t], [i, i], color=BLUE if at_risk else GRAY, lw=2.5 if at_risk else 1.5
        )
        ax.plot(
            t,
            i,
            "x" if e else "o",
            color=RED if e else GRAY,
            ms=8,
            mfc="none" if not e else None,
        )
    ax.axvline(t_event, color=RED, ls=":", lw=1)
    ax.text(
        t_event + 0.06,
        len(times) - 1.5,
        "a recurrence at 1.8 years",
        color=RED,
        fontsize=8,
    )
    ax.set_yticks(
        range(len(times)), [f"Patient {i + 1}" for i in range(len(times))], fontsize=8
    )
    ax.set_xlabel("Years after surgery")
    ax.set_xlim(0, 5.2)
    ax.set_title(
        "blue: still at risk at 1.8 years    x: event    o: censored",
        fontsize=8,
        color=GRAY,
        loc="left",
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _clip_row(model: str, fit, term: str, scale: str, transform=None) -> dict:
    t = fit.tidy().filter(pl.col("term") == term)
    b, se = float(t["estimate"][0]), float(t["std_error"][0])
    f = transform or (lambda v: v)
    lo, hi = sorted([f(b - 1.96 * se), f(b + 1.96 * se)])
    return {
        "model": model,
        "scale": scale,
        "estimate": float(np.exp(f(b))),
        "conf_low": float(np.exp(lo)),
        "conf_high": float(np.exp(hi)),
    }


def _at(curve_t: np.ndarray, curve_v: np.ndarray, t: float) -> float:
    """Value of a right-continuous step function at t."""
    i = np.searchsorted(curve_t, t, side="right") - 1
    return float(curve_v[i]) if i >= 0 else 0.0


def survival(out: Path) -> None:
    """Write ch15_*.csv and their figures into ``out``."""
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d = simulate_crc()
    events = d.filter(pl.col("event") == 1)
    _write_csv(
        out,
        "ch15_summary",
        pl.DataFrame(
            {
                "n": [d.height],
                "recurrences": [events.height],
                "deaths_without_recurrence": [int((d["cause"] == 2).sum())],
                "censored_before_5y": [
                    int(((d["cause"] == 0) & (d["time"] < FOLLOW_UP)).sum())
                ],
                "median_years_to_recurrence": [float(events["time"].median())],
                "recurrences_by_2y": [
                    int(((d["event"] == 1) & (d["time"] <= 2)).sum())
                ],
                "chemo_stage2": [
                    float(d.filter(pl.col("stage3") == 0)["chemo"].mean())
                ],
                "chemo_stage3": [
                    float(d.filter(pl.col("stage3") == 1)["chemo"].mean())
                ],
            }
        ),
    )

    # Kaplan-Meier: 1 - recurrence-free survival by chemotherapy.
    km_data = d.with_columns(
        pl.col("chemo").replace_strict({0: "No chemo", 1: "Chemo"}).alias("group")
    )
    ax = plot_survival(
        km_data,
        time="time",
        status="event",
        hue="group",
        cdf=True,
        show_censors=False,
    )
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Years after surgery")
    ax.set_ylabel("Cumulative incidence of recurrence")
    km_palette = ["#0073C2", "#EFC000"]
    first = km_data["group"].unique(maintain_order=True).to_list()
    house = {"Chemo": BLUE, "No chemo": GRAY}
    _recolor(ax.figure, {km_palette[i]: house[g] for i, g in enumerate(first)})
    ax.figure.savefig(
        out / "figures" / "ch15_km.png", dpi=180, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(ax.figure)

    rates = (
        d.group_by("chemo")
        .agg(
            pl.col("event").sum().alias("events"),
            pl.col("time").sum().alias("person_years"),
        )
        .with_columns(
            (100 * pl.col("events") / pl.col("person_years")).alias("per_100_years")
        )
        .sort("chemo")
    )
    _write_csv(out, "ch15_rates", rates)

    # The hazard over time: piecewise rates, true Weibull, constant.
    split = _split(d)
    piece = (
        split.group_by("interval", maintain_order=True)
        .agg(
            pl.col("event").sum().alias("events"),
            pl.col("years").sum().alias("person_years"),
        )
        .with_columns(
            (100 * pl.col("events") / pl.col("person_years")).alias("per_100_years")
        )
    )
    _write_csv(out, "ch15_piecewise", piece)
    lam = H5 / FOLLOW_UP**SHAPE
    lp = _true_lp(d)
    mean_risk = float(np.mean(np.exp(lp)))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (a, c), r in zip(itertools.pairwise(CUTS), piece["per_100_years"], strict=True):
        ax.plot([a, c], [r, r], color=RED, lw=2.5)
    ax.plot([], [], color=RED, lw=2.5, label="Observed rate in each interval (Poisson)")
    t = np.linspace(0.05, FOLLOW_UP, 300)
    true_h = 100 * SHAPE * lam * t ** (SHAPE - 1) * mean_risk
    ax.plot(t, true_h, color=BLUE, label=f"True hazard (Weibull, shape {SHAPE})")
    overall = 100 * d["event"].sum() / d["time"].sum()
    ax.axhline(overall, color=GRAY, ls="--", label="Constant hazard (exponential)")
    ax.set_ylim(0, float(piece["per_100_years"].max()) * 1.6)
    ax.set_xlabel("Years after surgery")
    ax.set_ylabel("Events per 100 person-years")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch15_hazard.png", dpi=180)
    plt.close(fig)

    _risk_set_schematic(out / "figures" / "ch15_risk_set.png")

    cox = cox_ph(d, SURV_FORMULA)
    ranked = d.with_columns(
        pl.col("time").rank("ordinal").cast(pl.Float64).alias("time")
    )
    cox_rank = cox_ph(ranked, SURV_FORMULA)
    squared = d.with_columns((pl.col("time") ** 2).alias("time"))
    cox_sq = cox_ph(squared, SURV_FORMULA)
    pois_const = fit_glm(
        d.with_columns(pl.col("time").log().alias("log_years")),
        "event ~ " + " + ".join(SURV_VARS),
        family="poisson",
        offset="log_years",
    )
    pois_piece = fit_glm(
        split,
        "event ~ interval + " + " + ".join(SURV_VARS),
        family="poisson",
        offset="log_years",
    )
    weibull = accelerated_failure(d, SURV_FORMULA, distribution="weibull")
    expo = accelerated_failure(d, SURV_FORMULA, distribution="exponential")
    lognormal = accelerated_failure(d, SURV_FORMULA, distribution="lognormal")
    rows = [
        _clip_row("Cox", cox, "chemo", "hazard ratio"),
        _clip_row("Cox, time replaced by its rank", cox_rank, "chemo", "hazard ratio"),
        _clip_row("Cox, time squared", cox_sq, "chemo", "hazard ratio"),
        _clip_row("Poisson, constant hazard", pois_const, "chemo", "hazard ratio"),
        _clip_row("Poisson, 5 intervals", pois_piece, "chemo", "hazard ratio"),
        _clip_row(
            "Weibull, as hazard ratio",
            weibull,
            "chemo",
            "hazard ratio",
            lambda v: -v / weibull.scale,
        ),
        _clip_row("Weibull AFT", weibull, "chemo", "time ratio"),
        _clip_row("Log-normal AFT", lognormal, "chemo", "time ratio"),
        _clip_row("Exponential AFT", expo, "chemo", "time ratio"),
    ]
    _write_csv(out, "ch15_models", pl.DataFrame(rows))
    _write_csv(
        out,
        "ch15_fit",
        pl.DataFrame(
            {
                "model": ["Exponential", "Weibull", "Log-normal"],
                "log_likelihood": [
                    expo.log_likelihood,
                    weibull.log_likelihood,
                    lognormal.log_likelihood,
                ],
                "scale": [expo.scale, weibull.scale, lognormal.scale],
            }
        ),
    )
    _write_csv(out, "ch15_cox", cox.tidy(exponentiate=True))
    _write_csv(out, "ch15_ph_test", proportional_hazards_test(cox))

    competing_risks(d, out)


def competing_risks(d: pl.DataFrame, out: Path) -> None:
    """Recurrence, with death without recurrence as a competing event."""
    d = d.with_columns(
        (pl.col("cause") == 1).cast(pl.Int64).alias("recurrence"),
        (pl.col("cause") == 2).cast(pl.Int64).alias("death"),
        pl.when(pl.col("age") >= 75)
        .then(pl.lit("75 or older"))
        .otherwise(pl.lit("Under 75"))
        .alias("age_group"),
    )
    times = [1.0, 3.0, 4.999]
    rows = []
    groups = [("All", d)] + [
        (g, d.filter(pl.col("age_group") == g)) for g in ["Under 75", "75 or older"]
    ]
    for name, part in groups:
        km = survival_curve(part, "time", "recurrence").at(times)
        aj = cumulative_incidence(part, "time", "cause").at(times)
        aj_rec = aj.filter(pl.col("cause") == "1")["estimate"].to_list()
        aj_death = aj.filter(pl.col("cause") == "2")["estimate"].to_list()
        for i, t in enumerate(times):
            rows.append(
                {
                    "group": name,
                    "n": part.height,
                    "years": round(t),
                    "one_minus_km": 1 - float(km["estimate"][i]),
                    "cif_recurrence": aj_rec[i],
                    "cif_death": aj_death[i],
                }
            )
    _write_csv(out, "ch15_cif", pl.DataFrame(rows))

    # Figure: 1 - KM for recurrence (death censored) vs the CIF.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, (name, part) in zip(
        axes,
        [("All patients", d), ("75 or older", d.filter(pl.col("age") >= 75))],
        strict=True,
    ):
        km = survival_curve(part, "time", "recurrence").frame()
        ax.step(
            np.r_[0, km["time"].to_numpy()],
            np.r_[0, 1 - km["estimate"].to_numpy()],
            where="post",
            color=GRAY,
            ls="--",
            label="1 - Kaplan-Meier (death censored)",
        )
        cif = cumulative_incidence(part, "time", "cause").frame()
        for cause, color, label in [
            ("1", RED, "CIF of recurrence"),
            ("2", BLUE, "CIF of death without recurrence"),
        ]:
            c = cif.filter(pl.col("cause") == cause)
            ax.step(
                np.r_[0, c["time"].to_numpy()],
                np.r_[0, c["estimate"].to_numpy()],
                where="post",
                color=color,
                label=label,
            )
        ax.set_title(f"{name} (n = {part.height})", fontsize=9)
        ax.set_xlabel("Years after surgery")
        ax.set_xlim(0, FOLLOW_UP)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative incidence")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch15_cif.png", dpi=180)
    plt.close(fig)

    rhs = " + ".join(SURV_VARS)
    cs_rec = cox_ph(d, f"Surv(time, recurrence) ~ {rhs}")
    cs_death = cox_ph(d, f"Surv(time, death) ~ {rhs}")
    fg = fine_gray_regression(d, f"Surv(time, cause) ~ {rhs}", cause=1)
    hr_rows = []
    for model, fit, scale in [
        ("Cause-specific Cox, recurrence", cs_rec, "cause-specific hazard ratio"),
        (
            "Cause-specific Cox, death without recurrence",
            cs_death,
            "cause-specific hazard ratio",
        ),
        ("Fine-Gray, recurrence", fg, "subdistribution hazard ratio"),
    ]:
        for term in ["age", "stage3", "chemo"]:
            hr_rows.append({"term": term, **_clip_row(model, fit, term, scale)})
    _write_csv(out, "ch15_competing", pl.DataFrame(hr_rows))


if __name__ == "__main__":
    survival(_HERE / "theory_bleeding_out")
