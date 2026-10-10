from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Figures and tables for the theory text, chapter 1 (the conventional "independent risk factor" analysis)."""


import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from support import ProjectPath, load_data
from statract.report.artifacts import write_csv_companion
from statract import (
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    fit_glm,
    plot_forest,
    write_tableone_artifacts,
)
from build import bleed_logit, simulate

project = ProjectPath(__file__)
CACHE = project.cache / "build" / "bleeding.parquet"
ANALYSIS_OUT = project.out

CANDIDATES = [
    "age",
    "male",
    "antithrombotic",
    "hypertension",
    "size_mm",
    "proximal",
    "clip",
]
LABELS = {
    "age": "Age (per year)",
    "male": "Male",
    "antithrombotic": "Antithrombotic",
    "hypertension": "Hypertension",
    "size_mm": "Lesion size (per mm)",
    "proximal": "Proximal colon",
    "clip": "Prophylactic clip",
}
# Adjustment set read off the DAG for the effect of clip (back-door paths).
DAG_ADJUST = ["age", "antithrombotic", "size_mm", "proximal"]
BLUE, GRAY, RED = "#0073C2", "#868686", "#CD534C"


def _clear_out(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    return out


def _write_csv(out: Path, stem: str, frame: pl.DataFrame) -> None:
    path = out / f"{stem}.csv"
    frame.write_csv(path)
    write_csv_companion(path, csv_link_prefix=out.name)


def _or_rows(fit, terms: list[str]) -> pl.DataFrame:
    return (
        fit.tidy(exponentiate=True)
        .filter(pl.col("term").is_in(terms))
        .select("term", "exp_estimate", "exp_conf_low", "exp_conf_high", "p_value")
    )


def _labelled(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("term").replace(LABELS))


def table1(df: pl.DataFrame, out: Path) -> None:
    params = {
        "Age": ("age", agg_mean_sd),
        "Male": ("male_label", agg_category),
        "Antithrombotic": ("antithrombotic_label", agg_category),
        "Hypertension": ("hypertension_label", agg_category),
        "Lesion size (mm)": ("size_mm", agg_median_iqr),
        "Proximal colon": ("proximal_label", agg_category),
        "Prophylactic clip": ("clip_label", agg_category),
    }
    yes_no = {0: "No", 1: "Yes"}
    shown = df.with_columns(
        *[
            pl.col(c).replace_strict(yes_no).alias(f"{c}_label")
            for c in ["male", "antithrombotic", "hypertension", "proximal", "clip"]
        ],
        pl.col("bleed")
        .replace_strict({0: "No bleeding", 1: "Delayed bleeding"})
        .alias("group"),
    )
    write_tableone_artifacts(
        out,
        "table1",
        df=shown,
        params=params,
        hue="group",
        add_all=True,
        add_pvalue=True,
        column_order=["No bleeding", "Delayed bleeding"],
    )


def conventional(df: pl.DataFrame, out: Path) -> tuple[pl.DataFrame, list[str]]:
    """Univariable screen at P < 0.05, then one multivariable logistic model."""
    uni = pl.concat(
        [
            _or_rows(fit_glm(df, f"bleed ~ {x}", family="binomial"), [x])
            for x in CANDIDATES
        ]
    )
    selected = uni.filter(pl.col("p_value") < 0.05)["term"].to_list()
    multi = _or_rows(
        fit_glm(df, "bleed ~ " + " + ".join(selected), family="binomial"), selected
    )
    _write_csv(out, "univariable", uni)
    _write_csv(out, "multivariable", multi)
    plot_forest(
        _labelled(uni),
        out / "figures" / "forest_univariable.png",
        title="Univariable logistic (OR)",
        xlabel="Odds ratio",
    )
    plot_forest(
        _labelled(multi),
        out / "figures" / "forest_multivariable.png",
        title="Multivariable logistic after P < 0.05 screen (OR)",
        xlabel="Odds ratio",
    )
    return uni, selected


def clip_estimates(df: pl.DataFrame, out: Path) -> pl.DataFrame:
    """The clip OR under three models, next to the truth used in the simulation."""
    crude = _or_rows(
        fit_glm(df, "bleed ~ clip", family="binomial"), ["clip"]
    ).with_columns(pl.lit("Crude (univariable)").alias("model"))
    dag = _or_rows(
        fit_glm(df, "bleed ~ clip + " + " + ".join(DAG_ADJUST), family="binomial"),
        ["clip"],
    ).with_columns(pl.lit("Adjusted for DAG confounders").alias("model"))
    # Truth: marginal risks if everyone / no one were clipped, from the true model on a large sample.
    big = simulate(n=1_000_000, seed=99)
    args = [big[c].to_numpy() for c in ["age", "antithrombotic", "size_mm", "proximal"]]
    r1 = float(np.mean(1 / (1 + np.exp(-bleed_logit(*args, np.ones(big.height))))))
    r0 = float(np.mean(1 / (1 + np.exp(-bleed_logit(*args, np.zeros(big.height))))))
    truth = pl.DataFrame(
        {
            "risk_clip": [r1],
            "risk_no_clip": [r0],
            "risk_difference": [r1 - r0],
            "risk_ratio": [r1 / r0],
            "odds_ratio_marginal": [(r1 / (1 - r1)) / (r0 / (1 - r0))],
        }
    )
    _write_csv(out, "clip_truth", truth)
    table = pl.concat([crude, dag]).select(
        "model", "exp_estimate", "exp_conf_low", "exp_conf_high", "p_value"
    )
    _write_csv(out, "clip_models", table)
    return table


def logistic_curve(df: pl.DataFrame, out: Path) -> None:
    """Size vs bleeding: log odds is a line, probability is an S curve."""
    fit = fit_glm(df, "bleed ~ size_mm", family="binomial")
    tidy = fit.tidy()
    b0 = float(tidy.filter(pl.col("term") == "(Intercept)")["estimate"][0])
    b1 = float(tidy.filter(pl.col("term") == "size_mm")["estimate"][0])
    x = np.linspace(0, 150, 300)
    eta = b0 + b1 * x
    p = 1 / (1 + np.exp(-eta))

    bins = (
        df.with_columns(
            ((pl.col("size_mm") // 10) * 10 + 5).clip(upper_bound=55).alias("mid")
        )
        .group_by("mid")
        .agg(pl.col("bleed").mean().alias("p"), pl.len().alias("n"))
        .sort("mid")
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    lo, hi = int(df["size_mm"].min()), int(df["size_mm"].max())
    for ax in axes:
        ax.axvspan(lo, hi, color=GRAY, alpha=0.12, lw=0)
    ax = axes[0]
    ax.plot(x, eta, color=BLUE)
    ax.set_xlabel("Lesion size (mm)")
    ax.set_ylabel("log odds of bleeding")
    ax.set_title("Linear in log odds")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(x, p, color=BLUE, label="Logistic model")
    ax.scatter(
        bins["mid"],
        bins["p"],
        s=np.sqrt(bins["n"]) * 3,
        color=GRAY,
        zorder=3,
        label="Observed (10 mm bins)",
    )
    ax.set_xlabel("Lesion size (mm)")
    ax.set_ylabel("Probability of bleeding")
    ax.set_ylim(0, 1)
    ax.set_title("S-shaped in probability")
    ax.text((lo + hi) / 2, 0.5, "range of\nthe data", ha="center", color=GRAY)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "logistic_curve.png", dpi=180)
    plt.close(fig)


def confounding_by_indication(df: pl.DataFrame, out: Path) -> None:
    """Within each risk stratum clip lowers bleeding, but clips go to high-risk lesions."""
    strata = df.with_columns(
        pl.when(pl.col("size_mm") >= 20)
        .then(pl.lit(">=20 mm"))
        .otherwise(pl.lit("<20 mm"))
        .alias("size_grp"),
        pl.when(pl.col("antithrombotic") == 1)
        .then(pl.lit("on AT"))
        .otherwise(pl.lit("no AT"))
        .alias("at_grp"),
    ).with_columns((pl.col("size_grp") + ", " + pl.col("at_grp")).alias("stratum"))
    order = ["<20 mm, no AT", "<20 mm, on AT", ">=20 mm, no AT", ">=20 mm, on AT"]
    rates = strata.group_by("stratum", "clip").agg(
        pl.col("bleed").mean().alias("p"), pl.len().alias("n")
    )
    share = strata.group_by("stratum").agg(
        pl.col("clip").mean().alias("clip_share"), pl.len().alias("n")
    )
    overall = df.group_by("clip").agg(
        pl.col("bleed").mean().alias("p"), pl.len().alias("n")
    )
    _write_csv(out, "stratified_rates", rates.sort("stratum", "clip"))
    _write_csv(out, "clip_share", share.sort("stratum"))

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]}
    )
    ax = axes[0]
    xs = np.arange(len(order) + 1)
    w = 0.38
    for k, (clip, color, name) in enumerate([(0, GRAY, "No clip"), (1, BLUE, "Clip")]):
        vals = [
            float(
                rates.filter((pl.col("stratum") == s) & (pl.col("clip") == clip))["p"][
                    0
                ]
            )
            for s in order
        ]
        vals.append(float(overall.filter(pl.col("clip") == clip)["p"][0]))
        ax.bar(xs + (k - 0.5) * w, np.array(vals) * 100, w, color=color, label=name)
    ax.axvline(len(order) - 0.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(xs, [s.replace(", ", "\n") for s in order] + ["All\n(crude)"])
    ax.set_ylabel("Delayed bleeding (%)")
    ax.set_title("Bleeding rate by clip, within strata")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    vals = [
        float(share.filter(pl.col("stratum") == s)["clip_share"][0]) * 100
        for s in order
    ]
    ax.bar(np.arange(len(order)), vals, 0.6, color=RED)
    ax.set_xticks(np.arange(len(order)), [s.replace(", ", "\n") for s in order])
    ax.set_ylabel("Clipped (%)")
    ax.set_title("Who gets a clip")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "confounding_by_indication.png", dpi=180)
    plt.close(fig)


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    df = load_data(CACHE)
    table1(df, out)
    _, selected = conventional(df, out)
    (out / "selected.md").write_text(
        ", ".join(f"`{s}`" for s in selected) + "\n", encoding="utf-8"
    )
    clip_estimates(df, out)
    logistic_curve(df, out)
    confounding_by_indication(df, out)
    n = df.height
    events = int(df["bleed"].sum())
    (out / "n.md").write_text(
        f"{n} 例、遅発性出血 {events} 例（{events / n:.1%}）\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
