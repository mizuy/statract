from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Figures and tables for the theory text.

Chapter 1: the conventional "independent risk factor" analysis.
Chapter 2: potential outcomes and causal effects.
Chapter 3: confounding and DAGs.
"""


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
    prop_test,
    standardize_glm,
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    fit_glm,
    plot_forest,
    write_tableone_artifacts,
)
from build import bleed_logit, potential_outcomes, simulate

project = ProjectPath(__file__)
CACHE = project.cache / "build" / "bleeding.parquet"
POTENTIAL = project.cache / "build" / "potential.parquet"
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


def _effect_plot(
    rows: pl.DataFrame, path: Path, *, title: str, truth: float | None = None
) -> None:
    """Risk differences as points with intervals; rows whose interval is a point are drawn as truths."""
    n = rows.height
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * n + 1.3))
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows.iter_rows(named=True)):
        known = r["conf_low"] == r["conf_high"]
        if known:
            ax.plot(r["estimate"], yi, marker="D", color=RED, ms=7, zorder=3)
        else:
            ax.plot([r["conf_low"], r["conf_high"]], [yi, yi], color=BLUE, lw=2)
            ax.plot(r["estimate"], yi, marker="s", color=BLUE, ms=7, zorder=3)
        ax.text(
            1.02,
            yi,
            f"{r['estimate']:.1f}"
            + ("" if known else f" ({r['conf_low']:.1f} to {r['conf_high']:.1f})"),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    if truth is not None:
        ax.axvline(truth, color=RED, lw=1, ls=":")
    ax.set_yticks(y, rows["term"].to_list())
    ax.set_xlabel("Risk difference, clip minus no clip (percentage points)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---- Chapter 2: potential outcomes ----------------------------------------


def _risk_rows(frame: pl.DataFrame, label: str) -> dict:
    r0, r1 = float(frame["y0"].mean()), float(frame["y1"].mean())
    o0, o1 = r0 / (1 - r0), r1 / (1 - r1)
    return {
        "group": label,
        "n": frame.height,
        "risk_no_clip": r0,
        "risk_clip": r1,
        "risk_difference": r1 - r0,
        "risk_ratio": r1 / r0,
        "odds_ratio": o1 / o0,
    }


def _signed(v: int) -> str:
    return "0" if v == 0 else f"{v:+d}"


def potential_table(po: pl.DataFrame, out: Path) -> None:
    """A few patients with both potential outcomes; the observed one is the arm they got."""
    picks = []
    for (y0, y1, clip), k in [
        ((0, 0, 0), 2),
        ((1, 0, 1), 2),
        ((1, 1, 0), 1),
        ((1, 0, 0), 1),
        ((0, 0, 1), 2),
    ]:
        rows = po.filter(
            (pl.col("y0") == y0) & (pl.col("y1") == y1) & (pl.col("clip") == clip)
        ).head(k)
        picks.append(rows)
    shown = (
        pl.concat(picks)
        .sort("id")
        .select(
            "id",
            "size_mm",
            "antithrombotic",
            "clip",
            "y0",
            "y1",
            pl.when(pl.col("clip") == 1)
            .then(pl.col("y1"))
            .otherwise(pl.col("y0"))
            .alias("bleed"),
        )
    )
    _write_csv(out, "potential_outcomes", shown)

    def cell(v: int, seen: bool) -> str:
        return str(v) if seen else f"({v})"

    lines = [
        "| 患者 | 病変径 (mm) | 抗血栓薬 | クリップ | $Y(0)$ | $Y(1)$ | 個人の効果 $Y(1) - Y(0)$ |",
        "|------|-------------|----------|----------|--------|--------|------------------------|",
    ]
    for r in shown.iter_rows(named=True):
        lines.append(
            f"| {r['id']} | {r['size_mm']} | {'あり' if r['antithrombotic'] else 'なし'} | "
            f"{'した' if r['clip'] else 'しない'} | {cell(r['y0'], r['clip'] == 0)} | "
            f"{cell(r['y1'], r['clip'] == 1)} | {_signed(r['y1'] - r['y0'])} |"
        )
    (out / "potential_outcomes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def causal_effects(po: pl.DataFrame, out: Path) -> pl.DataFrame:
    """Association vs ATE vs ATT, and what a randomized trial on the same patients would show."""
    obs = prop_test(
        [
            int(po.filter(pl.col("clip") == 1)["bleed"].sum()),
            int(po.filter(pl.col("clip") == 0)["bleed"].sum()),
        ],
        [po.filter(pl.col("clip") == 1).height, po.filter(pl.col("clip") == 0).height],
        correct=False,
    )
    rng = np.random.default_rng(7)
    rct = po.with_columns(
        pl.Series("arm", rng.binomial(1, 0.5, po.height))
    ).with_columns(
        pl.when(pl.col("arm") == 1)
        .then(pl.col("y1"))
        .otherwise(pl.col("y0"))
        .alias("y_rct")
    )
    t1, t0 = rct.filter(pl.col("arm") == 1), rct.filter(pl.col("arm") == 0)
    rct_test = prop_test(
        [int(t1["y_rct"].sum()), int(t0["y_rct"].sum())],
        [t1.height, t0.height],
        correct=False,
    )
    ate = float(po["y1"].mean() - po["y0"].mean())
    treated = po.filter(pl.col("clip") == 1)
    att = float(treated["y1"].mean() - treated["y0"].mean())
    obs_rd = obs.estimates["prop 1"] - obs.estimates["prop 2"]
    rct_rd = rct_test.estimates["prop 1"] - rct_test.estimates["prop 2"]
    rows = pl.DataFrame(
        {
            "term": [
                "Observed difference (clip vs no clip)",
                "True ATE (whole sample)",
                "True ATT (clipped patients)",
                "Randomized trial on the same patients",
            ],
            "estimate": [obs_rd * 100, ate * 100, att * 100, rct_rd * 100],
            "conf_low": [
                obs.conf_int[0] * 100,
                ate * 100,
                att * 100,
                rct_test.conf_int[0] * 100,
            ],
            "conf_high": [
                obs.conf_int[1] * 100,
                ate * 100,
                att * 100,
                rct_test.conf_int[1] * 100,
            ],
        }
    )
    _write_csv(out, "ch2_effects", rows)
    _effect_plot(
        rows,
        out / "figures" / "ch2_effects.png",
        title="Association and causal effects of clip",
    )

    # Exchangeability: the two groups differ in their risk without a clip.
    by_group = (
        po.group_by("clip")
        .agg(pl.col("y0").mean(), pl.col("y1").mean(), pl.len())
        .sort("clip")
    )
    _write_csv(out, "ch2_by_group", by_group)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = np.arange(2)
    w = 0.36
    for k, (col, color, name) in enumerate(
        [("y0", GRAY, "If not clipped, Y(0)"), ("y1", BLUE, "If clipped, Y(1)")]
    ):
        vals = by_group[col].to_numpy() * 100
        bars = ax.bar(xs + (k - 0.5) * w, vals, w, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(xs, ["Not clipped\n(actual group)", "Clipped\n(actual group)"])
    ax.set_ylabel("Delayed bleeding (%)")
    ax.set_title("Both potential outcomes, by the group patients were in")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, by_group["y0"].max() * 100 * 1.35)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch2_exchangeability.png", dpi=180)
    plt.close(fig)

    # Effect measures overall and by lesion size.
    large = po.with_columns((pl.col("size_mm") >= 20).alias("large"))
    measures = pl.DataFrame(
        [
            _risk_rows(po, "All"),
            _risk_rows(large.filter(~pl.col("large")), "<20 mm"),
            _risk_rows(large.filter(pl.col("large")), ">=20 mm"),
        ]
    )
    _write_csv(out, "ch2_measures", measures)

    return rows


# ---- Chapter 3: confounding and DAGs --------------------------------------

ADJUSTMENT_SETS = {
    "None": [],
    "Lesion size": ["size_mm"],
    "Antithrombotic": ["antithrombotic"],
    "Antithrombotic + size": ["antithrombotic", "size_mm"],
    "Antithrombotic + size + proximal": ["antithrombotic", "size_mm", "proximal"],
    "... + age": ["antithrombotic", "size_mm", "proximal", "age"],
    "... + age + hypertension": [
        "antithrombotic",
        "size_mm",
        "proximal",
        "age",
        "hypertension",
    ],
}


def adjustment_sets(out: Path) -> None:
    """Standardized risk difference of clip under each adjustment set.

    Uses 30,000 patients from the same mechanism, so that bias, not chance, drives the differences.
    """
    po = potential_outcomes(simulate(n=30_000, seed=2))
    # Lesion size enters as a line plus a step at 20 mm, the form the clip effect really has.
    df = po.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large"))
    rows = []
    for name, covs in ADJUSTMENT_SETS.items():
        terms = [
            t for c in covs for t in (["size_mm", "large"] if c == "size_mm" else [c])
        ]
        formula = "bleed ~ clip" + (" * (" + " + ".join(terms) + ")" if terms else "")
        std = standardize_glm(df, formula, values={"clip": [0, 1]})
        t = std.tidy(contrast="difference", reference=0).filter(pl.col("clip") == 1)
        rows.append(
            {
                "term": name,
                "estimate": float(t["estimate"][0]) * 100,
                "conf_low": float(t["conf_low"][0]) * 100,
                "conf_high": float(t["conf_high"][0]) * 100,
            }
        )
    ate = float(po["y1"].mean() - po["y0"].mean()) * 100
    rows.append(
        {"term": "Truth (ATE)", "estimate": ate, "conf_low": ate, "conf_high": ate}
    )
    table = pl.DataFrame(rows)
    _write_csv(out, "ch3_adjustment_sets", table)
    _effect_plot(
        table,
        out / "figures" / "ch3_adjustment_sets.png",
        truth=ate,
        title="Standardized effect of clip by adjustment set (30,000 patients)",
    )


def berkson(df: pl.DataFrame, out: Path) -> None:
    """Selection on a collider: only referred patients, where referral depends on size and antithrombotics."""
    rng = np.random.default_rng(11)
    large = (df["size_mm"] >= 20).to_numpy().astype(int)
    at = df["antithrombotic"].to_numpy()
    referred = rng.binomial(1, _expit(-3.0 + 3.0 * large + 3.0 * at))
    d = df.with_columns(pl.Series("large", large), pl.Series("referred", referred))
    rows = []
    for name, sub in [
        ("All patients", d),
        ("Referred only", d.filter(pl.col("referred") == 1)),
    ]:
        for lg in [0, 1]:
            g = sub.filter(pl.col("large") == lg)
            rows.append(
                {
                    "sample": name,
                    "large": lg,
                    "n": g.height,
                    "on_antithrombotic": float(g["antithrombotic"].mean()),
                }
            )
        f = _or_rows(
            fit_glm(sub, "antithrombotic ~ large", family="binomial"), ["large"]
        )
        rows[-1]["or_large_vs_small"] = float(f["exp_estimate"][0])
        rows[-2]["or_large_vs_small"] = float(f["exp_estimate"][0])
    table = pl.DataFrame(rows)
    _write_csv(out, "ch3_berkson", table)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = np.arange(2)
    w = 0.36
    for k, (lg, color, name) in enumerate(
        [(0, GRAY, "Lesion <20 mm"), (1, BLUE, "Lesion >=20 mm")]
    ):
        vals = [
            float(
                table.filter((pl.col("sample") == s) & (pl.col("large") == lg))[
                    "on_antithrombotic"
                ][0]
            )
            * 100
            for s in ["All patients", "Referred only"]
        ]
        bars = ax.bar(xs + (k - 0.5) * w, vals, w, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=9)
    ax.set_xticks(xs, ["All patients", "Referred only"])
    ax.set_ylabel("On antithrombotics (%)")
    ax.set_title("Selecting on a collider creates an association")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch3_berkson.png", dpi=180)
    plt.close(fig)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    po = load_data(POTENTIAL)
    potential_table(po, out)
    causal_effects(po, out)
    adjustment_sets(out)
    berkson(df, out)
    n = df.height
    events = int(df["bleed"].sum())
    (out / "n.md").write_text(
        f"{n} 例、遅発性出血 {events} 例（{events / n:.1%}）\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
