from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Figures and tables for the theory text.

Part I (chapter 1): the conventional "independent risk factor" analysis.
Part II (chapters 2-6): potential outcomes, adjustment, PS matching, IPTW, sensitivity.
Part III (chapters 7-11): prediction, overfitting, discrimination, calibration, regularization, trees.
Part IV (chapters 12-15): probability models, likelihood, hierarchical models, survival.
"""


import itertools
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl
from scipy import stats
from statract import (
    accelerated_failure,
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    brier_score,
    conditional_tree,
    cox_ph,
    decision_curve_table,
    fit_glm,
    fit_mixed,
    glmm_cluster_variance,
    hc_covariance,
    likelihood_ratio_test,
    match_sample,
    plot_forest,
    plot_love,
    plot_random_effects,
    plot_roc,
    plot_survival,
    plot_tree,
    prop_test,
    propensity_weights,
    proportional_hazards_test,
    roc_curve,
    spline_test,
    standardize_glm,
    threshold_tradeoff,
    validate_logistic,
    write_tableone_artifacts,
)
from statract.report.artifacts import write_csv_companion
from support import ProjectPath, load_data

import penalized as pen
from build import TRUE_BLEED, bleed_logit, potential_outcomes, simulate

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

    return rows


# ---- Chapter 3: adjustment methods; Chapter 4: propensity score matching ----

COVARIATES = ["age", "antithrombotic", "size_mm", "proximal"]
PS_FORMULA = "clip ~ " + " + ".join(COVARIATES)


def _rd_row(name: str, estimand: str, est: float, se: float) -> dict:
    return {
        "term": name,
        "estimand": estimand,
        "estimate": est * 100,
        "conf_low": (est - 1.96 * se) * 100,
        "conf_high": (est + 1.96 * se) * 100,
    }


def _weighted_rd(df: pl.DataFrame, weights: np.ndarray) -> tuple[float, float]:
    """Weighted risk difference with a sandwich (HC0) standard error."""
    fit = fit_glm(
        df.with_columns(pl.Series("w", weights)),
        "bleed ~ clip",
        family="gaussian",
        weights="w",
    )
    fit.covariance = hc_covariance(fit, kind="HC0")
    t = fit.tidy().filter(pl.col("term") == "clip")
    return float(t["estimate"][0]), float(t["std_error"][0])


def stratification(df: pl.DataFrame, out: Path) -> tuple[float, float]:
    """Risk difference within strata of size group x antithrombotic x location, averaged over the sample."""
    d = df.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large"))
    keys = ["large", "antithrombotic", "proximal"]
    strata = (
        d.group_by(keys)
        .agg(
            pl.len().alias("n"),
            pl.col("clip").sum().alias("n_clip"),
            pl.col("bleed").filter(pl.col("clip") == 1).mean().alias("risk_clip"),
            pl.col("bleed").filter(pl.col("clip") == 0).mean().alias("risk_no_clip"),
        )
        .with_columns(
            (pl.col("n") - pl.col("n_clip")).alias("n_no_clip"),
            (pl.col("risk_clip") - pl.col("risk_no_clip")).alias("risk_difference"),
        )
        .sort(keys)
    )
    _write_csv(
        out,
        "ch3_strata",
        strata.select(
            *keys,
            "n",
            "n_clip",
            "n_no_clip",
            "risk_no_clip",
            "risk_clip",
            "risk_difference",
        ),
    )
    w = strata["n"].to_numpy() / d.height
    rd = float(np.sum(w * strata["risk_difference"].to_numpy()))
    var = np.sum(
        w**2
        * (
            strata["risk_clip"].to_numpy()
            * (1 - strata["risk_clip"].to_numpy())
            / strata["n_clip"].to_numpy()
            + strata["risk_no_clip"].to_numpy()
            * (1 - strata["risk_no_clip"].to_numpy())
            / strata["n_no_clip"].to_numpy()
        )
    )
    return rd, float(np.sqrt(var))


def adjustment_methods(df: pl.DataFrame, po: pl.DataFrame, out: Path):
    """Crude, restriction, stratification, regression standardization, PS matching and IPTW side by side."""
    rows = []
    crude = _weighted_rd(df, np.ones(df.height))
    rows.append(_rd_row("Crude (no adjustment)", "-", *crude))

    small = df.filter((pl.col("size_mm") < 20) & (pl.col("antithrombotic") == 0))
    r = _weighted_rd(small, np.ones(small.height))
    rows.append(_rd_row("Restriction (<20 mm, no antithrombotic)", "restricted", *r))

    rows.append(_rd_row("Stratification (8 strata)", "ATE", *stratification(df, out)))

    std = standardize_glm(
        df, "bleed ~ clip + " + " + ".join(COVARIATES), values={"clip": [0, 1]}
    )
    t = std.tidy(contrast="difference", reference=0).filter(pl.col("clip") == 1)
    rows.append(
        _rd_row(
            "Regression + standardization",
            "ATE",
            float(t["estimate"][0]),
            float(t["std_error"][0]),
        )
    )

    matched = match_sample(df, "clip", COVARIATES, caliper=0.2)
    frame = matched.frame()
    kept = frame.filter(pl.col("weights") > 0)
    r = _weighted_rd(kept, kept["weights"].to_numpy())
    rows.append(_rd_row("Propensity score matching", "ATT", *r))

    ipw = propensity_weights(df, PS_FORMULA, estimand="ATE")
    r = _weighted_rd(df, np.asarray(ipw.weights))
    rows.append(_rd_row("Propensity score weighting (IPTW)", "ATE", *r))

    ate = float(po["y1"].mean() - po["y0"].mean())
    treated = po.filter(pl.col("clip") == 1)
    att = float(treated["y1"].mean() - treated["y0"].mean())
    restricted = po.filter((pl.col("size_mm") < 20) & (pl.col("antithrombotic") == 0))
    rest = float(restricted["y1"].mean() - restricted["y0"].mean())
    rows.append(_rd_row("Truth: ATE (all patients)", "ATE", ate, 0.0))
    rows.append(_rd_row("Truth: ATT (clipped patients)", "ATT", att, 0.0))
    rows.append(_rd_row("Truth: restricted group", "restricted", rest, 0.0))
    table = pl.DataFrame(rows)
    _write_csv(out, "ch3_methods", table)
    _effect_plot(
        table.drop("estimand"),
        out / "figures" / "ch3_methods.png",
        truth=ate * 100,
        title="Effect of clip by adjustment method",
    )
    return matched, ipw


def ps_matching(df: pl.DataFrame, po: pl.DataFrame, matched, out: Path) -> None:
    """Propensity score overlap, balance before and after matching, and the matched estimate."""
    frame = matched.frame()
    ps = frame["distance"].to_numpy()
    clip = frame["clip"].to_numpy()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    h1, _ = np.histogram(ps[clip == 1], bins=bins)
    h0, _ = np.histogram(ps[clip == 0], bins=bins)
    mids = (bins[:-1] + bins[1:]) / 2
    ax.bar(mids, h1, width=bins[1] - bins[0], color=BLUE, label="Clipped")
    ax.bar(mids, -h0, width=bins[1] - bins[0], color=GRAY, label="Not clipped")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Propensity score (probability of being clipped)")
    ax.set_ylabel("Patients")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: f"{abs(int(v))}")
    )
    ax.set_title("Propensity score by actual group")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch4_ps_overlap.png", dpi=180)
    plt.close(fig)

    balance = (
        matched.balance()
        .filter(pl.col("term") != "distance")
        .with_columns(
            pl.col("term").replace(
                {
                    "age": "Age",
                    "antithrombotic": "Antithrombotic",
                    "size_mm": "Lesion size",
                    "proximal": "Proximal colon",
                }
            )
        )
    )
    _write_csv(out, "ch4_balance", balance)
    plot_love(
        balance.rename({"smd_all": "diff_unadjusted", "smd_matched": "diff_adjusted"}),
        out / "figures" / "ch4_love_plot.png",
        threshold=0.1,
        title="Balance before and after matching",
        xlabel="Absolute standardized mean difference",
    )

    ps_fit = fit_glm(df, PS_FORMULA, family="binomial")
    _write_csv(out, "ch4_ps_model", _or_rows(ps_fit, COVARIATES))

    kept = frame.filter(pl.col("weights") > 0)
    n_treated = int(df["clip"].sum())
    n_kept_treated = int(kept.filter(pl.col("clip") == 1).height)
    risks = (
        kept.group_by("clip")
        .agg(pl.col("bleed").mean().alias("risk"), pl.len().alias("n"))
        .sort("clip")
    )
    rd, se = _weighted_rd(kept, kept["weights"].to_numpy())
    kept_ids = kept.filter(pl.col("clip") == 1)["id"]
    truth = po.filter(pl.col("id").is_in(kept_ids.implode()))
    summary = pl.DataFrame(
        {
            "n_treated": [n_treated],
            "n_treated_matched": [n_kept_treated],
            "n_controls_matched": [int(kept.height - n_kept_treated)],
            "risk_no_clip_matched": [
                float(risks.filter(pl.col("clip") == 0)["risk"][0])
            ],
            "risk_clip_matched": [float(risks.filter(pl.col("clip") == 1)["risk"][0])],
            "risk_difference": [rd],
            "conf_low": [rd - 1.96 * se],
            "conf_high": [rd + 1.96 * se],
            "truth_att_matched_treated": [
                float(truth["y1"].mean() - truth["y0"].mean())
            ],
        }
    )
    _write_csv(out, "ch4_matched_summary", summary)

    params = {
        "Age": ("age", agg_mean_sd),
        "Antithrombotic": ("antithrombotic_label", agg_category),
        "Lesion size (mm)": ("size_mm", agg_median_iqr),
        "Proximal colon": ("proximal_label", agg_category),
    }
    yes_no = {0: "No", 1: "Yes"}

    def shown(d: pl.DataFrame) -> pl.DataFrame:
        return d.with_columns(
            *[
                pl.col(c).replace_strict(yes_no).alias(f"{c}_label")
                for c in ["antithrombotic", "proximal"]
            ],
            pl.col("clip").replace_strict({0: "No clip", 1: "Clip"}).alias("group"),
        )

    write_tableone_artifacts(
        out,
        "ch4_table1_before",
        df=shown(df),
        params=params,
        hue="group",
        add_smd=True,
        column_order=["No clip", "Clip"],
    )
    write_tableone_artifacts(
        out,
        "ch4_table1_after",
        df=shown(kept),
        params=params,
        hue="group",
        add_smd=True,
        column_order=["No clip", "Clip"],
    )


# ---- Chapter 5: IPTW; Chapter 6: assumptions and sensitivity ------------


def _truth_weighted(po: pl.DataFrame, w: np.ndarray) -> float:
    return float(np.sum(w * (po["y1"].to_numpy() - po["y0"].to_numpy())) / np.sum(w))


def iptw(df: pl.DataFrame, po: pl.DataFrame, out: Path) -> None:
    """Weights, weighted balance, and estimates under several weighting choices."""
    ate = propensity_weights(df, PS_FORMULA, estimand="ATE")
    w = np.asarray(ate.weights)
    clip = df["clip"].to_numpy()
    _write_csv(out, "ch5_weight_summary", ate.summary())

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(0, np.log10(w.max() * 1.1), 40)
    ax.hist(w[clip == 1], bins=bins, color=BLUE, alpha=0.85, label="Clipped: 1 / e")
    ax.hist(
        w[clip == 0],
        bins=bins,
        color=GRAY,
        alpha=0.85,
        label="Not clipped: 1 / (1 - e)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weight (log scale)")
    ax.set_ylabel("Patients (log scale)")
    ax.set_title("ATE weights")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch5_weights.png", dpi=180)
    plt.close(fig)

    bal = (
        ate.balance()
        .filter(pl.col("term") != "prop.score")
        .with_columns(
            pl.col("term").replace(
                {
                    "age": "Age",
                    "antithrombotic": "Antithrombotic",
                    "size_mm": "Lesion size",
                    "proximal": "Proximal colon",
                }
            )
        )
    )
    _write_csv(out, "ch5_balance", bal)
    plot_love(
        bal,
        out / "figures" / "ch5_love_plot.png",
        threshold=0.1,
        title="Balance before and after weighting (ATE)",
        xlabel="Absolute standardized mean difference",
    )

    e = np.asarray(ate.ps)
    rows = []
    variants = [
        ("ATE weights", "ATE", {"estimand": "ATE"}, None),
        ("ATE, stabilized", "ATE", {"estimand": "ATE", "stabilize": True}, None),
        (
            "ATE, largest 1% of weights capped",
            "ATE",
            {"estimand": "ATE", "trim": 0.99},
            None,
        ),
        ("ATT weights", "ATT", {"estimand": "ATT"}, "att"),
        ("Overlap weights (ATO)", "ATO", {"estimand": "ATO"}, "ato"),
    ]
    truth_ate = float(po["y1"].mean() - po["y0"].mean())
    truth = {
        "ATE": truth_ate,
        "ATT": _truth_weighted(po, clip.astype(float)),
        "ATO": _truth_weighted(po, e * (1 - e)),
    }
    for name, estimand, kw, _ in variants:
        pw = propensity_weights(df, PS_FORMULA, **kw)
        r = _weighted_rd(df, np.asarray(pw.weights))
        row = _rd_row(name, estimand, *r)
        row["max_weight"] = float(np.max(pw.weights))
        rows.append(row)
    for k in ["ATE", "ATT", "ATO"]:
        rows.append({**_rd_row(f"Truth: {k}", k, truth[k], 0.0), "max_weight": None})
    table = pl.DataFrame(rows)
    _write_csv(out, "ch5_estimates", table)
    _effect_plot(
        table.drop("estimand", "max_weight"),
        out / "figures" / "ch5_estimates.png",
        truth=truth_ate * 100,
        title="IPTW estimates of the clip effect",
    )


def sensitivity(df: pl.DataFrame, out: Path) -> None:
    """What an unmeasured confounder does, and the E-value."""
    big = potential_outcomes(simulate(n=100_000, seed=2)).with_columns(
        (pl.col("size_mm") >= 20).cast(pl.Int64).alias("large")
    )
    ate = float(big["y1"].mean() - big["y0"].mean())
    rows = []
    # The outcome model has the true form: clip works differently at 20 mm or more.
    for name, formula in [
        (
            "All four confounders",
            "bleed ~ clip + clip:large + age + antithrombotic + size_mm + proximal",
        ),
        (
            "Antithrombotic not measured",
            "bleed ~ clip + clip:large + age + size_mm + proximal",
        ),
        ("Lesion size not measured", "bleed ~ clip + age + antithrombotic + proximal"),
    ]:
        std = standardize_glm(big, formula, values={"clip": [0, 1]})
        t = std.tidy(contrast="difference", reference=0).filter(pl.col("clip") == 1)
        rows.append(
            _rd_row(name, "ATE", float(t["estimate"][0]), float(t["std_error"][0]))
        )
    rows.append(_rd_row("Truth (ATE)", "ATE", ate, 0.0))
    table = pl.DataFrame(rows)
    _write_csv(out, "ch6_unmeasured", table)
    _effect_plot(
        table.drop("estimand"),
        out / "figures" / "ch6_unmeasured.png",
        truth=ate * 100,
        title="When a confounder is not measured (100,000 patients)",
    )

    # E-value for the standardized risk ratio in the 3000-patient data.
    std = standardize_glm(
        df, "bleed ~ clip + " + " + ".join(COVARIATES), values={"clip": [0, 1]}
    )
    rr = std.tidy(contrast="ratio", reference=0, ci="log").filter(pl.col("clip") == 1)
    est, lo, hi = (
        float(rr["estimate"][0]),
        float(rr["conf_low"][0]),
        float(rr["conf_high"][0]),
    )

    def evalue(r: float) -> float:
        r = 1 / r if r < 1 else r
        return r + np.sqrt(r * (r - 1))

    near = hi if est < 1 else lo
    ev = pl.DataFrame(
        {
            "risk_ratio": [est],
            "conf_low": [lo],
            "conf_high": [hi],
            "e_value": [evalue(est)],
            "e_value_ci": [evalue(near) if (near < 1) == (est < 1) else 1.0],
        }
    )
    _write_csv(out, "ch6_evalue", ev)

    r = 1 / est if est < 1 else est
    e_pt = evalue(est)
    x = np.linspace(r * 1.0001, 12, 400)
    y = r * (1 - x) / (r - x)  # RR_UY that, with RR_AU = x, just explains away r
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ok = (y > 0) & (y < 12)
    ax.plot(x[ok], y[ok], color=BLUE)
    ax.fill_between(x[ok], y[ok], 12, color=BLUE, alpha=0.12)
    ax.plot([e_pt], [e_pt], marker="o", color=RED)
    ax.annotate(
        f"E-value {e_pt:.2f}",
        (e_pt, e_pt),
        textcoords="offset points",
        xytext=(8, 8),
        color=RED,
    )
    ax.set_xlim(1, 12)
    ax.set_ylim(1, 12)
    ax.set_xlabel("Unmeasured confounder vs clip (risk ratio)")
    ax.set_ylabel("Unmeasured confounder vs bleeding (risk ratio)")
    ax.set_title("Strength needed to explain away the effect")
    ax.text(7.5, 9.5, "could explain\naway the effect", ha="center", color=BLUE)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch6_evalue.png", dpi=180)
    plt.close(fig)

    ps = np.asarray(propensity_weights(df, PS_FORMULA).ps)
    clip = df["clip"].to_numpy()
    pos = pl.DataFrame(
        {
            "range": ["PS < 0.05", "0.05 <= PS <= 0.95", "PS > 0.95"],
            "clipped": [
                int(((ps < 0.05) & (clip == 1)).sum()),
                int(((ps >= 0.05) & (ps <= 0.95) & (clip == 1)).sum()),
                int(((ps > 0.95) & (clip == 1)).sum()),
            ],
            "not_clipped": [
                int(((ps < 0.05) & (clip == 0)).sum()),
                int(((ps >= 0.05) & (ps <= 0.95) & (clip == 0)).sum()),
                int(((ps > 0.95) & (clip == 0)).sum()),
            ],
        }
    )
    _write_csv(out, "ch6_positivity", pos)


# Part III: prediction. The test set is a large new sample from the same population.
PRED_VARS = [
    "age",
    "male",
    "antithrombotic",
    "hypertension",
    "size_mm",
    "proximal",
    "clip",
]
PRED_FORMULA = "bleed ~ " + " + ".join(PRED_VARS)
NOISE_VARS = [f"lab{i}" for i in range(1, 11)]


def _test_set(n: int = 50_000, seed: int = 3) -> pl.DataFrame:
    return simulate(n=n, seed=seed)


def _with_noise(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Add ten lab values that have nothing to do with bleeding."""
    rng = np.random.default_rng(seed)
    return df.with_columns(
        [pl.Series(v, rng.normal(size=df.height)) for v in NOISE_VARS]
    )


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_curve(y, p).auc)


def _slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration intercept (with slope 1) and slope on new data."""
    lp = np.log(p / (1 - p))
    d = pl.DataFrame({"y": y, "lp": lp})
    slope = float(fit_glm(d, "y ~ lp", family="binomial").tidy()["estimate"][1])
    d = d.with_columns(pl.col("lp").alias("off"))
    citl = float(
        fit_glm(d, "y ~ 1", family="binomial", offset="off").tidy()["estimate"][0]
    )
    return citl, slope


def prediction_model(df: pl.DataFrame, out: Path) -> None:
    """Chapter 7: the same logistic regression, used to predict."""
    fit = fit_glm(df, PRED_FORMULA, family="binomial")
    _write_csv(out, "ch7_model", _labelled(_or_rows(fit, PRED_VARS)))
    p = np.asarray(fit.predict(df, kind="response"))
    y = df["bleed"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    bins = np.linspace(0, max(0.4, float(p.max())), 41)
    for ax, val, color, label in [
        (axes[0], 0, GRAY, "No bleeding"),
        (axes[1], 1, RED, "Delayed bleeding"),
    ]:
        ax.hist(p[y == val], bins=bins, color=color, alpha=0.85)
        ax.set_ylabel("Patients")
        ax.set_title(f"{label} (n = {int((y == val).sum())})", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Predicted probability of delayed bleeding")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch7_predicted.png", dpi=180)
    plt.close(fig)

    patients = pl.DataFrame(
        {
            "patient": ["A", "B", "C"],
            "age": [60, 75, 80],
            "male": [1, 0, 1],
            "antithrombotic": [0, 0, 1],
            "hypertension": [0, 1, 1],
            "size_mm": [8, 20, 40],
            "proximal": [0, 1, 1],
            "clip": [0, 1, 1],
        }
    )
    risk = np.asarray(fit.predict(patients, kind="response"))
    _write_csv(
        out, "ch7_patients", patients.with_columns(pl.Series("risk_pct", risk * 100))
    )

    # Without the antithrombotic column, hypertension (no arrow into bleeding) picks up its signal.
    no_at = [v for v in PRED_VARS if v != "antithrombotic"]
    fit2 = fit_glm(df, "bleed ~ " + " + ".join(no_at), family="binomial")
    test = _test_set()
    yt = test["bleed"].to_numpy()
    rows = []
    for name, f in [("All seven", fit), ("Antithrombotic not recorded", fit2)]:
        t = f.tidy(exponentiate=True).filter(pl.col("term") == "hypertension")
        rows.append(
            {
                "model": name,
                "hypertension_or": float(t["exp_estimate"][0]),
                "conf_low": float(t["exp_conf_low"][0]),
                "conf_high": float(t["exp_conf_high"][0]),
                "p_value": float(t["p_value"][0]),
                "auc_test": _auc(yt, np.asarray(f.predict(test, kind="response"))),
            }
        )
    _write_csv(out, "ch7_hypertension", pl.DataFrame(rows))


def overfitting(out: Path) -> None:
    """Chapter 8: apparent vs new-patient performance, learning curve, bootstrap."""
    test = _with_noise(_test_set(), seed=30)
    yt = test["bleed"].to_numpy()
    big = PRED_FORMULA + " + " + " + ".join(NOISE_VARS)

    dev = _with_noise(simulate(n=400, seed=4), seed=40)
    yd = dev["bleed"].to_numpy()
    rows = []
    for name, formula, k in [
        ("7 predictors", PRED_FORMULA, len(PRED_VARS)),
        ("7 predictors + 10 noise labs", big, len(PRED_VARS) + len(NOISE_VARS)),
    ]:
        fit = fit_glm(dev, formula, family="binomial")
        pa = np.asarray(fit.predict(dev, kind="response"))
        pt = np.asarray(fit.predict(test, kind="response"))
        _, slope = _slope(yt, pt)
        rows.append(
            {
                "model": name,
                "n": dev.height,
                "events": int(yd.sum()),
                "parameters": k,
                "events_per_parameter": yd.sum() / k,
                "auc_apparent": _auc(yd, pa),
                "auc_new": _auc(yt, pt),
                "slope_new": slope,
            }
        )
    _write_csv(out, "ch8_apparent", pl.DataFrame(rows))

    # Learning curve: average over repeated development samples of each size.
    sizes = [300, 600, 1200, 2400, 4800]
    reps = 30
    curve = []
    for n in sizes:
        app, new = [], []
        for r in range(reps):
            d = _with_noise(
                simulate(n=n, seed=1000 + 37 * n + r), seed=5000 + 37 * n + r
            )
            fit = fit_glm(d, big, family="binomial")
            app.append(
                _auc(d["bleed"].to_numpy(), np.asarray(fit.predict(d, kind="response")))
            )
            new.append(_auc(yt, np.asarray(fit.predict(test, kind="response"))))
        curve.append(
            {
                "n": n,
                "auc_apparent": np.mean(app),
                "auc_new": np.mean(new),
                "reps": len(app),
            }
        )
    curve = pl.DataFrame(curve)
    _write_csv(out, "ch8_learning", curve)
    best = _auc(
        yt,
        _expit(
            bleed_logit(
                *[
                    test[c].to_numpy()
                    for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
                ]
            )
        ),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        curve["n"],
        curve["auc_apparent"],
        marker="o",
        color=GRAY,
        label="Same patients (apparent)",
    )
    ax.plot(curve["n"], curve["auc_new"], marker="o", color=BLUE, label="New patients")
    ax.axhline(best, color=RED, linestyle=":", label=f"True model ({best:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Patients used to build the model")
    ax.set_ylabel("C statistic (AUC)")
    ax.set_title("17 candidate predictors, about 4.5% bleed")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch8_learning.png", dpi=180)
    plt.close(fig)

    # Bootstrap optimism on the 400-patient sample, checked against the new patients.
    val = validate_logistic(dev, big, B=200, seed=8)
    dxy = val.filter(pl.col("index") == "Dxy")
    slope = val.filter(pl.col("index") == "Slope")
    fit = fit_glm(dev, big, family="binomial")
    pt = np.asarray(fit.predict(test, kind="response"))
    _, slope_new = _slope(yt, pt)
    boot = pl.DataFrame(
        {
            "index": ["C statistic", "Calibration slope"],
            "apparent": [float(dxy["index_orig"][0]) / 2 + 0.5, 1.0],
            "optimism": [float(dxy["optimism"][0]) / 2, float(slope["optimism"][0])],
            "corrected": [
                float(dxy["index_corrected"][0]) / 2 + 0.5,
                float(slope["index_corrected"][0]),
            ],
            "new_patients": [_auc(yt, pt), slope_new],
        }
    )
    _write_csv(out, "ch8_bootstrap", boot)
    cross_validation(test, big, out)


def _cv_predictions(dev: pl.DataFrame, formula: str, folds: np.ndarray) -> np.ndarray:
    """Held-out predicted risk for every patient; ``folds`` gives each row's fold."""
    pred = np.empty(dev.height)
    for k in np.unique(folds):
        hold = folds == k
        fit = fit_glm(dev.filter(pl.Series(~hold)), formula, family="binomial")
        pred[hold] = np.asarray(
            fit.predict(dev.filter(pl.Series(hold)), kind="response")
        )
    return pred


def _cv_schematic(path: Path) -> None:
    """Who is used to build and who to test, for 5-fold CV and LOOCV."""
    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1, 1]}
    )
    for ax, n, k, title in [
        (axes[0], 20, 5, "5-fold cross-validation"),
        (axes[1], 10, 10, "Leave-one-out (n = 10)"),
    ]:
        size = n // k
        for r in range(k):
            for i in range(n):
                test = r * size <= i < (r + 1) * size
                ax.add_patch(
                    plt.Rectangle(
                        (i, k - 1 - r), 0.9, 0.8, color=RED if test else "#BFD7EA"
                    )
                )
        ax.set_xlim(-0.2, n)
        ax.set_ylim(-0.2, k)
        ax.set_yticks([k - 1 - r + 0.4 for r in range(k)])
        ax.set_yticklabels([f"Round {r + 1}" for r in range(k)], fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel("Patients")
        ax.set_title(title, fontsize=10)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#BFD7EA"),
        plt.Rectangle((0, 0), 1, 1, color=RED),
    ]
    fig.legend(
        handles, ["Build the model", "Test"], loc="lower center", ncol=2, frameon=False
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


INFO_ROWS: list[dict] = []


def _loglik(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def _info_criteria(fit, y: np.ndarray, loo: np.ndarray, label: str) -> None:
    """AIC, LOOCV deviance and WAIC on the -2 log-likelihood scale.

    WAIC uses a normal approximation of the posterior with a flat prior
    (mean = estimates, covariance = their covariance matrix).
    """
    x = np.asarray(fit.x, dtype=float)
    beta = np.asarray(fit.coefficients, dtype=float)
    draws = np.random.default_rng(12).multivariate_normal(
        beta, np.asarray(fit.covariance), size=4000
    )
    ll = _loglik(y[:, None], _expit(x @ draws.T))
    lppd = np.log(np.mean(np.exp(ll), axis=1))
    p_waic = np.var(ll, axis=1, ddof=1)
    INFO_ROWS.append(
        {
            "sample": label,
            "parameters": len(beta),
            "deviance_apparent": float(fit.deviance),
            "aic": float(fit.aic),
            "loocv": float(-2 * _loglik(y, loo).sum()),
            "waic": float(-2 * (lppd.sum() - p_waic.sum())),
            "p_waic": float(p_waic.sum()),
        }
    )


def _cv_estimates(
    dev: pl.DataFrame, test: pl.DataFrame, formula: str, label: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split-sample, k-fold, leave-one-out and bootstrap estimates of the C statistic."""
    yd = dev["bleed"].to_numpy()
    n = dev.height
    fit = fit_glm(dev, formula, family="binomial")
    truth = _auc(
        test["bleed"].to_numpy(), np.asarray(fit.predict(test, kind="response"))
    )
    apparent = _auc(yd, np.asarray(fit.predict(dev, kind="response")))
    rng = np.random.default_rng(11)
    rows = []
    # Split sample: 70% to build, 30% to test, 100 different random splits.
    for r in range(100):
        idx = rng.permutation(n)
        tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n) :]
        f = fit_glm(dev[tr], formula, family="binomial")
        yte = yd[te]
        if 0 < yte.sum() < len(yte):
            p_te = np.asarray(f.predict(dev[te], kind="response"))
            rows.append({"method": "Split 70/30", "repeat": r, "auc": _auc(yte, p_te)})
    # k-fold: the held-out predictions of all folds are pooled, then one C statistic.
    for k in (5, 10):
        for r in range(20):
            folds = rng.permutation(np.arange(n) % k)
            p_cv = _cv_predictions(dev, formula, folds)
            rows.append({"method": f"{k}-fold CV", "repeat": r, "auc": _auc(yd, p_cv)})
    loo = _cv_predictions(dev, formula, np.arange(n))
    rows.append({"method": "Leave-one-out", "repeat": 0, "auc": _auc(yd, loo)})
    _info_criteria(fit, yd, loo, label)
    val = validate_logistic(dev, formula, B=200, seed=8)
    dxy = val.filter(pl.col("index") == "Dxy")
    boot_c = float(dxy["index_corrected"][0]) / 2 + 0.5
    rows.append({"method": "Bootstrap", "repeat": 0, "auc": boot_c})
    est = pl.DataFrame(rows).with_columns(pl.lit(label).alias("sample"))
    summary = (
        est.group_by("sample", "method", maintain_order=True)
        .agg(
            pl.len().alias("repeats"),
            pl.col("auc").mean().alias("mean"),
            pl.col("auc").min().alias("min"),
            pl.col("auc").max().alias("max"),
            pl.col("auc").std().alias("sd"),
        )
        .with_columns(
            pl.lit(truth).alias("new_patients"), pl.lit(apparent).alias("apparent")
        )
    )
    return est, summary


def cross_validation(test: pl.DataFrame, formula: str, out: Path) -> None:
    """Chapter 8: internal validation with few events and with enough events."""
    _cv_schematic(out / "figures" / "ch8_cv_scheme.png")
    INFO_ROWS.clear()
    ests, sums = [], []
    for n, seed in [(400, 4), (2000, 6)]:
        dev = _with_noise(simulate(n=n, seed=seed), seed=10 * seed)
        events = int(dev["bleed"].sum())
        e, sm = _cv_estimates(dev, test, formula, f"{n} patients, {events} bleeds")
        ests.append(e)
        sums.append(sm)
    est = pl.concat(ests)
    summary = pl.concat(sums)
    _write_csv(out, "ch8_cv_estimates", est)
    _write_csv(out, "ch8_cv_summary", summary)
    _write_csv(out, "ch8_information", pl.DataFrame(INFO_ROWS))

    order = ["Split 70/30", "5-fold CV", "10-fold CV", "Leave-one-out", "Bootstrap"]
    labels = [
        "Split\n70/30",
        "5-fold\nCV",
        "10-fold\nCV",
        "Leave-\none-out",
        "Boot-\nstrap",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    jitter = np.random.default_rng(0)
    for ax, sample in zip(axes, est["sample"].unique(maintain_order=True), strict=True):
        part = est.filter(pl.col("sample") == sample)
        sm = summary.filter(pl.col("sample") == sample)
        for i, m in enumerate(order):
            v = part.filter(pl.col("method") == m)["auc"].to_numpy()
            many = len(v) > 1
            x = np.full(len(v), float(i)) + (
                jitter.uniform(-0.15, 0.15, len(v)) if many else 0
            )
            ax.scatter(
                x, v, s=12 if many else 50, color=BLUE, alpha=0.45 if many else 1
            )
        truth = float(sm["new_patients"][0])
        app = float(sm["apparent"][0])
        ax.axhline(truth, color=RED, linestyle=":", label=f"New patients ({truth:.2f})")
        ax.axhline(app, color=GRAY, linestyle="--", label=f"Apparent ({app:.2f})")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{sample}, 17 predictors", fontsize=10)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Estimated C statistic")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch8_cv.png", dpi=180)
    plt.close(fig)


def _decile_table(y: np.ndarray, p: np.ndarray, model: str) -> pl.DataFrame:
    """Observed rate and mean predicted risk in tenths of predicted risk."""
    edges = np.quantile(p, np.linspace(0, 1, 11))
    group = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    d = pl.DataFrame({"g": group, "y": y, "p": p})
    return (
        d.group_by("g")
        .agg(
            pl.len().alias("n"),
            pl.col("p").mean().alias("pred_mean"),
            pl.col("y").mean().alias("obs_rate"),
        )
        .sort("g")
        .with_columns(pl.lit(model).alias("model"), (pl.col("g") + 1).alias("tenth"))
        .select("model", "tenth", "n", "pred_mean", "obs_rate")
    )


def discrimination_calibration(df: pl.DataFrame, out: Path) -> None:
    """Chapter 9: ROC, calibration, Brier score and decision curves on new patients."""
    test = _test_set()
    yt = test["bleed"].to_numpy()
    full = fit_glm(df, PRED_FORMULA, family="binomial")
    simple = fit_glm(df, "bleed ~ age + antithrombotic", family="binomial")
    p_full = np.asarray(full.predict(test, kind="response"))
    p_simple = np.asarray(simple.predict(test, kind="response"))

    rocs = {
        "Seven predictors": roc_curve(yt, p_full),
        "Age + antithrombotic": roc_curve(yt, p_simple),
    }
    plot_roc(rocs, out / "figures" / "ch9_roc.png", title="ROC curve in new patients")

    # Another hospital: same patients and same effects, but a higher baseline risk.
    rng = np.random.default_rng(9)
    lp_true = bleed_logit(
        *[
            test[c].to_numpy()
            for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
        ]
    )
    y_other = rng.binomial(1, _expit(lp_true + 0.8))

    cal = pl.concat(
        [
            _decile_table(yt, p_full, "Same population"),
            _decile_table(y_other, p_full, "Hospital with higher risk"),
        ]
    )
    _write_csv(out, "ch9_calibration", cal)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = 0.22
    ax.plot([0, lim], [0, lim], color=GRAY, linestyle="--", label="Ideal")
    for name, color in [("Same population", BLUE), ("Hospital with higher risk", RED)]:
        part = cal.filter(pl.col("model") == name)
        ax.plot(
            part["pred_mean"], part["obs_rate"], marker="o", color=color, label=name
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Predicted risk (mean of each tenth)")
    ax.set_ylabel("Observed bleeding rate")
    ax.set_title("Calibration in new patients")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch9_calibration.png", dpi=180)
    plt.close(fig)

    rows = []
    for name, y, p in [
        ("Seven predictors, same population", yt, p_full),
        ("Age + antithrombotic, same population", yt, p_simple),
        ("Seven predictors, hospital with higher risk", y_other, p_full),
    ]:
        citl, slope = _slope(y, p)
        rows.append(
            {
                "model": name,
                "auc": _auc(y, p),
                "mean_predicted": float(p.mean()),
                "observed": float(y.mean()),
                "calibration_in_the_large": citl,
                "slope": slope,
                "brier": brier_score(y, p),
                "brier_null": brier_score(y, np.full(len(y), y.mean())),
            }
        )
    _write_csv(out, "ch9_metrics", pl.DataFrame(rows))

    trade = threshold_tradeoff(
        yt, p_full, thresholds=np.array([0.02, 0.05, 0.10, 0.20])
    )
    _write_csv(out, "ch9_thresholds", trade)

    th = np.linspace(0.01, 0.20, 20)
    dca = pl.concat(
        [
            decision_curve_table(yt, p_full, thresholds=th, model="Seven predictors"),
            decision_curve_table(
                yt, p_simple, thresholds=th, model="Age + antithrombotic"
            ),
        ],
        how="vertical_relaxed",
    )
    _write_csv(out, "ch9_dca", dca)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    first = dca.filter(pl.col("model") == "Seven predictors").sort("threshold")
    for name, color in [
        ("Seven predictors", BLUE),
        ("Age + antithrombotic", "#EFC000"),
    ]:
        part = dca.filter(pl.col("model") == name).sort("threshold")
        ax.plot(part["threshold"], part["net_benefit"], color=color, label=name)
    ax.plot(
        first["threshold"],
        first["treat_all"],
        color=GRAY,
        linestyle="--",
        label="Admit everyone",
    )
    ax.plot(
        first["threshold"],
        first["treat_none"],
        color="black",
        linestyle=":",
        label="Admit no one",
    )
    ax.set_ylim(-0.01, 0.04)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Threshold risk for admission")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve in new patients")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch9_dca.png", dpi=180)
    plt.close(fig)


ALL_VARS = PRED_VARS + NOISE_VARS
VAR_LABELS = {**LABELS, **{v: f"Lab {v[3:]} (noise)" for v in NOISE_VARS}}


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(float)


def _fit_three(
    dev: pl.DataFrame, test: pl.DataFrame, *, k: int = 10, n_lambda: int = 50
) -> dict:
    """Maximum likelihood, ridge and LASSO (lambda by CV deviance) on 17 predictors."""
    xs, mean, sd = pen.standardize(_matrix(dev, ALL_VARS))
    y = dev["bleed"].to_numpy().astype(float)
    xt = (_matrix(test, ALL_VARS) - mean) / sd
    yt = test["bleed"].to_numpy()
    lmax = pen.lambda_max(xs, y)
    grids = {
        "LASSO": lmax * np.logspace(0, -3, n_lambda),
        "Ridge": lmax * np.logspace(1.5, -3, n_lambda),
    }
    res = {"xs": xs, "y": y, "grids": grids}
    b0, b = pen.logistic_path(xs, y, np.array([0.0]), penalty="ridge")
    pt = pen._expit(b0[0] + xt @ b[0])
    res["Maximum likelihood"] = {
        "beta": b[0],
        "auc": _auc(yt, pt),
        "slope": _slope(yt, pt)[1],
    }
    for name, lams in grids.items():
        kind = "lasso" if name == "LASSO" else "ridge"
        cvm, cvs = pen.cv_deviance(xs, y, lams, penalty=kind, k=k)
        i = int(np.argmin(cvm))
        b0, b = pen.logistic_path(xs, y, lams, penalty=kind)
        pt = pen._expit(b0[i] + xt @ b[i])
        res[name] = {
            "beta": b[i],
            "path": b,
            "cv_mean": cvm,
            "cv_se": cvs,
            "best": i,
            "auc": _auc(yt, pt),
            # A model with every coefficient zero gives everyone the same risk: no slope.
            "slope": _slope(yt, pt)[1] if np.any(np.abs(b[i]) > 1e-10) else None,
            "nonzero": int((np.abs(b[i]) > 1e-10).sum()),
        }
    return res


def regularization(out: Path) -> None:
    """Chapter 10: ridge and LASSO paths, lambda by CV, and repeated samples."""
    test = _with_noise(_test_set(), seed=30)
    dev = _with_noise(simulate(n=1000, seed=4), seed=40)
    res = _fit_three(dev, test)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, name in zip(axes, ["Ridge", "LASSO"], strict=True):
        r = res[name]
        lams = res["grids"][name]
        for j, v in enumerate(ALL_VARS):
            noise = v in NOISE_VARS
            ax.plot(
                np.log10(lams),
                r["path"][:, j],
                color=GRAY if noise else BLUE,
                alpha=0.6 if noise else 1,
                lw=0.9 if noise else 1.6,
            )
            if not noise and abs(r["path"][-1, j]) > 0.15:
                ax.annotate(
                    VAR_LABELS[v].split(" (")[0],
                    (np.log10(lams[-1]), r["path"][-1, j]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=7.5,
                    va="center",
                )
        ax.axvline(
            np.log10(lams[r["best"]]), color=RED, linestyle=":", label="Chosen by CV"
        )
        ax.axhline(0, color="black", lw=0.6)
        ax.invert_xaxis()
        ax.set_xlabel("log10(lambda)  (strong penalty on the left)")
        ax.set_title(f"{name}: coefficient path", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("Coefficient (per 1 SD of the predictor)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch10_path.png", dpi=180)
    plt.close(fig)

    r = res["LASSO"]
    lams = res["grids"]["LASSO"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(
        np.log10(lams),
        r["cv_mean"],
        yerr=r["cv_se"],
        fmt="o",
        ms=3,
        color=BLUE,
        ecolor="#BFD7EA",
    )
    ax.axvline(np.log10(lams[r["best"]]), color=RED, linestyle=":")
    ax.invert_xaxis()
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Held-out deviance per patient")
    ax.set_title("LASSO: choosing lambda by 10-fold cross-validation", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch10_cv.png", dpi=180)
    plt.close(fig)

    rows = []
    for name in ["Maximum likelihood", "Ridge", "LASSO"]:
        b = res[name]["beta"]
        kept = np.abs(b) > 1e-10
        rows.append(
            {
                "model": name,
                "auc_new": res[name]["auc"],
                "slope_new": res[name]["slope"],
                "nonzero": int(kept.sum()),
                "noise_kept": int(
                    sum(kept[j] for j, v in enumerate(ALL_VARS) if v in NOISE_VARS)
                ),
            }
        )
    _write_csv(out, "ch10_single", pl.DataFrame(rows))
    coef = pl.DataFrame(
        {
            "term": [VAR_LABELS[v] for v in ALL_VARS],
            **{
                name: res[name]["beta"]
                for name in ["Maximum likelihood", "Ridge", "LASSO"]
            },
        }
    )
    _write_csv(out, "ch10_coefficients", coef)

    # Repeat with new development samples of the same size.
    reps = []
    for r_ in range(30):
        d = _with_noise(simulate(n=1000, seed=7000 + r_), seed=8000 + r_)
        rr = _fit_three(d, test, k=5, n_lambda=30)
        for name in ["Maximum likelihood", "Ridge", "LASSO"]:
            reps.append(
                {
                    "repeat": r_,
                    "model": name,
                    "auc_new": rr[name]["auc"],
                    "slope_new": rr[name]["slope"],
                    "nonzero": rr[name].get("nonzero", len(ALL_VARS)),
                }
            )
    reps = pl.DataFrame(reps)
    _write_csv(out, "ch10_repeats", reps)
    _write_csv(
        out,
        "ch10_repeats_summary",
        reps.group_by("model", maintain_order=True).agg(
            pl.col("auc_new").median().alias("auc_median"),
            pl.col("slope_new").median().alias("slope_median"),
            pl.col("slope_new").quantile(0.1).alias("slope_p10"),
            pl.col("slope_new").quantile(0.9).alias("slope_p90"),
            (pl.col("nonzero") == 0).sum().alias("empty_model"),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = ["Maximum likelihood", "Ridge", "LASSO"]
    jit = np.random.default_rng(0)
    for ax, col, ref, title in [
        (axes[0], "slope_new", 1.0, "Calibration slope in new patients"),
        (axes[1], "auc_new", None, "C statistic in new patients"),
    ]:
        for i, name in enumerate(names):
            v = reps.filter(pl.col("model") == name)[col].drop_nulls().to_numpy()
            ax.scatter(
                i + jit.uniform(-0.15, 0.15, len(v)), v, s=14, color=BLUE, alpha=0.55
            )
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black")
        if ref is not None:
            ax.axhline(ref, color=RED, linestyle=":", label="Ideal")
            ax.legend(frameon=False, fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Max.\nlikelihood", "Ridge", "LASSO"], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    if (reps["slope_new"] > 3).any():
        axes[0].set_ylim(0, 3)
    fig.suptitle(
        "30 development samples of 1000 patients (about 45 bleeds), 17 predictors",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch10_repeats.png", dpi=180)
    plt.close(fig)


def trees_ensembles(df: pl.DataFrame, out: Path) -> None:
    """Chapter 11: one tree, a random forest and boosting against logistic regression."""
    test = _test_set()
    yt = test["bleed"].to_numpy()
    x = _matrix(df, PRED_VARS)
    y = df["bleed"].to_numpy().astype(float)
    xt = _matrix(test, PRED_VARS)

    tree = conditional_tree(df, "bleed", PRED_VARS)
    (out / "ch11_tree.txt").write_text(tree.format(), encoding="utf-8")
    plot_tree(
        tree,
        out / "figures" / "ch11_tree.png",
        title="Conditional inference tree (3000 patients)",
    )

    # Boosting: choose the number of trees by 5-fold CV deviance.
    rounds = 400
    folds = np.random.default_rng(21).permutation(np.arange(len(y)) % 5)
    cv = np.zeros((5, rounds))
    for f in range(5):
        tr, te = folds != f, folds == f
        m = pen.boosting(x[tr], y[tr], rounds=rounds, depth=2, rate=0.05)
        eta = np.full(te.sum(), m[0])
        for r_, stage in enumerate(m[2]):
            eta += m[1] * pen.predict_cart(stage, x[te])
            p_ = np.clip(_expit(eta), 1e-12, 1 - 1e-12)
            cv[f, r_] = -2 * np.mean(_loglik(y[te], p_))
    cv_mean = cv.mean(axis=0)
    best = int(np.argmin(cv_mean)) + 1
    _write_csv(
        out,
        "ch11_boost_cv",
        pl.DataFrame({"trees": np.arange(1, rounds + 1), "cv_deviance": cv_mean}),
    )
    boost = pen.boosting(x, y, rounds=rounds, depth=2, rate=0.05)
    p_boost_all = {
        r_: pen.predict_boosting(boost, xt, rounds=r_) for r_ in (best, rounds)
    }
    p_boost_train = {
        r_: pen.predict_boosting(boost, x, rounds=r_) for r_ in (best, rounds)
    }

    forest = pen.random_forest(
        x, y, trees=200, max_depth=12, min_leaf=50, features=3, seed=3
    )
    logit = fit_glm(df, PRED_FORMULA, family="binomial")
    logit_int = fit_glm(
        df.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large")),
        PRED_FORMULA + " + clip:large",
        family="binomial",
    )
    test_l = test.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large"))
    df_l = df.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large"))
    preds = {
        "Logistic regression": (
            np.asarray(logit.predict(test, kind="response")),
            np.asarray(logit.predict(df, kind="response")),
        ),
        "Logistic + clip x (size >= 20 mm)": (
            np.asarray(logit_int.predict(test_l, kind="response")),
            np.asarray(logit_int.predict(df_l, kind="response")),
        ),
        "Single tree": (np.asarray(tree.predict(test)), np.asarray(tree.predict(df))),
        "Random forest (200 trees)": (
            pen.predict_forest(forest, xt),
            pen.predict_forest(forest, x),
        ),
        f"Boosting ({best} trees, chosen by CV)": (
            p_boost_all[best],
            p_boost_train[best],
        ),
        f"Boosting ({rounds} trees)": (p_boost_all[rounds], p_boost_train[rounds]),
    }
    rows = []
    for name, (pt, pa) in preds.items():
        pt = np.clip(pt, 1e-4, 1 - 1e-4)
        rows.append(
            {
                "model": name,
                "auc_apparent": _auc(y, pa),
                "auc_new": _auc(yt, pt),
                "slope_new": _slope(yt, pt)[1],
                "brier_new": brier_score(yt, pt),
            }
        )
    _write_csv(out, "ch11_compare", pl.DataFrame(rows))


def _network_schematic(path: Path) -> None:
    """Logistic regression as one unit, and a network with one hidden layer."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    inputs = ["Age", "Antithrombotic", "Lesion size", "Proximal", "Clip"]
    ys = np.linspace(0.85, 0.15, len(inputs))
    for ax, hidden, title in [
        (axes[0], 0, "Logistic regression = one unit"),
        (axes[1], 4, "Neural network: logistic units in layers"),
    ]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        xin, xh, xo = 0.12, 0.5, 0.88
        out = (xo, 0.5)
        hs = [(xh, yy) for yy in np.linspace(0.8, 0.2, hidden)] if hidden else []
        for yy, lab in zip(ys, inputs, strict=True):
            ax.add_patch(plt.Circle((xin, yy), 0.035, color="#BFD7EA"))
            ax.text(xin - 0.05, yy, lab, ha="right", va="center", fontsize=8)
            for tgt in hs or [out]:
                ax.plot([xin + 0.035, tgt[0] - 0.04], [yy, tgt[1]], color=GRAY, lw=0.6)
        for hx, hy in hs:
            ax.add_patch(plt.Circle((hx, hy), 0.045, color=BLUE))
            ax.plot([hx + 0.045, out[0] - 0.05], [hy, out[1]], color=GRAY, lw=0.8)
        ax.add_patch(plt.Circle(out, 0.055, color=RED))
        ax.text(out[0], out[1] - 0.11, "risk p", ha="center", fontsize=8.5)
        if hidden:
            ax.text(
                xh,
                0.06,
                "hidden units\n(each a logistic curve)",
                ha="center",
                fontsize=8,
            )
        else:
            ax.text(
                0.5,
                0.06,
                "p = 1 / (1 + exp(-(b0 + b1 x1 + ...)))",
                ha="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def neural_network(df: pl.DataFrame, out: Path) -> None:
    """Chapter 11: a network as stacked logistic regressions."""
    _network_schematic(out / "figures" / "ch11_network.png")

    # The bleeding data: same seven predictors as chapter 7.
    test = _test_set()
    yt = test["bleed"].to_numpy().astype(float)
    xs, mean, sd = pen.standardize(_matrix(df, PRED_VARS))
    xt = (_matrix(test, PRED_VARS) - mean) / sd
    y = df["bleed"].to_numpy().astype(float)
    logit = fit_glm(df, PRED_FORMULA, family="binomial")
    p_log = np.asarray(logit.predict(test, kind="response"))
    rows = [
        {
            "model": "Logistic regression (chapter 7)",
            "hidden": 0,
            "decay": 0.0,
            "auc_apparent": _auc(y, np.asarray(logit.predict(df, kind="response"))),
            "auc_new": _auc(yt, p_log),
            "slope_new": _slope(yt, p_log)[1],
        }
    ]
    for hidden, decay in [(1, 0.0), (5, 0.0), (20, 0.0), (20, 0.01)]:
        net = pen.mlp_fit(xs, y, hidden=hidden, decay=decay, epochs=2000, rate=0.01)
        pt = np.clip(pen.mlp_predict(net, xt), 1e-6, 1 - 1e-6)
        rows.append(
            {
                "model": f"Network, {hidden} hidden"
                + (f", weight decay {decay}" if decay else ""),
                "hidden": hidden,
                "decay": decay,
                "auc_apparent": _auc(y, pen.mlp_predict(net, xs)),
                "auc_new": _auc(yt, pt),
                "slope_new": _slope(yt, pt)[1],
            }
        )
    _write_csv(out, "ch11_network", pl.DataFrame(rows))


# --- Part IV: statistical modelling -------------------------------------------------

SIZE_GROUPS = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 81)]


def _size_group(size: np.ndarray) -> np.ndarray:
    return np.digitize(size, [g[1] for g in SIZE_GROUPS[:-1]])


def _generative_schematic(path: Path) -> None:
    """x -> eta -> p -> coin -> y, the story a logistic model tells."""
    fig, ax = plt.subplots(figsize=(9, 2.2))
    boxes = [
        ("Patient\n$x_i$", "age, size, ..."),
        ("Linear predictor\n$\\eta_i = \\beta_0 + \\beta_1 x_{i1} + \\cdots$", ""),
        ("Probability\n$p_i = 1/(1+e^{-\\eta_i})$", ""),
        ("Coin flip\nBernoulli($p_i$)", "chance"),
        ("Outcome\n$y_i$ = 0 or 1", "observed"),
    ]
    xs = np.linspace(0.09, 0.91, len(boxes))
    for i, (x, (label, note)) in enumerate(zip(xs, boxes, strict=True)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": "white",
                "ec": RED if i == 3 else BLUE,
            },
        )
        if note:
            ax.text(x, 0.08, note, ha="center", fontsize=8, color=GRAY)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.075, 0.55),
                xytext=(x + 0.075, 0.55),
                arrowprops={"arrowstyle": "->", "color": GRAY},
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _group_check(df: pl.DataFrame, p: np.ndarray, model: str, rng) -> pl.DataFrame:
    """Observed bleeding rate by size group vs 95% range from data simulated by the model."""
    group = _size_group(df["size_mm"].to_numpy())
    y = df["bleed"].to_numpy()
    sims = rng.binomial(1, p, size=(1000, len(p)))
    rows = []
    for g, (lo, hi) in enumerate(SIZE_GROUPS):
        m = group == g
        rates = sims[:, m].mean(axis=1)
        rows.append(
            {
                "model": model,
                "size_group": f"{lo}-{hi - 1} mm" if hi < 81 else f">= {lo} mm",
                "n": int(m.sum()),
                "observed": float(y[m].mean()),
                "predicted": float(p[m].mean()),
                "sim_low": float(np.quantile(rates, 0.025)),
                "sim_high": float(np.quantile(rates, 0.975)),
            }
        )
    return pl.DataFrame(rows)


def probability_model(df: pl.DataFrame, out: Path) -> None:
    """Chapter 12: the logistic model as a story of how the data were made."""
    _generative_schematic(out / "figures" / "ch12_generative.png")
    truth = pl.DataFrame(
        {
            "term": list(TRUE_BLEED),
            "true_log_odds": [float(v) for v in TRUE_BLEED.values()],
        }
    )
    _write_csv(out, "ch12_truth", truth)

    # Same patients, same true risks, new coin flips.
    rng = np.random.default_rng(12)
    args = [
        df[c].to_numpy()
        for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
    ]
    p_true = _expit(bleed_logit(*args))
    counts = rng.binomial(1, p_true, size=(5000, len(p_true))).sum(axis=1)
    observed = int(df["bleed"].sum())
    _write_csv(
        out,
        "ch12_replicates",
        pl.DataFrame(
            {
                "observed": [observed],
                "expected": [float(p_true.sum())],
                "sd_theory": [float(np.sqrt(np.sum(p_true * (1 - p_true))))],
                "sim_low": [float(np.quantile(counts, 0.025))],
                "sim_high": [float(np.quantile(counts, 0.975))],
            }
        ),
    )
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(counts, bins=30, color=BLUE, alpha=0.6)
    ax.axvline(observed, color=RED, lw=2, label=f"This data set ({observed})")
    ax.set_xlabel("Number of bleeds among the same 3000 patients")
    ax.set_ylabel("Simulated data sets")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch12_replicates.png", dpi=180)
    plt.close(fig)

    # Simulate from fitted models and check bleeding by size group.
    fits = {
        "Size in mm": fit_glm(df, PRED_FORMULA, family="binomial"),
        "Size >= 20 mm (yes/no)": fit_glm(
            df.with_columns((pl.col("size_mm") >= 20).cast(pl.Int64).alias("large")),
            PRED_FORMULA.replace("size_mm", "large"),
            family="binomial",
        ),
    }
    data = {
        "Size in mm": df,
        "Size >= 20 mm (yes/no)": df.with_columns(
            (pl.col("size_mm") >= 20).cast(pl.Int64).alias("large")
        ),
    }
    check = pl.concat(
        [
            _group_check(df, fit.predict(data[k], kind="response"), k, rng)
            for k, fit in fits.items()
        ]
    )
    _write_csv(out, "ch12_check", check)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, sub in zip(
        axes, check.partition_by("model", maintain_order=True), strict=True
    ):
        x = np.arange(sub.height)
        ax.vlines(
            x,
            sub["sim_low"],
            sub["sim_high"],
            color=BLUE,
            lw=6,
            alpha=0.4,
            label="Simulated from the model (95%)",
        )
        ax.scatter(x, sub["observed"], color=RED, zorder=3, label="Observed")
        ax.set_xticks(x, sub["size_group"].to_list(), fontsize=8)
        ax.set_title(sub["model"][0])
        ax.set_xlabel("Lesion size")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Bleeding rate")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch12_check.png", dpi=180)
    plt.close(fig)

    # Is log odds linear in size? Restricted cubic spline.
    spline = fit_glm(
        df, PRED_FORMULA.replace("size_mm", "rcs(size_mm, 4)"), family="binomial"
    )
    test = spline_test(spline, "rcs(size_mm, 4)")
    _write_csv(out, "ch12_spline_test", test)
    grid = pl.DataFrame(
        {
            "age": [70] * 60,
            "male": [1] * 60,
            "antithrombotic": [0] * 60,
            "hypertension": [0] * 60,
            "size_mm": np.linspace(5, 60, 60),
            "proximal": [0] * 60,
            "clip": [0] * 60,
        }
    )
    linear = fits["Size in mm"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(
        grid["size_mm"], linear.predict(grid, kind="link"), color=BLUE, label="Linear"
    )
    ax.plot(
        grid["size_mm"],
        spline.predict(grid, kind="link"),
        color=RED,
        label="Restricted cubic spline (4 knots)",
    )
    sizes = df["size_mm"].to_numpy()
    ax.plot(sizes, np.full_like(sizes, -5.0, dtype=float), "|", color=GRAY, alpha=0.2)
    ax.set_xlabel("Lesion size (mm)")
    ax.set_ylabel("log odds of bleeding")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch12_spline.png", dpi=180)
    plt.close(fig)
    _write_csv(
        out,
        "ch12_models",
        pl.DataFrame(
            {
                "model": ["Size in mm", "Size >= 20 mm (yes/no)", "rcs(size, 4)"],
                "aic": [
                    fits["Size in mm"].aic,
                    fits["Size >= 20 mm (yes/no)"].aic,
                    spline.aic,
                ],
            }
        ),
    )


def _binom_loglik(p: np.ndarray, events: int, n: int) -> np.ndarray:
    return events * np.log(p) + (n - events) * np.log(1 - p)


def _lr_interval(grid: np.ndarray, ll: np.ndarray) -> tuple[float, float]:
    """Where the log likelihood is within 1.92 (chi-square 3.84 / 2) of its top."""
    inside = grid[ll >= ll.max() - 1.92]
    return float(inside.min()), float(inside.max())


def likelihood(df: pl.DataFrame, out: Path) -> None:
    """Chapter 13: likelihood, maximum likelihood, standard errors, tests, separation."""
    # One parameter: the bleeding rate.
    small = df.sample(300, seed=13)
    sets = {"300 patients": small, "3000 patients": df}
    grid = np.linspace(0.005, 0.14, 2701)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for (name, d), color in zip(sets.items(), [GRAY, BLUE], strict=True):
        n, events = d.height, int(d["bleed"].sum())
        ll = _binom_loglik(grid, events, n)
        mle = events / n
        se = float(np.sqrt(mle * (1 - mle) / n))
        lo, hi = _lr_interval(grid, ll)
        rows.append(
            {
                "data": name,
                "n": n,
                "events": events,
                "mle": mle,
                "se": se,
                "wald_low": mle - 1.96 * se,
                "wald_high": mle + 1.96 * se,
                "lr_low": lo,
                "lr_high": hi,
            }
        )
        axes[0].plot(
            grid, np.exp(ll - ll.max()), color=color, label=f"{name} ({events} bleeds)"
        )
        axes[1].plot(grid, ll - ll.max(), color=color, label=name)
        if name == "300 patients":
            quad = -((grid - mle) ** 2) / (2 * se**2)
            axes[1].plot(
                grid, quad, color=color, ls="--", label="Quadratic approximation"
            )
    _write_csv(out, "ch13_rate", pl.DataFrame(rows))
    axes[0].set_ylabel("Likelihood (relative to its maximum)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(-1.92, color=RED, lw=1, ls=":")
    axes[1].text(0.13, -1.75, "95% interval", color=RED, ha="right", fontsize=8)
    axes[1].set_ylim(-8, 0.5)
    axes[1].set_ylabel("log likelihood (minus its maximum)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xlabel("Bleeding rate p")
        ax.set_xlim(0, 0.10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch13_rate.png", dpi=180)
    plt.close(fig)

    # Several parameters: the profile log likelihood of the clip coefficient.
    full = fit_glm(df, PRED_FORMULA, family="binomial")
    tidy = full.tidy().filter(pl.col("term") == "clip")
    b_hat, se_hat = float(tidy["estimate"][0]), float(tidy["std_error"][0])
    reduced_formula = PRED_FORMULA.replace(" + clip", "")
    bgrid = np.linspace(-1.8, 0.2, 81)
    prof = np.array(
        [
            fit_glm(
                df.with_columns((pl.col("clip") * b).alias("off")),
                reduced_formula,
                family="binomial",
                offset="off",
            ).log_likelihood
            for b in bgrid
        ]
    )
    fine = np.linspace(bgrid[0], bgrid[-1], 4001)
    prof_fine = np.interp(fine, bgrid, prof)
    p_lo, p_hi = _lr_interval(fine, prof_fine)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(
        bgrid, prof - full.log_likelihood, color=BLUE, label="Profile log likelihood"
    )
    ax.plot(
        bgrid,
        -((bgrid - b_hat) ** 2) / (2 * se_hat**2),
        color=GRAY,
        ls="--",
        label="Quadratic (Wald)",
    )
    ax.axhline(-1.92, color=RED, lw=1, ls=":")
    ax.set_ylim(-8, 0.5)
    ax.set_xlabel("Clip coefficient (log odds ratio)")
    ax.set_ylabel("log likelihood (minus its maximum)")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch13_profile.png", dpi=180)
    plt.close(fig)

    # Three tests of "clip has no effect".
    reduced = fit_glm(df, reduced_formula, family="binomial")
    lr = likelihood_ratio_test(full, reduced)
    y = df["bleed"].to_numpy().astype(float)
    mu0 = reduced.predict(df, kind="response")
    x = full.x
    info = x.T @ (x * (mu0 * (1 - mu0))[:, None])
    score_vec = x.T @ (y - mu0)
    score = float(score_vec @ np.linalg.solve(info, score_vec))
    wald = (b_hat / se_hat) ** 2
    _write_csv(
        out,
        "ch13_tests",
        pl.DataFrame(
            {
                "test": ["Wald", "Likelihood ratio", "Score"],
                "chi2": [wald, lr.statistic, score],
                "p_value": [
                    float(stats.chi2.sf(v, 1)) for v in [wald, lr.statistic, score]
                ],
                "ci_low": [np.exp(b_hat - 1.96 * se_hat), np.exp(p_lo), None],
                "ci_high": [np.exp(b_hat + 1.96 * se_hat), np.exp(p_hi), None],
            }
        ),
    )

    # Newton-Raphson from zero.
    history = pen.newton_logistic(x, y, iters=7)
    names = list(full.tidy()["term"])
    clip_col = names.index("clip")
    _write_csv(
        out,
        "ch13_newton",
        pl.DataFrame(
            {
                "iteration": list(range(len(history))),
                "log_likelihood": [
                    float(np.sum(_loglik(y, _expit(x @ b)))) for b in history
                ],
                "clip": [float(b[clip_col]) for b in history],
            }
        ),
    )

    _bayes_rate(sets, out)
    _bayes_clip(full, y, out)

    # Separation: a small study where no clipped patient bled.
    sep = _separated_sample(df)
    sx = fit_glm(sep, "bleed ~ antithrombotic + size_mm + clip", family="binomial")
    st = sx.tidy().filter(pl.col("term") == "clip")
    fb, fse = pen.firth_logistic(sx.x, sep["bleed"].to_numpy().astype(float))
    ci = list(sx.tidy()["term"]).index("clip")
    _write_csv(
        out,
        "ch13_separation",
        pl.DataFrame(
            {
                "method": ["Maximum likelihood", "Firth"],
                "n": [sep.height] * 2,
                "events": [int(sep["bleed"].sum())] * 2,
                "clipped": [int(sep["clip"].sum())] * 2,
                "clipped_events": [int(sep.filter(pl.col("clip") == 1)["bleed"].sum())]
                * 2,
                "estimate": [float(st["estimate"][0]), float(fb[ci])],
                "std_error": [float(st["std_error"][0]), float(fse[ci])],
            }
        ),
    )


BETA_PRIORS = {
    "Flat": (1.0, 1.0),
    "Past studies (about 5%)": (5.0, 95.0),
    "Off target (about 10%)": (20.0, 180.0),
}


def _bayes_rate(sets: dict[str, pl.DataFrame], out: Path) -> None:
    """Bleeding rate with a beta prior: the posterior is beta again."""
    grid = np.linspace(0.0005, 0.16, 2000)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    colors = [GRAY, BLUE, RED]
    for ax, (name, d) in zip(axes, sets.items(), strict=True):
        n, events = d.height, int(d["bleed"].sum())
        for (prior, (a, b)), color in zip(BETA_PRIORS.items(), colors, strict=True):
            post = stats.beta(a + events, b + n - events)
            rows.append(
                {
                    "data": name,
                    "prior": prior,
                    "prior_mean": a / (a + b),
                    "posterior_mean": float(post.mean()),
                    "cri_low": float(post.ppf(0.025)),
                    "cri_high": float(post.ppf(0.975)),
                }
            )
            if prior != "Flat":
                ax.plot(grid, stats.beta(a, b).pdf(grid), color=color, ls=":", lw=1)
            ax.plot(
                grid,
                post.pdf(grid),
                color=color,
                label=f"Posterior, prior: {prior.lower()}",
            )
        ax.set_title(f"{name} ({events} bleeds)")
        ax.set_xlabel("Bleeding rate p")
        ax.set_xlim(0, 0.12)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].plot([], [], color=GRAY, ls=":", lw=1, label="Prior (dotted)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch13_bayes_rate.png", dpi=180)
    plt.close(fig)
    _write_csv(out, "ch13_bayes_rate", pl.DataFrame(rows))


def _bayes_clip(full, y: np.ndarray, out: Path) -> None:
    """Posterior of the clip coefficient by Metropolis sampling."""
    names = list(full.tidy()["term"])
    clip = names.index("clip")
    start = np.asarray(full.coefficients, dtype=float)
    cov = np.asarray(full.covariance, dtype=float) * 2.38**2 / len(start)
    flat = np.full(len(start), np.inf)
    sceptical = flat.copy()
    sceptical[clip] = 0.35  # 95% of the prior between OR 0.5 and 2
    runs = {"Flat prior": flat, "Sceptical prior": sceptical}
    draws = {
        k: pen.metropolis_logistic(full.x, y, sd, start, cov, n_iter=30_000, seed=13)
        for k, sd in runs.items()
    }
    b_hat = float(start[clip])
    se = float(np.sqrt(full.covariance[clip, clip]))
    rows = [
        {
            "method": "Maximum likelihood",
            "or": float(np.exp(b_hat)),
            "low": float(np.exp(b_hat - 1.96 * se)),
            "high": float(np.exp(b_hat + 1.96 * se)),
            "prob_or_below_1": None,
            "acceptance": None,
        }
    ]
    for k, dr in draws.items():
        c = dr[:, clip]
        rows.append(
            {
                "method": k,
                "or": float(np.exp(np.median(c))),
                "low": float(np.exp(np.quantile(c, 0.025))),
                "high": float(np.exp(np.quantile(c, 0.975))),
                "prob_or_below_1": float(np.mean(c < 0)),
                "acceptance": float(np.mean(np.any(np.diff(dr, axis=0) != 0, axis=1))),
            }
        )
    _write_csv(out, "ch13_bayes_clip", pl.DataFrame(rows))

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    ax = axes[0]
    ax.plot(np.exp(draws["Flat prior"][:3000, clip]), color=BLUE, lw=0.5)
    ax.set_xlabel("Step of the chain")
    ax.set_ylabel("Clip odds ratio")
    ax.set_title("Metropolis chain (first 3000 steps)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    edges = np.linspace(-1.8, 0.4, 70)
    for (k, dr), color in zip(draws.items(), [BLUE, RED], strict=True):
        ax.hist(
            dr[:, clip],
            bins=edges,
            density=True,
            color=color,
            alpha=0.4,
            label=f"Posterior, {k.lower()}",
        )
    b = np.linspace(-1.8, 0.4, 300)
    ax.plot(
        b,
        stats.norm(b_hat, se).pdf(b),
        color=GRAY,
        ls="--",
        label="Maximum likelihood (normal)",
    )
    ax.plot(
        b, stats.norm(0, 0.35).pdf(b), color=RED, ls=":", lw=1, label="Sceptical prior"
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Clip coefficient (log odds ratio)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch13_bayes_clip.png", dpi=180)
    plt.close(fig)


def _separated_sample(df: pl.DataFrame, n: int = 120) -> pl.DataFrame:
    """The first seed whose sample has bleeds, clipped patients, and no clipped bleed."""
    for seed in range(1000):
        d = df.sample(n, seed=seed)
        clipped = d.filter(pl.col("clip") == 1)
        if (
            d["bleed"].sum() >= 5
            and clipped.height >= 20
            and clipped["bleed"].sum() == 0
        ):
            return d
    raise RuntimeError("no separated sample found")


HOSPITAL_SD = 0.5  # true SD of the hospital effect on the log odds


def _multicenter(seed: int = 14) -> tuple[pl.DataFrame, pl.DataFrame]:
    """20 hospitals of very different sizes; each adds its own shift to the log odds."""
    rng = np.random.default_rng(seed)
    sizes = (
        np.clip(np.exp(rng.normal(np.log(110), 0.9, 20)), 15, 700).round().astype(int)
    )
    df = simulate(int(sizes.sum()), seed=seed).drop("bleed")
    hospital = np.repeat(np.arange(20), sizes)
    u = rng.normal(0, HOSPITAL_SD, 20)
    args = [
        df[c].to_numpy()
        for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
    ]
    p = _expit(bleed_logit(*args) + u[hospital])
    labels = np.array([f"H{i + 1:02d}" for i in range(20)])
    df = df.with_columns(
        pl.Series("hospital", labels[hospital]),
        pl.Series("bleed", rng.binomial(1, p)),
        pl.Series("p_true", p),
    )
    truth = (
        df.group_by("hospital")
        .agg(
            pl.len().alias("n"),
            pl.col("bleed").sum().alias("events"),
            pl.col("p_true").mean().alias("true_rate"),
        )
        .join(pl.DataFrame({"hospital": labels, "true_effect": u}), on="hospital")
        .sort("hospital")
    )
    return df, truth


def hierarchical(out: Path) -> None:
    """Chapter 14: hospitals, partial pooling, ranking, and the Bayes view."""
    df, truth = _multicenter()
    overall = float(df["bleed"].mean())

    # Bleeding rate by hospital: own data only vs partial pooling.
    empty = fit_mixed(df, "bleed ~ 1 + (1 | hospital)", family="binomial")
    b0 = float(empty.tidy()["estimate"][0])
    sigma0 = float(np.sqrt(glmm_cluster_variance(empty)[0]))
    re = empty.random_effects().rename({"group": "hospital"}).select("hospital", "blup")
    rates = (
        truth.join(re, on="hospital")
        .with_columns(
            (pl.col("events") / pl.col("n")).alias("own"),
            pl.col("blup")
            .map_batches(lambda b: _expit(b0 + b.to_numpy()))
            .alias("pooled"),
        )
        .sort("n")
    )
    _write_csv(out, "ch14_rates", rates)
    err = rates.select(
        ((pl.col("own") - pl.col("true_rate")) ** 2).mean().sqrt().alias("rmse_own"),
        ((pl.col("pooled") - pl.col("true_rate")) ** 2)
        .mean()
        .sqrt()
        .alias("rmse_pooled"),
        ((overall - pl.col("true_rate")) ** 2).mean().sqrt().alias("rmse_complete"),
    )
    _write_csv(out, "ch14_error", err)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    ax = axes[0]
    for row in rates.iter_rows(named=True):
        ax.plot([0, 1], [row["own"], row["pooled"]], color=GRAY, lw=0.8, alpha=0.7)
    size = np.sqrt(rates["n"].to_numpy()) * 2.5
    ax.scatter(np.zeros(rates.height), rates["own"], s=size, color=RED, zorder=3)
    ax.scatter(np.ones(rates.height), rates["pooled"], s=size, color=BLUE, zorder=3)
    ax.axhline(overall, color=GRAY, ls=":", lw=1)
    ax.text(1.05, overall, "all hospitals", va="center", fontsize=8, color=GRAY)
    ax.set_xticks([0, 1], ["Own data only", "Partial pooling"])
    ax.set_xlim(-0.3, 1.45)
    ax.set_ylabel("Bleeding rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Estimates move toward the mean")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    n = rates["n"].to_numpy()
    ax.scatter(
        n, np.abs(rates["own"] - rates["true_rate"]), color=RED, label="Own data only"
    )
    ax.scatter(
        n,
        np.abs(rates["pooled"] - rates["true_rate"]),
        color=BLUE,
        label="Partial pooling",
    )
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.set_xlabel("Patients in the hospital")
    ax.set_ylabel("Error from the true rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Small hospitals gain the most")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch14_shrinkage.png", dpi=180)
    plt.close(fig)

    # Adjusted models: ignore hospital vs random intercept.
    glm = fit_glm(df, PRED_FORMULA, family="binomial")
    mixed = fit_mixed(df, PRED_FORMULA + " + (1 | hospital)", family="binomial")
    sigma2 = glmm_cluster_variance(mixed)[0]
    rows = []
    for name, fit in [
        ("Ignore hospital (GLM)", glm),
        ("Random intercept (GLMM)", mixed),
    ]:
        t = fit.tidy().filter(pl.col("term") == "clip")
        rows.append(
            {
                "model": name,
                "clip_or": float(np.exp(t["estimate"][0])),
                "clip_se": float(t["std_error"][0]),
                "hospital_sd": None if fit is glm else float(np.sqrt(sigma2)),
                "icc": None if fit is glm else float(sigma2 / (sigma2 + np.pi**2 / 3)),
                "mor": None
                if fit is glm
                else float(np.exp(np.sqrt(2 * sigma2) * 0.6745)),
                "log_likelihood": float(fit.log_likelihood),
            }
        )
    _write_csv(out, "ch14_models", pl.DataFrame(rows))

    # Ranking hospitals: caterpillar plot of the adjusted hospital effects.
    effects = mixed.random_effects()
    # Approximate intervals: the curvature of the conditional log likelihood
    # of each hospital effect plus the normal prior (a Laplace approximation).
    blup = dict(zip(effects["group"], effects["blup"], strict=True))
    hosp = df["hospital"].to_numpy()
    eta = mixed.predict(df, kind="link") + np.array([blup[h] for h in hosp])
    w = _expit(eta) * (1 - _expit(eta))
    se = {h: float(1 / np.sqrt(w[hosp == h].sum() + 1 / sigma2)) for h in blup}
    effects = effects.with_columns(
        (pl.col("blup") - 1.96 * pl.col("group").replace_strict(se)).alias("conf_low"),
        (pl.col("blup") + 1.96 * pl.col("group").replace_strict(se)).alias("conf_high"),
    )
    _write_csv(out, "ch14_effects", effects)
    plot_random_effects(
        effects,
        out / "figures" / "ch14_caterpillar.png",
        xlabel="Hospital effect (log odds, adjusted)",
        figsize=(6, 5),
    )
    ranks = (
        effects.rename({"group": "hospital"})
        .select("hospital", "blup")
        .join(truth.select("hospital", "true_effect", "n"), on="hospital")
        .with_columns(
            pl.col("blup").rank(descending=True).alias("rank_estimated"),
            pl.col("true_effect").rank(descending=True).alias("rank_true"),
        )
        .sort("rank_estimated")
    )
    _write_csv(out, "ch14_ranks", ranks)

    # Bayes: the hospital distribution as a prior for the smallest hospital.
    zero = rates.filter(pl.col("events") == 0)
    small = (zero if zero.height else rates).row(0, named=True)
    grid = np.linspace(0.0005, 0.30, 3000)
    logit = np.log(grid / (1 - grid))
    prior = np.exp(-((logit - b0) ** 2) / (2 * sigma0**2)) / (grid * (1 - grid))
    like = grid ** small["events"] * (1 - grid) ** (small["n"] - small["events"])
    post = prior * like
    curves = {
        "Prior (other hospitals)": prior,
        "Likelihood (own data)": like,
        "Posterior": post,
    }
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (name, c), color, ls in zip(
        curves.items(), [GRAY, RED, BLUE], ["--", ":", "-"], strict=True
    ):
        ax.plot(grid, c / np.trapezoid(c, grid), color=color, ls=ls, label=name)
    ax.set_xlim(0, 0.25)
    ax.set_xlabel(
        f"Bleeding rate of hospital {small['hospital']} "
        f"({small['events']} of {small['n']})"
    )
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch14_bayes.png", dpi=180)
    plt.close(fig)
    post = post / np.trapezoid(post, grid)
    cdf = np.cumsum(post) * (grid[1] - grid[0])
    _write_csv(
        out,
        "ch14_bayes",
        pl.DataFrame(
            {
                "hospital": [small["hospital"]],
                "n": [small["n"]],
                "events": [small["events"]],
                "own_rate": [small["own"]],
                "pooled": [small["pooled"]],
                "posterior_mean": [float(np.trapezoid(grid * post, grid))],
                "posterior_low": [float(grid[np.searchsorted(cdf, 0.025)])],
                "posterior_high": [float(grid[np.searchsorted(cdf, 0.975)])],
                "prior_center": [float(_expit(np.array(b0)))],
                "prior_sd_logit": [sigma0],
                "true_rate": [small["true_rate"]],
            }
        ),
    )


SHAPE = 0.5  # Weibull shape of the true baseline hazard: high at first, then falling
FOLLOW_UP = 30.0
CUTS = [0.0, 1.0, 3.0, 7.0, 14.0, 30.0]
SURV_FORMULA = "Surv(time, event) ~ " + " + ".join(PRED_VARS)


def _survival_data(
    df: pl.DataFrame, seed: int = 15
) -> tuple[pl.DataFrame, float, float]:
    """Days to delayed bleeding, with loss to follow-up and the end of follow-up at day 30."""
    rng = np.random.default_rng(seed)
    args = [
        df[c].to_numpy()
        for c in ["age", "antithrombotic", "size_mm", "proximal", "clip"]
    ]
    lp = bleed_logit(*args) - TRUE_BLEED["intercept"]
    mean_risk = float(np.mean(np.exp(lp)))
    lam = 0.05 / (FOLLOW_UP**SHAPE * mean_risk)
    t_bleed = (rng.exponential(1.0, df.height) / (lam * np.exp(lp))) ** (1 / SHAPE)
    t_loss = rng.exponential(1 / 0.004, df.height)
    time = np.minimum.reduce([t_bleed, t_loss, np.full(df.height, FOLLOW_UP)])
    d = df.drop("bleed").with_columns(
        pl.Series("time", np.maximum(time, 0.01)),
        pl.Series("event", (t_bleed <= np.minimum(t_loss, FOLLOW_UP)).astype(int)),
    )
    return d, lam, mean_risk


def _split(d: pl.DataFrame) -> pl.DataFrame:
    """One row per patient and interval: person-days and whether bleeding happened in it."""
    parts = []
    for a, b in itertools.pairwise(CUTS):
        part = d.filter(pl.col("time") > a).with_columns(
            (pl.min_horizontal(pl.col("time"), pl.lit(b)) - a).alias("days"),
            ((pl.col("event") == 1) & (pl.col("time") <= b))
            .cast(pl.Int64)
            .alias("event"),
            pl.lit(f"{a:g}-{b:g} d").alias("interval"),
        )
        parts.append(part)
    return pl.concat(parts).with_columns(pl.col("days").log().alias("log_days"))


def _risk_set_schematic(path: Path) -> None:
    """Partial likelihood: at each bleed, who was still at risk?"""
    times = [3, 6, 9, 11, 14, 17, 21, 25]
    status = [1, 0, 1, 0, 0, 1, 0, 0]
    fig, ax = plt.subplots(figsize=(7, 3))
    t_event = 9
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
    ax.text(t_event + 0.3, len(times) - 1.5, "a bleed on day 9", color=RED, fontsize=8)
    ax.set_yticks(
        range(len(times)), [f"Patient {i + 1}" for i in range(len(times))], fontsize=8
    )
    ax.set_xlabel("Days after resection")
    ax.set_xlim(0, 27)
    ax.set_title(
        "blue: still at risk on day 9    x: bleed    o: censored",
        fontsize=8,
        color=GRAY,
        loc="left",
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def survival(df: pl.DataFrame, out: Path) -> None:
    """Chapter 15: censoring, hazards, Cox, Poisson person-time, and AFT models."""
    d, lam, mean_risk = _survival_data(df)
    _write_csv(
        out,
        "ch15_summary",
        pl.DataFrame(
            {
                "n": [d.height],
                "bleeds": [int(d["event"].sum())],
                "lost": [int(((d["event"] == 0) & (d["time"] < FOLLOW_UP)).sum())],
                "median_day_of_bleed": [
                    float(d.filter(pl.col("event") == 1)["time"].median())
                ],
                "bleeds_by_day3": [int(((d["event"] == 1) & (d["time"] <= 3)).sum())],
            }
        ),
    )

    # Kaplan-Meier: cumulative incidence by clip.
    ax = plot_survival(
        d.with_columns(
            pl.col("clip").replace_strict({0: "No clip", 1: "Clip"}).alias("group")
        ),
        time="time",
        status="event",
        hue="group",
        cdf=True,
        show_censors=False,
    )
    ax.set_ylim(0, 0.08)
    ax.set_yticks(np.arange(0, 0.081, 0.02))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Days after resection")
    ax.set_ylabel("Cumulative incidence of bleeding")
    ax.figure.savefig(out / "figures" / "ch15_km.png", dpi=180)
    plt.close(ax.figure)

    # Person-time: the classic rate.
    rates = (
        d.group_by("clip")
        .agg(
            pl.col("event").sum().alias("events"),
            pl.col("time").sum().alias("person_days"),
        )
        .with_columns(
            (1000 * pl.col("events") / pl.col("person_days")).alias("per_1000_days")
        )
        .sort("clip")
    )
    _write_csv(out, "ch15_rates", rates)

    # The hazard over time: piecewise rates, Weibull, constant.
    split = _split(d)
    piece = (
        split.group_by("interval", maintain_order=True)
        .agg(
            pl.col("event").sum().alias("events"),
            pl.col("days").sum().alias("person_days"),
        )
        .with_columns(
            (1000 * pl.col("events") / pl.col("person_days")).alias("per_1000_days")
        )
    )
    _write_csv(out, "ch15_piecewise", piece)
    weibull = accelerated_failure(d, SURV_FORMULA, distribution="weibull")
    expo = accelerated_failure(d, SURV_FORMULA, distribution="exponential")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (a, b), r in zip(itertools.pairwise(CUTS), piece["per_1000_days"], strict=True):
        ax.plot([a, b], [r, r], color=RED, lw=2.5)
    ax.plot([], [], color=RED, lw=2.5, label="Observed rate in each interval (Poisson)")
    t = np.linspace(0.2, FOLLOW_UP, 300)
    # Hazard of the average patient: h(t) = shape * lam * t^(shape - 1) * mean(exp(lp)).
    true_h = 1000 * SHAPE * lam * t ** (SHAPE - 1) * mean_risk
    ax.plot(t, true_h, color=BLUE, label="True hazard (Weibull, shape 0.5)")
    overall = 1000 * d["event"].sum() / d["time"].sum()
    ax.axhline(overall, color=GRAY, ls="--", label="Constant hazard (exponential)")
    ax.set_ylim(0, max(true_h[5], float(piece["per_1000_days"].max())) * 1.1)
    ax.set_xlabel("Days after resection")
    ax.set_ylabel("Bleeds per 1000 patient-days")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ch15_hazard.png", dpi=180)
    plt.close(fig)

    _risk_set_schematic(out / "figures" / "ch15_risk_set.png")

    # Cox, and Cox on the ranks of time.
    cox = cox_ph(d, SURV_FORMULA)
    ranked = d.with_columns(
        pl.col("time").rank("ordinal").cast(pl.Float64).alias("time")
    )
    cox_rank = cox_ph(ranked, SURV_FORMULA)
    squared = d.with_columns((pl.col("time") ** 2).alias("time"))
    cox_sq = cox_ph(squared, SURV_FORMULA)

    pois_const = fit_glm(
        d.with_columns(pl.col("time").log().alias("log_days")),
        "event ~ " + " + ".join(PRED_VARS),
        family="poisson",
        offset="log_days",
    )
    pois_piece = fit_glm(
        split,
        "event ~ interval + " + " + ".join(PRED_VARS),
        family="poisson",
        offset="log_days",
    )
    lognormal = accelerated_failure(d, SURV_FORMULA, distribution="lognormal")

    def clip_row(model: str, fit, scale: str, transform=None) -> dict:
        t = fit.tidy().filter(pl.col("term") == "clip")
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

    rows = [
        clip_row("Cox", cox, "hazard ratio"),
        clip_row("Cox, time replaced by its rank", cox_rank, "hazard ratio"),
        clip_row("Cox, time squared", cox_sq, "hazard ratio"),
        clip_row("Poisson, constant hazard", pois_const, "hazard ratio"),
        clip_row("Poisson, 5 intervals", pois_piece, "hazard ratio"),
        clip_row(
            "Weibull, as hazard ratio",
            weibull,
            "hazard ratio",
            lambda v: -v / weibull.scale,
        ),
        clip_row("Weibull AFT", weibull, "time ratio"),
        clip_row("Log-normal AFT", lognormal, "time ratio"),
        clip_row("Exponential AFT", expo, "time ratio"),
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
    matched, _ = adjustment_methods(df, po, out)
    ps_matching(df, po, matched, out)
    iptw(df, po, out)
    sensitivity(df, out)
    prediction_model(df, out)
    overfitting(out)
    discrimination_calibration(df, out)
    regularization(out)
    trees_ensembles(df, out)
    neural_network(df, out)
    probability_model(df, out)
    likelihood(df, out)
    hierarchical(out)
    survival(df, out)
    n = df.height
    events = int(df["bleed"].sum())
    (out / "n.md").write_text(
        f"{n} 例、遅発性出血 {events} 例（{events / n:.1%}）\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
