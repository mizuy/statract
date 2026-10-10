from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""Figures and tables for the theory text.

Part I (chapter 1): the conventional "independent risk factor" analysis.
Part II (chapters 2-6): potential outcomes, adjustment, PS matching, IPTW, sensitivity.
Part III (chapters 7-9): prediction, overfitting, discrimination and calibration.
"""


import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl
from build import bleed_logit, potential_outcomes, simulate
from support import ProjectPath, load_data

from statract import (
    agg_category,
    agg_mean_sd,
    agg_median_iqr,
    brier_score,
    decision_curve_table,
    fit_glm,
    hc_covariance,
    match_sample,
    plot_forest,
    plot_love,
    plot_roc,
    prop_test,
    propensity_weights,
    roc_curve,
    standardize_glm,
    threshold_tradeoff,
    validate_logistic,
    write_tableone_artifacts,
)
from statract.report.artifacts import write_csv_companion

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
    n = df.height
    events = int(df["bleed"].sum())
    (out / "n.md").write_text(
        f"{n} 例、遅発性出血 {events} 例（{events / n:.1%}）\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
