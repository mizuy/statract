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
import matplotlib.ticker
import numpy as np
import polars as pl

from support import ProjectPath, load_data
from statract.report.artifacts import write_csv_companion
from statract import (
    hc_covariance,
    match_sample,
    plot_love,
    propensity_weights,
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
    n = df.height
    events = int(df["bleed"].sum())
    (out / "n.md").write_text(
        f"{n} 例、遅発性出血 {events} 例（{events / n:.1%}）\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
