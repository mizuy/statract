from __future__ import annotations

import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

"""DARTH Sick-Sicker CEA: cohort Markov, ICER, one-way DSA, PSA."""


import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from build import (
    AGE_END,
    AGE_START,
    BASE_PARAMS,
    DSA_RANGES,
    DISCOUNT_RATE,
    N_PSA,
    PSA_DISTS,
    PSA_SEED,
    STRATEGIES,
    WTP,
)
from support import ProjectPath, load_parquet_dir
from statract.cea import (
    calculate_icers,
    ce_plane,
    ceac,
    evpi,
    net_monetary_benefit,
    one_way_dsa,
    run_psa,
    simulate_cohort_markov,
    tornado_table,
)
from statract.report.artifacts import write_csv_companion

project = ProjectPath(__file__)
CACHE = project.cache
ANALYSIS_OUT = project.out

STATES = ("H", "S1", "S2", "D")
AGES = range(AGE_START, AGE_END + 1)


def _clear_out(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    return out


def _write_csv(out: Path, stem: str, frame: pl.DataFrame) -> None:
    path = out / f"{stem}.csv"
    frame.write_csv(path)
    write_csv_companion(path, csv_link_prefix=out.name)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p / (1.0 - p)))


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _prob_from_hr(p_base: float, hr: float) -> float:
    r = -np.log(1.0 - float(p_base))
    return float(1.0 - np.exp(-float(hr) * r))


def transition_matrix(params: dict[str, float], *, treatment_b: bool) -> np.ndarray:
    """Time-homogeneous P (rows from, cols to). Disease model is not in statract.cea."""
    p_hd = float(params["p_HD"])
    p_hs1 = float(params["p_HS1"])
    p_s1h = float(params["p_S1H"])
    p_s1s2 = float(params["p_S1S2"])
    if treatment_b:
        p_s1s2 = _expit(_logit(p_s1s2) + np.log(float(params["or_S1S2"])))
    p_s1d = _prob_from_hr(p_hd, params["hr_S1"])
    p_s2d = _prob_from_hr(p_hd, params["hr_S2"])
    stay_s1 = max(0.0, 1.0 - p_s1h - p_s1s2)
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = (1.0 - p_hd) * (1.0 - p_hs1)
    p[0, 1] = (1.0 - p_hd) * p_hs1
    p[0, 3] = p_hd
    p[1, 0] = (1.0 - p_s1d) * p_s1h
    p[1, 1] = (1.0 - p_s1d) * stay_s1
    p[1, 2] = (1.0 - p_s1d) * p_s1s2
    p[1, 3] = p_s1d
    p[2, 2] = 1.0 - p_s2d
    p[2, 3] = p_s2d
    p[3, 3] = 1.0
    return p


def simulate_strategy(name: str, params: dict[str, float], *, record_trace: bool = False):
    use_a = name in {"Strategy A", "Strategy AB"}
    use_b = name in {"Strategy B", "Strategy AB"}
    extra_c = 0.0
    if use_a:
        extra_c += float(params["c_trtA"])
    if use_b:
        extra_c += float(params["c_trtB"])
    cost = {
        "H": float(params["c_H"]),
        "S1": float(params["c_S1"]) + extra_c,
        "S2": float(params["c_S2"]) + extra_c,
        "D": float(params["c_D"]),
    }
    u_s1 = float(params["u_trtA"]) if use_a else float(params["u_S1"])
    utility = {
        "H": float(params["u_H"]),
        "S1": u_s1,
        "S2": float(params["u_S2"]),
        "D": float(params["u_D"]),
    }
    return simulate_cohort_markov(
        states=STATES,
        initial={"H": 1.0},
        ages=AGES,
        transition=transition_matrix(params, treatment_b=use_b),
        utility=utility,
        cost=cost,
        discount_rate=DISCOUNT_RATE,
        record_trace=record_trace,
    )


def evaluate(params: dict[str, float]) -> tuple[list[float], list[float]]:
    costs: list[float] = []
    effects: list[float] = []
    for name in STRATEGIES:
        res = simulate_strategy(name, params)
        costs.append(res.cost)
        effects.append(res.qaly)
    return costs, effects


def _draw_psa(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    draws: dict[str, np.ndarray] = {}
    for key, value in BASE_PARAMS.items():
        spec = PSA_DISTS.get(key)
        if spec is None:
            draws[key] = np.full(n, value, dtype=np.float64)
            continue
        kind, a, b = spec
        if kind == "beta":
            draws[key] = rng.beta(a, b, size=n)
        elif kind == "gamma":
            draws[key] = rng.gamma(a, b, size=n)
        elif kind == "lognormal":
            draws[key] = rng.lognormal(a, b, size=n)
        else:
            raise ValueError(kind)
    # Keep utilities ordered: u_H >= u_trtA >= u_S1 >= u_S2 >= u_D.
    u_h = np.clip(draws["u_H"], 1e-6, 1.0)
    u_trt = np.minimum(draws["u_trtA"], u_h)
    u_s1 = np.minimum(draws["u_S1"], u_trt)
    u_s2 = np.minimum(draws["u_S2"], u_s1)
    draws["u_H"] = u_h
    draws["u_trtA"] = u_trt
    draws["u_S1"] = u_s1
    draws["u_S2"] = u_s2
    draws["u_D"] = np.zeros(n, dtype=np.float64)
    draws["c_D"] = np.zeros(n, dtype=np.float64)
    return draws


def _plot_trace(trace: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    age = trace["age"].to_numpy()
    for state, color in zip(STATES, ("#2e7d32", "#f9a825", "#e65100", "#424242")):
        ax.plot(age, trace[state].to_numpy(), label=state, color=color)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Cohort membership")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_frontier(icers: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    nd = icers.filter(pl.col("status") == "ND")
    other = icers.filter(pl.col("status") != "ND")
    if nd.height:
        ax.plot(nd["effect"].to_numpy(), nd["cost"].to_numpy(), color="#1565c0", zorder=1)
        ax.scatter(
            nd["effect"].to_numpy(),
            nd["cost"].to_numpy(),
            color="#1565c0",
            zorder=2,
            label="ND",
        )
    if other.height:
        ax.scatter(
            other["effect"].to_numpy(),
            other["cost"].to_numpy(),
            color="#9e9e9e",
            zorder=2,
            label="dominated",
        )
    for row in icers.iter_rows(named=True):
        ax.annotate(row["strategy"], (row["effect"], row["cost"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("QALYs")
    ax.set_ylabel("Cost")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tornado(dsa: pl.DataFrame, path: Path, *, base: float) -> None:
    ordered = dsa.sort("spread")
    labels = ordered["parameter"].to_list()
    y = np.arange(len(labels))
    left = ordered["outcome_min"].to_numpy()
    width = ordered["outcome_max"].to_numpy() - left
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(y, width, left=left, color="#90caf9", edgecolor="#1565c0", height=0.7)
    ax.axvline(base, color="#c62828", linewidth=1.2, label="base")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("NMB of the optimal strategy, WTP = $100,000/QALY")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ce_plane(plane: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    wtp = WTP
    xmax = float(max(plane["delta_effect"].max(), 0.05))
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.plot([0, xmax], [0, wtp * xmax], color="#c62828", linewidth=1.0, label=f"WTP ${wtp:,.0f}")
    for name, color in (
        ("Strategy A", "#1565c0"),
        ("Strategy B", "#2e7d32"),
        ("Strategy AB", "#6a1b9a"),
    ):
        sub = plane.filter(pl.col("strategy") == name)
        ax.scatter(
            sub["delta_effect"].to_numpy(),
            sub["delta_cost"].to_numpy(),
            s=8,
            alpha=0.25,
            color=color,
            label=name,
        )
    ax.set_xlabel("Incremental QALYs vs SoC")
    ax.set_ylabel("Incremental cost vs SoC")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ceac(ac: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, color in zip(
        STRATEGIES,
        ("#424242", "#1565c0", "#2e7d32", "#6a1b9a"),
    ):
        sub = ac.filter(pl.col("strategy") == name).sort("wtp")
        ax.plot(sub["wtp"].to_numpy() / 1000.0, sub["prob_ce"].to_numpy(), label=name, color=color)
    ax.set_xlabel("WTP ($1,000 / QALY)")
    ax.set_ylabel("P(max NMB)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_evpi(voi: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(voi["wtp"].to_numpy() / 1000.0, voi["evpi"].to_numpy(), color="#1565c0")
    ax.set_xlabel("WTP ($1,000 / QALY)")
    ax.set_ylabel("EVPI (per person)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = _clear_out(ANALYSIS_OUT)
    data = load_parquet_dir(CACHE / "build")
    params_tbl: pl.DataFrame = data["params"]
    _write_csv(out, "params", params_tbl)

    soc = simulate_strategy("Standard of care", BASE_PARAMS, record_trace=True)
    assert soc.trace is not None
    _write_csv(out, "trace_soc", soc.trace)
    _plot_trace(soc.trace, out / "figures" / "trace_soc.png")

    costs, effects = evaluate(BASE_PARAMS)
    icers = calculate_icers(costs, effects, strategies=list(STRATEGIES))
    nmb = net_monetary_benefit(icers["cost"], icers["effect"], wtp=WTP)
    icers = icers.with_columns(pl.Series("nmb_wtp100k", nmb))
    _write_csv(out, "icer", icers)
    _plot_frontier(icers, out / "figures" / "ce_frontier.png")

    dsa = one_way_dsa(
        strategies=list(STRATEGIES),
        base_params=BASE_PARAMS,
        ranges=DSA_RANGES,
        evaluate=evaluate,
        outcome="nmb",
        wtp=WTP,
    )
    dsa = tornado_table(dsa)
    _write_csv(out, "dsa", dsa)
    base_delta = float(dsa["outcome_base"][0])
    _plot_tornado(dsa, out / "figures" / "tornado.png", base=base_delta)

    rng = np.random.default_rng(PSA_SEED)
    psa = run_psa(
        strategies=list(STRATEGIES),
        param_draws=_draw_psa(rng, N_PSA),
        evaluate=evaluate,
    )
    plane = ce_plane(psa, comparator="Standard of care")
    wtp_grid = np.linspace(0.0, 200_000.0, 41)
    ac = ceac(psa, wtp=wtp_grid)
    voi = evpi(psa, wtp=wtp_grid)
    _write_csv(out, "ce_plane", plane)
    _write_csv(out, "ceac", ac)
    _write_csv(out, "evpi", voi)
    _plot_ce_plane(plane, out / "figures" / "ce_plane.png")
    _plot_ceac(ac, out / "figures" / "ceac.png")
    _plot_evpi(voi, out / "figures" / "evpi.png")

    mean_cost = psa.cost.mean(axis=0)
    mean_eff = psa.effect.mean(axis=0)
    psa_mean = calculate_icers(mean_cost, mean_eff, strategies=list(STRATEGIES))
    _write_csv(out, "psa_mean_icer", psa_mean)

    (out / "cohort_n.md").write_text(
        "Virtual closed cohort: 100% start in H at age 25; ages 25–100 inclusive "
        f"({len(list(AGES))} cycles); no patient-level inclusion/exclusion.\n",
        encoding="utf-8",
    )
    (out / "model_notes.md").write_text(
        "Time-homogeneous 4-state STM (H, S1, S2, D). State rewards only "
        "(no DARTH transition rewards / half-cycle / age-specific mortality). "
        f"Discount {DISCOUNT_RATE:.0%} on cost, QALY, and LY. PSA n={N_PSA}, seed={PSA_SEED}. "
        f"WTP for NMB/DSA tornado = ${WTP:,.0f}/QALY. "
        "Tornado is NMB of the optimal strategy at WTP $100,000/QALY.\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {project.project_root}")


if __name__ == "__main__":
    main()
