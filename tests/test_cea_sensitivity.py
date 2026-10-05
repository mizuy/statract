"""Tests for statract.cea.sensitivity (DSA / PSA post-processing)."""

from __future__ import annotations

import numpy as np

from statract.cea.sensitivity import ce_plane, ceac, evpi, one_way_dsa, run_psa, tornado_table


def _toy_evaluate(params: dict[str, float]) -> tuple[list[float], list[float]]:
    """Two strategies; intervention cost scales with param 'c_int'."""
    c0, e0 = 1000.0, 10.0
    c1 = float(params["c_int"])
    e1 = 10.0 + float(params.get("delta_e", 1.0))
    return [c0, c1], [e0, e1]


def test_run_psa_and_ceac_evpi() -> None:
    rng = np.random.default_rng(0)
    n = 200
    draws = {
        "c_int": rng.normal(5000.0, 500.0, size=n),
        "delta_e": rng.normal(1.0, 0.1, size=n),
    }
    psa = run_psa(
        strategies=["SOC", "INT"],
        param_draws=draws,
        evaluate=_toy_evaluate,
    )
    assert psa.cost.shape == (n, 2)
    plane = ce_plane(psa, comparator="SOC")
    assert plane.height == n
    assert set(plane["strategy"].unique().to_list()) == {"INT"}

    wtps = [0.0, 1000.0, 10_000.0, 100_000.0]
    ac = ceac(psa, wtp=wtps)
    assert ac.height == len(wtps) * 2
    # At each WTP, probabilities sum to 1
    for w in wtps:
        s = ac.filter(ac["wtp"] == w)["prob_ce"].sum()
        np.testing.assert_allclose(s, 1.0)

    ev = evpi(psa, wtp=wtps)
    assert ev.height == len(wtps)
    assert (ev["evpi"] >= -1e-9).all()


def test_one_way_dsa_tornado() -> None:
    dsa = one_way_dsa(
        strategies=["SOC", "INT"],
        base_params={"c_int": 5000.0, "delta_e": 1.0},
        ranges={
            "c_int": (3000.0, 8000.0),
            "delta_e": (0.5, 1.5),
        },
        evaluate=_toy_evaluate,
        outcome="nmb",
        wtp=50_000.0,
    )
    assert dsa.height == 2
    assert dsa["spread"][0] >= dsa["spread"][1]
    torn = tornado_table(dsa)
    assert torn["parameter"].to_list() == dsa["parameter"].to_list()


def test_delta_nmb_dsa() -> None:
    dsa = one_way_dsa(
        strategies=["SOC", "INT"],
        base_params={"c_int": 5000.0, "delta_e": 1.0},
        ranges={"c_int": (4000.0, 6000.0)},
        evaluate=_toy_evaluate,
        outcome="delta_nmb",
        comparator="SOC",
        wtp=50_000.0,
    )
    # base: ΔC=4000, ΔE=1 → NMB = 50_000 - 4000 = 46_000
    np.testing.assert_allclose(dsa["outcome_base"][0], 46_000.0)
    np.testing.assert_allclose(dsa["outcome_low"][0], 47_000.0)
    np.testing.assert_allclose(dsa["outcome_high"][0], 45_000.0)


def test_icer_dsa() -> None:
    dsa = one_way_dsa(
        strategies=["SOC", "INT"],
        base_params={"c_int": 5000.0, "delta_e": 1.0},
        ranges={"c_int": (4000.0, 6000.0)},
        evaluate=_toy_evaluate,
        outcome="icer",
        comparator="SOC",
    )
    # base ICER = (5000-1000)/1 = 4000
    np.testing.assert_allclose(dsa["outcome_base"][0], 4000.0)
