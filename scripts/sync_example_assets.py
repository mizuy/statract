#!/usr/bin/env python3
"""Copy curated example ``*_out/`` artifacts into ``docs/examples/assets/``.

Source of truth is the example analysis output. Docs Pages embeds must use the
exact files produced by ``examples/<stem>/<stem>.py`` (via ``task analysis`` /
``task all``). This script only copies; it never regenerates plots.

Usage (repo root)::

    uv run python scripts/sync_example_assets.py
    uv run python scripts/sync_example_assets.py --stem surv_colon aft_rotterdam
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ASSETS = ROOT / "docs" / "examples" / "assets"

# Relative paths under ``examples/<stem>/<stem>_out/`` → same relative path under assets.
CURATED: dict[str, list[str]] = {
    "surv_colon": [
        "table1_gt.md",
        "figures/km_rx.png",
        "figures/cox_forest.png",
        "figures/cox_schoenfeld.png",
        "figures/cox_loglog.png",
        "figures/cox_dfbeta.png",
        "figures/cox_martingale.png",
        "figures/cox_deviance.png",
        "table1.csv",
        "km_at_3y.csv",
        "cox_tidy.csv",
        "ph_test.csv",
        "mermaid_flowchart.md",
        "text_flowchart.md",
        "logrank.md",
        "ph_test.md",
    ],
    "aft_rotterdam": [
        "table1.csv",
        "table1_gt.md",
        "figures/km_hormon.png",
        "figures/cox_forest.png",
        "figures/cox_schoenfeld.png",
        "figures/cox_loglog.png",
        "figures/cox_dfbeta.png",
        "figures/cox_martingale.png",
        "figures/cox_deviance.png",
        "km_at.csv",
        "cox_tidy.csv",
        "ph_test.csv",
        "aft_weibull_tidy.csv",
        "n.md",
        "logrank.md",
        "ph_test.md",
    ],
    "cif_pbc": [
        "table1.csv",
        "table1_gt.md",
        "figures/cif_death.png",
        "figures/finegray_forest.png",
        "cif_death_at.csv",
        "km_naive_at.csv",
        "finegray_tidy.csv",
        "mermaid_flowchart.md",
        "text_flowchart.md",
    ],
    "htest_licorice": [
        "table1.csv",
        "table1_gt.md",
        "figures/incidence.png",
        "figures/risk_difference_forest.png",
        "incidence.csv",
        "sore_throat_tests.csv",
        "pain_tests.csv",
        "paired_tests.csv",
        "mermaid_flowchart.md",
        "text_flowchart.md",
    ],
    "logit_indo": [
        "table1.csv",
        "table1_gt.md",
        "figures/glmm_fixed_forest.png",
        "figures/glmm_site_blups.png",
        "figures/glm_adjusted_forest.png",
        "glm_unadjusted.csv",
        "glm_adjusted.csv",
        "glmm_tidy.csv",
        "glmm_random_effects.csv",
        "glmm_mor.csv",
        "n.md",
        "glmm.md",
    ],
    "psm_rhc": [
        "table1_unmatched.csv",
        "table1_unmatched_gt.md",
        "table1_matched.csv",
        "table1_matched_gt.md",
        "figures/love_plot.png",
        "figures/glm_matched_forest.png",
        "balance.csv",
        "glm_matched.csv",
        "glm_unmatched.csv",
        "n.md",
    ],
    "iptw_nhefs": [
        "table1.csv",
        "table1_gt.md",
        "figures/cem_love.png",
        "figures/ols_iptw_forest.png",
        "iptw_weight_summary.csv",
        "ols_unadjusted.csv",
        "ols_iptw_hc3.csv",
        "mermaid_flowchart.md",
        "text_flowchart.md",
    ],
    "cox_retinopathy": [
        "table1_gt.md",
        "figures/km_trt.png",
        "figures/cox_forest.png",
        "figures/cox_schoenfeld.png",
        "figures/cox_loglog.png",
        "figures/cox_dfbeta.png",
        "figures/cox_martingale.png",
        "figures/cox_deviance.png",
        "table1.csv",
        "km_at.csv",
        "cox_tidy.csv",
        "cox_se_compare.csv",
        "ph_test.csv",
        "n.md",
        "logrank.md",
        "ph_test.md",
    ],
    "pred_support": [
        "table1.csv",
        "table1_gt.md",
        "ctree.txt",
        "ctree_tests.csv",
        "model_compare_val.csv",
        "figures/glm_full_forest.png",
        "figures/fig_calibration.png",
        "figures/fig_calibration_apparent.png",
        "figures/fig_dca.png",
        "glm_full_train.csv",
        "table_brier.csv",
        "table_calibration.csv",
        "table_calibration_apparent.csv",
        "table_dca.csv",
        "binary_perf_val.csv",
        "threshold_tradeoff_val.csv",
        "mermaid_flowchart.md",
        "text_flowchart.md",
        "n.md",
        "split.md",
    ],
    "lmm_pbcseq": [
        "table1.csv",
        "table1_gt.md",
        "figures/gam_day.png",
        "lmm_fixed.csv",
        "ols_cluster.csv",
        "gam_tidy.csv",
        "n.md",
        "visit_n.md",
    ],
    "cea_sicksicker": [
        "figures/trace_soc.png",
        "figures/ce_frontier.png",
        "figures/tornado.png",
        "figures/ce_plane.png",
        "figures/ceac.png",
        "figures/evpi.png",
        "params.csv",
        "icer.csv",
        "dsa.csv",
        "ceac.csv",
        "evpi.csv",
        "psa_mean_icer.csv",
        "cohort_n.md",
        "model_notes.md",
    ],
}


def sync_stem(stem: str, *, missing_ok: bool = False) -> list[Path]:
    if stem not in CURATED:
        raise KeyError(f"unknown stem {stem!r}; known: {sorted(CURATED)}")
    src_root = EXAMPLES / stem / f"{stem}_out"
    dst_root = ASSETS / stem
    if not src_root.is_dir():
        msg = f"missing analysis output: {src_root} (run examples/{stem} task all first)"
        if missing_ok:
            print(f"skip {stem}: {msg}", file=sys.stderr)
            return []
        raise FileNotFoundError(msg)

    copied: list[Path] = []
    for rel in CURATED[stem]:
        src = src_root / rel
        if not src.is_file():
            # Figures may live under figures/; also allow flat names already in assets.
            alt = src_root / Path(rel).name
            if alt.is_file():
                src = alt
            else:
                print(f"warn: missing {src}", file=sys.stderr)
                continue
        dst = dst_root / Path(rel).name if rel.startswith("figures/") else dst_root / Path(rel).name
        # Keep forest/KM/diagnostic PNGs flat under assets/<stem>/ (existing convention).
        dst = dst_root / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stem",
        nargs="*",
        default=None,
        help="Example stems to sync (default: all curated stems with an *_out/ dir)",
    )
    parser.add_argument(
        "--missing-ok",
        action="store_true",
        help="Skip stems whose *_out/ is absent instead of failing",
    )
    args = parser.parse_args(argv)
    stems = args.stem or list(CURATED)
    n = 0
    for stem in stems:
        n += len(sync_stem(stem, missing_ok=args.missing_ok or args.stem is None))
    print(f"synced {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
