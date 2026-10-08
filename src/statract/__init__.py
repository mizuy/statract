"""Statistical analysis: Table One, survival analysis, regression, and more.

This module provides statistical analysis utilities for Polars DataFrames, including:
- Table One generation
- Survival analysis (Kaplan-Meier, log-rank test)
- Statistical tests (ANOVA, chi-square, Fisher's exact, etc.)
- Aggregation functions for descriptive statistics
- Propensity score matching (experimental)
- Linear and generalized linear mixed models (``fit_mixed``; lme-rs / lme-python)
- Experimental extras (``glmm_gpboost`` via optional ``statract[gpboost]``)
- Confusion matrix and related metrics (experimental)
- Binary classification metrics (``binary_perf``) and probability evaluation
  (calibration / Brier / decision curve)

.. warning::
    Some functions in this module are experimental implementations:
    - ``confusion_matrix``, ``ratio``, ``sm_summary2df`` (from ``.confusion``)
    - ``glmm_gpboost`` (optional extra; not the default GLMM engine),
      ``glmm_forestplot``
    - ``median_odds_ratio``, ``glmm_random_effects``, ``plot_random_effects``
      (MOR / BLUP helpers for ``fit_mixed``)
    - ``plot_forest`` (from ``.forest``) — canonical Cox HR / GLM OR / OLS forest

    These may change or be removed in future versions. Use with caution.

Statistical figures: ``plot_forest`` / ``plot_survival`` (and diagnostic helpers).
Plotly funnel / image concat stay in ``statract.figure``.

R integration lives in ``statract.r`` and needs the optional ``statract[r]`` extra plus a
working R installation. Importing ``statract`` does not load rpy2; rpy2
loads on the first call to ``run`` / ``assign`` / ``get`` etc.
"""

from __future__ import annotations

import polars as pl

from .agg import (
    agg_bool_category,
    agg_bool_ci,
    agg_bool_n,
    agg_bool_np,
    agg_bool_p,
    agg_bool_pnn,
    agg_bool_pnnci,
    agg_category,
    agg_category_base,
    agg_category_n,
    agg_category_np,
    agg_category_p,
    agg_category_pnn,
    agg_count,
    agg_mean_sd,
    agg_median,
    agg_median_iqr,
    agg_median_range,
    agg_nunique,
    agg_range,
    agg_ratio,
    agg_ratio_ci,
    agg_ratio_n,
    agg_ratio_np,
    agg_ratio_pnn,
    agg_ratio_pnnci,
    agg_size,
)
from .binary import binary_perf, decision_rates, threshold_tradeoff
from .confusion import confusion_matrix, ratio, sm_summary2df
from .probability import (
    brier_score,
    calibration_table,
    decision_curve_table,
    net_benefit,
    plot_calibration,
    plot_dca,
    write_probability_artifacts,
)
from .covariance import (
    bootstrap_covariance,
    cluster_covariance,
    hc_covariance,
    meat,
    newey_west_covariance,
)
from .fit import fit_glm, fit_ols
from .formula import model_matrix
from .mixed import MixedFit, fit_mixed
from .gam import compare_gams, gam, smooth, tensor_interaction, tensor_smooth
from .tree import conditional_tree
from .linear_tests import (
    breusch_godfrey_test,
    breusch_pagan_test,
    coefficient_interval,
    coefficient_test,
    durbin_watson_test,
    check_collinearity,
    likelihood_ratio_test,
    ramsey_reset_test,
    wald_test,
)
from .matching import match_sample
from .forest import plot_forest
from .regression import (
    glmm_cluster_variance,
    glmm_forestplot,
    glmm_gpboost,
    glmm_random_effects,
    median_odds_ratio,
    plot_random_effects,
)
from .stat import (
    proportion_ci,
    cohen_d,
    format_pvalue,
    notnull_mean,
    odds,
    oddsratio,
    stat_anova,
    stat_auto,
    stat_chisq,
    weighted_corr,
    weighted_cov,
    weighted_mean,
    weighted_qcut,
    stat_fisher,
)
from .surv import (
    accelerated_failure,
    conditional_logit,
    cox_ph,
    fine_gray,
    log_rank,
    plot_cox_residuals,
    plot_loglog,
    proportional_hazards_test,
    split_follow_up,
    survival_curve,
    write_cox_diagnostic_suite,
)
from .survival import (
    add_at_risk_counts,
    cumulative_survival_ci,
    default_at_risk_xticks,
    log_rank_pvalue,
    plot_survival,
    plot_survival_grid,
)
from .tableone import (
    TableOneStyle,
    tableone,
    tableone_gt_from_frame,
    tableone_raw,
    write_tableone_artifacts,
)
from .task_io import (
    clear_task_output_dir,
    prepare_task_output,
    print_saved,
    save_frames,
    task_output_dir,
)
from .validation import (
    CalibrationCurve,
    SurvivalCalibration,
    calibrate_cox,
    calibrate_logistic,
    gini_mean_difference,
    plot_calibration_curve,
    somers_dxy,
    validate_cox,
    validate_logistic,
)




__all__ = [
    "agg_bool_category",
    "agg_bool_ci",
    "agg_bool_n",
    "agg_bool_np",
    "agg_bool_p",
    "agg_bool_pnn",
    "agg_bool_pnnci",
    "agg_category",
    "agg_category_base",
    "agg_category_n",
    "agg_category_np",
    "agg_category_p",
    "agg_category_pnn",
    "agg_count",
    "agg_mean_sd",
    "agg_median",
    "agg_median_iqr",
    "agg_median_range",
    "agg_nunique",
    "agg_range",
    "agg_ratio",
    "agg_ratio_ci",
    "agg_ratio_n",
    "agg_ratio_np",
    "agg_ratio_pnn",
    "agg_ratio_pnnci",
    "agg_size",
    "binary_perf",
    "brier_score",
    "calibration_table",
    "confusion_matrix",
    "decision_curve_table",
    "decision_rates",
    "net_benefit",
    "plot_calibration",
    "plot_dca",
    "plot_forest",
    "glmm_cluster_variance",
    "glmm_forestplot",
    "glmm_random_effects",
    "median_odds_ratio",
    "plot_random_effects",
    "accelerated_failure",
    "bootstrap_covariance",
    "breusch_godfrey_test",
    "breusch_pagan_test",
    "check_collinearity",
    "cluster_covariance",
    "coefficient_interval",
    "coefficient_test",
    "compare_gams",
    "conditional_logit",
    "conditional_tree",
    "cox_ph",
    "durbin_watson_test",
    "fine_gray",
    "fit_glm",
    "fit_mixed",
    "MixedFit",
    "fit_ols",
    "model_matrix",
    "gam",
    "hc_covariance",
    "likelihood_ratio_test",
    "log_rank",
    "match_sample",
    "meat",
    "newey_west_covariance",
    "proportional_hazards_test",
    "plot_cox_residuals",
    "plot_loglog",
    "write_cox_diagnostic_suite",
    "ramsey_reset_test",
    "smooth",
    "split_follow_up",
    "survival_curve",
    "tensor_interaction",
    "tensor_smooth",
    "wald_test",
    "glmm_gpboost",
    "proportion_ci",
    "threshold_tradeoff",
    "write_probability_artifacts",
    "cohen_d",
    "format_pvalue",
    "notnull_mean",
    "odds",
    "oddsratio",
    "stat_anova",
    "stat_auto",
    "stat_chisq",
    "stat_fisher",
    "weighted_corr",
    "weighted_cov",
    "weighted_mean",
    "weighted_qcut",
    "add_at_risk_counts",
    "cumulative_survival_ci",
    "default_at_risk_xticks",
    "log_rank_pvalue",
    "plot_survival",
    "plot_survival_grid",
    "clear_task_output_dir",
    "prepare_task_output",
    "print_saved",
    "save_frames",
    "task_output_dir",
    "tableone",
    "tableone_gt_from_frame",
    "tableone_raw",
    "write_tableone_artifacts",
    "TableOneStyle",
    "CalibrationCurve",
    "SurvivalCalibration",
    "calibrate_cox",
    "calibrate_logistic",
    "gini_mean_difference",
    "plot_calibration_curve",
    "somers_dxy",
    "validate_cox",
    "validate_logistic",
]
