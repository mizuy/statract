"""Statistics on Polars frames: Table One, models, survival, figures, and CEA.

The names below are re-exported from these subpackages:

- ``statract.tableone``: Table One, aggregation columns, and group tests
- ``statract.models``: OLS / GLM, sandwich covariance, tests, mixed and additive
  models, trees, imputation, matching, ROC, calibration, and validation
- ``statract.surv``: survival curves, log-rank, Cox, AFT, Fine–Gray
- ``statract.viz``: forest, Kaplan–Meier, and tree plots
- ``statract.report``: CSV companions and task output folders
- ``statract.cea``: cost-effectiveness analysis (import it directly)

``glmm_gpboost`` needs the optional ``statract[gpboost]`` extra and is experimental.

R integration lives in ``statract.r`` and needs the optional ``statract[r]`` extra plus a
working R installation. Importing ``statract`` does not load rpy2; rpy2
loads on the first call to ``run`` / ``assign`` / ``get`` etc.
"""

from __future__ import annotations

import polars as pl

from .tableone.agg import (
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
from .models.binary import binary_perf, decision_rates, threshold_tradeoff
from .models.probability import (
    brier_score,
    calibration_table,
    decision_curve_table,
    net_benefit,
    plot_calibration,
    plot_dca,
    write_probability_artifacts,
)
from .models.covariance import (
    bootstrap_covariance,
    cluster_covariance,
    hc_covariance,
    meat,
    newey_west_covariance,
)
from .models.fit import fit_glm, fit_ols
from .models.multinom import MultinomialFit, multinomial_regression
from .models.ordinal import BrantTest, OrdinalFit, brant_test, ordinal_regression
from .models.risk import RiskRatioFit, fit_risk_ratio
from .models.formula import model_matrix
from .models.mixed import MixedFit, fit_mixed
from .models.impute import MultipleImputation, impute_chained, pool
from .models.gamm import GammFit, gamm
from .models.gam import compare_gams, gam, smooth, tensor_interaction, tensor_smooth
from .models.tree import conditional_tree
from .models.htest import HTest, binom_test, mcnemar_test, p_adjust, prop_test, t_test, wilcox_test
from .viz.tree import plot_tree
from .models.linear_tests import (
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
from .models.matching import match_sample
from .models.roc import RocCurve, RocTest, plot_roc, roc_curve, roc_test
from .viz.forest import plot_forest
from .models.glmm_extras import (
    glmm_cluster_variance,
    glmm_forestplot,
    glmm_gpboost,
    glmm_random_effects,
    median_odds_ratio,
    plot_random_effects,
)
from .tableone.stat import (
    proportion_ci,
    cohen_d,
    format_pvalue,
    notnull_mean,
    odds,
    oddsratio,
    stat_anova,
    stat_auto,
    stat_chisq,
    stat_kruskal,
    standardized_difference,
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
    cumulative_incidence,
    fine_gray,
    fine_gray_regression,
    log_rank,
    plot_cox_residuals,
    plot_loglog,
    proportional_hazards_test,
    restricted_mean_survival,
    split_follow_up,
    standardize_cox,
    survival_curve,
    write_cox_diagnostic_suite,
)
from .surv.rmst import RmstResult
from .viz.km import (
    add_at_risk_counts,
    cumulative_survival_ci,
    default_at_risk_xticks,
    log_rank_pvalue,
    plot_survival,
)
from .tableone.table import (
    TableOneStyle,
    tableone,
    tableone_gt_from_frame,
    tableone_raw,
    write_tableone_artifacts,
)
from .report.task_io import (
    clear_task_output_dir,
    prepare_task_output,
    print_saved,
    save_frames,
    task_output_dir,
)
from .models.validation import (
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
    "HTest",
    "binom_test",
    "mcnemar_test",
    "p_adjust",
    "prop_test",
    "t_test",
    "wilcox_test",
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
    "plot_tree",
    "cox_ph",
    "cumulative_incidence",
    "durbin_watson_test",
    "fine_gray",
    "fine_gray_regression",
    "fit_glm",
    "fit_mixed",
    "MixedFit",
    "fit_ols",
    "fit_risk_ratio",
    "RiskRatioFit",
    "ordinal_regression",
    "OrdinalFit",
    "brant_test",
    "BrantTest",
    "multinomial_regression",
    "MultinomialFit",
    "restricted_mean_survival",
    "RmstResult",
    "gamm",
    "GammFit",
    "impute_chained",
    "MultipleImputation",
    "pool",
    "model_matrix",
    "gam",
    "hc_covariance",
    "likelihood_ratio_test",
    "log_rank",
    "match_sample",
    "RocCurve",
    "RocTest",
    "plot_roc",
    "roc_curve",
    "roc_test",
    "meat",
    "newey_west_covariance",
    "proportional_hazards_test",
    "plot_cox_residuals",
    "plot_loglog",
    "write_cox_diagnostic_suite",
    "ramsey_reset_test",
    "smooth",
    "split_follow_up",
    "standardize_cox",
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
    "stat_kruskal",
    "standardized_difference",
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
