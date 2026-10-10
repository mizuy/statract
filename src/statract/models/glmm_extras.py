"""Experimental implementations for regression models.

.. warning::
    This module contains experimental implementations that may change
    or be removed in future versions. Use with caution.

.. note::
    The API and behavior of functions in this module may change without notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats

from .formula import model_matrix
from ..tableone.stat import format_pvalue

_Z075 = float(scipy.stats.norm.ppf(0.75))


def median_odds_ratio(
    variance: float | Any,
    *,
    std_error: float | None = None,
) -> pl.DataFrame:
    """Median odds ratio (MOR) for a logit-scale random intercept.

    Larsen et al.: ``MOR = exp(sqrt(2 * sigma^2) * Phi^{-1}(0.75))``. This is the
    median odds ratio between two randomly sampled clusters with the same
    covariates (higher-risk vs lower-risk cluster).

    Args:
        variance: Cluster (random-intercept) variance, a ``MixedFit`` from
            ``fit_mixed(..., family="binomial")``, or (experimental) a GPBoost
            ``GPModel``.
        std_error: Optional SE of the variance. When omitted, a ``GPModel`` may
            supply one; ``MixedFit`` does not currently report a variance SE.

    Returns:
        One-row frame with ``variance`` and ``median_odds_ratio`` (and optional CI).
    """
    se = std_error
    if not isinstance(variance, (int, float, np.floating)):
        variance, extracted_se = glmm_cluster_variance(variance)
        if se is None:
            se = extracted_se
    var = float(variance)
    if var < 0:
        raise ValueError("random-intercept variance must be non-negative")
    if var == 0:
        mor = 1.0
    else:
        mor = float(np.exp(np.sqrt(2.0 * var) * _Z075))
    row: dict[str, float] = {
        "variance": var,
        "median_odds_ratio": mor,
        "z_075": _Z075,
    }
    if se is not None and var > 0:
        se_v = float(se)
        se_log = abs(_Z075 / np.sqrt(2.0 * var)) * se_v
        log_mor = float(np.log(mor))
        row["std_error_variance"] = se_v
        row["log_mor"] = log_mor
        row["std_error_log_mor"] = float(se_log)
        row["conf_low"] = float(np.exp(log_mor - 1.96 * se_log))
        row["conf_high"] = float(np.exp(log_mor + 1.96 * se_log))
    return pl.DataFrame(row)


def glmm_cluster_variance(model: Any) -> tuple[float, float | None]:
    """Random-intercept variance (and SE if present) from a mixed fit.

    Accepts ``MixedFit`` (canonical) or an experimental GPBoost ``GPModel``.
    """
    from .mixed import MixedFit

    if isinstance(model, MixedFit):
        names = model.random_names
        if "(Intercept)" in names:
            i = names.index("(Intercept)")
        elif names:
            i = 0
        else:
            raise ValueError("fit has no random-effect terms")
        return float(model.group_covariance[i, i]), None
    cov = model.get_cov_pars(std_err=True)
    col = cov.columns[0]
    if "Param." in cov.index:
        variance = float(cov.loc["Param.", col])
        se = float(cov.loc["Std. err.", col]) if "Std. err." in cov.index else None
    else:
        variance = float(cov.iloc[0, 0])
        se = float(cov.iloc[1, 0]) if cov.shape[0] > 1 else None
    return variance, se


def glmm_random_effects(
    model: Any,
    group: pl.DataFrame | pl.Series | None = None,
    *,
    predict_var: bool = True,
) -> pl.DataFrame:
    """Unique-group BLUPs from a mixed fit.

    Canonical input is ``MixedFit`` from ``fit_mixed``. The experimental
    GPBoost ``GPModel`` path still accepts observation-level ``group`` labels.
    """
    from .mixed import MixedFit

    if isinstance(model, MixedFit):
        frame = model.random_effects()
        if group is None:
            return frame
        if isinstance(group, pl.Series):
            labels = group.cast(pl.Utf8).to_list()
        else:
            labels = group.get_column(group.columns[0]).cast(pl.Utf8).to_list()
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        n_col = pl.Series("n", [int(counts.get(str(g), 0)) for g in frame["group"].to_list()])
        return frame.with_columns(n_col)

    if group is None:
        raise ValueError("group labels are required for a GPBoost GPModel")
    if isinstance(group, pl.Series):
        gdf = group.to_frame()
    else:
        gdf = group
    if gdf.width < 1 or gdf.height < 1:
        raise ValueError("group data is empty")
    group_name = gdf.columns[0]
    labels = gdf.get_column(group_name).to_list()
    all_re = model.predict_training_data_random_effects(predict_var=predict_var)
    first_idx: list[int] = []
    seen: set[object] = set()
    counts: dict[object, int] = {}
    for i, lab in enumerate(labels):
        counts[lab] = counts.get(lab, 0) + 1
        if lab not in seen:
            seen.add(lab)
            first_idx.append(i)
    unique = all_re.iloc[first_idx]
    mean_col = unique.columns[0]
    blup = np.asarray(unique[mean_col], dtype=float)
    var_col = next((c for c in unique.columns if str(c).endswith("_var")), None)
    frame = pl.DataFrame(
        {
            "group": [str(labels[i]) for i in first_idx],
            "blup": blup,
            "n": [int(counts[labels[i]]) for i in first_idx],
        }
    )
    if var_col is not None:
        var = np.asarray(unique[var_col], dtype=float)
        se = np.sqrt(np.maximum(var, 0.0))
        frame = frame.with_columns(
            pl.Series("variance", var),
            pl.Series("se", se),
            pl.Series("conf_low", blup - 1.96 * se),
            pl.Series("conf_high", blup + 1.96 * se),
        )
    return frame.sort("blup")


def plot_random_effects(
    effects: pl.DataFrame,
    path: Path | str | None = None,
    *,
    group: str = "group",
    estimate: str = "blup",
    conf_low: str | None = "conf_low",
    conf_high: str | None = "conf_high",
    title: str | None = None,
    xlabel: str = "Random intercept (logit)",
    figsize: tuple[float, float] | None = None,
) -> Path | Any:
    """Caterpillar plot of cluster BLUPs (logit scale).

    Draws a vertical reference at 0. When ``conf_low`` / ``conf_high`` exist they
    are used as error bars.

    Args:
        effects: Table from ``glmm_random_effects`` / ``MixedFit.random_effects``
            (``group``, ``blup``, optional CI).
        path: PNG path; when omitted, returns a matplotlib ``Figure``.
        group: Label column.
        estimate: Point-estimate column (BLUP).
        conf_low: Lower interval column (ignored if missing).
        conf_high: Upper interval column (ignored if missing).
        title: Optional title.
        xlabel: X-axis label.
        figsize: Figure size; height scales with the number of groups when omitted.

    Returns:
        ``Path`` when ``path`` is set, otherwise a matplotlib ``Figure``.
    """
    if effects.height == 0:
        raise ValueError("random-effect frame is empty")
    if group not in effects.columns or estimate not in effects.columns:
        raise ValueError(f"frame needs {group!r} and {estimate!r}")
    labels = effects[group].cast(pl.Utf8).to_list()
    est = np.asarray(effects[estimate].to_list(), dtype=float)
    has_ci = (
        conf_low is not None
        and conf_high is not None
        and conf_low in effects.columns
        and conf_high in effects.columns
    )
    n = len(labels)
    if figsize is None:
        figsize = (7.0, max(2.2, 0.42 * n + 1.0))
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    if has_ci:
        lo = np.asarray(effects[conf_low].to_list(), dtype=float)
        hi = np.asarray(effects[conf_high].to_list(), dtype=float)
        ax.errorbar(
            est,
            y,
            xerr=[est - lo, hi - est],
            fmt="o",
            color="#1f4e79",
            ecolor="#1f4e79",
            elinewidth=1.4,
            capsize=3,
            markersize=5,
        )
    else:
        ax.scatter(est, y, color="#1f4e79", s=28, zorder=3)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if path is None:
        return fig
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def glmm_gpboost(
    formula: str,
    cols_group: list[str],
    data: pl.DataFrame,
    return_dict: bool = False,
    likelihood: str = "bernoulli_logit",
) -> pl.DataFrame | dict[str, Any] | None:
    """Experimental GPBoost GLMM (not the default mixed-model engine).

    Prefer ``fit_mixed(..., family="binomial")`` (lme-python / lme-rs). This
    helper remains for the optional ``statract[gpboost]`` extra.

    Args:
        formula: Formula string specifying the model (e.g., "outcome ~ treatment")
        cols_group: List of column names for random effects grouping
        data: Polars DataFrame containing the data
        return_dict: If True, return dictionary with model summary, BLUPs, MOR,
            and a fixed-effect forest plot
        likelihood: Likelihood function ("bernoulli_logit" for logistic regression)

    Returns:
        Polars DataFrame with GLMM results (odds ratios, CI, p-values), or dict if
        return_dict=True, or None if model fails
    """
    try:
        import gpboost
    except ImportError as exc:
        raise ImportError(
            "glmm_gpboost requires the optional extra: pip install 'statract[gpboost]'. "
            "The canonical binomial GLMM is fit_mixed(..., family='binomial')."
        ) from exc

    built = model_matrix(formula, data)
    if built.random_effects:
        raise ValueError("pass grouping columns via cols_group, not (1 | g) in the formula")
    y_arr = np.asarray(built.y)
    x_arr = np.asarray(built.design.x)
    names = list(built.design.names)
    idx = np.asarray(built.design.row_index, dtype=int)
    group_pl = data.select(cols_group).gather(idx)
    group = group_pl.to_pandas()
    if y_arr.ndim > 1:
        y_arr = y_arr[:, 0]

    try:
        model = gpboost.GPModel(group_data=group, likelihood=likelihood)
        model.fit(y=y_arr, X=x_arr)
    except gpboost.basic.GPBoostError as e:
        print(e)
        if return_dict:
            return dict(model=None, error=str(e))
        else:
            return None

    coef_table = (
        model.get_coef(std_err=True)
        .T.reset_index()
        .rename(columns={"index": "variable", "Param.": "coef", "Std. err.": "se"})
    )
    if len(names) == len(coef_table):
        coef_table = coef_table.assign(variable=names)
    summary_table = (
        pl.DataFrame(coef_table)
        .with_columns((pl.col("coef") / pl.col("se")).alias("z"))
        .with_columns(
            (
                2
                * pl.col("z")
                .abs()
                .map_elements(
                    lambda x: scipy.stats.norm.cdf(-x),
                    return_dtype=pl.Float64,
                )
            ).alias("p"),
        )
        .with_columns(
            pl.col("p").alias("p_value"),
            pl.col("p").map_elements(format_pvalue, return_dtype=pl.String).alias("pvalue"),
            pl.col("coef").exp().alias("oddsratio"),
            (pl.col("coef") + 1.96 * pl.col("se")).exp().alias("hi"),
            (pl.col("coef") - 1.96 * pl.col("se")).exp().alias("lo"),
        )
        .with_columns(
            pl.struct(["oddsratio", "lo", "hi"])
            .map_elements(
                lambda s: f"{s['oddsratio']:.3f} ({s['lo']:.3f} - {s['hi']:.3f})",
                return_dtype=pl.String,
            )
            .alias("desc"),
        )
        .drop("z")
    )

    if return_dict:
        from ..viz.forest import plot_forest

        plot = plot_forest(
            summary_table,
            term="variable",
            estimate="oddsratio",
            conf_low="lo",
            conf_high="hi",
            title="GLMM odds ratios",
            xlabel="Odds ratio",
            layout="table",
        )
        re_frame = glmm_random_effects(model, group_pl.select(cols_group[:1]))
        mor = median_odds_ratio(model)
        cov_pars = pl.from_pandas(model.get_cov_pars(std_err=True).reset_index())
        return dict(
            model=summary_table,
            plot=plot,
            gp_model=model,
            random_effects=re_frame,
            median_odds_ratio=mor,
            cov_pars=cov_pars,
        )
    return summary_table


def glmm_forestplot(
    summary: pl.DataFrame,
    cell_height: int = 36,
    width: int = 900,
    lim: int = 50,
    path: Any = None,
) -> Any:
    """Create a forest plot for GLMM / tidy OR tables (matplotlib).

    .. warning::
        This is an experimental implementation and may change in future versions.

    Args:
        summary: Summary DataFrame from ``glmm_gpboost`` (``oddsratio`` / ``lo`` /
            ``hi``) or a ``Fit.tidy(exponentiate=True)`` frame.
        cell_height: Unused; kept for API compatibility with the old R helper.
        width: Unused; kept for API compatibility.
        lim: Upper x-axis limit for the odds-ratio scale.
        path: Optional PNG path. When omitted, returns a matplotlib ``Figure``.

    Returns:
        ``Path`` when ``path`` is set, otherwise a matplotlib ``Figure``.
    """
    del cell_height, width  # accepted for backward-compatible call sites
    from ..viz.forest import plot_forest

    kwargs: dict[str, Any] = {
        "null_value": 1.0,
        "log_scale": True,
        "xlabel": "Odds ratio",
        "layout": "table",
    }
    if {"oddsratio", "lo", "hi"} <= set(summary.columns):
        kwargs.update(
            term="variable" if "variable" in summary.columns else None,
            estimate="oddsratio",
            conf_low="lo",
            conf_high="hi",
        )
    if lim and lim > 0:
        lo_name = kwargs.get("conf_low")
        if lo_name is None:
            lo_name = "exp_conf_low" if "exp_conf_low" in summary.columns else "conf_low"
        lo_min = float(summary[lo_name].drop_nulls().min())
        kwargs["xlim"] = (max(1e-3, lo_min * 0.8), float(lim))

    return plot_forest(summary, path=path, **kwargs)
