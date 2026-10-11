# Oracle for restricted_mean_survival.
# Rscript tests/r_oracle/scripts/rmst.R  (writes tests/r_oracle/fixtures/rmst.json)
#
# Three independent sources per case:
#   * "survfit": rmean and se(rmean) from summary(survfit(...), rmean = tau).
#   * "formulas": the RMST, its variance and the contrasts written here in
#     base R from the published formulas, not from any package's code:
#     area under the Kaplan-Meier curve and its Greenwood-type variance
#     (Klein and Moeschberger 2003, section 4.5; Royston and Parmar 2013),
#     the difference and log-scale ratios of Uno et al. (2014), and the
#     default tau rule of the survRM2::rmst2 help page.
#   * "survrm2": survRM2::rmst2 run as a black box, when survRM2 is installed
#     (null otherwise). The script stops if it disagrees with "formulas".
suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

out_path <- file.path("tests", "r_oracle", "fixtures", "rmst.json")
data_dir <- file.path("tests", "r_oracle", "data")
have_rm2 <- requireNamespace("survRM2", quietly = TRUE)

# Kaplan-Meier step function by hand: distinct times, number at risk, events.
km_area <- function(time, status, tau) {
  u <- sort(unique(time))
  u <- u[u <= tau]
  at_risk <- vapply(u, function(s) sum(time >= s), numeric(1))
  events <- vapply(u, function(s) sum(time == s & status == 1), numeric(1))
  surv_after <- cumprod(1 - events / at_risk)
  # Step j spans [lo[j], hi[j]) at height h[j].
  lo <- c(0, u)
  hi <- c(u, tau)
  h <- c(1, surv_after)
  pieces <- h * (hi - lo)
  area <- sum(pieces)
  # Area from u[i] to tau.
  tail_area <- vapply(seq_along(u), function(i) sum(pieces[(i + 1):length(pieces)]), numeric(1))
  ok <- events > 0 & at_risk > events
  v <- sum(tail_area[ok]^2 * events[ok] / (at_risk[ok] * (at_risk[ok] - events[ok])))
  list(rmst = area, var = v)
}

# Largest tau with each group's curve known (survRM2::rmst2 help, Details):
# a group's curve is known past its last time only if it ends with events.
largest_tau <- function(time, status, group) {
  lv <- unique(group)
  last <- vapply(lv, function(g) max(time[group == g]), numeric(1))
  ends_zero <- vapply(lv, function(g) {
    t <- time[group == g]
    all(status[group == g][t == max(t)] == 1)
  }, logical(1))
  earlier <- last < max(last)
  if (all(ends_zero[earlier])) max(last) else min(last)
}

run_case <- function(data_name, arms, tau, level) {
  d <- read.csv(file.path(data_dir, paste0(data_name, ".csv")), stringsAsFactors = FALSE)
  tau_used <- if (is.null(tau)) largest_tau(d$time, d$status, d$arm) else tau
  z <- qnorm(0.5 + level / 2)

  sf <- summary(survfit(Surv(time, status) ~ arm, data = d), rmean = tau_used)$table
  # survfit names strata "arm=<value>"; order them as `arms`.
  rows <- paste0("arm=", arms)
  sf_rmean <- unname(sf[rows, "rmean"])
  sf_se <- unname(sf[rows, "se(rmean)"])

  fits <- lapply(arms, function(a) km_area(d$time[d$arm == a], d$status[d$arm == a], tau_used))
  rmst <- vapply(fits, `[[`, numeric(1), "rmst")
  v <- vapply(fits, `[[`, numeric(1), "var")
  se <- sqrt(v)
  rmtl <- tau_used - rmst
  arms_out <- list(
    rmst = rmst, std_error = se, conf_low = rmst - z * se, conf_high = rmst + z * se,
    rmtl = rmtl, rmtl_conf_low = rmtl - z * se, rmtl_conf_high = rmtl + z * se
  )
  # arms[1] is the reference (arm = 0 in survRM2's coding).
  diff <- rmst[2] - rmst[1]
  diff_se <- sqrt(v[1] + v[2])
  lr <- log(rmst[2] / rmst[1])
  lr_se <- sqrt(v[2] / rmst[2]^2 + v[1] / rmst[1]^2)
  ll <- log(rmtl[2] / rmtl[1])
  ll_se <- sqrt(v[2] / rmtl[2]^2 + v[1] / rmtl[1]^2)
  est <- c(diff, exp(lr), exp(ll))
  stat <- c(diff / diff_se, lr / lr_se, ll / ll_se)
  contrasts_out <- list(
    estimate = est,
    std_error = c(diff_se, lr_se, ll_se),
    conf_low = c(diff - z * diff_se, exp(lr - z * lr_se), exp(ll - z * ll_se)),
    conf_high = c(diff + z * diff_se, exp(lr + z * lr_se), exp(ll + z * ll_se)),
    p_value = 2 * pnorm(-abs(stat))
  )

  stopifnot(isTRUE(all.equal(rmst, sf_rmean, tolerance = 1e-12)))
  stopifnot(isTRUE(all.equal(se, sf_se, tolerance = 1e-12)))

  rm2 <- NULL
  if (have_rm2) {
    arm01 <- as.integer(d$arm == arms[2])
    fit <- if (is.null(tau)) survRM2::rmst2(d$time, d$status, arm01, alpha = 1 - level)
           else survRM2::rmst2(d$time, d$status, arm01, tau = tau, alpha = 1 - level)
    a0 <- unname(fit$RMST.arm0$result)
    a1 <- unname(fit$RMST.arm1$result)
    u <- unname(fit$unadjusted.result)
    rm2 <- list(
      tau = fit$tau,
      arms = list(
        rmst = c(a0[1, 1], a1[1, 1]), std_error = c(a0[1, 2], a1[1, 2]),
        conf_low = c(a0[1, 3], a1[1, 3]), conf_high = c(a0[1, 4], a1[1, 4]),
        rmtl = c(a0[2, 1], a1[2, 1]),
        rmtl_conf_low = c(a0[2, 3], a1[2, 3]), rmtl_conf_high = c(a0[2, 4], a1[2, 4])
      ),
      contrasts = list(estimate = u[, 1], conf_low = u[, 2], conf_high = u[, 3], p_value = u[, 4])
    )
    stopifnot(isTRUE(all.equal(rm2$tau, tau_used, tolerance = 1e-15)))
    for (k in names(rm2$arms)) stopifnot(isTRUE(all.equal(rm2$arms[[k]], arms_out[[k]], tolerance = 1e-12)))
    for (k in names(rm2$contrasts)) stopifnot(isTRUE(all.equal(rm2$contrasts[[k]], contrasts_out[[k]], tolerance = 1e-12)))
  }

  list(
    data = data_name, arms = arms, tau = tau, level = level, tau_used = tau_used,
    survfit = list(rmean = sf_rmean, se = sf_se),
    formulas = list(arms = arms_out, contrasts = contrasts_out),
    survrm2 = rm2
  )
}

cases <- list(
  a_default = run_case("rmst_a", c(0, 1), NULL, 0.95),
  a_tau5 = run_case("rmst_a", c(0, 1), 5, 0.90),
  b_strings = run_case("rmst_b", c("control", "treat"), NULL, 0.95),
  b_tau_on_time = run_case("rmst_b", c("control", "treat"), 4, 0.95),
  c_default = run_case("rmst_c", c(0, 1), NULL, 0.95),
  c_past_shorter_arm = run_case("rmst_c", c(0, 1), 4.5, 0.95)
)

versions <- list(
  survival = as.character(packageVersion("survival")),
  R = R.version.string,
  survRM2 = if (have_rm2) as.character(packageVersion("survRM2")) else NULL
)
out <- list(versions = versions, cases = cases)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), out_path)
