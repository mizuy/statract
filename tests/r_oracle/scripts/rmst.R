# Restricted mean survival time for tests/r_oracle/test_rmst.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/rmst.R
#
# survRM2 is not installable here (CRAN is blocked), so there are two references:
#
# 1. survival::survfit: summary(survfit(...), rmean = tau)$table gives rmean
#    and se(rmean) for each arm.
# 2. rmst1_reimpl and rmst2_reimpl below are a RE-IMPLEMENTATION of
#    survRM2::rmst1 / rmst2 (unadjusted) written from the published formulas,
#    not the package itself: area under the KM curve to tau, variance
#    sum_{t_i <= tau} A(t_i)^2 d_i / (Y_i (Y_i - d_i)), and the difference,
#    RMST ratio, and RMTL ratio (arm 1 over arm 0) with log-scale intervals.
#
# The script checks that survfit's se(rmean) equals the re-implemented
# variance, so the Greenwood-type sum without an n / (n - 1) factor is what
# survival 3.5 reports.
suppressMessages({
  library(survival)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/rmst.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

rmst1_reimpl <- function(time, status, tau, alpha = 0.05) {
  ft <- survfit(Surv(time, status) ~ 1)
  idx <- ft$time <= tau
  wk.time <- sort(c(ft$time[idx], tau))
  wk.surv <- ft$surv[idx]
  wk.n.risk <- ft$n.risk[idx]
  wk.n.event <- ft$n.event[idx]
  time.diff <- diff(c(0, wk.time))
  areas <- time.diff * c(1, wk.surv)
  rmst <- sum(areas)
  wk.var <- ifelse((wk.n.risk - wk.n.event) == 0, 0, wk.n.event / (wk.n.risk * (wk.n.risk - wk.n.event)))
  wk.var <- c(wk.var, 0)
  rmst.var <- sum(cumsum(rev(areas[-1]))^2 * rev(wk.var)[-1])
  list(rmst = rmst, var = rmst.var, se = sqrt(rmst.var), rmtl = tau - rmst)
}

rmst2_reimpl <- function(time, status, arm, tau, alpha = 0.05) {
  z <- qnorm(1 - alpha / 2)
  w1 <- rmst1_reimpl(time[arm == 1], status[arm == 1], tau, alpha)
  w0 <- rmst1_reimpl(time[arm == 0], status[arm == 0], tau, alpha)
  diff <- w1$rmst - w0$rmst
  diff.se <- sqrt(w1$var + w0$var)
  lr <- log(w1$rmst) - log(w0$rmst)
  lr.se <- sqrt(w1$var / w1$rmst^2 + w0$var / w0$rmst^2)
  ll <- log(w1$rmtl) - log(w0$rmtl)
  ll.se <- sqrt(w1$var / w1$rmtl^2 + w0$var / w0$rmtl^2)
  arms <- list(
    rmst = c(w0$rmst, w1$rmst), std_error = c(w0$se, w1$se),
    conf_low = c(w0$rmst, w1$rmst) - z * c(w0$se, w1$se),
    conf_high = c(w0$rmst, w1$rmst) + z * c(w0$se, w1$se),
    rmtl = c(w0$rmtl, w1$rmtl),
    rmtl_conf_low = c(w0$rmtl, w1$rmtl) - z * c(w0$se, w1$se),
    rmtl_conf_high = c(w0$rmtl, w1$rmtl) + z * c(w0$se, w1$se)
  )
  contrasts <- list(
    estimate = c(diff, exp(lr), exp(ll)),
    conf_low = c(diff - z * diff.se, exp(lr - z * lr.se), exp(ll - z * ll.se)),
    conf_high = c(diff + z * diff.se, exp(lr + z * lr.se), exp(ll + z * ll.se)),
    p_value = 2 * pnorm(-abs(c(diff / diff.se, lr / lr.se, ll / ll.se)))
  )
  list(arms = arms, contrasts = contrasts)
}

run_case <- function(case) {
  d <- read_sample(case$data)
  arm <- as.integer(d$arm == case$arms[2])
  tau <- if (is.null(case$tau)) min(max(d$time[arm == 0]), max(d$time[arm == 1])) else case$tau
  re <- rmst2_reimpl(d$time, d$status, arm, tau, alpha = 1 - case$level)
  sf <- survfit(Surv(time, status) ~ arm, data = data.frame(time = d$time, status = d$status, arm = arm))
  tab <- summary(sf, rmean = tau)$table
  stopifnot(isTRUE(all.equal(unname(tab[, "rmean"]), re$arms$rmst, tolerance = 1e-12)))
  stopifnot(isTRUE(all.equal(unname(tab[, "se(rmean)"]), re$arms$std_error, tolerance = 1e-12)))
  list(
    data = case$data, arms = case$arms, tau = case$tau, level = case$level, tau_used = tau,
    survfit = list(rmean = unname(tab[, "rmean"]), se = unname(tab[, "se(rmean)"])),
    reimpl = re
  )
}

cases <- list(
  a_default = list(data = "rmst_a", arms = c(0, 1), level = 0.95),
  a_tau5 = list(data = "rmst_a", arms = c(0, 1), tau = 5, level = 0.9),
  b_strings = list(data = "rmst_b", arms = c("control", "treat"), level = 0.95),
  b_tau_on_time = list(data = "rmst_b", arms = c("control", "treat"), tau = 4, level = 0.95),
  c_default = list(data = "rmst_c", arms = c(0, 1), level = 0.95),
  c_past_shorter_arm = list(data = "rmst_c", arms = c(0, 1), tau = 4.5, level = 0.95)
)

result <- list(
  versions = list(survival = as.character(packageVersion("survival")), R = R.version.string),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
