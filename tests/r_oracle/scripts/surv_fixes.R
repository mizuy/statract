# survival oracle for the 2026-11 survival fixes, for tests/r_oracle/test_surv_fixes.py:
# Kaplan-Meier quantiles with confidence limits, Nelson-Aalen (Fleming-
# Harrington) curves with each confidence transform, and conditional logit
# with weights and offsets.
# Run from the repository root: Rscript tests/r_oracle/scripts/surv_fixes.R
suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

data_dir <- file.path("tests", "r_oracle", "data")
out <- file.path("tests", "r_oracle", "fixtures", "surv_fixes.json")
na_num <- function(x) ifelse(is.na(x), NA_real_, x)

# quantile.survfit on the rmst samples, by arm, for each confidence transform.
quantiles <- list()
for (name in c("rmst_a", "rmst_b", "rmst_c")) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  for (conf in c("log", "log-log", "plain")) {
    fit <- survfit(Surv(time, status) ~ arm, data = d, conf.type = conf)
    q <- quantile(fit, probs = c(0.25, 0.5, 0.75))
    quantiles[[paste(name, conf, sep = "_")]] <- list(
      data = name, confidence = conf, probability = c(0.25, 0.5, 0.75),
      group = sub("^arm=", "", rownames(q$quantile)),
      quantile = lapply(seq_len(nrow(q$quantile)), function(i) na_num(unname(q$quantile[i, ]))),
      conf_low = lapply(seq_len(nrow(q$lower)), function(i) na_num(unname(q$lower[i, ]))),
      conf_high = lapply(seq_len(nrow(q$upper)), function(i) na_num(unname(q$upper[i, ])))
    )
  }
}

# Nelson-Aalen: survfit with stype = 2, ctype = 1 gives exp(-H) and its interval.
nelson_aalen <- list()
for (name in c("rmst_a", "rmst_b")) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  for (conf in c("log", "log-log", "plain")) {
    fit <- survfit(Surv(time, status) ~ 1, data = d, stype = 2, ctype = 1, conf.type = conf)
    nelson_aalen[[paste(name, conf, sep = "_")]] <- list(
      data = name, confidence = conf, time = fit$time, n_risk = fit$n.risk,
      estimate = fit$surv, cumhaz = fit$cumhaz, std_error = fit$std.err,
      conf_low = na_num(fit$lower), conf_high = na_num(fit$upper)
    )
  }
}

# clogit: exact and approximate (Breslow), with an offset, whole-number and
# fractional weights. survival treats non-integer weights with a robust variance.
cl <- read.csv(file.path(data_dir, "clogit_a.csv"))
cl$z <- factor(cl$z, levels = c("lo", "mid", "hi"))
clogit_case <- function(formula, method, weights = NULL) {
  f <- as.formula(formula)
  fit <- if (is.null(weights)) {
    clogit(f, data = cl, method = method)
  } else {
    cl$.w <- cl[[weights]]
    suppressWarnings(clogit(f, data = cl, method = method, weights = .w))
  }
  list(
    formula = formula, method = method, weights = if (is.null(weights)) NA else weights,
    terms = names(coef(fit)), coefficients = unname(coef(fit)),
    std_errors = unname(sqrt(diag(vcov(fit)))), log_likelihood = fit$loglik[2]
  )
}
clogit_cases <- list(
  exact = clogit_case("case ~ x + z + strata(set)", "exact"),
  exact_offset = clogit_case("case ~ x + z + offset(o) + strata(set)", "exact"),
  breslow = clogit_case("case ~ x + z + strata(set)", "approximate"),
  breslow_offset = clogit_case("case ~ x + z + offset(o) + strata(set)", "approximate"),
  breslow_weights = clogit_case("case ~ x + z + strata(set)", "approximate", "w"),
  breslow_fractional_weights = clogit_case("case ~ x + z + strata(set)", "approximate", "wf")
)

meta <- list(r = R.version.string, survival = as.character(packageVersion("survival")))
write_json(list(meta = meta, quantile = quantiles, nelson_aalen = nelson_aalen, clogit = clogit_cases),
           out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null")
