# Oracle for SurvivalCurve.at().
# Rscript tests/r_oracle/scripts/km_at.R  (writes tests/r_oracle/fixtures/km_at.json)
#
# summary(survfit(...), times = grid, extend = TRUE) run as a black box on the
# rmst_* data. The grid has times before the first time, on tied times,
# between times, and past the last time. The "entry" cases add a delayed
# entry of (row index mod 3) * 0.5. The "aj" cases are Aalen-Johansen curves
# from survfit with a factor status on fg_a (estimate and n_risk only, since
# survfit's standard error uses another variance estimator).
suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

out_path <- file.path("tests", "r_oracle", "fixtures", "km_at.json")
data_dir <- file.path("tests", "r_oracle", "data")

one <- function(data_name, by, entry) {
  d <- read.csv(file.path(data_dir, paste0(data_name, ".csv")))
  d$entry <- ((seq_len(nrow(d)) - 1) %% 3) * 0.5
  if (entry) d <- d[d$time > d$entry, ]
  u <- sort(unique(d$time))
  grid <- sort(unique(c(0, u[1] / 2, u[c(1, 2, 5)], (u[10] + u[11]) / 2, median(u), max(u), max(u) + 1)))
  lhs <- if (entry) "Surv(entry, time, status)" else "Surv(time, status)"
  rhs <- if (by) "arm" else "1"
  fit <- survfit(as.formula(paste(lhs, "~", rhs)), data = d)
  s <- summary(fit, times = grid, extend = TRUE)
  group <- if (is.null(s$strata)) NULL else sub("^arm=", "", as.character(s$strata))
  list(
    data = data_name, by = by, entry = entry, grid = grid,
    group = group, time = s$time, n_risk = s$n.risk,
    estimate = s$surv, std_error = s$std.err,
    conf_low = s$lower, conf_high = s$upper
  )
}

aj <- function(by, entry) {
  d <- read.csv(file.path(data_dir, "fg_a.csv"))
  d$entry <- ((seq_len(nrow(d)) - 1) %% 3) * 0.5
  if (entry) d <- d[d$time > d$entry, ]
  d$state <- factor(d$status, 0:2)
  lhs <- if (entry) "Surv(entry, time, state)" else "Surv(time, state)"
  rhs <- if (by) "stage" else "1"
  fit <- survfit(as.formula(paste(lhs, "~", rhs)), data = d, id = seq_len(nrow(d)))
  grid <- c(0, 0.05, 0.5, 1, 2, 3.75, 5, 8.5, 20)
  s <- summary(fit, times = grid, extend = TRUE)
  group <- if (is.null(s$strata)) NULL else sub("^stage=", "", as.character(s$strata))
  list(
    by = by, entry = entry, grid = grid, group = group, time = s$time,
    n_risk = s$n.risk[, 1], cause1 = s$pstate[, 2], cause2 = s$pstate[, 3]
  )
}

cases <- list()
for (data_name in c("rmst_a", "rmst_b", "rmst_c")) {
  for (by in c(FALSE, TRUE)) {
    for (entry in c(FALSE, TRUE)) {
      key <- paste0(data_name, if (by) "_by" else "", if (entry) "_entry" else "")
      cases[[key]] <- one(data_name, by, entry)
    }
  }
}

aj_cases <- list()
for (by in c(FALSE, TRUE)) {
  for (entry in c(FALSE, TRUE)) {
    aj_cases[[paste0("fg_a", if (by) "_by" else "", if (entry) "_entry" else "")]] <- aj(by, entry)
  }
}

out <- list(versions = list(survival = as.character(packageVersion("survival")), R = R.version.string), cases = cases, aj = aj_cases)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), out_path)
