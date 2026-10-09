# Oracle for statract.models.validation: rms::validate.lrm, calibrate (lrm), validate.cph and
# calibrate.cph (cmethod = "KM").
#
#   Rscript tests/r_oracle/scripts/rms.R
#
# writes tests/r_oracle/fixtures/rms.json. predab.resample with method="boot" draws
# one sample(n, replace=TRUE) per repetition and nothing else, so the bootstrap rows
# are rebuilt here with the same seed. The script checks them against the training
# subscripts that predab.resample prints with debug=TRUE before it writes anything.

suppressPackageStartupMessages({
  library(rms)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
out_dir <- normalizePath(file.path(dirname(script), "..", "fixtures"), mustWork = FALSE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

boot_rows <- function(seed, n, B) {
  set.seed(seed)
  t(vapply(seq_len(B), function(i) sample(n, replace = TRUE), integer(n)))
}

# The training subscripts that predab.resample prints with debug=TRUE.
debug_rows <- function(expr_fun, B, n) {
  out <- capture.output(invisible(expr_fun()))
  starts <- grep("Subscripts of training sample", out)
  ends <- grep("Subscripts of test sample", out)
  stopifnot(length(starts) == B)
  t(vapply(seq_len(B), function(i) {
    lines <- out[(starts[i] + 1):(ends[i] - 1)]
    lines <- sub("^ *\\[[0-9]+\\]", "", lines)
    v <- as.integer(unlist(strsplit(trimws(paste(lines, collapse = " ")), " +")))
    stopifnot(length(v) == n)
    v
  }, integer(n)))
}

check_rows <- function(seed, B, n, call) {
  idx <- boot_rows(seed, n, B)
  set.seed(seed)
  seen <- debug_rows(call, B, n)
  stopifnot(identical(idx, seen))
  idx
}

as_table <- function(z) {
  m <- unclass(z)
  rows <- rownames(m)
  out <- list(index = rows)
  for (col in colnames(m)) out[[col]] <- unname(as.numeric(m[, col]))
  out
}

# ---- logistic samples ------------------------------------------------------

logistic_samples <- function() {
  set.seed(101)
  n <- 200
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- rbinom(n, 1, plogis(-0.2 + 0.9 * x1 - 0.6 * x2))
  s1 <- list(name = "balanced", data = data.frame(y = y, x1 = x1, x2 = x2),
             formula = "y ~ x1 + x2", B = 30L, seed = 11L)

  set.seed(202)
  n <- 300
  x1 <- rnorm(n)
  grp <- sample(c("a", "b", "c"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  x3 <- sample(0:4, n, replace = TRUE)
  lp <- -2.2 + 0.7 * x1 + ifelse(grp == "b", 0.5, ifelse(grp == "c", -0.4, 0)) + 0.25 * x3
  y <- rbinom(n, 1, plogis(lp))
  s2 <- list(name = "unbalanced_factor",
             data = data.frame(y = y, x1 = x1, grp = grp, x3 = x3),
             formula = "y ~ x1 + grp + x3", B = 30L, seed = 22L)

  set.seed(303)
  n <- 40
  x1 <- round(rnorm(n), 1)
  y <- rbinom(n, 1, plogis(0.3 + 1.2 * x1))
  s3 <- list(name = "small", data = data.frame(y = y, x1 = x1),
             formula = "y ~ x1", B = 20L, seed = 33L)
  list(s1, s2, s3)
}

validate_lrm_case <- function(s) {
  d <- s$data
  f <- lrm(as.formula(s$formula), data = d, x = TRUE, y = TRUE)
  n <- nrow(d)
  idx <- check_rows(s$seed, s$B, n, function() rms::validate(f, B = s$B, debug = TRUE))
  set.seed(s$seed)
  v <- rms::validate(f, B = s$B)
  list(name = s$name, formula = s$formula, B = s$B, seed = s$seed, data = d,
       indices = idx - 1L, coefficients = unname(coef(f)), result = as_table(v))
}

calibrate_lrm_case <- function(s) {
  d <- s$data
  f <- lrm(as.formula(s$formula), data = d, x = TRUE, y = TRUE)
  n <- nrow(d)
  idx <- check_rows(s$seed, s$B, n, function() rms::calibrate(f, B = s$B, debug = TRUE))
  set.seed(s$seed)
  cal <- rms::calibrate(f, B = s$B)
  m <- unclass(cal)
  at <- attributes(cal)
  ok <- !is.na(m[, "predy"] + m[, "calibrated.corrected"])
  err <- at$predicted - approx(m[ok, "predy"], m[ok, "calibrated.corrected"],
                               xout = at$predicted, ties = mean)$y
  list(name = s$name, formula = s$formula, B = s$B, seed = s$seed, data = d,
       indices = idx - 1L,
       predy = unname(m[, "predy"]),
       calibrated_orig = unname(m[, "calibrated.orig"]),
       calibrated_corrected = unname(m[, "calibrated.corrected"]),
       optimism = unname(m[, "optimism"]),
       n = unname(m[, "n"]),
       mean_absolute_error = mean(abs(err), na.rm = TRUE),
       mean_squared_error = mean(err^2, na.rm = TRUE),
       quantile_90 = unname(quantile(abs(err), 0.9, na.rm = TRUE)))
}

# ---- Cox samples -----------------------------------------------------------

cox_samples <- function() {
  set.seed(404)
  n <- 150
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  t <- rexp(n, 0.1 * exp(0.7 * x1 - 0.4 * x2))
  c <- rexp(n, 0.05)
  s1 <- list(name = "continuous",
             data = data.frame(time = pmin(t, c), status = as.integer(t <= c), x1 = x1, x2 = x2),
             formula = "Surv(time, status) ~ x1 + x2", B = 30L, seed = 44L)

  set.seed(505)
  n <- 200
  x1 <- rnorm(n)
  grp <- sample(c("a", "b", "c"), n, replace = TRUE)
  t <- rexp(n, 0.2 * exp(0.5 * x1 + ifelse(grp == "b", 0.6, ifelse(grp == "c", -0.3, 0))))
  c <- runif(n, 2, 12)
  s2 <- list(name = "ties_factor",
             data = data.frame(time = ceiling(pmin(t, c)), status = as.integer(t <= c),
                               x1 = x1, grp = grp),
             formula = "Surv(time, status) ~ x1 + grp", B = 30L, seed = 55L)

  set.seed(606)
  n <- 40
  x1 <- rnorm(n)
  t <- rexp(n, exp(0.8 * x1))
  c <- rexp(n, 0.5)
  s3 <- list(name = "small",
             data = data.frame(time = pmin(t, c), status = as.integer(t <= c), x1 = x1),
             formula = "Surv(time, status) ~ x1", B = 20L, seed = 66L)
  list(s1, s2, s3)
}

validate_cph_case <- function(s) {
  d <- s$data
  f <- cph(as.formula(s$formula), data = d, x = TRUE, y = TRUE)
  n <- nrow(d)
  idx <- check_rows(s$seed, s$B, n, function() rms::validate(f, B = s$B, debug = TRUE))
  set.seed(s$seed)
  v <- rms::validate(f, B = s$B)
  list(name = s$name, formula = s$formula, B = s$B, seed = s$seed, data = d,
       indices = idx - 1L, coefficients = unname(coef(f)), result = as_table(v))
}

calibrate_cph_case <- function(s, u, m) {
  d <- s$data
  f <- cph(as.formula(s$formula), data = d, x = TRUE, y = TRUE, surv = TRUE, time.inc = u)
  n <- nrow(d)
  idx <- check_rows(s$seed, s$B, n, function()
    rms::calibrate(f, cmethod = "KM", u = u, m = m, B = s$B, debug = TRUE))
  set.seed(s$seed)
  cal <- rms::calibrate(f, cmethod = "KM", u = u, m = m, B = s$B)
  tab <- unclass(cal)
  out <- list(name = s$name, formula = s$formula, B = s$B, seed = s$seed, u = u, m = m,
              data = d, indices = idx - 1L, predicted = attr(cal, "predicted"))
  for (col in colnames(tab)) out[[gsub(".", "_", col, fixed = TRUE)]] <- unname(as.numeric(tab[, col]))
  out
}

# Apparent Somers' Dxy and C for logistic and survival predictions.
concordance_cases <- function() {
  lr <- logistic_samples()[[2]]
  f <- lrm(as.formula(lr$formula), data = lr$data, x = TRUE, y = TRUE)
  s2 <- somers2(f$linear.predictors, lr$data$y)
  cx <- cox_samples()[[2]]
  g <- cph(as.formula(cx$formula), data = cx$data, x = TRUE, y = TRUE)
  dx <- rms:::dxy.cens(g$linear.predictors, g$y, type = "hazard")
  list(logistic = list(sample = lr$name, C = unname(s2["C"]), Dxy = unname(s2["Dxy"])),
       survival = list(sample = cx$name, Dxy = unname(dx["Dxy"])))
}

# stats::lowess (clowess) with its defaults and with iter = 0, on data with ties.
lowess_cases <- function() {
  set.seed(707)
  x <- round(runif(120, 0, 10), 1)
  y <- sin(x) + rnorm(120, sd = 0.4) + ifelse(runif(120) < 0.05, 4, 0)
  a <- lowess(x, y)
  b <- lowess(x, y, iter = 0)
  c <- lowess(x, y, f = 0.2, delta = 0)
  list(x = x, y = y, sorted_x = a$x, iter3 = a$y, iter0 = b$y, f02_delta0 = c$y)
}

lrs <- logistic_samples()
cxs <- cox_samples()
out <- list(
  versions = list(R = R.version.string,
                  rms = as.character(packageVersion("rms")),
                  Hmisc = as.character(packageVersion("Hmisc")),
                  survival = as.character(packageVersion("survival"))),
  validate_lrm = lapply(lrs, validate_lrm_case),
  calibrate_lrm = lapply(lrs, calibrate_lrm_case),
  validate_cph = lapply(cxs, validate_cph_case),
  calibrate_cph = list(calibrate_cph_case(cxs[[1]], u = 5, m = 50),
                       calibrate_cph_case(cxs[[2]], u = 3, m = 40),
                       calibrate_cph_case(cxs[[3]], u = 0.5, m = 20)),
  concordance = concordance_cases(),
  lowess = lowess_cases()
)
path <- file.path(out_dir, "rms.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "columns", matrix = "rowmajor",
                  na = "null"), path)
cat("wrote", path, "\n")
