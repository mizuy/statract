# Two-way cluster-robust Cox variance for tests/r_oracle/test_cox_two_way.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/cox_two_way.R
#
# The reference is sandwich::vcovCL on coxph. vcovCL fails on a coxph with a
# single coefficient (estfun returns a vector), so that case uses the same
# inclusion-exclusion written out by hand. The script checks that the hand
# version equals vcovCL wherever vcovCL runs.
suppressMessages({
  library(survival)
  library(sandwich)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/cox_two_way.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

# Inclusion-exclusion over every combination of cluster columns, as meatCL.
by_hand <- function(fit, clusters, type = "HC0", cadjust = TRUE) {
  ef <- as.matrix(residuals(fit, type = "score"))
  if (!is.null(fit$weights)) ef <- ef * fit$weights
  n <- nrow(ef)
  k <- ncol(ef)
  if (!is.null(fit$na.action)) clusters <- clusters[-fit$na.action, , drop = FALSE]
  p <- ncol(clusters)
  meat <- matrix(0, k, k)
  for (size in seq_len(p)) {
    for (cols in combn(p, size, simplify = FALSE)) {
      label <- do.call(paste, c(unname(as.list(clusters[, cols, drop = FALSE])), sep = "_"))
      g <- length(unique(label))
      summed <- rowsum(ef, label)
      adj <- if (cadjust) g / (g - 1) else 1
      meat <- meat + (-1)^(size + 1) * adj * crossprod(summed) / n
    }
  }
  if (type == "HC1") meat <- (n - 1) / (n - k) * meat
  bread <- n * if (is.null(fit$naive.var)) fit$var else fit$naive.var
  bread %*% meat %*% bread / n
}

case <- function(sample, formula, ties, weights = NULL, cluster = c("id_patient", "e_examiner"),
                 type = "HC0", cadjust = TRUE) {
  d <- read_sample(sample)
  fit <- if (is.null(weights)) {
    coxph(as.formula(formula), data = d, ties = ties)
  } else {
    d$.w <- d[[weights]]
    coxph(as.formula(formula), data = d, ties = ties, weights = .w)
  }
  v <- by_hand(fit, d[, cluster, drop = FALSE], type = type, cadjust = cadjust)
  if (length(coef(fit)) > 1) {
    cl <- as.formula(paste("~", paste(cluster, collapse = " + ")))
    ref <- vcovCL(fit, cluster = cl, type = type, cadjust = cadjust)
    stopifnot(isTRUE(all.equal(unname(ref), unname(v), tolerance = 1e-12)))
  }
  list(
    sample = sample, formula = formula, ties = ties, weights = weights, cluster = as.list(cluster),
    type = type, adjust = cadjust, vcovcl = length(coef(fit)) > 1,
    terms = names(coef(fit)), coefficients = unname(coef(fit)),
    covariance = unname(v), std_errors = unname(sqrt(diag(v)))
  )
}

f3 <- "Surv(time, status) ~ x + arm"
cases <- list(
  two_way_efron = case("cox2way_a", f3, "efron"),
  two_way_hc1 = case("cox2way_a", f3, "efron", type = "HC1"),
  two_way_no_adjust = case("cox2way_a", f3, "efron", cadjust = FALSE),
  one_way_patient = case("cox2way_a", f3, "efron", cluster = "id_patient"),
  single_coefficient = case("cox2way_a", "Surv(time, status) ~ x", "efron"),
  single_coefficient_breslow = case("cox2way_b", "Surv(time, status) ~ x", "breslow"),
  weighted_strata_breslow = case("cox2way_b", "Surv(time, status) ~ x + arm + strata(site)", "breslow", weights = "w"),
  three_way = case("cox2way_b", f3, "efron", cluster = c("id_patient", "e_examiner", "site"))
)

meta <- list(r = R.version.string, survival = as.character(packageVersion("survival")),
             sandwich = as.character(packageVersion("sandwich")))
write_json(list(meta = meta, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
