# Mixed-effects Cox models for tests/r_oracle/test_cox_mixed.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/cox_mixed.R
#
# coxme is LGPL. It is only run here as a black box; statract's code follows
# dev-notes/design/cox-mixed.md. Each case is fitted twice: once with the
# variances estimated, and once with them fixed (vfixed) at those estimates,
# so that the rest of the fit can be compared tightly even where coxme's
# variance search stops early.
suppressMessages({
  library(survival)
  library(coxme)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/cox_mixed.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

# VarCorr gives a variance per term, or a matrix with variances on the
# diagonal and correlations off it. Return covariance matrices.
covariances <- function(fit) {
  lapply(VarCorr(fit), function(v) {
    if (is.null(dim(v))) return(matrix(unname(v), 1, 1))
    sd <- sqrt(diag(v))
    corr <- v
    diag(corr) <- 1
    unname(diag(sd, length(sd)) %*% corr %*% diag(sd, length(sd)))
  })
}

blups <- function(fit) {
  lapply(ranef(fit), function(b) {
    if (is.null(dim(b))) return(list(levels = names(b), values = matrix(unname(b), ncol = 1)))
    list(levels = rownames(b), values = unname(b))
  })
}

summary_of <- function(fit) {
  list(
    coefficients = unname(fixef(fit)),
    covariance = unname(as.matrix(vcov(fit))),
    covariances = unname(covariances(fit)),
    loglik = unname(fit$loglik),
    df = unname(fit$df),
    blups = unname(blups(fit))
  )
}

# groups: the statract group of each term, in coxme's term order.
case <- function(sample, formula, ours, groups, ties = "efron", weights = NULL, fixed = TRUE, sparse = c(50, 0.02)) {
  d <- read_sample(sample)
  control <- coxme.control(sparse = sparse)
  fit_with <- function(...) {
    if (is.null(weights)) {
      coxme(as.formula(formula), data = d, ties = ties, control = control, ...)
    } else {
      d$.w <- d[[weights]]
      coxme(as.formula(formula), data = d, ties = ties, control = control, weights = .w, ...)
    }
  }
  fit <- fit_with()
  res <- list(
    sample = sample, formula = formula, ours = ours, groups = groups, ties = ties,
    weights = weights, terms = names(fixef(fit)), n_events = fit$n[1], estimated = summary_of(fit)
  )
  if (fixed) {
    vfixed <- unname(lapply(VarCorr(fit), unname))
    res$fixed <- summary_of(fit_with(vfixed = vfixed))
    res$fixed$variances <- unname(lapply(vfixed, as.numeric))
  }
  res
}

exact <- c(1000, 0.02)
cases <- list(
  intercept = case("coxme_a", "Surv(time, status) ~ x + arm + (1 | hosp)",
    "Surv(time, status) ~ x + arm + (1 | hosp)", list("hosp")),
  breslow = case("coxme_a", "Surv(time, status) ~ x + arm + (1 | hosp)",
    "Surv(time, status) ~ x + arm + (1 | hosp)", list("hosp"), ties = "breslow"),
  crossed = case("coxme_a", "Surv(time, status) ~ x + arm + (1 | hosp) + (1 | surgeon)",
    "Surv(time, status) ~ x + arm + (1 | hosp) + (1 | surgeon)", list("hosp", "surgeon")),
  nested = case("coxme_a", "Surv(time, status) ~ x + arm + (1 | hosp/ward)",
    "Surv(time, status) ~ x + arm + (1 | hosp/ward)", list("hosp:ward", "hosp")),
  strata = case("coxme_a", "Surv(time, status) ~ x + arm + strata(sex) + (1 | hosp)",
    "Surv(time, status) ~ x + arm + strata(sex) + (1 | hosp)", list("hosp")),
  # coxme's (x | g) is a slope without an intercept: statract's (0 + x | g).
  slope_separate = case("coxme_a", "Surv(time, status) ~ x + arm + (1 | hosp) + (x | hosp)",
    "Surv(time, status) ~ x + arm + (1 | hosp) + (0 + x | hosp)", list("hosp", "hosp")),
  slope_correlated = case("coxme_a", "Surv(time, status) ~ x + arm + (1 + x | hosp)",
    "Surv(time, status) ~ x + arm + (1 + x | hosp)", list("hosp"), fixed = FALSE),
  weighted = case("coxme_b", "Surv(time, status) ~ x + arm + (1 | hosp)",
    "Surv(time, status) ~ x + arm + (1 | hosp)", list("hosp"), weights = "w", sparse = exact),
  counting = case("coxme_b", "Surv(start, time, status) ~ x + arm + (1 | hosp)",
    "Surv(start, time, status) ~ x + arm + (1 | hosp)", list("hosp"), sparse = exact),
  # 80 hospitals: with the default sparse = c(50, 0.02) coxme drops part of
  # the information matrix. statract matches the exact fit.
  many_levels = case("coxme_b", "Surv(time, status) ~ x + arm + sex + (1 | hosp)",
    "Surv(time, status) ~ x + arm + sex + (1 | hosp)", list("hosp"), sparse = exact),
  many_levels_sparse = case("coxme_b", "Surv(time, status) ~ x + arm + sex + (1 | hosp)",
    "Surv(time, status) ~ x + arm + sex + (1 | hosp)", list("hosp"), fixed = FALSE)
)
meta <- list(
  r = R.version.string,
  survival = as.character(packageVersion("survival")),
  coxme = as.character(packageVersion("coxme"))
)
write_json(list(meta = meta, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
