# Crossed random intercepts and ar1() GLMMs for tests/r_oracle/test_glmm_crossed.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/glmm_crossed.R
suppressMessages({
  library(glmmTMB)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/glmm_crossed_a.csv")
d$arm <- factor(d$arm, levels = c("control", "low", "high"))
# ar1() runs over the levels of a factor; the test passes the integer column.
d$year <- factor(d$year)
out <- "tests/r_oracle/fixtures/glmm_crossed.json"

tight <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4, rel.tol = 1e-13))

fit <- function(formula, family) {
  m <- glmmTMB(as.formula(formula), data = d, family = family, control = tight)
  vc <- VarCorr(m)$cond
  re <- ranef(m)$cond
  terms <- lapply(names(vc), function(g) {
    v <- vc[[g]]
    corr <- attr(v, "correlation")
    list(
      group = g,
      names = colnames(v),
      covariance = unname(as.matrix(v)),
      correlation = if (ncol(v) > 1) corr[2, 1] else NULL,
      levels = rownames(re[[g]]),
      blup = unname(as.matrix(re[[g]]))
    )
  })
  list(
    formula = formula,
    family = family$family,
    terms = names(fixef(m)$cond),
    coefficients = unname(fixef(m)$cond),
    std_errors = unname(sqrt(diag(vcov(m)$cond))),
    log_likelihood = as.numeric(logLik(m)),
    theta = if (family$family == "nbinom2") unname(sigma(m)) else NULL,
    random = terms
  )
}

crossed <- "y ~ x + arm + offset(log(exposure)) + (1 | id_patient) + (1 | e_examiner)"
ar1 <- "y ~ x + arm + offset(log(exposure)) + (1 | id_patient) + ar1(year + 0 | e_examiner)"
cases <- list(
  nb_crossed = fit(crossed, nbinom2()),
  poisson_crossed = fit(crossed, poisson()),
  binomial_crossed = fit("yb ~ x + arm + (1 | id_patient) + (1 | e_examiner)", binomial()),
  nb_ar1 = fit(ar1, nbinom2()),
  binomial_ar1 = fit("yb ~ x + arm + ar1(year + 0 | e_examiner)", binomial())
)
meta <- list(r = R.version.string, glmmTMB = as.character(packageVersion("glmmTMB")))
write_json(list(meta = meta, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
