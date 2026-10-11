# Binomial, zero-inflated, and hurdle GLMMs for tests/r_oracle/test_glmm_binary_zero.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/glmm_binary_zero.R
suppressMessages({
  library(glmmTMB)
  library(lme4)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/glmm_binary_zero.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  if ("arm" %in% names(d)) d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

tmb_fit <- function(d, formula, family, zi = ~0) {
  m <- glmmTMB(as.formula(formula), ziformula = zi, data = d, family = family)
  vc <- VarCorr(m)$cond[[1]]
  res <- list(
    terms = names(fixef(m)$cond),
    coefficients = unname(fixef(m)$cond),
    std_errors = unname(sqrt(diag(vcov(m)$cond))),
    log_likelihood = as.numeric(logLik(m)),
    group_covariance = unname(as.matrix(vc)),
    theta = if (family$family %in% c("nbinom2", "truncated_nbinom2")) unname(sigma(m)) else NULL
  )
  if (length(fixef(m)$zi) > 0) {
    res$zero_terms <- names(fixef(m)$zi)
    res$zero_coefficients <- unname(fixef(m)$zi)
    res$zero_std_errors <- unname(sqrt(diag(vcov(m)$zi)))
  }
  res
}

binary <- list()
for (name in c("glmm_binary_a", "glmm_binary_b", "glmm_binary_c")) {
  d <- read_sample(name)
  formula <- if (name == "glmm_binary_b") "y ~ x + arm + (1 + x | site)" else "y ~ x + arm + (1 | site)"
  agq <- NULL
  if (name != "glmm_binary_b") {
    g <- glmer(as.formula(formula), data = d, family = binomial, nAGQ = 9L,
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
    agq <- list(coefficients = unname(fixef(g)), variance = unname(as.numeric(VarCorr(g)[[1]])))
  }
  binary[[name]] <- list(formula = formula, laplace = tmb_fit(d, formula, binomial()), agq = agq)
}

zero <- list()
za <- read_sample("glmm_zero_a")
zb <- read_sample("glmm_zero_b")
f <- "y ~ x + offset(log(years)) + (1 | site)"
zero$poisson_zi_const <- list(sample = "glmm_zero_a", family = "poisson", zero = "inflated", zero_columns = list(),
                              formula = f, fit = tmb_fit(za, f, poisson(), ~1))
zero$poisson_zi_z <- list(sample = "glmm_zero_a", family = "poisson", zero = "inflated", zero_columns = list("z"),
                          formula = f, fit = tmb_fit(za, f, poisson(), ~z))
zero$nb_zi_z <- list(sample = "glmm_zero_b", family = "negative_binomial", zero = "inflated", zero_columns = list("z"),
                     formula = f, fit = tmb_fit(zb, f, nbinom2(), ~z))
zero$poisson_hurdle_z <- list(sample = "glmm_zero_a", family = "poisson", zero = "hurdle", zero_columns = list("z"),
                              formula = f, fit = tmb_fit(za, f, truncated_poisson(), ~z))
zero$nb_hurdle_const <- list(sample = "glmm_zero_b", family = "negative_binomial", zero = "hurdle", zero_columns = list(),
                             formula = f, fit = tmb_fit(zb, f, truncated_nbinom2(), ~1))

meta <- list(
  r = R.version.string,
  glmmTMB = as.character(packageVersion("glmmTMB")),
  lme4 = as.character(packageVersion("lme4"))
)
write_json(list(meta = meta, binary = binary, zero = zero), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
