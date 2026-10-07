# Poisson and negative binomial GLMMs for tests/r_oracle/test_glmm_count.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/glmm_count.R
# Laplace fits use glmmTMB (exact Hessian). Adaptive Gauss-Hermite uses glmer.
suppressMessages({
  library(glmmTMB)
  library(lme4)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/glmm_count.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

tmb_fit <- function(d, formula, family) {
  m <- glmmTMB(as.formula(formula), data = d, family = family)
  vc <- VarCorr(m)$cond[[1]]
  list(
    terms = names(fixef(m)$cond),
    coefficients = unname(fixef(m)$cond),
    std_errors = unname(sqrt(diag(vcov(m)$cond))),
    log_likelihood = as.numeric(logLik(m)),
    group_covariance = unname(as.matrix(vc)),
    theta = if (family$family == "poisson") NULL else unname(sigma(m))
  )
}

agq_fit <- function(d, formula, n_agq) {
  m <- glmer(as.formula(formula), data = d, family = poisson, nAGQ = n_agq,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
  list(
    terms = names(fixef(m)),
    coefficients = unname(fixef(m)),
    variance = unname(as.numeric(VarCorr(m)[[1]]))
  )
}

intercept <- "y ~ x + arm + offset(log(years)) + (1 | site)"
slope <- "y ~ x + arm + offset(log(years)) + (1 + x | site)"

cases <- list()
for (name in c("glmm_count_a", "glmm_count_b", "glmm_count_c")) {
  d <- read_sample(name)
  formula <- if (name == "glmm_count_b") slope else intercept
  cases[[name]] <- list(
    formula = formula,
    poisson = tmb_fit(d, formula, poisson()),
    negative_binomial = tmb_fit(d, formula, nbinom2()),
    agq = if (name == "glmm_count_b") NULL else agq_fit(d, formula, 9L)
  )
}

meta <- list(
  r = R.version.string,
  glmmTMB = as.character(packageVersion("glmmTMB")),
  lme4 = as.character(packageVersion("lme4"))
)
write_json(list(meta = meta, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
