# Binomial glmer (Laplace and adaptive quadrature) for tests/r_oracle/test_glmm_binary_glmer.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/glmm_binary_glmer.R
#
# Covers what downstream code reads from glmer: fixed effects and SE, the
# cluster variance and its median odds ratio, the log-likelihood, and
# population-level predictions (re.form = NA) at reference rows.
#
# glmer's default inner tolerance (tolPwrss = 1e-7) leaves its Laplace
# objective about 1e-4 off on the log-likelihood scale. Each Laplace case is
# fitted twice: with the default control, as downstream code runs it, and
# with tolPwrss = 1e-13, which is the exact Laplace optimum.
suppressMessages({
  library(lme4)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/glmm_binary_glmer.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

reference <- data.frame(
  x = c(0, 0, 0, 1, -1),
  arm = factor(c("control", "low", "high", "control", "high"), levels = c("control", "low", "high"))
)

fit_glmer <- function(d, formula, nagq, tight = TRUE) {
  control <- if (tight) {
    glmerControl(optimizer = "bobyqa", tolPwrss = 1e-13, optCtrl = list(maxfun = 1e5, rhobeg = 1e-2, rhoend = 1e-10))
  } else {
    glmerControl()
  }
  g <- glmer(as.formula(formula), data = d, family = binomial, nAGQ = nagq, control = control)
  vc <- as.matrix(VarCorr(g)[[1]])
  res <- list(
    nagq = nagq,
    tight = tight,
    terms = names(fixef(g)),
    coefficients = unname(fixef(g)),
    std_errors = unname(sqrt(diag(as.matrix(vcov(g))))),
    log_likelihood = as.numeric(logLik(g)),
    group_covariance = unname(vc),
    reference_link = unname(predict(g, newdata = reference, re.form = NA, type = "link")),
    reference_response = unname(predict(g, newdata = reference, re.form = NA, type = "response"))
  )
  if (nrow(vc) == 1) res$median_odds_ratio <- exp(sqrt(2 * vc[1, 1]) * qnorm(0.75))
  res
}

cases <- list()
for (name in c("glmm_binary_a", "glmm_binary_c")) {
  d <- read_sample(name)
  f <- "y ~ x + arm + (1 | site)"
  cases[[paste0(name, "_laplace")]] <- c(list(sample = name, formula = f), fit_glmer(d, f, 1L))
  cases[[paste0(name, "_laplace_default")]] <- c(list(sample = name, formula = f), fit_glmer(d, f, 1L, tight = FALSE))
  cases[[paste0(name, "_agq25")]] <- c(list(sample = name, formula = f), fit_glmer(d, f, 25L))
}
d <- read_sample("glmm_binary_b")
f <- "y ~ x + arm + (1 + x | site)"
cases[["glmm_binary_b_laplace"]] <- c(list(sample = "glmm_binary_b", formula = f), fit_glmer(d, f, 1L))
cases[["glmm_binary_b_laplace_default"]] <- c(list(sample = "glmm_binary_b", formula = f), fit_glmer(d, f, 1L, tight = FALSE))

meta <- list(r = R.version.string, lme4 = as.character(packageVersion("lme4")))
write_json(list(meta = meta, reference = reference, cases = cases), out,
           digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
