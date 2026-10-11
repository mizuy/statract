# Multinomial logistic regression for tests/r_oracle/test_multinom.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/multinom.R
#
# nnet::multinom stops BFGS at reltol 1e-8. Each case is fitted twice: with
# the defaults, and with reltol = 1e-12, abstol = 1e-14, maxit = 1000 so the
# estimates sit at the optimum. The standard errors are summary.multinom's,
# from vcov.multinom (the inverse of the exact Hessian at the estimates).
suppressMessages({
  library(nnet)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/multinom.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

pack <- function(m) {
  s <- summary(m)
  co <- s$coefficients
  se <- s$standard.errors
  if (is.null(dim(co))) {
    co <- matrix(co, nrow = 1, dimnames = list(m$lev[2], names(co)))
    se <- matrix(se, nrow = 1)
  }
  list(
    coefficients = unname(co), std_errors = unname(se), names = colnames(co), levels = m$lev,
    vcov = unname(vcov(m)), deviance = deviance(m), aic = AIC(m),
    loglik = as.numeric(logLik(m)), edf = m$edf, fitted = unname(as.matrix(fitted(m)))
  )
}

run_case <- function(case) {
  d <- read_sample(case$data)
  d$y <- factor(d$y, levels = case$levels)
  d$.w <- if (is.null(case$weights)) rep(1, nrow(d)) else d[[case$weights]]
  f <- as.formula(case$formula)
  default <- multinom(f, data = d, weights = .w, trace = FALSE)
  tight <- multinom(f, data = d, weights = .w, trace = FALSE, reltol = 1e-12, abstol = 1e-14, maxit = 1000)
  list(
    data = case$data, formula = case$formula, weights = case$weights, levels = case$levels,
    tight = pack(tight), default = pack(default)
  )
}

cases <- list(
  three = list(data = "multinom_a", formula = "y ~ x + age + sex", levels = c("A", "B", "C")),
  four_weighted = list(data = "multinom_b", formula = "y ~ x * sex + age", levels = c("ctrl", "low", "mid", "high"),
                       weights = "w"),
  two = list(data = "multinom_c", formula = "y ~ x + age + sex", levels = c("no", "yes"))
)

result <- list(
  versions = list(nnet = as.character(packageVersion("nnet")), R = R.version.string),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
