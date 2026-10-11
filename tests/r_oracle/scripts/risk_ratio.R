# Risk ratios for tests/r_oracle/test_risk_ratio.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/risk_ratio.R
#
# Modified Poisson (Zou 2004): glm(family = poisson) on the 0/1 outcome with
# sandwich::vcovHC(type = "HC0"), and vcovCL(cluster = ~site, type = "HC0").
# Log-binomial: glm(family = binomial(link = "log")) with summary.glm's
# covariance and HC0, from glm's default start or from start =.
suppressMessages({
  library(sandwich)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/risk_ratio.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

run_case <- function(case) {
  d <- read_sample(case$data)
  d$.w <- if (is.null(case$weights)) rep(1, nrow(d)) else d[[case$weights]]
  f <- as.formula(case$formula)
  fam <- if (case$method == "poisson") poisson() else binomial(link = "log")
  args <- list(formula = f, family = fam, data = d, weights = d$.w)
  if (!is.null(case$start)) args$start <- case$start
  m <- suppressWarnings(do.call(glm, args))
  out <- list(
    data = case$data, formula = case$formula, method = case$method, weights = case$weights,
    start = case$start, cluster = case$cluster,
    names = names(coef(m)), coefficients = unname(coef(m)),
    vcov_model = unname(vcov(m)), vcov_hc0 = unname(vcovHC(m, type = "HC0")),
    deviance = deviance(m), loglik = as.numeric(logLik(m)), aic = AIC(m), iter = m$iter,
    fitted = unname(fitted(m))
  )
  if (!is.null(case$cluster)) {
    out$vcov_cluster <- unname(vcovCL(m, cluster = d[[case$cluster]], type = "HC0"))
  }
  out
}

cases <- list(
  poisson_a = list(data = "risk_a", formula = "y ~ arm + age + sex", method = "poisson", cluster = "site"),
  poisson_b_weighted = list(data = "risk_b", formula = "y ~ arm + age + sex", method = "poisson",
                            weights = "w", cluster = "site"),
  poisson_c = list(data = "risk_c", formula = "y ~ arm * sex + age", method = "poisson", cluster = "site"),
  logbin_a = list(data = "risk_a", formula = "y ~ arm + age + sex", method = "log-binomial", cluster = "site"),
  logbin_b_weighted = list(data = "risk_b", formula = "y ~ arm + age + sex", method = "log-binomial", weights = "w"),
  logbin_c_start = list(data = "risk_c", formula = "y ~ arm + age + sex", method = "log-binomial",
                        start = c(-0.8, 0, 0, 0))
)

result <- list(
  versions = list(sandwich = as.character(packageVersion("sandwich")), R = R.version.string),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
