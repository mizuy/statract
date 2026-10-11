# Negative binomial GLMs for tests/r_oracle/test_glm_nb.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/glm_nb.R
# The data are the GLMM count samples with the site ignored.
suppressMessages({
  library(MASS)
  library(sandwich)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/glm_nb.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  d$arm <- factor(d$arm, levels = c("control", "low", "high"))
  d
}

cases <- list(
  b_offset = list(data = "glmm_count_b", formula = "y ~ x + arm + offset(log(years))"),
  c_offset = list(data = "glmm_count_c", formula = "y ~ x + arm + offset(log(years))"),
  c_weighted = list(data = "glmm_count_c", formula = "y ~ x * arm", weights = "years"),
  b_no_intercept = list(data = "glmm_count_b", formula = "y ~ 0 + arm + x")
)

fit_case <- function(case) {
  d <- read_sample(case$data)
  args <- list(formula = as.formula(case$formula), data = d)
  if (!is.null(case$weights)) args$weights <- d[[case$weights]]
  m <- do.call(glm.nb, args)
  list(
    data = case$data,
    formula = case$formula,
    weights = case$weights,
    terms = names(coef(m)),
    coefficients = unname(coef(m)),
    std_errors = unname(sqrt(diag(vcov(m)))),
    hc0 = unname(vcovHC(m, type = "HC0")),
    theta = m$theta,
    theta_std_error = m$SE.theta,
    log_likelihood = as.numeric(logLik(m)),
    aic = AIC(m),
    bic = BIC(m),
    deviance = deviance(m),
    df_residual = df.residual(m),
    fitted = unname(fitted(m))[1:5]
  )
}

result <- list(
  meta = list(r = R.version.string, MASS = as.character(packageVersion("MASS"))),
  cases = lapply(cases, fit_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, pretty = TRUE), out)
