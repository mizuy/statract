# lmtest and stats::glm oracle for statract.models.linear_tests and the gamma
# GLM likelihood, for tests/r_oracle/test_linear_tests.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/linear_tests.R
#
# resettest, bptest, dwtest (exact = FALSE: the normal approximation), and
# bgtest on an unweighted lm fit; logLik, AIC, BIC, deviance, and
# coefficients of Gamma GLMs with and without weights.
#
# lmtest rebuilds the regression from the model terms and drops the weights
# and offset of an lm fit (resettest on lm(y ~ x + z, weights = w) gives the
# unweighted statistic), so weighted and offset fits have no R reference here.
suppressPackageStartupMessages({
  library(lmtest)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/gam_a.csv")
out <- "tests/r_oracle/fixtures/linear_tests.json"

htest <- function(h) {
  list(statistic = unname(h$statistic), df = unname(as.numeric(h$parameter)), p_value = unname(h$p.value))
}

f <- lm(y ~ x + z, data = d)
lin <- list(
  formula = "y ~ x + z",
  reset = htest(resettest(f, power = 2:3, type = "fitted")),
  reset_2 = htest(resettest(f, power = 2, type = "fitted")),
  bp = htest(bptest(f, studentize = TRUE)),
  bp_raw = htest(bptest(f, studentize = FALSE)),
  dw = htest(dwtest(f, alternative = "greater", exact = FALSE)),
  dw_two = htest(dwtest(f, alternative = "two.sided", exact = FALSE)),
  bg_chi2 = htest(bgtest(f, order = 2, type = "Chisq")),
  bg_f = htest(bgtest(f, order = 2, type = "F"))
)

gamma_case <- function(weights) {
  f <- if (is.na(weights)) {
    glm(cost ~ x + z, family = Gamma(), data = d)
  } else {
    glm(cost ~ x + z, family = Gamma(), data = d, weights = w)
  }
  s <- summary(f)
  list(
    weights = weights, coefficients = unname(coef(f)), std_errors = unname(s$coefficients[, 2]),
    deviance = deviance(f), log_likelihood = as.numeric(logLik(f)), aic = AIC(f), bic = BIC(f),
    dispersion = s$dispersion
  )
}

meta <- list(r = R.version.string, lmtest = as.character(packageVersion("lmtest")))
write_json(list(meta = meta, linear = lin, gamma = list(plain = gamma_case(NA), weighted = gamma_case("w"))),
           out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null")
