# Multiple imputation and Rubin's rules for tests/r_oracle/test_mice_pool.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/mice_pool.R
#
# mice imputes the sample, and the completed data sets go into the fixture.
# The test refits the models on those exact data sets in Python and checks
# that pooling them matches mice::pool and summary(pool(...)).
suppressMessages({
  library(mice)
  library(survival)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/mi_a.csv", na.strings = "")
d$g <- factor(d$g, levels = c("a", "b", "c"))
d$b <- factor(d$b, levels = c(0, 1))
out <- "tests/r_oracle/fixtures/mice_pool.json"

imp <- mice(d, m = 5, maxit = 10, seed = 20261019, printFlag = FALSE)
completed <- lapply(seq_len(imp$m), function(i) {
  x <- complete(imp, i)
  list(x2 = x$x2, g = as.character(x$g), b = as.integer(as.character(x$b)))
})

pooled <- function(fits, exponentiate = FALSE) {
  p <- pool(fits)
  s <- summary(p, conf.int = TRUE, exponentiate = exponentiate)
  list(
    terms = as.character(p$pooled$term),
    estimate = p$pooled$estimate,
    ubar = p$pooled$ubar,
    b = p$pooled$b,
    t = p$pooled$t,
    dfcom = p$pooled$dfcom[1],
    df = p$pooled$df,
    riv = p$pooled$riv,
    lambda = p$pooled$lambda,
    fmi = p$pooled$fmi,
    std_error = s$std.error,
    statistic = s$statistic,
    p_value = s$p.value,
    conf_low = s[["2.5 %"]],
    conf_high = s[["97.5 %"]]
  )
}

with_b <- function(x) { x$b <- as.integer(as.character(x$b)); x }
ols <- lapply(seq_len(imp$m), function(i) lm(y ~ x1 + x2 + g, data = complete(imp, i)))
logit <- lapply(seq_len(imp$m), function(i) glm(b ~ x1 + x2 + y, family = binomial, data = with_b(complete(imp, i))))
cox <- lapply(seq_len(imp$m), function(i) coxph(Surv(time, status) ~ x1 + x2 + b, data = with_b(complete(imp, i))))

meta <- list(r = R.version.string, mice = as.character(packageVersion("mice")))
write_json(
  list(
    meta = meta,
    methods = as.list(imp$method[imp$method != ""]),
    completed = completed,
    ols = pooled(ols),
    logit = pooled(logit),
    cox = pooled(cox)
  ),
  out, digits = NA, auto_unbox = TRUE, pretty = TRUE
)
