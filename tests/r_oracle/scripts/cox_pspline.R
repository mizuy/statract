# Penalized-spline Cox for tests/r_oracle/test_cox_pspline.py:
# coxph(Surv(time, status) ~ pspline(x, df)) with its coefficients, variances,
# df, log-likelihoods, the linear/nonlin tests and termplot values.
# Run from the repository root: Rscript tests/r_oracle/scripts/cox_pspline.R
suppressMessages({
  library(survival)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/cox_pspline.json"

one <- function(data, df) {
  d <- read.csv(file.path(data_dir, paste0(data, ".csv")))
  fit <- coxph(Surv(time, status) ~ pspline(x, df = df), data = d)
  tab <- summary(fit)$coefficients
  tp <- termplot(fit, se = TRUE, plot = FALSE)$x
  list(
    data = data, df_target = df,
    coefficients = unname(coef(fit)),
    variance = unname(fit$var),
    variance_sandwich = unname(fit$var2),
    df = unname(fit$df),
    log_likelihood = unname(fit$loglik[2]),
    log_likelihood_null = unname(fit$loglik[1]),
    aic = AIC(fit),
    linear_coef = unname(tab[1, "coef"]),
    linear_se = unname(tab[1, "se(coef)"]),
    chisq = unname(tab[, "Chisq"]),
    test_df = unname(tab[, "DF"]),
    p_value = unname(tab[, "p"]),
    term_x = tp$x, term_y = tp$y, term_se = tp$se
  )
}

cases <- list(
  cox_a_df3 = one("cox_a", 3),
  cox_a_df4 = one("cox_a", 4),
  cox_b_df3 = one("cox_b", 3)
)
writeLines(toJSON(list(cases = cases), digits = NA, auto_unbox = TRUE, pretty = TRUE), out)
