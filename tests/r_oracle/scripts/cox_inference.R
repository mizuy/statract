# Cox inference for tests/r_oracle/test_cox_inference.py: residuals, cox.zph,
# concordance, basehaz, predict, and cluster-robust variance.
# Run from the repository root: Rscript tests/r_oracle/scripts/cox_inference.R
suppressMessages({
  library(survival)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/cox_inference.json"

read_sample <- function(name) {
  d <- read.csv(file.path(data_dir, paste0(name, ".csv")))
  if ("stage" %in% names(d)) d$stage <- factor(d$stage, levels = c("I", "II", "III"))
  d
}

summarize_fit <- function(fit) {
  zph <- cox.zph(fit, transform = "km")
  conc <- concordance(fit)
  bh <- basehaz(fit, centered = FALSE)
  list(
    terms = names(coef(fit)),
    coefficients = unname(coef(fit)),
    covariance = unname(vcov(fit)),
    naive_covariance = unname(if (is.null(fit$naive.var)) vcov(fit) else fit$naive.var),
    log_likelihood = unname(fit$loglik[2]),
    martingale = unname(residuals(fit, type = "martingale")),
    deviance = unname(residuals(fit, type = "deviance")),
    score = unname(as.matrix(residuals(fit, type = "score"))),
    schoenfeld = unname(as.matrix(residuals(fit, type = "schoenfeld"))),
    scaled_schoenfeld = unname(as.matrix(residuals(fit, type = "scaledsch"))),
    dfbeta = unname(as.matrix(residuals(fit, type = "dfbeta"))),
    dfbetas = unname(as.matrix(residuals(fit, type = "dfbetas"))),
    zph_terms = rownames(zph$table),
    zph_chisq = unname(zph$table[, "chisq"]),
    zph_df = unname(zph$table[, "df"]),
    concordance = unname(conc$concordance),
    concordance_se = unname(sqrt(conc$var)),
    basehaz_time = bh$time,
    basehaz_hazard = bh$hazard,
    basehaz_strata = if (is.null(bh$strata)) NULL else as.character(bh$strata),
    expected = unname(predict(fit, type = "expected")),
    lp = unname(predict(fit, type = "lp", reference = "zero"))
  )
}

cases <- list()

a <- read_sample("cox_a")
cases$cox_a_efron <- c(
  list(formula = "Surv(time, status) ~ x + stage", ties = "efron"),
  summarize_fit(coxph(Surv(time, status) ~ x + stage, data = a, ties = "efron"))
)
cases$cox_a_breslow <- c(
  list(formula = "Surv(time, status) ~ x + stage", ties = "breslow"),
  summarize_fit(coxph(Surv(time, status) ~ x + stage, data = a, ties = "breslow"))
)

b <- read_sample("cox_b")
cases$cox_b_strata_weights <- c(
  list(formula = "Surv(time, status) ~ x + stage + strata(site)", ties = "efron", weights = "w"),
  summarize_fit(coxph(Surv(time, status) ~ x + stage + strata(site), data = b, weights = w, ties = "efron"))
)

cc <- read_sample("cox_c")
cases$cox_c_counting <- c(
  list(formula = "Surv(start, stop, status) ~ age + dose", ties = "efron"),
  summarize_fit(coxph(Surv(start, stop, status) ~ age + dose, data = cc, ties = "efron"))
)
cases$cox_c_counting_cluster <- c(
  list(formula = "Surv(start, stop, status) ~ age + dose + cluster(id)", ties = "efron"),
  summarize_fit(coxph(Surv(start, stop, status) ~ age + dose, cluster = id, data = cc, ties = "efron"))
)
cases$cox_c_counting_breslow_cluster <- c(
  list(formula = "Surv(start, stop, status) ~ age + dose + cluster(id)", ties = "breslow"),
  summarize_fit(coxph(Surv(start, stop, status) ~ age + dose, cluster = id, data = cc, ties = "breslow"))
)

fg_data <- read_sample("fg_a")
fg_data$status <- factor(fg_data$status, levels = 0:2)
fg <- finegray(Surv(time, status) ~ ., data = fg_data, etype = "1")
fg_fit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x + stage, data = fg, weights = fgwt)
fine_gray <- list(
  terms = names(coef(fg_fit)),
  coefficients = unname(coef(fg_fit)),
  covariance = unname(vcov(fg_fit)),
  naive_covariance = unname(fg_fit$naive.var),
  log_likelihood = unname(fg_fit$loglik[2]),
  n_rows = nrow(fg)
)

meta <- list(r = R.version.string, survival = as.character(packageVersion("survival")))
write_json(list(meta = meta, cases = cases, fine_gray = fine_gray), out, digits = NA, auto_unbox = TRUE, pretty = FALSE, null = "null")
