# Edge-case oracle for tests/r_oracle/test_edge_cases.py: small hand-made data
# shapes (tests/r_oracle/data/edge_*.csv) that the larger samples lack. Groups
# of one, all events at one time, a stratum with no events, delayed entry that
# equals other subjects' event times, a cause that never occurs in one arm,
# singleton groups and a zero group variance, and an ordinal category with one row.
# Run from the repository root: Rscript tests/r_oracle/scripts/edge_cases.R
suppressPackageStartupMessages({
  library(survival)
  library(cmprsk)
  library(lme4)
  library(MASS)
  library(jsonlite)
})

data_dir <- file.path("tests", "r_oracle", "data")
out <- file.path("tests", "r_oracle", "fixtures", "edge_cases.json")
read <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE)
na_num <- function(x) ifelse(is.na(x), NA_real_, x)

km_table <- function(fit) {
  s <- summary(fit, censored = TRUE)
  strata <- if (is.null(s$strata)) rep("all", length(s$time)) else sub("^[^=]*=", "", as.character(s$strata))
  list(
    group = strata, time = s$time, n_risk = s$n.risk, n_event = s$n.event,
    estimate = s$surv, std_error = na_num(s$std.err),
    conf_low = na_num(s$lower), conf_high = na_num(s$upper)
  )
}

cox_case <- function(formula, data, ties) {
  fit <- coxph(as.formula(formula), data = data, ties = ties)
  list(
    formula = formula, ties = ties, terms = names(coef(fit)),
    coefficients = unname(coef(fit)), std_error = unname(sqrt(diag(vcov(fit)))),
    loglik = unname(fit$loglik), score = unname(fit$score), n_event = fit$nevent
  )
}

result <- list()

# Kaplan-Meier with log intervals, and medians.
km <- read("edge_km")
fit <- survfit(Surv(time, status) ~ arm, data = km)
q <- quantile(fit, probs = 0.5)
result$km <- c(
  km_table(fit),
  list(median_group = sub("^arm=", "", names(q$quantile)), median = na_num(unname(q$quantile)),
       median_low = na_num(unname(q$lower)), median_high = na_num(unname(q$upper)))
)

# Cox with every event at one time.
ties <- read("edge_cox_ties")
result$cox_ties <- lapply(c("efron", "breslow"), function(t) cox_case("Surv(time, status) ~ x + g", ties, t))

# Cox with a stratum without events and a stratum of one subject.
strata <- read("edge_cox_strata")
result$cox_strata <- lapply(c("efron", "breslow"), function(t) cox_case("Surv(time, status) ~ x + strata(stratum)", strata, t))
result$km_strata <- km_table(survfit(Surv(time, status) ~ stratum, data = strata))

# Delayed entry.
entry <- read("edge_entry")
result$cox_entry <- lapply(c("efron", "breslow"), function(t) cox_case("Surv(entry, time, status) ~ x + arm", entry, t))
result$km_entry <- km_table(survfit(Surv(entry, time, status) ~ arm, data = entry))

# Log-rank: a group of one; a stratum with only one arm.
lr <- read("edge_logrank")
logrank <- function(formula) {
  sd <- survdiff(as.formula(formula), data = lr)
  list(formula = formula, statistic = sd$chisq, df = length(sd$n) - 1,
       group = sub("^arm=", "", names(sd$n)),
       observed = unname(if (is.matrix(sd$obs)) rowSums(sd$obs) else sd$obs),
       expected = unname(if (is.matrix(sd$exp)) rowSums(sd$exp) else sd$exp))
}
result$logrank <- list(logrank("Surv(time, status) ~ arm"), logrank("Surv(time, status) ~ arm + strata(site)"))

# Competing risks.
cr <- read("edge_cr")
ci <- cuminc(cr$time, cr$cause, cr$arm, cencode = 0)
curves <- list()
for (name in setdiff(names(ci), "Tests")) {
  parts <- strsplit(name, " ")[[1]]
  curves[[length(curves) + 1]] <- list(group = parts[1], cause = parts[2], time = ci[[name]]$time,
                                       estimate = ci[[name]]$est, variance = ci[[name]]$var)
}
tests <- ci$Tests
result$cuminc <- list(curves = curves, test_cause = rownames(tests), test_statistic = unname(tests[, "stat"]),
                      test_p = unname(tests[, "pv"]), test_df = unname(tests[, "df"]))
cov <- model.matrix(~ x + arm, data = cr)[, -1]
crr_fit <- crr(cr$time, cr$cause, cov, failcode = 1, cencode = 0)
result$crr <- list(terms = colnames(cov), coefficients = unname(crr_fit$coef),
                   std_error = unname(sqrt(diag(crr_fit$var))), loglik = crr_fit$loglik)

# Mixed models: singleton groups. y has a group effect only by chance; y0 is
# centred within groups, so the group variance is on the boundary (0).
mx <- read("edge_mixed")
mixed_case <- function(formula, reml) {
  fit <- lmer(as.formula(formula), data = mx, REML = reml)
  vc <- as.data.frame(VarCorr(fit))
  list(formula = formula, reml = reml, terms = names(fixef(fit)), coefficients = unname(fixef(fit)),
       std_error = unname(sqrt(diag(as.matrix(vcov(fit))))), group_variance = vc$vcov[1],
       residual_variance = vc$vcov[2], loglik = as.numeric(logLik(fit)), singular = isSingular(fit))
}
result$lmer <- list(mixed_case("y ~ x + arm + (1 | site)", TRUE), mixed_case("y ~ x + arm + (1 | site)", FALSE),
                    mixed_case("y0 ~ x + arm + (1 | site)", TRUE), mixed_case("y0 ~ x + arm + (1 | site)", FALSE))
gfit <- glmer(yb ~ x + (1 | site), data = mx, family = binomial,
              control = glmerControl(tolPwrss = 1e-13))
result$glmer <- list(formula = "yb ~ x + (1 | site)", terms = names(fixef(gfit)), coefficients = unname(fixef(gfit)),
                     std_error = unname(sqrt(diag(as.matrix(vcov(gfit))))),
                     group_variance = as.data.frame(VarCorr(gfit))$vcov[1], loglik = as.numeric(logLik(gfit)),
                     singular = isSingular(gfit))

# Firth with complete separation (n = 12) is the case firth_edge_tiny in firth.R.

# Proportional odds with a one-row middle category.
od <- read("edge_ordinal")
od$y <- factor(od$y, levels = c("low", "mid", "high"), ordered = TRUE)
pf <- polr(y ~ x, data = od, Hess = TRUE, control = list(reltol = 1e-14, maxit = 2000))
v <- vcov(pf)
result$polr <- list(formula = "y ~ x", coefficients = unname(coef(pf)), zeta = unname(pf$zeta),
                    std_error = unname(sqrt(diag(v))), loglik = as.numeric(logLik(pf)))

result$versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                        cmprsk = as.character(packageVersion("cmprsk")), lme4 = as.character(packageVersion("lme4")),
                        MASS = as.character(packageVersion("MASS")))
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), out)
