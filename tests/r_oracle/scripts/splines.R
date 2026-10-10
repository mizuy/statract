# splines::ns / bs and rms::rcs in formulas, for tests/r_oracle/test_splines.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/splines.R
suppressMessages({
  library(splines)
  library(survival)
  library(rms)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/splines.json"

# --- data -------------------------------------------------------------------
set.seed(20261010)
n <- 400
a <- data.frame(
  age = round(runif(n, 30, 85), 1),
  sex = sample(c("F", "M"), n, replace = TRUE),
  bmi = round(rnorm(n, 23, 3.5), 1)
)
lp <- -1 + 0.002 * (a$age - 55)^2 + 0.4 * (a$sex == "M") - 0.05 * (a$bmi - 23)
a$y <- rbinom(n, 1, plogis(lp))
a$time <- round(rexp(n, 0.05 * exp(0.03 * (a$age - 55) + 0.0008 * (a$age - 55)^2)), 2)
a$status <- rbinom(n, 1, 0.75)
a$age[c(7, 50, 222)] <- NA
a$bmi[c(9, 300)] <- NA

b <- data.frame(x = round(rgamma(80, 3, 0.2), 0), z = rnorm(80))
b$y <- b$z + sin(b$x / 5) + rnorm(80, 0, 0.3)

c <- data.frame(x = ifelse(runif(300) < 0.15, 0, round(rexp(300, 0.1), 1)), z = rnorm(300))
c$y <- log1p(c$x) + c$z + rnorm(300)

write.csv(a, file.path(data_dir, "splines_a.csv"), row.names = FALSE, na = "")
write.csv(b, file.path(data_dir, "splines_b.csv"), row.names = FALSE, na = "")
write.csv(c, file.path(data_dir, "splines_c.csv"), row.names = FALSE, na = "")
samples <- list(splines_a = a, splines_b = b, splines_c = c)

# --- model matrices, fitted and on new rows ---------------------------------
matrices <- list(
  list(data = "splines_a", formula = "y ~ ns(age, df = 3)", new = "age", at = c(20, 30, 57.5, 85, 95)),
  list(data = "splines_a", formula = "y ~ 0 + ns(age, 4, intercept = TRUE) + bmi", new = "age", at = c(25, 60, 90)),
  list(data = "splines_a", formula = "y ~ ns(age, knots = c(45, 65), Boundary.knots = c(35, 80)) * sex",
       new = "age", at = c(30, 50, 82)),
  list(data = "splines_a", formula = "y ~ bs(age, df = 5)", new = "age", at = c(20, 40, 85, 99)),
  list(data = "splines_a", formula = "y ~ bs(bmi, knots = c(20, 25), degree = 2) + age", new = "bmi", at = c(10, 22, 40)),
  list(data = "splines_a", formula = "y ~ rcs(age, 4) + sex", new = "age", at = c(20, 50, 99)),
  list(data = "splines_a", formula = "y ~ rcs(age) + rcs(bmi, 3)", new = "age", at = c(31, 60)),
  list(data = "splines_a", formula = "y ~ rcs(age, c(40, 55, 70))", new = "age", at = c(35, 72)),
  list(data = "splines_a", formula = "y ~ rcs(log(bmi), 4)", new = "bmi", at = c(15, 30)),
  list(data = "splines_b", formula = "y ~ rcs(x, 5) + z", new = "x", at = c(0, 10, 60)),
  list(data = "splines_b", formula = "y ~ ns(x, 3) + z", new = "x", at = c(0, 10, 60)),
  list(data = "splines_c", formula = "y ~ rcs(x, 4) + z", new = "x", at = c(0, 5, 40)),
  list(data = "splines_c", formula = "y ~ rcs(x, 6)", new = "x", at = c(0, 5, 40))
)

matrix_cases <- lapply(matrices, function(case) {
  d <- samples[[case$data]]
  fit <- lm(as.formula(case$formula), data = d)
  x <- model.matrix(fit)
  newdata <- d[rep(1, length(case$at)), , drop = FALSE]
  newdata[[case$new]] <- case$at
  for (col in names(newdata)) if (anyNA(newdata[[col]])) newdata[[col]] <- d[[col]][which(!is.na(d[[col]]))[1]]
  tt <- delete.response(terms(fit))
  new_x <- model.matrix(tt, model.frame(tt, newdata, xlev = fit$xlevels))
  list(
    data = case$data, formula = case$formula, new = case$new, at = case$at,
    names = colnames(x), rows = as.integer(rownames(x)) - 1L,
    x = unname(x), new_x = unname(new_x), coef = unname(coef(fit))
  )
})

# --- fits, Wald tests, and contrasts ----------------------------------------
dd <- datadist(a)
options(datadist = "dd")
grid <- c(35, 45, 60, 70, 80)

glm_fit <- glm(y ~ rcs(age, 4) + sex + ns(bmi, 3), family = binomial, data = a)
lrm_fit <- lrm(y ~ rcs(age, 4) + sex + bmi, data = a, eps = 1e-12, maxit = 50)
lrm_glm <- glm(y ~ rcs(age, 4) + sex + bmi, family = binomial, data = a)
lrm_anova <- anova(lrm_fit)
lrm_con <- contrast(lrm_fit, list(age = grid), list(age = 50))

cox_fit <- coxph(Surv(time, status) ~ rcs(age, 5) + sex + ns(bmi, df = 2), data = a)
cph_fit <- cph(Surv(time, status) ~ rcs(age, 4) + sex, data = a, x = TRUE, y = TRUE, eps = 1e-12, iter.max = 50)
cph_anova <- anova(cph_fit)
cph_con <- contrast(cph_fit, list(age = grid), list(age = 50))

anova_rows <- function(tab, term) {
  list(
    overall = unname(tab[term, c("Chi-Square", "d.f.", "P")]),
    nonlinear = unname(tab[" Nonlinear", c("Chi-Square", "d.f.", "P")])
  )
}

contrast_rows <- function(con) {
  list(estimate = unname(con$Contrast), std_error = unname(con$SE),
       conf_low = unname(con$Lower), conf_high = unname(con$Upper))
}

fits <- list(
  glm = list(
    formula = "y ~ rcs(age, 4) + sex + ns(bmi, 3)", names = names(coef(glm_fit)),
    coef = unname(coef(glm_fit)), se = unname(sqrt(diag(vcov(glm_fit))))
  ),
  lrm = list(
    formula = "y ~ rcs(age, 4) + sex + bmi", term = "rcs(age, 4)",
    coef = unname(coef(lrm_glm)), anova = anova_rows(lrm_anova, "age"),
    grid = grid, reference = 50, contrast = contrast_rows(lrm_con)
  ),
  cox = list(
    formula = "Surv(time, status) ~ rcs(age, 5) + sex + ns(bmi, df = 2)", names = names(coef(cox_fit)),
    coef = unname(coef(cox_fit)), se = unname(sqrt(diag(vcov(cox_fit))))
  ),
  cph = list(
    formula = "Surv(time, status) ~ rcs(age, 4) + sex", term = "rcs(age, 4)",
    coef = unname(coef(cph_fit)), anova = anova_rows(cph_anova, "age"),
    grid = grid, reference = 50, contrast = contrast_rows(cph_con)
  )
)

writeLines(
  toJSON(
    list(
      versions = list(R = as.character(getRversion()), rms = as.character(packageVersion("rms")),
                      survival = as.character(packageVersion("survival"))),
      matrices = matrix_cases, fits = fits
    ),
    digits = NA, auto_unbox = TRUE, pretty = FALSE
  ),
  out_path
)
