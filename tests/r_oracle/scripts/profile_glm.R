# Profile-likelihood intervals for tests/r_oracle/test_profile_glm.py (statract.profile_interval).
# Run from the repository root: Rscript tests/r_oracle/scripts/profile_glm.R
#
# confint() on a glm: MASS::confint.glm (R < 4.4) or stats::confint.glm
# (R >= 4.4, the same code moved into stats). It profiles the deviance on a
# grid and interpolates a spline, so the limits carry an interpolation error.
# The script also solves for the exact crossing with uniroot on
# sign(b - b_hat) sqrt((D(b) - D) / phi) = qnorm(level), with the coefficient
# fixed through the offset and glm.fit refitted to epsilon = 1e-14.
suppressPackageStartupMessages({
  library(MASS)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/profile_glm.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

exact_limits <- function(m, level) {
  x <- model.matrix(m)
  y <- m$y
  w <- m$prior.weights
  off <- if (is.null(m$offset)) rep(0, nrow(x)) else m$offset
  fam <- family(m)
  ctrl <- glm.control(epsilon = 1e-14, maxit = 200)
  full <- glm.fit(x, y, weights = w, offset = off, family = fam, control = ctrl,
                  etastart = drop(x %*% coef(m) + off))
  dev0 <- full$deviance
  phi <- summary(m)$dispersion
  se <- sqrt(diag(vcov(m)))
  q <- qnorm(0.5 + level / 2)
  b0 <- full$coefficients
  t(sapply(seq_along(b0), function(j) {
    f <- function(b) {
      o <- off + x[, j] * b
      xj <- x[, -j, drop = FALSE]
      fm <- tryCatch(
        glm.fit(xj, y, weights = w, offset = o, family = fam, control = ctrl,
                etastart = drop(xj %*% b0[-j] + o)),
        error = function(e) glm.fit(xj, y, weights = w, offset = o, family = fam, control = ctrl,
                                    mustart = fitted(m))
      )
      sign(b - b0[j]) * sqrt(max(fm$deviance - dev0, 0) / phi)
    }
    side <- function(sgn) {
      lo <- b0[j]
      d <- q * se[j]
      repeat {
        hi <- b0[j] + sgn * d
        if (sgn * f(hi) >= q) break
        lo <- hi
        d <- 2 * d
      }
      uniroot(function(b) f(b) - sgn * q, sort(c(lo, hi)), tol = 1e-12, maxiter = 500)$root
    }
    c(side(-1), side(1))
  }))
}

run_case <- function(case) {
  d <- read_sample(case$data)
  d$.w <- if (is.null(case$weights)) rep(1, nrow(d)) else d[[case$weights]]
  fam <- switch(case$family, binomial = binomial(), poisson = poisson(), gamma = Gamma(), gaussian = gaussian())
  m <- suppressWarnings(glm(as.formula(case$formula), family = fam, data = d, weights = .w))
  level <- if (is.null(case$level)) 0.95 else case$level
  ci <- suppressMessages(confint(m, level = level))
  exact <- suppressWarnings(exact_limits(m, level))
  list(
    data = case$data, formula = case$formula, family = case$family, weights = case$weights, level = level,
    names = names(coef(m)), coefficients = unname(coef(m)),
    confint_lower = unname(ci[, 1]), confint_upper = unname(ci[, 2]),
    exact_lower = unname(exact[, 1]), exact_upper = unname(exact[, 2]),
    std_error = unname(sqrt(diag(vcov(m))))
  )
}

cases <- list(
  binomial = list(data = "profile_a", formula = "y ~ x + arm", family = "binomial"),
  binomial_weighted = list(data = "profile_a", formula = "y ~ x + arm", family = "binomial", weights = "k"),
  binomial_90 = list(data = "profile_a", formula = "y ~ x * arm", family = "binomial", level = 0.90),
  poisson_offset = list(data = "profile_a", formula = "count ~ x + arm + offset(log(years))", family = "poisson"),
  gamma = list(data = "profile_a", formula = "cost ~ x + arm", family = "gamma"),
  gaussian_weighted = list(data = "profile_a", formula = "score ~ x + arm", family = "gaussian", weights = "w"),
  firth_a_binomial = list(data = "firth_a", formula = "y ~ arm + x + sex", family = "binomial")
)

result <- list(
  versions = list(R = R.version.string, MASS = as.character(packageVersion("MASS"))),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
