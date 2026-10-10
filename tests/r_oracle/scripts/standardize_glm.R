# GLM standardization (g-computation) for tests/r_oracle/test_standardize_glm.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/standardize_glm.R
#
# covariates = "fixed": marginaleffects 0.18 avg_predictions / avg_comparisons
# with type = "response" and Richardson derivatives (the default forward
# difference is accurate to only about 1e-5).
#
# covariates = "sampled": HAND-CODED RE-IMPLEMENTATION, NOT stdReg2 ITSELF.
# stdReg2 is not installable here (CRAN is blocked). stacked_vcov() below
# writes out the estimator that stdReg2::standardize_glm documents (Sjolander
# 2016): stack w * (mu_k - theta_k) with the GLM score, take var() of the
# per-row (or per-cluster) estimating functions, and form
# A^-1 J A^-T * m / n^2 (m rows or clusters). Only base R, stats, and sandwich
# (for estfun) are used.
suppressMessages({
  library(marginaleffects)
  library(sandwich)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/standardize_glm.json"
read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

fit_model <- function(case, d) {
  args <- list(formula = as.formula(case$formula), family = get(case$family), data = d)
  if (!is.null(case$weights)) args$weights <- d[[case$weights]]
  suppressWarnings(do.call(glm, args))
}

set_value <- function(d, exposure, v) {
  d[[exposure]] <- if (is.character(d[[exposure]])) as.character(v) else v
  d
}

stacked_vcov <- function(m, d, case, cluster) {
  x <- model.matrix(m)
  n <- nrow(x)
  k <- length(case$values)
  w <- if (is.null(case$weights)) rep(1, n) else d[[case$weights]]
  rhs <- delete.response(terms(m))
  preds <- matrix(0, n, k)
  dtheta <- matrix(0, k, ncol(x))
  for (j in seq_len(k)) {
    nd <- set_value(d, case$exposure, case$values[[j]])
    xj <- model.matrix(rhs, model.frame(rhs, nd, xlev = m$xlevels))
    eta <- predict(m, newdata = nd, type = "link")
    preds[, j] <- m$family$linkinv(eta)
    dtheta[j, ] <- colMeans(w * m$family$mu.eta(eta) * xj)
  }
  theta <- colSums(w * preds) / sum(w)
  # estfun.glm divides by the dispersion; it is 1 for binomial and poisson.
  score <- estfun(m) * summary(m)$dispersion
  info <- -crossprod(x * weights(m, "working"), x) / n
  res <- cbind(w * sweep(preds, 2, theta), score)
  if (!is.null(cluster)) res <- rowsum(res, d[[cluster]])
  J <- var(res)
  A <- rbind(cbind(-mean(w) * diag(k), dtheta), cbind(matrix(0, ncol(x), k), info))
  Ainv <- solve(A)
  V <- (Ainv %*% J %*% t(Ainv) * nrow(res) / n^2)[1:k, 1:k, drop = FALSE]
  list(estimates = theta, covariance = V)
}

me_vcov <- function(spec) {
  if (!is.null(spec$cluster)) return(as.formula(paste0("~", spec$cluster)))
  if (is.null(spec$vcov)) return(TRUE)
  spec$vcov
}

odds_ratio_avg <- function(hi, lo, w) {
  h <- weighted.mean(hi, w)
  l <- weighted.mean(lo, w)
  (h / (1 - h)) / (l / (1 - l))
}

run_fixed <- function(case, m, d, spec) {
  vc <- me_vcov(spec)
  d$.one <- 1
  wts <- if (is.null(case$weights)) ".one" else case$weights
  var_list <- setNames(list(case$values), case$exposure)
  p <- avg_predictions(m, newdata = d, variables = var_list, by = case$exposure, type = "response",
                       vcov = vc, wts = wts, numderiv = "richardson")
  pd <- as.data.frame(p)
  pos <- match(as.character(case$values), as.character(pd[[case$exposure]]))
  cmp_var <- if (length(case$values) == 2) case$exposure else setNames(list("reference"), case$exposure)
  one <- function(contrast, ci) {
    comparison <- switch(paste(contrast, ci),
      "difference plain" = "difference",
      "ratio plain" = "ratioavg",
      "ratio log" = "lnratioavg",
      "odds_ratio plain" = odds_ratio_avg,
      "odds_ratio log" = "lnoravg"
    )
    args <- list(m, newdata = d, variables = cmp_var, comparison = comparison, type = "response",
                 vcov = vc, wts = wts, numderiv = "richardson")
    if (ci == "log") args$transform <- "exp"
    r <- as.data.frame(do.call(avg_comparisons, args))
    list(contrast = contrast, ci = ci, value = case$values[-1], estimate = r$estimate,
         std_error = if (ci == "log") NULL else r$std.error, conf_low = r$conf.low, conf_high = r$conf.high)
  }
  specs <- list(c("difference", "plain"), c("ratio", "plain"), c("ratio", "log"))
  if (case$family == "binomial") specs <- c(specs, list(c("odds_ratio", "plain"), c("odds_ratio", "log")))
  list(
    estimates = pd$estimate[pos],
    covariance = unname(vcov(p))[pos, pos, drop = FALSE],
    tables = c(
      list(list(contrast = NULL, ci = "plain", value = case$values, estimate = pd$estimate[pos],
                std_error = pd$std.error[pos], conf_low = pd$conf.low[pos], conf_high = pd$conf.high[pos])),
      lapply(specs, function(s) one(s[1], s[2]))
    )
  )
}

run_case <- function(case) {
  d <- read_sample(case$data)
  m <- fit_model(case, d)
  case$variances <- lapply(case$variances, function(spec) {
    out <- if (identical(spec$covariates, "sampled")) {
      stacked_vcov(m, d, case, spec$cluster)
    } else {
      run_fixed(case, m, d, spec)
    }
    c(spec, out)
  })
  case
}

fixed <- function(vcov = NULL, cluster = NULL) list(covariates = "fixed", vcov = vcov, cluster = cluster)
sampled <- function(cluster = NULL) list(covariates = "sampled", vcov = NULL, cluster = cluster)

cases <- list(
  binary_main = list(
    data = "stdglm_a", formula = "y ~ trt + age + sex", family = "binomial",
    exposure = "trt", values = c(0, 1),
    variances = list(fixed(), fixed("HC0"), fixed("HC3"), fixed(cluster = "site"), sampled(), sampled("site"))
  ),
  binary_interaction = list(
    data = "stdglm_a", formula = "y ~ trt * age + sex", family = "binomial",
    exposure = "trt", values = c(0, 1),
    variances = list(fixed(), fixed("HC3"), sampled())
  ),
  factor_main = list(
    data = "stdglm_a", formula = "y ~ arm + age + sex", family = "binomial",
    exposure = "arm", values = c("a", "b", "c"),
    variances = list(fixed(), fixed(cluster = "site"), sampled(), sampled("site"))
  ),
  factor_interaction = list(
    data = "stdglm_b", formula = "y ~ arm * sex + age", family = "binomial",
    exposure = "arm", values = c("a", "b", "c"),
    variances = list(fixed(), fixed("HC0"), sampled())
  ),
  poisson_offset = list(
    data = "stdglm_a", formula = "count ~ trt * age + arm + offset(log(years))", family = "poisson",
    exposure = "trt", values = c(0, 1),
    variances = list(fixed(), fixed("HC3"), fixed(cluster = "site"), sampled())
  ),
  weighted = list(
    data = "stdglm_b", formula = "y ~ trt + age + sex", family = "binomial",
    exposure = "trt", values = c(0, 1), weights = "w",
    variances = list(fixed(), fixed("HC0"), sampled(), sampled("site"))
  )
)

out <- list(
  r_version = R.version.string,
  marginaleffects = as.character(packageVersion("marginaleffects")),
  sandwich = as.character(packageVersion("sandwich")),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), out_path)
cat("wrote", out_path, "\n")
