# Cumulative-link models for tests/r_oracle/test_ordinal.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/ordinal.R
#
# MASS::polr and ordinal::clm on the same data. polr is refitted with a tight
# optim reltol so its estimates sit at the optimum; its Hessian is still the
# finite-difference one from optim. clm reports the analytic Hessian.
#
# The brant package is not installed in this oracle environment, so the Brant
# test comes from brant_by_hand below, written from Brant (1990, Biometrics
# 46:1171); its output was compared with the brant package.
#
# The cauchit link is left out of the comparison.
suppressMessages({
  library(MASS)
  library(ordinal)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/ordinal.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

polr_method <- c(logit = "logistic", probit = "probit", cloglog = "cloglog", loglog = "loglog", cauchit = "cauchit")

brant_by_hand <- function(formula, d, w = NULL) {
  # Brant (1990): fit one logistic regression per split "response above cut"
  # and test with a Wald statistic whether the slopes agree across splits.
  # Two split fits share the data, so their estimates are correlated; the
  # covariance follows from stacking their score equations (sandwich form).
  frame <- model.frame(formula, d)
  level <- as.integer(model.response(frame))
  design <- model.matrix(formula, frame)
  if (is.null(w)) w <- rep(1, nrow(design))
  n_split <- max(level) - 1
  n_slope <- ncol(design) - 1

  fits <- lapply(seq_len(n_split), function(cut) {
    above <- as.numeric(level > cut)
    suppressWarnings(glm.fit(design, above, weights = w, family = binomial(),
                             control = glm.control(epsilon = 1e-14, maxit = 100)))
  })
  split_coef <- t(sapply(fits, function(f) f$coefficients))  # one row per split
  prob <- sapply(fits, function(f) f$fitted.values)          # one column per split

  bread <- lapply(seq_len(n_split), function(cut)
    solve(crossprod(design, design * (w * prob[, cut] * (1 - prob[, cut])))))
  # For splits a <= b, "above b" implies "above a", so the scores of the two
  # fits have cross moment p_b - p_a p_b per row; with a == b this is p (1 - p).
  split_cov <- function(a, b) {
    upper <- max(a, b)
    meat <- crossprod(design, design * (w * (prob[, upper] - prob[, a] * prob[, b])))
    bread[[a]] %*% meat %*% bread[[b]]
  }
  slot <- function(cut) (cut - 1) * n_slope + seq_len(n_slope)
  slope_cov <- matrix(0, n_split * n_slope, n_split * n_slope)
  for (a in seq_len(n_split)) for (b in seq_len(n_split)) {
    slope_cov[slot(a), slot(b)] <- split_cov(a, b)[-1, -1, drop = FALSE]
  }
  slopes <- as.vector(t(split_coef[, -1, drop = FALSE]))     # split 1 slopes, then split 2, ...

  # Wald test that the slopes in `terms` are equal across splits: contrast
  # every later split with the first one.
  equal_slopes <- function(terms) {
    pairs <- expand.grid(term = terms, split = 2:n_split)
    contrast <- matrix(0, nrow(pairs), n_split * n_slope)
    contrast[cbind(seq_len(nrow(pairs)), pairs$term)] <- 1
    contrast[cbind(seq_len(nrow(pairs)), (pairs$split - 1) * n_slope + pairs$term)] <- -1
    gap <- contrast %*% slopes
    middle <- contrast %*% slope_cov %*% t(contrast)
    c(as.numeric(crossprod(gap, solve(middle, gap))), nrow(contrast))
  }
  tests <- rbind(equal_slopes(seq_len(n_slope)), t(sapply(seq_len(n_slope), equal_slopes)))
  chisq <- tests[, 1]
  df <- tests[, 2]
  list(term = c("Omnibus", colnames(design)[-1]), statistic = unname(chisq), df = unname(df),
       p_value = unname(pchisq(chisq, df, lower.tail = FALSE)), binary_coefficients = unname(split_coef))
}

run_case <- function(case) {
  d <- read_sample(case$data)
  d$y <- factor(d$y, levels = case$levels, ordered = TRUE)
  f <- as.formula(case$formula)
  w <- if (is.null(case$weights)) NULL else d[[case$weights]]
  d$.w <- if (is.null(w)) rep(1, nrow(d)) else w
  m <- clm(f, data = d, weights = .w, link = case$link, control = clm.control(gradTol = 1e-10, maxIter = 500))
  q <- length(m$alpha)
  p <- length(m$beta)
  ord <- c(q + seq_len(p), seq_len(q))
  out <- list(
    data = case$data, formula = case$formula, link = case$link, weights = case$weights,
    levels = case$levels,
    clm = list(
      coefficients = unname(m$beta), thresholds = unname(m$alpha), names = names(m$beta),
      threshold_names = names(m$alpha), vcov = unname(vcov(m)[ord, ord]),
      loglik = as.numeric(logLik(m)), aic = AIC(m), n = m$nobs,
      fitted = unname(predict(m, newdata = m$model[, -1, drop = FALSE], type = "prob")$fit)
    )
  )
  if (case$link %in% names(polr_method)) {
    pf <- polr(f, data = d, weights = .w, method = polr_method[[case$link]], Hess = TRUE,
               control = list(reltol = 1e-14, maxit = 2000))
    out$polr <- list(
      coefficients = unname(coef(pf)), zeta = unname(pf$zeta), vcov = unname(vcov(pf)),
      loglik = as.numeric(logLik(pf)), aic = AIC(pf), deviance = deviance(pf),
      fitted_head = unname(fitted(pf)[1:20, ])
    )
  }
  if (isTRUE(case$brant)) out$brant <- brant_by_hand(f, d, w = if (is.null(case$weights)) NULL else d$.w)
  out
}

lev_a <- c("none", "mild", "moderate", "severe")
lev_c <- c("low", "mid", "high", "top")
cases <- list(
  a_logit = list(data = "ordinal_a", formula = "y ~ x + age + stage", link = "logit", levels = lev_a, brant = TRUE),
  a_probit = list(data = "ordinal_a", formula = "y ~ x + age + stage", link = "probit", levels = lev_a),
  a_cloglog = list(data = "ordinal_a", formula = "y ~ x + age + stage", link = "cloglog", levels = lev_a),
  a_loglog = list(data = "ordinal_a", formula = "y ~ x + age + stage", link = "loglog", levels = lev_a),
  b_weighted = list(data = "ordinal_b", formula = "y ~ x + age + stage", link = "logit", levels = as.character(1:5),
                    weights = "w", brant = TRUE),
  b_probit_weighted = list(data = "ordinal_b", formula = "y ~ x + stage", link = "probit", levels = as.character(1:5),
                           weights = "w"),
  c_interaction = list(data = "ordinal_c", formula = "y ~ x * stage + age", link = "logit", levels = lev_c, brant = TRUE)
)

result <- list(
  versions = list(
    MASS = as.character(packageVersion("MASS")),
    ordinal = as.character(packageVersion("ordinal")),
    R = R.version.string
  ),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
