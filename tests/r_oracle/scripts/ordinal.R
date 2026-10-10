# Cumulative-link models for tests/r_oracle/test_ordinal.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/ordinal.R
#
# MASS::polr and ordinal::clm on the same data. polr is refitted with a tight
# optim reltol so its estimates sit at the optimum; its Hessian is still the
# finite-difference one from optim. clm reports the analytic Hessian.
#
# The Brant test has no R package here (brant is not installed). The function
# brant_by_hand below is a re-implementation of Brant (1990) as the brant
# package computes it: one binomial glm per cutpoint and the cross-fit
# covariance (X'W_m X)^-1 X'W_ml X (X'W_l X)^-1 with W_ml = pi_l - pi_m pi_l.
#
# The cauchit link is left out: polr clamps the linear predictor to +-100 and
# clm also bounds it, which changes the heavy-tailed likelihood, so neither
# reports the exact cauchit optimum.
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
  mf <- model.frame(formula, d)
  y <- as.integer(model.response(mf))
  X <- model.matrix(formula, mf)
  if (is.null(w)) w <- rep(1, nrow(X))
  J <- max(y)
  K <- ncol(X) - 1
  coefs <- matrix(0, J - 1, K + 1)
  pis <- matrix(0, nrow(X), J - 1)
  for (m in 1:(J - 1)) {
    z <- as.numeric(y > m)
    g <- suppressWarnings(glm.fit(X, z, weights = w, family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 100)))
    coefs[m, ] <- g$coefficients
    pis[, m] <- g$fitted.values
  }
  inv <- lapply(1:(J - 1), function(m) solve(t(X) %*% (X * (w * pis[, m] * (1 - pis[, m])))))
  V <- matrix(0, (J - 1) * (K + 1), (J - 1) * (K + 1))
  blk <- function(m) ((m - 1) * (K + 1) + 1):(m * (K + 1))
  for (m in 1:(J - 1)) {
    V[blk(m), blk(m)] <- inv[[m]]
    if (m < J - 1) for (l in (m + 1):(J - 1)) {
      Wml <- w * (pis[, l] - pis[, m] * pis[, l])
      B <- inv[[m]] %*% (t(X) %*% (X * Wml)) %*% inv[[l]]
      V[blk(m), blk(l)] <- B
      V[blk(l), blk(m)] <- t(B)
    }
  }
  slope <- unlist(lapply(1:(J - 1), function(m) blk(m)[-1]))
  bstar <- as.vector(t(coefs[, -1, drop = FALSE]))
  Vstar <- V[slope, slope]
  wald <- function(cols) {
    D <- NULL
    for (m in 2:(J - 1)) for (j in cols) {
      r <- rep(0, (J - 1) * K)
      r[j] <- 1
      r[(m - 1) * K + j] <- -1
      D <- rbind(D, r)
    }
    db <- D %*% bstar
    c(stat = as.numeric(t(db) %*% solve(D %*% Vstar %*% t(D)) %*% db), df = nrow(D))
  }
  res <- rbind(wald(1:K), t(sapply(1:K, function(j) wald(j))))
  list(term = c("Omnibus", colnames(X)[-1]), statistic = unname(res[, 1]), df = unname(res[, 2]),
       p_value = pchisq(res[, 1], res[, 2], lower.tail = FALSE), binary_coefficients = unname(coefs))
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
