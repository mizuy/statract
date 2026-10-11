# Additive mixed models for tests/r_oracle/test_gamm4.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/gamm4.R
#
# gamm4 fits the smooth as an iid random effect through glmer (Laplace ML).
# Cases marked tight pass tolPwrss = 1e-13 and tight optimiser tolerances,
# which reach the exact optimum; the default case keeps glmer's defaults.
#
# With Matrix >= 1.6, the edf and SEs that gamm4 0.2-6 reports differ by
# about 1% from the mixed-model formula below. The fixture keeps gamm4's
# reported values and the formula's values ("corrected"); statract follows
# the formula.
suppressMessages({
  library(gamm4)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/gamm_binary_a.csv")
d$examiner <- factor(d$examiner)
out <- "tests/r_oracle/fixtures/gamm4.json"
grid <- list(pre_size_mm = c(5, 10, 20, 30, 38), age = c(45, 55, 60, 70, 80))

tight <- glmerControl(optimizer = "nloptwrap", tolPwrss = 1e-13,
                      optCtrl = list(xtol_abs = 1e-12, ftol_abs = 1e-14, xtol_rel = 0, ftol_rel = 0, maxeval = 1e5))

# Bayesian posterior covariance of the GAM coefficients from the mixed-model
# representation of the penalized fit (Wood 2017, sec. 6.6 and 6.10-6.11;
# Wood & Scheipl, ?gamm4 "Value"), built with an unpivoted Cholesky factor of
# the marginal working covariance V.
#
# In the mixed-model parametrization the model matrix is A = [Xf, Zs Ls]:
# the unpenalized columns Xf = getME(mer, "X"), and the smooths' random-effect
# columns Zs scaled by their relative covariance factor Ls, so that their
# coefficients u_s have prior N(0, I) (binomial and Poisson: dispersion 1).
# The examiner intercept stays random: V = diag(1 / w) + Ze Le Le' Ze', with
# w the working weights at convergence. Then
#   Vu = (A' V^-1 A + blockdiag(0, I))^-1.
# A spans the same space as the gam model matrix X (lpmatrix), A = X M, so the
# coefficients map as beta = M (beta_f, u_s) and
#   Vp = M Vu M'  ( = (X' V^-1 X + S_lambda)^-1 when M is invertible ),
#   edf = diag(Vp X' V^-1 X).
# M is found by least squares, which undoes the smooths' reparametrization
# without reading it from the fitted objects.
corrected_vp <- function(fit) {
  mer <- fit$mer
  x <- predict(fit$gam, type = "lpmatrix")
  xf <- as.matrix(lme4::getME(mer, "X"))
  z <- as.matrix(lme4::getME(mer, "Z"))
  lambda <- as.matrix(lme4::getME(mer, "Lambda"))
  gp <- lme4::getME(mer, "Gp")
  groups <- names(lme4::getME(mer, "cnms"))
  term_cols <- lapply(seq_along(groups), function(j) seq.int(gp[j] + 1, gp[j + 1]))
  is_examiner <- groups == "examiner"
  stopifnot(sum(is_examiner) == 1)
  ce <- term_cols[[which(is_examiner)]]
  cs <- unlist(term_cols[!is_examiner])
  w <- weights(mer, type = "working")
  stopifnot(nrow(x) == nrow(z), length(w) == nrow(x), ncol(xf) + length(cs) == ncol(x))

  ze <- z[, ce, drop = FALSE] %*% lambda[ce, ce, drop = FALSE]
  v <- diag(1 / w, nrow(x)) + tcrossprod(ze)
  r <- chol(v)  # upper triangular, V = R'R, no pivoting
  a <- cbind(xf, z[, cs, drop = FALSE] %*% lambda[cs, cs, drop = FALSE])
  ra <- backsolve(r, a, transpose = TRUE)  # R'^-1 A
  rx <- backsolve(r, x, transpose = TRUE)  # R'^-1 X
  prior <- diag(c(rep(0, ncol(xf)), rep(1, length(cs))), ncol(a))
  vu <- solve(crossprod(ra) + prior)
  m <- qr.coef(qr(x), a)
  stopifnot(max(abs(x %*% m - a)) < 1e-8 * max(1, max(abs(a))))
  vp <- m %*% vu %*% t(m)
  vp <- (vp + t(vp)) / 2
  edf <- rowSums(vp * t(crossprod(rx)))
  list(vp = vp, edf = edf)
}

run <- function(formula, family, smooths, tight_control = TRUE, predictors = list()) {
  fit <- if (tight_control) {
    gamm4(as.formula(formula), random = ~ (1 | examiner), family = family, data = d, control = tight)
  } else {
    gamm4(as.formula(formula), random = ~ (1 | examiner), family = family, data = d)
  }
  s <- summary(fit$gam)
  v <- as.numeric(VarCorr(fit$mer)$examiner)
  nd <- data.frame(pre_size_mm = grid$pre_size_mm, age = grid$age)
  terms <- predict(fit$gam, newdata = nd, type = "terms", se.fit = TRUE)
  smooth_fits <- lapply(seq_along(smooths), function(i) {
    label <- rownames(s$s.table)[i]
    list(column = smooths[[i]]$column, label = label,
         fit = unname(terms$fit[, label]), std_error = unname(terms$se.fit[, label]))
  })
  re <- ranef(fit$mer)$examiner
  fixed <- corrected_vp(fit)
  lp <- predict(fit$gam, newdata = nd, type = "lpmatrix")
  corrected <- list(
    std_errors = unname(sqrt(diag(fixed$vp))[seq_len(nrow(s$p.table))]),
    edf = unname(sapply(fit$gam$smooth, function(sm) sum(fixed$edf[sm$first.para:sm$last.para]))),
    smooth_std_errors = lapply(fit$gam$smooth, function(sm) {
      ii <- sm$first.para:sm$last.para
      unname(sqrt(rowSums((lp[, ii] %*% fixed$vp[ii, ii]) * lp[, ii])))
    })
  )
  list(
    formula = formula, family = family$family, tight = tight_control,
    smooths = smooths, predictors = predictors,
    terms = rownames(s$p.table), coefficients = unname(s$p.table[, 1]), std_errors = unname(s$p.table[, 2]),
    edf = unname(s$edf), log_likelihood = as.numeric(logLik(fit$mer)),
    examiner_variance = v, median_odds_ratio = exp(sqrt(2 * v) * qnorm(0.75)),
    smooth_fits = smooth_fits,
    link = unname(predict(fit$gam, newdata = nd, type = "link")),
    blup_levels = rownames(re), blup = unname(re[, 1]),
    corrected = corrected
  )
}

tp <- list(list(column = "pre_size_mm", basis = "tp", k = 10))
cases <- list(
  binomial_tp = run("y ~ s(pre_size_mm)", binomial(), tp),
  binomial_tp_default = run("y ~ s(pre_size_mm)", binomial(), tp, tight_control = FALSE),
  binomial_cr = run("y ~ s(pre_size_mm, bs = 'cr', k = 8)", binomial(),
                    list(list(column = "pre_size_mm", basis = "cr", k = 8))),
  binomial_age_two_smooths = run("y ~ s(pre_size_mm) + s(age, bs = 'cr', k = 6)", binomial(),
                                 list(list(column = "pre_size_mm", basis = "tp", k = 10),
                                      list(column = "age", basis = "cr", k = 6))),
  binomial_age_linear = run("y ~ age + s(pre_size_mm, k = 6)", binomial(),
                            list(list(column = "pre_size_mm", basis = "tp", k = 6)), predictors = list("age")),
  poisson_tp = run("n ~ s(pre_size_mm)", poisson(), tp)
)
meta <- list(r = R.version.string, gamm4 = as.character(packageVersion("gamm4")),
             mgcv = as.character(packageVersion("mgcv")), lme4 = as.character(packageVersion("lme4")))
write_json(list(meta = meta, grid = grid, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null")
