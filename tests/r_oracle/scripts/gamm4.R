# Additive mixed models for tests/r_oracle/test_gamm4.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/gamm4.R
#
# gamm4 fits the smooth as an iid random effect through glmer (Laplace ML).
# Cases marked tight pass tolPwrss = 1e-13 and tight optimiser tolerances,
# which reach the exact optimum; the default case keeps glmer's defaults.
#
# gamm4 0.2-6 builds Vp (and so edf and every SE) from
# Matrix::chol(V, pivot = TRUE) and reads the permutation from
# attr(R, "pivot"). Matrix 1.6 no longer sets that attribute, so gamm4
# solves against a permuted factor and Vp is off by about 1%. The fixture
# keeps gamm4's reported values and the same quantities with an unpivoted
# Cholesky ("corrected"), which is what gamm4 computes when the pivot is
# honoured.
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

# gamm4's Vp with the Cholesky factor of V itself (no fill-reducing pivot).
corrected_vp <- function(fit) {
  gm <- fit$gam
  mer <- fit$mer
  X <- predict(gm, type = "lpmatrix")
  p <- ncol(X)
  B <- diag(p)
  Sp <- matrix(0, p, p)
  k <- 1
  for (sm in gm$smooth) {
    ii <- sm$first.para:sm$last.para
    B[ii, ii] <- t(sm$trans.D * t(sm$trans.U))
    diag(Sp)[ii[sm$pen.ind == 1]] <- sqrt(gm$sp[k])
    k <- k + 1
  }
  Zt <- getME(mer, "Zt")
  keep <- which(rownames(Zt) %in% levels(d$examiner))
  V <- Diagonal(x = 1 / gm$weights) + crossprod(getME(mer, "Lambdat")[keep, keep] %*% Zt[keep, ])
  R <- Matrix::chol(V)
  WX <- as.matrix(solve(t(R), X %*% B))
  XVX <- crossprod(as.matrix(solve(t(R), X)))
  Ri <- solve(crossprod(WX) + crossprod(Sp))
  Vb <- B %*% Ri %*% t(B)
  list(vp = Vb, edf = rowSums(Vb * t(XVX)))
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
