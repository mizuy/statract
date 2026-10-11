# Penalized regression for tests/r_oracle/test_penalized.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/penalized.R
#
# glmnet() paths (gaussian, binomial, poisson, cox) with alpha, weights,
# offset, penalty.factor, standardize, and intercept; the automatic lambda
# sequence and a user lambda grid; coef() and predict() between path lambdas
# (exact = FALSE); and cv.glmnet() with foldid for each type.measure.
suppressMessages({
  library(glmnet)
  library(survival)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/penalized.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

design <- function(case) {
  d <- read_sample(case$data)
  f <- as.formula(case$formula)
  mf <- model.frame(f, d)
  x <- model.matrix(f, d)
  intercept <- attr(terms(f, data = d), "intercept") == 1
  if (intercept) x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  y <- model.response(mf)
  off <- model.offset(mf)
  w <- if (is.null(case$weights)) rep(1, nrow(d)) else d[[case$weights]]
  pf <- rep(1, ncol(x))
  if (!is.null(case$penalty)) for (nm in names(case$penalty)) pf[colnames(x) == nm] <- case$penalty[[nm]]
  list(d = d, x = x, y = y, offset = off, weights = w, intercept = intercept, pf = pf)
}

fit_args <- function(case, des) {
  args <- list(x = des$x, y = des$y, family = case$family, weights = des$weights,
               alpha = if (is.null(case$alpha)) 1 else case$alpha,
               standardize = if (is.null(case$standardize)) TRUE else case$standardize,
               penalty.factor = des$pf)
  if (case$family != "cox") args$intercept <- des$intercept
  if (!is.null(des$offset)) args$offset <- des$offset
  args
}

path_out <- function(fit, is_cox) {
  beta <- as.matrix(fit$beta)
  list(lambda = unname(fit$lambda), a0 = if (is_cox) NULL else unname(fit$a0), beta = unname(beta),
       df = unname(fit$df), dev_ratio = unname(fit$dev.ratio), nulldev = fit$nulldev)
}

run_path <- function(case) {
  des <- design(case)
  is_cox <- case$family == "cox"
  args <- fit_args(case, des)
  fit <- do.call(glmnet, args)
  lam <- fit$lambda
  grid <- exp(seq(log(lam[1] * 0.9), log(lam[1] * 0.002), length.out = 15))
  user <- do.call(glmnet, c(args, list(lambda = grid)))
  refit <- do.call(glmnet, c(args, list(lambda = lam[-1])))
  s <- c(lam[1] * 1.5, lam[2], sqrt(lam[3] * lam[4]), (lam[10] + lam[11]) / 2, lam[length(lam)] / 3)
  newx <- des$x[1:20, , drop = FALSE]
  newoff <- if (is.null(des$offset)) NULL else des$offset[1:20]
  pr <- function(type) {
    a <- list(fit, newx = newx, s = s, type = type)
    if (!is.null(newoff)) a$newoffset <- newoff
    unname(as.matrix(do.call(predict, a)))
  }
  out <- list(
    data = case$data, formula = case$formula, family = case$family, alpha = case$alpha,
    weights = case$weights, penalty = case$penalty, standardize = case$standardize,
    names = colnames(des$x), auto = path_out(fit, is_cox), grid = grid, user = path_out(user, is_cox),
    refit = path_out(refit, is_cox),
    s = s, coef_s = unname(as.matrix(coef(fit, s = s))),
    link = pr("link"), response = pr("response")
  )
  if (case$family == "binomial") out$class <- matrix(as.integer(pr("class")), nrow = 20)
  out
}

run_cv <- function(case) {
  des <- design(case)
  args <- fit_args(case, des)
  args$foldid <- des$d[[case$fold]]
  args$type.measure <- case$measure
  if (!is.null(case$lambda_scale)) {
    ref <- do.call(glmnet, fit_args(case, des))
    args$lambda <- ref$lambda[1] * case$lambda_scale
  }
  cv <- do.call(cv.glmnet, args)
  list(
    data = case$data, formula = case$formula, family = case$family, alpha = case$alpha,
    weights = case$weights, penalty = case$penalty, standardize = case$standardize,
    fold = case$fold, measure = case$measure, lambda = unname(cv$lambda), cvm = unname(cv$cvm),
    cvsd = unname(cv$cvsd), nzero = unname(cv$nzero), lambda_min = cv$lambda.min,
    lambda_1se = cv$lambda.1se, name = unname(cv$name)
  )
}

full_a <- "x1 + x2 + x3 + x4 + x5 + x6 + grp + flag"
paths <- list(
  gaussian_lasso = list(data = "penalized_a", formula = paste("y ~", full_a), family = "gaussian"),
  gaussian_enet_weighted = list(data = "penalized_a", formula = paste("y ~", full_a), family = "gaussian",
                                alpha = 0.5, weights = "w", penalty = list(x1 = 0, x6 = 2, grpc = 0.5)),
  gaussian_ridge_raw = list(data = "penalized_a", formula = paste("y ~", full_a), family = "gaussian",
                            alpha = 0, standardize = FALSE),
  gaussian_noint = list(data = "penalized_a", formula = "y ~ x1 + x2 + x3 + x4 + grp - 1", family = "gaussian",
                        alpha = 0.7),
  gaussian_offset = list(data = "penalized_a", formula = "y ~ x1 + x2 + x3 + x4 + grp + offset(0.5 * x6)",
                         family = "gaussian", weights = "w"),
  gaussian_wide = list(data = "penalized_b", formula = "y ~ . - yb - fold", family = "gaussian"),
  binomial_lasso = list(data = "penalized_a", formula = paste("yb ~", full_a), family = "binomial"),
  binomial_enet_weighted_offset = list(data = "penalized_a",
                                       formula = "yb ~ x1 + x2 + x3 + x4 + x6 + grp + flag + offset(0.3 * x5)",
                                       family = "binomial", alpha = 0.3, weights = "w",
                                       penalty = list(flag = 0, x2 = 3)),
  binomial_noint_raw = list(data = "penalized_a", formula = "yb ~ x1 + x2 + x4 + x6 + grp - 1", family = "binomial",
                            standardize = FALSE),
  binomial_wide = list(data = "penalized_b", formula = "yb ~ . - y - fold", family = "binomial", alpha = 0.8),
  poisson_offset = list(data = "penalized_a", formula = paste("yc ~", full_a, "+ offset(log(exposure))"),
                        family = "poisson"),
  poisson_ridge_weighted = list(data = "penalized_a", formula = paste("yc ~", full_a), family = "poisson",
                                alpha = 0, weights = "w"),
  poisson_noint_raw = list(data = "penalized_a", formula = "yc ~ x1 + x2 + x4 + grp - 1 + offset(log(exposure))",
                           family = "poisson", alpha = 0.5, standardize = FALSE),
  cox_lasso = list(data = "penalized_a", formula = paste("Surv(time, status) ~", full_a), family = "cox"),
  cox_enet_weighted = list(data = "penalized_a", formula = paste("Surv(time, status) ~", full_a), family = "cox",
                           alpha = 0.5, weights = "w", penalty = list(x4 = 0, grpb = 2)),
  cox_offset_raw = list(data = "penalized_a", formula = "Surv(time, status) ~ x1 + x2 + x3 + x4 + grp + flag + offset(0.2 * x6)",
                        family = "cox", alpha = 0.9, standardize = FALSE)
)

cvs <- list(
  cv_gaussian_mse = list(data = "penalized_a", formula = paste("y ~", full_a), family = "gaussian",
                         fold = "fold", measure = "mse"),
  cv_gaussian_mae_weighted = list(data = "penalized_a", formula = paste("y ~", full_a), family = "gaussian",
                                  alpha = 0.5, weights = "w", fold = "fold", measure = "mae"),
  cv_gaussian_wide = list(data = "penalized_b", formula = "y ~ . - yb - fold", family = "gaussian",
                          fold = "fold", measure = "deviance"),
  cv_binomial_deviance = list(data = "penalized_a", formula = paste("yb ~", full_a), family = "binomial",
                              fold = "fold", measure = "deviance"),
  cv_binomial_class = list(data = "penalized_a", formula = paste("yb ~", full_a), family = "binomial",
                           fold = "fold", measure = "class"),
  cv_binomial_auc_weighted = list(data = "penalized_a", formula = paste("yb ~", full_a), family = "binomial",
                                  alpha = 0.5, weights = "w", fold = "fold", measure = "auc"),
  cv_binomial_mse_lambda = list(data = "penalized_a", formula = paste("yb ~", full_a), family = "binomial",
                                fold = "fold", measure = "mse",
                                lambda_scale = exp(seq(log(0.95), log(0.005), length.out = 30))),
  cv_poisson_offset = list(data = "penalized_a", formula = paste("yc ~", full_a, "+ offset(log(exposure))"),
                           family = "poisson", fold = "fold", measure = "deviance"),
  cv_cox = list(data = "penalized_a", formula = paste("Surv(time, status) ~", full_a), family = "cox",
                fold = "fold", measure = "deviance"),
  cv_cox_weighted = list(data = "penalized_a", formula = paste("Surv(time, status) ~", full_a), family = "cox",
                         alpha = 0.5, weights = "w", fold = "fold", measure = "deviance")
)

result <- list(
  versions = list(glmnet = as.character(packageVersion("glmnet")), R = R.version.string),
  paths = lapply(paths, run_path),
  cv = lapply(cvs, run_cv)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null", matrix = "rowmajor"), out_path)
