# Firth logistic regression for tests/r_oracle/test_firth.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/firth.R
#
# logistf is not installable here (CRAN is not reachable), so this script fits
# Firth's model directly from the papers, in base R:
#   Firth (1993), Biometrika 80, 27-38: the penalty 0.5 * log det I(beta).
#   Heinze and Schemper (2002), Statistics in Medicine 21, 2409-2419: the
#   modified score X'(w (y - p) + h (1/2 - p)), Newton-Raphson with the Fisher
#   information, profile penalized-likelihood intervals, and penalized
#   likelihood-ratio tests whose restricted fits keep the full-design penalty.
# The iteration controls are those described on the help page of
# logistf.control(): maxit = 25, maxstep = 5, maxhs = 0, and 1e-5 for the
# step, the score, and the log-likelihood change. No logistf code is used.
suppressMessages(library(jsonlite))

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/firth.json"

log1pexp <- function(e) ifelse(e > 0, e + log1p(exp(-e)), log1p(exp(e)))

pen_loglik <- function(beta, x, y, w, off) {
  eta <- drop(x %*% beta) + off
  p <- plogis(eta)
  info <- crossprod(x * (w * p * (1 - p)), x)
  ch <- tryCatch(chol(info), error = function(e) NULL)
  if (is.null(ch)) return(-Inf)
  sum(w * (y * eta - log1pexp(eta))) + sum(log(diag(ch)))
}

score_info <- function(beta, x, y, w, off) {
  p <- plogis(drop(x %*% beta) + off)
  v <- w * p * (1 - p)
  info <- crossprod(x * v, x)
  h <- v * rowSums((x %*% solve(info)) * x)
  list(score = drop(crossprod(x, w * (y - p) + h * (0.5 - p))), info = info)
}

# Newton-Raphson over the coefficients in `free`; the others stay at `start`.
newton <- function(x, y, w, off, start, free, maxit, maxstep, maxhs, tol, slack = 0) {
  beta <- start
  ll <- pen_loglik(beta, x, y, w, off)
  if (length(free) == 0) return(list(beta = beta, loglik = ll, iter = 0, converged = TRUE))
  for (it in seq_len(maxit)) {
    si <- score_info(beta, x, y, w, off)
    u <- si$score[free]
    delta <- drop(solve(si$info[free, free, drop = FALSE], u))
    big <- max(abs(delta))
    if (big > maxstep) delta <- delta * (maxstep / big)
    trial <- beta
    trial[free] <- beta[free] + delta
    new <- pen_loglik(trial, x, y, w, off)
    halvings <- 0
    while (halvings < maxhs && new < ll - slack) {
      delta <- delta / 2
      trial <- beta
      trial[free] <- beta[free] + delta
      new <- pen_loglik(trial, x, y, w, off)
      halvings <- halvings + 1
    }
    done <- max(abs(delta)) <= tol && max(abs(u)) < tol && abs(new - ll) < tol
    beta <- trial
    ll <- new
    if (done) return(list(beta = beta, loglik = ll, iter = it, converged = TRUE))
  }
  list(beta = beta, loglik = ll, iter = maxit, converged = FALSE)
}

# Restricted fits are solved tightly, with halving so the likelihood rises.
restricted <- function(x, y, w, off, start, free) {
  newton(x, y, w, off, start, free, maxit = 500, maxstep = 5, maxhs = 60, tol = 1e-10,
         slack = 1e-12 * max(1, sum(w)))
}

profile_limit <- function(drop_at, est, se, q, side) {
  target <- q^2
  dist <- q * se
  lo <- est
  for (k in 1:40) {
    hi <- est + side * dist
    if (drop_at(hi) >= target) break
    lo <- hi
    dist <- dist * 2
  }
  f <- function(b) drop_at(b) - target
  interval <- sort(c(lo, hi))
  uniroot(f, interval, tol = 1e-13, maxiter = 1000)$root
}

run_case <- function(case) {
  d <- read.csv(file.path(data_dir, paste0(case$data, ".csv")))
  f <- as.formula(case$formula)
  mf <- model.frame(f, d)
  x <- model.matrix(f, d)
  y <- model.response(mf)
  off <- rep(0, nrow(x))
  w <- if (is.null(case$weights)) rep(1, nrow(x)) else d[[case$weights]]
  maxit <- if (is.null(case$maxit)) 25 else case$maxit
  level <- if (is.null(case$level)) 0.95 else case$level
  p <- ncol(x)
  fit <- newton(x, y, w, off, rep(0, p), seq_len(p), maxit = maxit, maxstep = 5, maxhs = 0, tol = 1e-5)
  stopifnot(fit$converged)
  beta <- fit$beta
  vcov <- solve(score_info(beta, x, y, w, off)$info)
  se <- sqrt(diag(vcov))
  lr <- numeric(p)
  low <- numeric(p)
  high <- numeric(p)
  q <- qnorm(0.5 + level / 2)
  for (j in seq_len(p)) {
    free <- setdiff(seq_len(p), j)
    start <- beta
    start[j] <- 0
    lr[j] <- max(2 * (fit$loglik - restricted(x, y, w, off, start, free)$loglik), 0)
    for (side in c(-1, 1)) {
      warm <- beta
      drop_at <- function(b) {
        s <- warm
        s[j] <- b
        r <- restricted(x, y, w, off, s, free)
        if (is.finite(r$loglik)) warm <<- r$beta
        2 * (fit$loglik - r$loglik)
      }
      limit <- profile_limit(drop_at, beta[j], se[j], q, side)
      if (side < 0) low[j] <- limit else high[j] <- limit
    }
  }
  has_int <- "(Intercept)" %in% colnames(x)
  if (has_int) {
    null <- restricted(x, y, w, off, rep(0, p), which(colnames(x) == "(Intercept)"))
    loglik_null <- null$loglik
  } else {
    loglik_null <- pen_loglik(rep(0, p), x, y, w, off)
  }
  global_df <- p - as.integer(has_int)
  glm_fit <- suppressWarnings(glm(f, data = d, family = binomial(), weights = w))
  list(
    data = case$data, formula = case$formula, weights = case$weights, level = level, maxit = maxit,
    names = colnames(x), coefficients = unname(beta), iter = fit$iter, vcov = unname(vcov),
    loglik = fit$loglik, loglik_null = loglik_null, lr_stat = lr, prob = pchisq(lr, 1, lower.tail = FALSE),
    ci_lower = low, ci_upper = high, global_stat = 2 * (fit$loglik - loglik_null), global_df = global_df,
    fitted = unname(plogis(drop(x %*% beta) + off)), glm_coefficients = unname(coef(glm_fit))
  )
}

cases <- list(
  firth_a = list(data = "firth_a", formula = "y ~ arm + x + sex"),
  firth_a_90 = list(data = "firth_a", formula = "y ~ arm + x + sex", level = 0.9),
  firth_b_weighted = list(data = "firth_b", formula = "y ~ arm + x + sex", weights = "w"),
  firth_b_interaction = list(data = "firth_b", formula = "y ~ arm * x"),
  firth_sep_quasi = list(data = "firth_sep", formula = "y ~ g + z"),
  firth_sep_complete = list(data = "firth_sep", formula = "yc ~ z", maxit = 200),
  firth_noint = list(data = "firth_a", formula = "y ~ 0 + arm + x")
)

result <- list(
  versions = list(R = R.version.string, method = "Firth (1993) and Heinze and Schemper (2002), in base R"),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null", matrix = "rowmajor"), out_path)
