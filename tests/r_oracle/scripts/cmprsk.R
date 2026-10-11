# Oracle for cumulative_incidence and fine_gray_regression.
# Rscript tests/r_oracle/scripts/cmprsk.R  (writes tests/r_oracle/fixtures/cmprsk.json)
suppressPackageStartupMessages({
  library(cmprsk)
  library(jsonlite)
})

out_path <- file.path("tests", "r_oracle", "fixtures", "cmprsk.json")

make_sample <- function(name, n, seed, ngroup, ncause, round_to, censor_rate, extra = NULL) {
  set.seed(seed)
  x1 <- round(rnorm(n), 3)
  x2 <- rbinom(n, 1, 0.4)
  grp <- sample(letters[1:ngroup], n, replace = TRUE)
  st <- sample(c("s1", "s2"), n, replace = TRUE)
  rate <- exp(0.5 * x1 - 0.4 * x2)
  t_event <- rexp(n, rate)
  cause <- sample(seq_len(ncause), n, replace = TRUE, prob = rev(seq_len(ncause)))
  t_cens <- rexp(n, censor_rate)
  time <- pmin(t_event, t_cens)
  status <- ifelse(t_event <= t_cens, cause, 0L)
  if (!is.null(round_to)) time <- pmax(round(time / round_to) * round_to, round_to)
  d <- data.frame(time = time, status = as.integer(status), x1 = x1, x2 = x2, grp = grp, st = st)
  if (!is.null(extra)) d <- extra(d)
  d
}

run_cuminc <- function(d, by, strata, rho) {
  fit <- if (is.null(strata)) cuminc(d$time, d$status, d[[by]], rho = rho)
         else cuminc(d$time, d$status, d[[by]], d[[strata]], rho = rho)
  tests <- fit$Tests
  curves <- fit[names(fit) != "Tests"]
  grid <- as.numeric(quantile(d$time, c(0.1, 0.25, 0.5, 0.75, 0.9, 1)))
  grid <- c(grid, max(d$time) + 1)
  tp <- timepoints(fit, grid)
  list(
    curves = lapply(names(curves), function(k) list(name = k, time = curves[[k]]$time,
                                                     est = curves[[k]]$est, var = curves[[k]]$var)),
    tests = if (is.null(tests)) NULL else list(cause = rownames(tests), stat = unname(tests[, "stat"]),
                                                pv = unname(tests[, "pv"]), df = unname(tests[, "df"])),
    grid = grid,
    tp_names = rownames(tp$est),
    tp_est = unname(tp$est),
    tp_var = unname(tp$var)
  )
}

run_crr <- function(d, rhs, failcode, cengroup = NULL) {
  mm <- model.matrix(as.formula(paste("~", rhs)), d)[, -1, drop = FALSE]
  fit <- if (is.null(cengroup)) crr(d$time, d$status, mm, failcode = failcode)
         else crr(d$time, d$status, mm, failcode = failcode, cengroup = d[[cengroup]])
  newx <- mm[1:3, , drop = FALSE]
  pr <- predict(fit, newx)
  list(
    rhs = rhs, failcode = failcode, cengroup = cengroup,
    names = I(names(fit$coef)), coef = unname(fit$coef), var = unname(fit$var),
    inf = unname(fit$inf), score = unname(fit$score), loglik = fit$loglik,
    loglik_null = fit$loglik.null, converged = fit$converged, uftime = fit$uftime,
    bfitj = fit$bfitj, res = unname(fit$res),
    pred_time = unname(pr[, 1]), pred = unname(pr[, -1, drop = FALSE])
  )
}

samples <- list()

d1 <- make_sample("balanced", 200, 101, 2, 2, NULL, 0.3)
samples[[1]] <- list(
  name = "balanced", data = d1,
  cuminc = list(list(by = "grp", strata = NULL, rho = 0, fit = run_cuminc(d1, "grp", NULL, 0))),
  crr = list(run_crr(d1, "x1 + x2", 1), run_crr(d1, "x1 + x2", 2))
)

d2 <- make_sample("ties", 300, 202, 3, 3, 0.25, 0.4)
samples[[2]] <- list(
  name = "ties_strata", data = d2,
  cuminc = list(
    list(by = "grp", strata = "st", rho = 0, fit = run_cuminc(d2, "grp", "st", 0)),
    list(by = "grp", strata = NULL, rho = 1, fit = run_cuminc(d2, "grp", NULL, 1))
  ),
  crr = list(run_crr(d2, "x1 + grp", 1), run_crr(d2, "x1 + x2 + grp", 3, cengroup = "st"))
)

d3 <- make_sample("small", 40, 303, 2, 2, 0.5, 0.5, extra = function(d) {
  # The largest time is a cause-1 failure, tied with a censoring and a cause-2 failure.
  m <- max(d$time) + 0.5
  d$time[1:3] <- m
  d$status[1:3] <- c(1L, 0L, 2L)
  d
})
samples[[3]] <- list(
  name = "small_edge", data = d3,
  cuminc = list(
    list(by = "grp", strata = NULL, rho = 0, fit = run_cuminc(d3, "grp", NULL, 0)),
    list(by = NULL, strata = NULL, rho = 0, fit = run_cuminc(transform(d3, one = 1L), "one", NULL, 0))
  ),
  crr = list(run_crr(d3, "x1", 1))
)

out <- list(
  r_version = R.version.string,
  cmprsk_version = as.character(packageVersion("cmprsk")),
  samples = samples
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null", matrix = "rowmajor", dataframe = "columns"), out_path)
cat("wrote", out_path, "\n")
