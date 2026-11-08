# mgcv fits for the bases, tensors, `by`, `select`, and GCV paths of
# statract.gam, for tests/r_oracle/test_gam_fit.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/gam_fit.R
#
# Only public outputs are kept: edf per smooth (s.table), total edf,
# deviance, scale, the minimized criterion, and predictions with standard
# errors on a fixed grid. Smoothing parameters depend on how each penalty is
# scaled, so they are kept for reference only.
suppressMessages({
  library(mgcv)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/gam_b.csv", stringsAsFactors = TRUE)
out <- "tests/r_oracle/fixtures/gam_fit.json"

grid <- data.frame(
  x = c(0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95),
  z = c(0.9, 0.1, 0.5, 0.3, 0.7, 0.4, 0.6),
  noise = c(0.5, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9),
  hour = c(1, 4.5, 8, 12, 15.5, 19, 23),
  grp = factor(c("a", "b", "c", "a", "b", "c", "a"), levels = levels(d$grp)),
  site = factor(c("s01", "s03", "s05", "s07", "s09", "s11", "s12"), levels = levels(d$site))
)

one <- function(formula, family = gaussian(), method = "REML", ...) {
  m <- gam(as.formula(formula), family = family, data = d, method = method, ...)
  st <- summary(m)$s.table
  p <- predict(m, grid, type = "link", se.fit = TRUE)
  list(
    formula = formula, family = family$family, method = method,
    smooths = I(rownames(st)), edf = I(unname(st[, 1])), total_edf = sum(m$edf),
    deviance = deviance(m), scale = m$sig2, criterion = m$gcv.ubre, sp = I(unname(m$sp)),
    predict = unname(as.numeric(p$fit)), predict_se = unname(as.numeric(p$se.fit))
  )
}

cases <- list(
  tp = one("y ~ s(x, bs = 'tp', k = 10)"),
  tp_ml = one("y ~ s(x, bs = 'tp', k = 10)", method = "ML"),
  cc = one("y ~ s(hour, bs = 'cc', k = 10)"),
  ps = one("y ~ s(x, bs = 'ps', k = 10)"),
  re = one("y ~ s(x, bs = 'cr', k = 10) + s(site, bs = 're')"),
  te = one("y ~ te(x, z, k = c(5, 5))"),
  ti = one("y ~ s(x, bs = 'cr', k = 5) + s(z, bs = 'cr', k = 5) + ti(x, z, k = c(5, 5))"),
  by_factor = one("y ~ grp + s(x, by = grp, bs = 'cr', k = 8)"),
  by_numeric = one("y ~ s(x, by = z, bs = 'cr', k = 8)"),
  select = one("y ~ s(x, bs = 'cr', k = 10) + s(noise, bs = 'cr', k = 10)", select = TRUE),
  select_te = one("y ~ s(noise, bs = 'cr', k = 8) + te(x, z, k = c(5, 5))", select = TRUE),
  select_by_factor = one("y ~ grp + s(x, by = grp, bs = 'cr', k = 8)", select = TRUE),
  gcv = one("y ~ s(x, bs = 'cr', k = 10)", method = "GCV.Cp"),
  gcv_gamma = one("y ~ s(x, bs = 'cr', k = 10)", method = "GCV.Cp", gamma = 1.4),
  poisson_by_cc = one("count ~ grp + s(x, by = grp, bs = 'cr', k = 8) + s(hour, bs = 'cc', k = 8)",
                      family = poisson()),
  poisson_select = one("count ~ s(x, bs = 'cr', k = 8) + s(noise, bs = 'cr', k = 8)",
                       family = poisson(), select = TRUE)
)

meta <- list(r = R.version.string, mgcv = as.character(packageVersion("mgcv")))
write_json(list(meta = meta, grid = as.list(grid), cases = cases), out,
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
