# Deviance, deviance explained, adjusted R-squared, and nested-model deviance
# changes from mgcv, for tests/r_oracle/test_gam_summary.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/gam_summary.R
#
# Values come from the public summary.gam fields (dev.expl, r.sq), deviance(),
# null.deviance, and anova(..., test = ...). mgcv's anova takes the df change
# from a smoothing-parameter-uncertainty corrected edf, so the fixture keeps
# its df and p-value for reference only; statract uses the plain edf.
suppressMessages({
  library(mgcv)
  library(jsonlite)
})

b <- read.csv("tests/r_oracle/data/gamm_binary_a.csv")
b$off <- log(b$age / 50)
g <- read.csv("tests/r_oracle/data/gam_a.csv")
out <- "tests/r_oracle/fixtures/gam_summary.json"

one <- function(m) {
  s <- summary(m)
  list(deviance = deviance(m), null_deviance = m$null.deviance, deviance_explained = s$dev.expl,
       adj_r_squared = s$r.sq, edf = sum(m$edf))
}
pair <- function(m1, m2, test) {
  a <- anova(m1, m2, test = test)
  list(small = one(m1), large = one(m2), deviance_change = a$Deviance[2],
       mgcv_df = a$Df[2], mgcv_p_value = a[2, ncol(a)])
}

b1 <- gam(y ~ s(pre_size_mm, k = 10, bs = "cr"), family = binomial, data = b, method = "REML")
b2 <- gam(y ~ s(pre_size_mm, k = 10, bs = "cr") + s(age, k = 10, bs = "cr"), family = binomial, data = b, method = "REML")
g1 <- gam(y ~ s(x, k = 10, bs = "cr"), weights = w, data = g, method = "REML")
g2 <- gam(y ~ s(x, k = 10, bs = "cr") + s(z, k = 10, bs = "cr"), weights = w, data = g, method = "REML")

cases <- list(
  binomial = pair(b1, b2, "Chisq"),
  gaussian_weighted = pair(g1, g2, "F"),
  poisson_offset = one(gam(n ~ s(age, k = 8, bs = "cr") + offset(off), family = poisson, data = b, method = "REML")),
  gaussian = one(gam(y ~ s(x, k = 10, bs = "cr"), data = g, method = "REML")),
  gamma = one(gam(cost ~ s(x, k = 8, bs = "cr"), family = Gamma, data = g, method = "REML"))
)

meta <- list(r = R.version.string, mgcv = as.character(packageVersion("mgcv")))
write_json(list(meta = meta, cases = cases), out, digits = NA, auto_unbox = TRUE, pretty = TRUE)
