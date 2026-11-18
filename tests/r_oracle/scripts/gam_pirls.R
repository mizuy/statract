# Binomial GAMs whose REML optimum needs a weakly penalized fit, for
# tests/r_oracle/test_gam_pirls.py. P-IRLS without step halving diverged
# there and the smoothing parameter search stopped short.
# Data: synthetic (tests/r_oracle/data/gam_pirls_*.csv).
# Run from the repository root: Rscript tests/r_oracle/scripts/gam_pirls.R
suppressMessages({
  library(mgcv)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out <- "tests/r_oracle/fixtures/gam_pirls.json"

one <- function(data, by) {
  d <- read.csv(file.path(data_dir, paste0(data, ".csv")))
  d$tech <- factor(d$tech, levels = c("EMR", "ESD", "UEMR"))
  if (by) {
    f <- gam(y ~ tech + s(diameter, k = 5, bs = "tp", by = tech), family = binomial, data = d, method = "REML")
  } else {
    d <- d[d$tech == "EMR", ]
    f <- gam(y ~ s(diameter, k = 5, bs = "tp"), family = binomial, data = d, method = "REML")
  }
  s <- summary(f)
  list(data = data, by = by, edf = unname(s$edf), chisq = unname(s$s.table[, "Chi.sq"]),
       p_value = unname(s$s.table[, "p-value"]), deviance = deviance(f))
}

cases <- list(
  a_single = one("gam_pirls_a", FALSE),
  a_by = one("gam_pirls_a", TRUE),
  b_by = one("gam_pirls_b", TRUE)
)
writeLines(toJSON(list(cases = cases), digits = NA, auto_unbox = TRUE, pretty = TRUE), out)
