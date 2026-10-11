# epiR::epi.tests oracle for statract.models.diagnostic. Run from the repo root:
#   Rscript tests/r_oracle/scripts/diagnostic.R
# epiR 2.0.67. The table is test (row) by disease (column): tp, fp / fn, tn.
suppressPackageStartupMessages({
  library(epiR)
  library(jsonlite)
})

out <- "tests/r_oracle/fixtures/diagnostic.json"

tables <- list(
  epir_example = c(tp = 670, fp = 202, fn = 74, tn = 640),
  bleeding_1000 = c(tp = 23, fp = 259, fn = 12, tn = 706),
  small = c(tp = 9, fp = 3, fn = 2, tn = 17),
  zero_fp = c(tp = 12, fp = 0, fn = 4, tn = 25)
)
methods <- c("wilson", "exact", "clopper-pearson", "agresti", "jeffreys")

cases <- list()
for (tn in names(tables)) {
  t <- tables[[tn]]
  dat <- as.table(matrix(unname(t), nrow = 2, byrow = TRUE))
  for (m in methods) {
    for (cl in c(0.95, 0.90)) {
      r <- epi.tests(dat, method = m, conf.level = cl)$detail
      cases[[length(cases) + 1]] <- list(
        table = tn, counts = as.list(t), method = m, conf_level = cl,
        statistic = r$statistic, est = r$est, lower = r$lower, upper = r$upper
      )
    }
  }
}

writeLines(toJSON(list(epiR = as.character(packageVersion("epiR")), cases = cases),
                  digits = NA, auto_unbox = TRUE, na = "string", pretty = TRUE), out)
