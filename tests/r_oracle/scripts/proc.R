# pROC oracle for statract.roc. Run from the repo root:
#   Rscript tests/r_oracle/scripts/proc.R
suppressPackageStartupMessages({
  library(pROC)
  library(jsonlite)
})

out <- "tests/r_oracle/fixtures/proc.json"

make_samples <- function() {
  # balanced, continuous scores, no ties
  set.seed(20261008)
  n <- 200
  y <- rep(c(0, 1), each = n / 2)
  s1 <- round(rnorm(n) + 1.0 * y, 6)
  s2 <- round(0.5 * s1 + rnorm(n) + 0.3 * y, 6)
  yb <- rep(c(0, 1), times = c(70, 50))
  sb <- round(rnorm(120) + 0.7 * yb, 6)
  balanced <- list(y = y, score1 = s1, score2 = s2, y_other = yb, score_other = sb,
                   thresholds = c(-0.5, 0, 0.37, 1.2))

  # unbalanced, many ties, NA, and a predictor with controls > cases
  set.seed(7)
  n <- 150
  y <- rbinom(n, 1, 0.25)
  s1 <- round(rnorm(n) + 0.8 * y, 1)
  s2 <- as.numeric(pmin(pmax(round(3 - 1.2 * y + rnorm(n)), 1), 5))
  s1[c(5, 40)] <- NA
  s2[c(12, 40, 99)] <- NA
  y[77] <- NA
  yb <- rbinom(90, 1, 0.3)
  sb <- round(rnorm(90) + 0.4 * yb, 1)
  ties <- list(y = y, score1 = s1, score2 = s2, y_other = yb, score_other = sb,
               thresholds = c(-1, 0.2, 0.5, 3))

  # small n, perfect separation in score1
  y <- c(0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
  s1 <- c(0.1, 0.25, 0.3, 0.3, 0.42, 0.5, 0.55, 0.6, 0.7, 0.7, 0.85, 0.9)
  s2 <- c(0.2, 0.6, 0.1, 0.4, 0.7, 0.3, 0.5, 0.45, 0.8, 0.35, 0.9, 0.6)
  yb <- c(0, 0, 0, 0, 1, 1, 1, 1, 1)
  sb <- c(1, 2, 2, 3, 2, 3, 4, 4, 5)
  small <- list(y = y, score1 = s1, score2 = s2, y_other = yb, score_other = sb,
                thresholds = c(0.3, 0.575, 0.7))

  list(balanced = balanced, ties = ties, small = small)
}

roc_out <- function(r, thresholds) {
  ci95 <- ci.auc(r, method = "delong", conf.level = 0.95)
  ci90 <- ci.auc(r, method = "delong", conf.level = 0.90)
  best_y <- coords(r, "best", best.method = "youden", transpose = FALSE,
                   ret = c("threshold", "specificity", "sensitivity", "accuracy", "npv", "ppv", "youden"))
  best_t <- coords(r, "best", best.method = "closest.topleft", transpose = FALSE,
                   ret = c("threshold", "specificity", "sensitivity", "closest.topleft"))
  at <- coords(r, thresholds, input = "threshold", transpose = FALSE,
               ret = c("threshold", "specificity", "sensitivity", "accuracy", "npv", "ppv"))
  list(
    direction = r$direction,
    n_controls = length(r$controls),
    n_cases = length(r$cases),
    auc = as.numeric(r$auc),
    var = as.numeric(var(r, method = "delong")),
    ci95 = as.numeric(ci95),
    ci90 = as.numeric(ci90),
    thresholds = r$thresholds,
    sensitivities = r$sensitivities,
    specificities = r$specificities,
    best_youden = as.list(best_y),
    best_topleft = as.list(best_t),
    coords_at = as.list(at)
  )
}

test_out <- function(t) {
  list(
    statistic = unname(as.numeric(t$statistic)),
    p_value = t$p.value,
    df = if (is.null(t$parameter)) NULL else unname(as.numeric(t$parameter)),
    auc1 = unname(as.numeric(t$estimate[1])),
    auc2 = unname(as.numeric(t$estimate[2])),
    method = t$method
  )
}

samples <- make_samples()
res <- list(
  r_version = R.version.string,
  proc_version = as.character(packageVersion("pROC")),
  samples = list()
)
for (name in names(samples)) {
  s <- samples[[name]]
  r1 <- roc(s$y, s$score1, quiet = TRUE)
  r2 <- roc(s$y, s$score2, quiet = TRUE)
  ro <- roc(s$y_other, s$score_other, quiet = TRUE)
  paired <- suppressWarnings(roc.test(r1, r2, method = "delong"))
  unpaired <- suppressWarnings(roc.test(r1, ro, method = "delong"))
  greater <- suppressWarnings(roc.test(r1, r2, method = "delong", alternative = "greater"))
  res$samples[[name]] <- list(
    data = s,
    roc1 = roc_out(r1, s$thresholds),
    roc2 = roc_out(r2, s$thresholds),
    roc_other = roc_out(ro, s$thresholds),
    test_paired = test_out(paired),
    test_unpaired = test_out(unpaired),
    test_paired_greater = test_out(greater)
  )
}

writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), na = "string", null = "null", pretty = TRUE), out)
cat("wrote", out, "\n")
