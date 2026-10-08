# Cox regression standardization for tests/r_oracle/test_stdreg_cox.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/stdreg_cox.R
suppressMessages({
  library(survival)
  library(stdReg2)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/stdreg_cox.json"

# summary_std_coxph checks `out$measure` before `out` exists, so the logit and
# odds transforms stop with an error. A global `out` lets that check read the
# survival measure, which is what it means to test.
out <- list(measure = "survival")

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

# statract takes the group of each event from the event row itself. stdReg2
# matches the event time to the first row with that time, which can be a
# censored row or a row of the other group. This copy fixes only that line, so
# the tied case checks everything else against R.
patched_standardize_coxph <- local({
  f <- standardize_coxph
  b <- deparse(body(f))
  i <- grep("Ai <- data[[expname]][match(etimes, survobj[, 1])]", b, fixed = TRUE)
  stopifnot(length(i) == 1)
  b[i] <- paste(
    "ev <- which(survobj[, 2] == 1); ev <- ev[order(survobj[ev, 1])];",
    "ev <- ev[survobj[ev, 1] <= tstar]; Ai <- data[[expname]][ev]"
  )
  body(f) <- parse(text = paste(b, collapse = "\n"))[[1]]
  environment(f) <- asNamespace("stdReg2")
  f
})

run_case <- function(case) {
  d <- read_sample(case$data)
  fun <- if (isTRUE(case$patched)) patched_standardize_coxph else standardize_coxph
  call_one <- function(contrast, transform, ci) {
    args <- list(
      formula = as.formula(case$formula), data = d, values = setNames(list(case$values), case$exposure),
      times = case$times, measure = case$measure, ci_type = ci
    )
    if (!is.null(case$cluster)) args$clusterid <- case$cluster
    if (!is.null(contrast)) {
      args$contrasts <- contrast
      args$reference <- case$reference
    }
    if (!is.null(transform)) args$transforms <- transform
    suppressWarnings(do.call(fun, args))
  }
  base <- call_one(NULL, NULL, "plain")
  tables <- lapply(case$specs, function(spec) {
    fit <- call_one(spec$contrast, spec$transform, spec$ci)
    tab <- tidy(fit)
    want <- if (is.null(spec$contrast)) "none" else spec$contrast
    tab <- tab[tab$contrast == want, ]
    list(
      contrast = spec$contrast, transform = spec$transform, ci = spec$ci,
      time = tab$time, value = tab[[case$exposure]], estimate = tab$Estimate,
      std_error = tab$Std.Error, conf_low = tab[[4]], conf_high = tab[[5]]
    )
  })
  list(
    data = case$data, formula = case$formula, measure = case$measure, exposure = case$exposure,
    values = case$values, times = case$times, cluster = case$cluster, reference = case$reference,
    patched = isTRUE(case$patched),
    estimates = unname(base$res$est),
    covariances = lapply(base$res$vcov, function(v) unname(as.matrix(v))),
    tables = tables
  )
}

spec <- function(contrast = NULL, transform = NULL, ci = "plain") {
  list(contrast = contrast, transform = transform, ci = ci)
}

cases <- list(
  survival_binary = list(
    data = "stdreg_a", formula = "Surv(time, status) ~ ope * age + sex", measure = "survival",
    exposure = "ope", values = c(0, 1), times = c(1, 2, 3, 5), reference = 0,
    specs = list(spec(), spec("difference"), spec("ratio"), spec("ratio", ci = "log"))
  ),
  survival_cluster = list(
    data = "stdreg_a", formula = "Surv(time, status) ~ ope + age + sex", measure = "survival",
    exposure = "ope", values = c(0, 1), times = c(1, 3, 5), cluster = "site", reference = 0,
    specs = list(spec(), spec("difference"))
  ),
  survival_dose = list(
    data = "stdreg_b", formula = "Surv(time, status) ~ dose * age + sex", measure = "survival",
    exposure = "dose", values = c(0, 1, 2), times = c(0.5, 1.5, 3), reference = 0,
    specs = list(spec(), spec("difference"))
  ),
  survival_transform = list(
    data = "stdreg_a", formula = "Surv(time, status) ~ ope + age + sex", measure = "survival",
    exposure = "ope", values = c(0, 1), times = c(2, 4), reference = 0,
    specs = list(
      spec(transform = "log"), spec(transform = "logit"), spec(transform = "odds"),
      spec("difference", "logit"), spec("ratio", "log"), spec(ci = "log")
    )
  ),
  rmean_main = list(
    data = "stdreg_c", formula = "Surv(time, status) ~ ope + age", measure = "rmean",
    exposure = "ope", values = c(0, 1), times = 3, reference = 0,
    specs = list(spec(), spec("difference"), spec("ratio"), spec(transform = "log"))
  ),
  rmean_interaction = list(
    data = "stdreg_c", formula = "Surv(time, status) ~ ope * age + sex", measure = "rmean",
    exposure = "ope", values = c(0, 1), times = 3, reference = 0,
    specs = list(spec(), spec("difference"))
  ),
  rmean_ties = list(
    data = "stdreg_a", formula = "Surv(time, status) ~ ope + age + sex", measure = "rmean",
    exposure = "ope", values = c(0, 1), times = 3, reference = 0, patched = TRUE,
    specs = list(spec(), spec("difference"))
  )
)

result <- list(
  versions = list(
    stdReg2 = as.character(packageVersion("stdReg2")),
    survival = as.character(packageVersion("survival")),
    R = R.version.string
  ),
  cases = lapply(cases, run_case)
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
