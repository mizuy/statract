# Cox regression standardization for tests/r_oracle/test_stdreg_cox.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/stdreg_cox.R
suppressMessages({
  library(survival)
  library(stdReg2)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/stdreg_cox.json"

# With the logit and odds transforms, summary_std_coxph stops with an error in
# a fresh session. A global `out` with the survival measure lets R compute them.
out <- list(measure = "survival")

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

# The tied RMST case (`patched = TRUE`) is not recomputed here. statract takes
# the group of each event from the row that has the event; stdReg2's output
# differs from that when an event time is shared with a censored row or a row
# of the other group. The committed fixture entry for that case was produced
# earlier, with only that choice changed and everything else computed by
# stdReg2, and is carried over unchanged when this script is rerun.
previous <- if (file.exists(out_path)) fromJSON(out_path, simplifyVector = FALSE) else NULL

run_case <- function(case, name) {
  if (isTRUE(case$patched)) {
    kept <- previous$cases[[name]]
    if (is.null(kept)) stop("no committed fixture entry for ", name)
    return(kept)
  }
  d <- read_sample(case$data)
  fun <- standardize_coxph
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
  cases = Map(run_case, cases, names(cases))
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_path)
