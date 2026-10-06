# Time the R estimators on the prepared slices and write python-comparable JSON.

Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
  library(lmtest)
  library(survival)
  library(MatchIt)
  library(mgcv)
  library(lme4)
})

cache <- "/tmp/statract-bench"
manifest <- fromJSON(file.path(cache, "prepared", "manifest.json"), simplifyVector = FALSE)
repeats <- 5

ols_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "sex", "race",
  "marital3", "workclass4", "relationship3", "us_native"
)
glm_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "sex", "race",
  "marital3", "workclass4", "us_native", "hours_per_week"
)
match_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "race", "marital3",
  "workclass4", "relationship3", "us_native", "hours_per_week"
)
surv_x <- c("age", "sex", "num_co", "scoma", "meanbp", "hrt", "temp", "resp", "diabetes", "ca")
star_x <- c(
  "gender", "race", "freelunch", "birthyear", "grade", "classtype", "urban",
  "tyears", "tgen", "classsize"
)

time_call <- function(fun) {
  fun()
  samples <- numeric(repeats)
  result <- NULL
  for (i in seq_len(repeats)) {
    start <- proc.time()[["elapsed"]]
    result <- fun()
    samples[[i]] <- proc.time()[["elapsed"]] - start
  }
  list(seconds = median(samples), result = result)
}

rhs <- function(names) paste(unlist(names), collapse = " + ")

apply_factors <- function(frame, factors) {
  for (name in factors) {
    if (!name %in% names(frame)) next
    values <- as.character(frame[[name]])
    frame[[name]] <- factor(values, levels = sort(unique(values)))
  }
  frame
}

load_slice <- function(info) {
  # Factor columns stay character. read.csv would otherwise turn zero-padded
  # codes such as "01" into integers and change the dummy names.
  factors <- unlist(info$factors)
  header <- names(read.csv(info$csv, nrows = 0, check.names = FALSE))
  classes <- ifelse(header %in% factors, "character", NA)
  frame <- read.csv(
    info$csv,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = classes
  )
  apply_factors(frame, factors)
}

fit_terms <- function(fit) {
  coefs <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  list(
    terms = names(coefs),
    coef = as.numeric(coefs),
    se = as.numeric(se),
    loglik = as.numeric(logLik(fit))
  )
}

ols_terms <- function(fit) {
  tab <- summary(fit)$coefficients
  list(
    terms = rownames(tab),
    coef = as.numeric(tab[, "Estimate"]),
    se = as.numeric(tab[, "Std. Error"]),
    t = as.numeric(tab[, "t value"]),
    p = as.numeric(tab[, "Pr(>|t|)"]),
    loglik = as.numeric(logLik(fit))
  )
}

cov_payload <- function(fit, cov) {
  list(terms = names(coef(fit)), cov = unname(as.matrix(cov)))
}

curve_rows <- function(fit, times, grouped) {
  summary_fit <- summary(fit, times = times, extend = TRUE)
  groups <- if (grouped) sub("^[^=]+=", "", summary_fit$strata) else rep(NA_character_, length(summary_fit$time))
  lapply(seq_along(summary_fit$time), function(i) {
    list(
      group = if (is.na(groups[[i]])) NULL else groups[[i]],
      time = summary_fit$time[[i]],
      estimate = summary_fit$surv[[i]],
      low = summary_fit$lower[[i]],
      high = summary_fit$upper[[i]]
    )
  })
}

pair_payload <- function(matched) {
  matrix <- matched$match.matrix
  pairs <- list()
  if (!is.null(matrix) && length(matrix)) {
    treated <- as.integer(rownames(matrix))
    controls <- as.integer(matrix[, 1])
    keep <- !is.na(controls)
    if (any(keep)) {
      pairs <- lapply(which(keep), function(i) c(treated[[i]], controls[[i]]))
    }
  }
  list(weights = as.numeric(matched$weights), pairs = pairs)
}

run_task <- function(task, frame, manifest) {
  kind <- task$kind
  option <- task$option
  record <- list(id = task$id, slice = task$slice, n = nrow(frame), error = NULL, parts = list())
  tryCatch({
    if (kind == "ols") {
      timed <- time_call(function() lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame))
      record$parts$main <- c(ols_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "ols_w") {
      timed <- time_call(function() lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame, weights = fnlwgt))
      record$parts$main <- c(ols_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_bin") {
      timed <- time_call(function() glm(as.formula(paste("income_gt_50k ~", rhs(glm_x))), data = frame, family = binomial()))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_gamma") {
      timed <- time_call(function() glm(
        as.formula(paste("hours_per_week ~", rhs(ols_x))),
        data = frame,
        family = Gamma(link = "inverse")
      ))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_pois") {
      predictors <- manifest$slices[[task$slice]]$predictors
      timed <- time_call(function() glm(as.formula(paste("cnt ~", rhs(predictors))), data = frame, family = poisson()))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "hc") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      for (hc_type in option$types) {
        timed <- time_call(function() vcovHC(fit, type = hc_type))
        record$parts[[hc_type]] <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
      }
    } else if (kind == "cl1") {
      fit <- lm(as.formula(paste("read ~", rhs(star_x))), data = frame)
      timed <- time_call(function() vcovCL(fit, cluster = frame$school, type = "HC1"))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind == "cl2") {
      fit <- lm(as.formula(paste("read ~", rhs(star_x))), data = frame)
      cluster <- cbind(frame$school, frame$grade)
      timed <- time_call(function() vcovCL(fit, cluster = cluster, type = "HC1"))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind == "nw") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      lag <- floor(4 * (nrow(frame) / 100) ^ (2 / 9))
      timed <- time_call(function() NeweyWest(fit, lag = lag, prewhite = FALSE, adjust = FALSE))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind %in% c("wald", "lr")) {
      full <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      reduced_x <- ols_x[ols_x != "us_native"]
      reduced <- lm(as.formula(paste("hours_per_week ~", rhs(reduced_x))), data = frame)
      if (kind == "wald") {
        timed <- time_call(function() waldtest(full, reduced))
        record$parts$main <- list(stat = timed$result[2, "F"], p = timed$result[2, "Pr(>F)"], seconds = timed$seconds)
      } else {
        timed <- time_call(function() lrtest(full, reduced))
        record$parts$main <- list(
          stat = timed$result[2, "Chisq"],
          p = timed$result[2, "Pr(>Chisq)"],
          seconds = timed$seconds
        )
      }
    } else if (kind == "bp") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      for (studentize in option$studentize) {
        label <- if (isTRUE(studentize)) "student" else "raw"
        timed <- time_call(function() bptest(fit, studentize = studentize))
        record$parts[[label]] <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
      }
    } else if (kind == "reset") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      timed <- time_call(function() resettest(fit, power = 2:3, type = "fitted"))
      record$parts$main <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
    } else if (kind == "dw") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      timed <- time_call(function() dwtest(fit, alternative = "greater"))
      record$parts$main <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
    } else if (kind == "bg") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      for (order in option$orders) {
        timed <- time_call(function() bgtest(fit, order = order, type = "Chisq"))
        record$parts[[as.character(order)]] <- list(
          stat = unname(timed$result$statistic),
          p = unname(timed$result$p.value),
          seconds = timed$seconds
        )
      }
    } else if (kind %in% c("km", "km_sex", "na")) {
      times <- unlist(manifest$km_times)
      grouped <- kind == "km_sex"
      formula <- if (grouped) Surv(d_time, death) ~ sex else Surv(d_time, death) ~ 1
      timed <- time_call(function() {
        if (kind == "na") survfit(formula, data = frame, stype = 2) else survfit(formula, data = frame)
      })
      record$parts$main <- list(rows = curve_rows(timed$result, times, grouped), seconds = timed$seconds)
    } else if (kind == "lrk") {
      specs <- list(
        rho0 = list(formula = Surv(d_time, death) ~ sex, rho = 0),
        rho1 = list(formula = Surv(d_time, death) ~ sex, rho = 1),
        strata = list(formula = Surv(d_time, death) ~ sex + strata(ca), rho = 0)
      )
      for (label in names(specs)) {
        spec <- specs[[label]]
        timed <- time_call(function() survdiff(spec$formula, data = frame, rho = spec$rho))
        stat <- timed$result$chisq
        df <- length(timed$result$n) - 1
        record$parts[[label]] <- list(stat = stat, p = pchisq(stat, df, lower.tail = FALSE), seconds = timed$seconds)
      }
    } else if (kind == "cox") {
      formula_text <- paste("Surv(d_time, death) ~", rhs(surv_x))
      if (isTRUE(option$strata)) formula_text <- paste(formula_text, "+ strata(dzgroup)")
      ties <- option$ties
      timed <- time_call(function() coxph(as.formula(formula_text), data = frame, ties = ties))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "aft") {
      dist <- option$distribution
      formula <- as.formula(paste("Surv(d_time, death) ~", rhs(surv_x)))
      timed <- time_call(function() survreg(formula, data = frame, dist = dist))
      fit <- timed$result
      payload <- fit_terms(fit)
      # coef() omits the scale. Weibull and lognormal report it as Log(scale),
      # and vcov() already carries that extra row.
      if (dist != "exponential") {
        coefs <- coef(fit)
        vc <- vcov(fit)
        if (nrow(vc) != length(coefs) + 1L) {
          stop(paste("survreg vcov rows", nrow(vc), "coef", length(coefs)))
        }
        payload$terms <- c(names(coefs), "Log(scale)")
        payload$coef <- c(as.numeric(coefs), unname(log(fit$scale)))
        payload$se <- as.numeric(sqrt(diag(vc)))
      }
      record$parts$main <- c(payload, list(seconds = timed$seconds))
    } else if (kind == "match") {
      work <- frame
      work$treat <- as.integer(work$sex == "Female")
      rownames(work) <- as.character(seq_len(nrow(work)) - 1L)
      distance <- option$distance
      timed <- time_call(function() {
        if (distance == "exact") {
          matchit(treat ~ race, data = work, method = "exact", estimand = "ATT")
        } else if (distance == "cem") {
          matchit(as.formula(paste("treat ~", rhs(match_x))), data = work, method = "cem", estimand = "ATT")
        } else if (distance == "logit") {
          matchit(
            as.formula(paste("treat ~", rhs(match_x))),
            data = work, method = "nearest", distance = "glm", link = "logit",
            m.order = "data", estimand = "ATT", replace = FALSE
          )
        } else {
          matchit(
            as.formula(paste("treat ~", rhs(match_x))),
            data = work, method = "nearest", distance = "mahalanobis",
            m.order = "data", estimand = "ATT", replace = FALSE
          )
        }
      })
      payload <- pair_payload(timed$result)
      if (distance == "logit") {
        balance <- summary(timed$result)$sum.matched
        payload$smd_age <- unname(balance["age", "Std. Mean Diff."])
      }
      payload$seconds <- timed$seconds
      record$parts$main <- payload
    } else if (kind == "gam") {
      k <- option$k
      timed <- time_call(function() gam(cnt ~ s(temp, bs = "cr", k = k), data = frame, method = "REML"))
      # The default outer Newton stops with a gradient near 1e-3 in log sp,
      # which moves the coefficients by up to 1e-3 on 10,000 rows. The time is
      # the default call; the numbers come from a refit run to convergence.
      converged <- gam(
        cnt ~ s(temp, bs = "cr", k = k), data = frame, method = "REML",
        control = gam.control(newton = list(conv.tol = 1e-12), epsilon = 1e-12)
      )
      record$parts$main <- list(
        coef = as.numeric(coef(converged)),
        sp = unname(converged$sp[[1]]),
        # sum(edf) is the trace of the hat matrix, including the intercept.
        # summary()$s.table is the smooth alone and is short by 1.
        edf = sum(converged$edf),
        reml = unname(converged$gcv.ubre),
        seconds = timed$seconds
      )
    } else if (kind == "lmm") {
      if (isTRUE(option$slopes)) {
        formula <- as.formula(paste("read ~", rhs(star_x), "+ (1 + grade | student)"))
      } else {
        formula <- as.formula(paste("read ~", rhs(star_x), "+ (1 | class_id)"))
      }
      reml <- option$method == "reml"
      timed <- time_call(function() lmer(formula, data = frame, REML = reml))
      fit <- timed$result
      re <- as.matrix(VarCorr(fit)[[1]])
      record$parts$main <- list(
        terms = names(fixef(fit)),
        coef = as.numeric(fixef(fit)),
        se = as.numeric(sqrt(diag(vcov(fit)))),
        re = unname(re),
        sigma2 = unname(sigma(fit)^2),
        loglik = as.numeric(logLik(fit)),
        seconds = timed$seconds
      )
    } else {
      stop(paste("unknown kind", kind))
    }
    record
  }, error = function(err) {
    record$error <- conditionMessage(err)
    record
  })
}

slices <- list()
for (name in names(manifest$slices)) {
  message("load ", name)
  slices[[name]] <- load_slice(manifest$slices[[name]])
}

out <- list()
for (task in manifest$tasks) {
  message("R ", task$id, " ", task$slice)
  out[[length(out) + 1]] <- run_task(task, slices[[task$slice]], manifest)
}

dir.create(file.path(cache, "results"), showWarnings = FALSE, recursive = TRUE)
write_json(out, file.path(cache, "results", "r.json"), auto_unbox = TRUE, digits = 16, null = "null")
failed <- Filter(function(row) !is.null(row$error), out)
message("R failed ", length(failed))
for (row in failed) message("  ", row$id, " ", row$slice, " ", row$error)
